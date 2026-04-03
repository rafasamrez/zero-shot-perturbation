"""Zero-shot in-silico latent-space perturbation pipeline for EVA-RNA.

Implements equations (19)–(20) from the EVA technical report (Scienta Lab, 2026):

    z'  = z + ∇_z L_pert     (eq. 19)   z = gene_embeddings (B, S, H)
    x'  = f_dec(z')           (eq. 20)   x' = decoded expression (B, S)

Here z refers to the *gene-level* hidden states (output of the transformer
encoder, shape (B, S, H)), which are the direct input to the expression
decoder — not the CLS embedding.  Perturbing z and decoding yields a predicted
expression profile x' that can be compared to healthy reference expressions.

For each drug-disease pair in the benchmark matrix, this script:
  1. Loads the disease cohort (bulk RNA-seq, FPKM → log1p(CPM×1e4)).
  2. Prepares tokenisation: gene vocab filtering, token ID tensor (shared
     across all samples in a cohort, mirroring utils.encode_from_anndata).
  3. Encodes healthy samples → decodes → builds healthy expression centroid.
  4. For each disease sample (processed in batches for the forward pass,
     one-by-one for the backward pass):
       a. Slice precomputed gene_embeddings z[i]  (shape 1, S, H)
       b. Decode → predicted expression x  (shape 1, S)  [under enable_grad]
       c. Compute perturbation loss, backpropagate to get ∇_z L
       d. L2-normalise ∇_z per sample (EVA report eq. 25)
       e. z' = z + ∇_z L  (gradient ascent, eq. 19)
       f. x' = model.decode(z')  (eq. 20)
  5. Scores each sample by its shift toward the healthy expression centroid
     (delegated to scoring.compute_shift_score — currently NotImplementedError).
  6. Saves per-patient decoded expressions (.npy) and a summary CSV.

Usage
-----
    python perturb.py
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import torch
from scipy.sparse import issparse
from huggingface_hub import hf_hub_download
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm


from eva_rna.utils import _normalize_and_log

from gene_alias_map import MissingTargetGenesList, GENE_ALIAS_MAP
from encode_and_save import load_cohort_data
from gradient_flow_pert_loss import perturbation_loss
from scoring import compute_healthy_centroid, compute_shift_score

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BENCHMARK_PATH = "data/benchmark_drug_target_disease_matrix.csv"
BATCH_SIZE     = 16      # samples per forward pass (backward is always per-sample)
N_TOP_GENES    = 2500    # HVG subset, matching encode_and_save.py
EPS            = 1e-8    # numerical stability for gradient L2 normalisation
PERTURBATION_DIR = -1    # δ = -1: knockdown for all targets (mission statement)
_TARGET_SUM    = 1e4     # library-size normalisation target, matching utils.py
OUTPUT_DIR     = Path(f"data/perturbation_scores_v2/{N_TOP_GENES}_top_genes")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)



# ---------------------------------------------------------------------------
# Tokenisation helpers (mirroring utils.encode_from_anndata)
# ---------------------------------------------------------------------------

def prepare_tokenisation(
    adata: ad.AnnData,
    tokenizer,
    device: torch.device,
) -> tuple[np.ndarray, torch.Tensor, list[int]]:
    """Filter genes to the model vocabulary and build the shared token ID tensor.

    Mirrors the vocab-filtering and tokenisation logic in
    ``utils.encode_from_anndata``:
      1. Extract gene IDs from ``adata.var`` (Entrez IDs as strings in index).
      2. Filter to genes present in the tokenizer vocabulary.
      3. Convert to integer token IDs via ``tokenizer.convert_tokens_to_ids``.
      4. Preprocess and filter the expression matrix to the same gene subset.

    Parameters
    ----------
    adata : ad.AnnData
        AnnData cohort (HVG-filtered).  Gene Entrez IDs are in ``adata.var_names``.
    tokenizer : EvaRnaTokenizer
        EVA-RNA tokenizer.
    device : torch.device
        Target device for the token ID tensor.

    Returns
    -------
    X_filtered : np.ndarray
        Log-normalised expression matrix filtered to vocab genes,
        shape ``(n_samples, n_vocab_genes)``, dtype float32.
    token_ids_tensor : torch.Tensor
        Shared integer token IDs, shape ``(n_vocab_genes,)``.
        Expand to ``(batch, n_vocab_genes)`` before passing to the model.
    gene_indices : list[int]
        Column indices into the original ``adata.X`` that were kept.
    """
    gene_ids = adata.var_names.astype(str).tolist()

    # Vocab filtering — same as utils.py lines 197–231
    gene_mask    = [tokenizer.gene_in_vocab(g) for g in gene_ids]
    gene_indices = [i for i, m in enumerate(gene_mask) if m]
    n_matched    = len(gene_indices)

    if n_matched == 0:
        raise ValueError(
            "No genes in adata match the model vocabulary.  Check that "
            "adata.var_names contains NCBI Entrez IDs (e.g. '7157' for TP53)."
        )

    filtered_gene_ids = [gene_ids[i] for i in gene_indices]
    token_ids = tokenizer.convert_tokens_to_ids(filtered_gene_ids)
    if isinstance(token_ids, int):
        token_ids = [token_ids]
    token_ids_tensor = torch.tensor(token_ids, dtype=torch.long, device=device)

    # Expression matrix: densify if sparse, filter columns, normalise
    X = adata.X
    if issparse(X):
        X = X.toarray()
    X = X[:, gene_indices]
    X = _normalize_and_log(X).astype(np.float32)

    return X, token_ids_tensor, gene_indices


def make_batch_tensors(
    X_filtered: np.ndarray,
    token_ids_tensor: torch.Tensor,
    sample_indices: list[int] | np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build gene_ids and expression_values tensors for a batch of samples.

    The gene_ids tensor is shared across samples in a batch (same genes in the
    same order for every sample), matching the pattern in ``utils.py`` line 257:
        batch_genes = token_ids_tensor.unsqueeze(0).expand(current_batch_size, -1)

    Parameters
    ----------
    X_filtered : np.ndarray
        Full log-normalised expression matrix, shape ``(n_samples, n_genes)``.
    token_ids_tensor : torch.Tensor
        Shared token IDs, shape ``(n_genes,)``.
    sample_indices : list[int] | np.ndarray
        Row indices into ``X_filtered`` for this batch.
    device : torch.device
        Target device.

    Returns
    -------
    gene_ids : torch.Tensor
        Shape ``(batch, n_genes)``.
    expression_values : torch.Tensor
        Shape ``(batch, n_genes)``.
    """
    B = len(sample_indices)
    batch_X = X_filtered[sample_indices]                             # (B, n_genes)
    gene_ids = token_ids_tensor.unsqueeze(0).expand(B, -1)          # (B, n_genes)
    expression_values = torch.from_numpy(batch_X).to(device)        # (B, n_genes)
    return gene_ids, expression_values


# ---------------------------------------------------------------------------
# Encoding and decoding helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_and_decode_samples(
    model,
    X_filtered: np.ndarray,
    token_ids_tensor: torch.Tensor,
    device: torch.device,
    desc: str = "Encoding",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode all samples and return both gene embeddings and decoded expression.

    Runs in no-grad / inference mode (frozen weights).

    Parameters
    ----------
    model : EvaRnaModel
        Frozen EVA-RNA model.
    X_filtered : np.ndarray
        Log-normalised expression, shape ``(n_samples, n_genes)``, float32.
    token_ids_tensor : torch.Tensor
        Shared token IDs, shape ``(n_genes,)``.
    device : torch.device
        Computation device.
    desc : str
        Progress bar label.

    Returns
    -------
    all_gene_embeddings : torch.Tensor
        Gene-level hidden states, shape ``(n_samples, n_genes, hidden_size)``,
        on CPU.
    all_decoded_expr : torch.Tensor
        Decoded expression profiles, shape ``(n_samples, n_genes)``, on CPU.
    """
    n_samples = len(X_filtered)
    all_gene_emb  = []
    all_decoded   = []

    for start in tqdm(range(0, n_samples, BATCH_SIZE), desc=desc, leave=False):
        idx = list(range(start, min(start + BATCH_SIZE, n_samples)))
        gene_ids, expr_vals = make_batch_tensors(
            X_filtered, token_ids_tensor, idx, device
        )
        out          = model.encode(gene_ids, expr_vals)
        decoded      = model.decode(out.gene_embeddings)       # (B, S)

        all_gene_emb.append(out.gene_embeddings.cpu())
        all_decoded.append(decoded.cpu())

    return (
        torch.cat(all_gene_emb, dim=0),   # (n_samples, S, H)
        torch.cat(all_decoded,  dim=0),   # (n_samples, S)
    )


# ---------------------------------------------------------------------------
# Per-sample perturbation
# ---------------------------------------------------------------------------

def perturb_one_sample(
    model,
    gene_embeddings_1: torch.Tensor,
    gene_ids_1: torch.Tensor,
    target_gene_ids: list[int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply latent-space perturbation to a single expression profile.

    Implements EVA report equations (19)–(20):
        z'  = z + ∇_z L_pert    (∇_z L normalised to unit L2 norm, eq. 25)
        x'  = f_dec(z')

    where z = gene_embeddings (shape 1, S, H) — the gene-level hidden states
    that are the direct input to the expression decoder.

    The encoder forward pass is intentionally omitted here: the caller
    (``run_perturbation_pipeline``) already ran ``encode_and_decode_samples``
    over all disease samples in batches and holds the resulting gene embeddings
    in ``disease_gene_embs``.  Passing them in directly avoids re-encoding each
    sample a second time inside the per-sample perturbation loop.

    Parameters
    ----------
    model : EvaRnaModel
        EVA-RNA model (weights frozen via ``requires_grad_(False)``).
    gene_embeddings_1 : torch.Tensor
        Precomputed gene-level hidden states for this sample, shape
        ``(1, n_genes, hidden_size)``, on CPU (moved to ``device`` internally).
    gene_ids_1 : torch.Tensor
        Token IDs for this sample, shape ``(1, n_genes)``.  Still required by
        ``perturbation_loss`` to locate target-gene positions in the sequence.
    target_gene_ids : list[int]
        Entrez IDs of the drug's target genes to knock down (δ = -1).
    device : torch.device
        Computation device.

    Returns
    -------
    z_prime : torch.Tensor
        Perturbed gene embeddings, shape ``(1, S, H)``, detached, on CPU.
    x_prime : torch.Tensor
        Decoded expression from z', shape ``(1, S)``, detached, on CPU.
    """
    model.eval()

    # Re-attach the precomputed embedding as a fresh leaf tensor that requires
    # a gradient.  Backward will accumulate ∇_z directly into z.grad without
    # touching any model parameter (all weights are frozen).
    # .to(device) is a no-op when already on device; .clone() ensures we don't
    # mutate the precomputed tensor stored in disease_gene_embs.
    z = gene_embeddings_1.to(device).clone().detach().requires_grad_(True)  # (1, S, H) — leaf

    # decode(z) must run with grad enabled so PyTorch records the computation
    # graph between z and pred_expr.  torch.enable_grad() is explicit and
    # cannot be overridden by any outer no_grad context (e.g. the @no_grad
    # decorator on encode_and_decode_samples or model.eval()).
    with torch.enable_grad():
        pred_expr = model.decode(z)                                # (1, S) — on graph

    directions = [PERTURBATION_DIR] * len(target_gene_ids)

    try:
        with torch.enable_grad():
            loss = perturbation_loss(
                predicted_expression=pred_expr,
                gene_ids=gene_ids_1,
                target_gene_ids=target_gene_ids,
                perturbation_directions=directions,
            )                                                # scalar or (1,)
            loss.mean().backward()
    except KeyError as exc:
        log.warning("Skipping perturbation (target gene absent): %s", exc)
        with torch.no_grad():
            x_orig = model.decode(z.detach())
        return z.detach().cpu(), x_orig.detach().cpu()

    grad_z = z.grad                                      # (1, S, H)

    if grad_z is None:
        log.warning(
            "∇_z is None — target gene unreachable through decoder graph. "
            "Returning original embedding and decoded expression."
        )
        with torch.no_grad():
            x_orig = model.decode(z.detach())
        return z.detach().cpu(), x_orig.detach().cpu()

    # L2-normalise the gradient per sample (EVA report eq. 25).
    # norm is taken over the full (S, H) representation for each sample.
    grad_norm = grad_z.norm(dim=(-2, -1), keepdim=True)            # (1, 1, 1)
    grad_z_normalised = grad_z / (grad_norm + EPS)                 # (1, S, H)

    # z' = z + ∇_z L  (gradient ascent, eq. 19)
    z_prime = z.detach() + grad_z_normalised.detach()              # (1, S, H)

    # x' = f_dec(z')  (eq. 20)
    with torch.no_grad():
        x_prime = model.decode(z_prime)                            # (1, S)

    return z_prime.cpu(), x_prime.cpu()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_perturbation_pipeline(
    model,
    tokenizer,
    benchmark: pd.DataFrame,
    device: torch.device,
) -> pd.DataFrame:
    """Run the full zero-shot perturbation pipeline over the benchmark.

    For each unique disease in the benchmark:
      - Loads and preprocesses the cohort.
      - Builds the shared token ID tensor (mirroring utils.encode_from_anndata).
      - Encodes healthy samples → decodes → builds healthy expression centroid.
      - For each drug targeting that disease:
          * Perturbs each disease sample (forward in batch, backward per-sample).
          * Scores samples via shift toward the healthy expression centroid.
          * Saves per-patient decoded expressions as .npy.
          * Records the median score for AUROC evaluation.

    Parameters
    ----------
    model : EvaRnaModel
        Frozen EVA-RNA model, on ``device``.
    tokenizer : EvaRnaTokenizer
        Matching tokenizer.
    benchmark : pd.DataFrame
        Drug-disease efficacy matrix with columns:
        ``drug_name``, ``target_genes``, ``disease_name``,
        ``disease_abbrev``, ``tissue``, ``expected_efficacy``.
    device : torch.device
        Computation device.

    Returns
    -------
    pd.DataFrame
        Results with columns:
        ``drug_name``, ``disease_abbrev``, ``median_score``, ``expected_efficacy``.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    with MissingTargetGenesList(out=f"missing_target_genes_top{N_TOP_GENES}.json") as missing:
        for disease_abbrev, disease_group in benchmark.groupby("disease_abbrev"):
            log.info("=== Disease: %s ===", disease_abbrev)

            # ---- Load cohort and prepare shared tokenisation ------------------
            # Select genes that must be included in the expression, from list of target genes of the tested drugs
            target_genes_list = []
            for target_gene_1 in disease_group.target_genes:
                target_gene_1_split = target_gene_1.split(';')
                for gene in  target_gene_1_split:
                    if gene in GENE_ALIAS_MAP:
                        target_genes_list += GENE_ALIAS_MAP[gene]
                    else:
                        target_genes_list.append(gene)

            adata = load_cohort_data(disease_abbrev, N_TOP_GENES, target_genes_list)
            
            X_filtered, token_ids_tensor, _ = prepare_tokenisation(
                adata, tokenizer, device
            )

            # Build symbol → Entrez ID map from adata.var.
            # adata.var_names contains Entrez IDs as strings (e.g. "7157");
            # adata.var["gene_symbols"] contains HGNC symbols (e.g. "TP53").
            # The benchmark uses HGNC symbols, so we need this lookup to convert
            # target gene symbols to the integer token IDs expected by the model.
            symbol_to_entrez: dict[str, str] = {
                symbol: entrez
                for entrez, symbol in zip(
                    adata.var_names, adata.var["gene_symbols"]
                )
            }

            disease_mask = (adata.obs["disease"] != "Control").values
            healthy_mask = (adata.obs["disease"] == "Control").values

            disease_expr = X_filtered[disease_mask]   # (n_disease, n_genes)
            healthy_expr = X_filtered[healthy_mask]   # (n_healthy, n_genes)

            n_disease = disease_mask.sum()
            n_healthy = healthy_mask.sum()
            log.info("  %d disease samples, %d healthy controls", n_disease, n_healthy)

            # ---- Encode healthy samples, decode → healthy expression centroid -
            log.info("  Building healthy expression centroid...")
            _, healthy_decoded = encode_and_decode_samples(
                model, healthy_expr, token_ids_tensor, device, desc="Healthy"
            )                                          # (n_healthy, S)
            healthy_centroid = compute_healthy_centroid(healthy_decoded)  # (S,)

            # ---- Encode disease samples (original decoded expression) ----------
            # gene embeddings are captured here and reused in the perturbation
            # loop below, avoiding a redundant encoder forward pass per sample.
            log.info("  Encoding disease samples...")
            disease_gene_embs, disease_decoded_orig = encode_and_decode_samples(
                model, disease_expr, token_ids_tensor, device, desc="Disease"
            )                                          # (n_disease, S, H), (n_disease, S)

            # ---- Per-drug perturbation loop -----------------------------------
            for _, row in disease_group.iterrows():
                drug_name        = row["drug_name"]
                target_genes_raw = row["target_genes"]   # semicolon-separated HGNC symbols
                expected         = row["expected_efficacy"]

                # Resolve HGNC symbols → Entrez integer IDs via adata.var.
                # Symbols absent from the HVG-filtered cohort (e.g. low-variance
                # genes dropped by highly_variable_genes) are skipped with a warning.
                # Then, resolve Entrez -> gene IDs via tokenizer.convert_tokens_to_ids
                target_gene_ids: list[int] = []
                for symbol in str(target_genes_raw).split(";"):
                    symbol = symbol.strip()
                    entrez = symbol_to_entrez.get(symbol)
                    if entrez is None:
                        # Look for entrez from gene alias map
                        if symbol in GENE_ALIAS_MAP:
                            for s in GENE_ALIAS_MAP[symbol]:
                                entrez_alias = symbol_to_entrez.get(s)
                                if entrez_alias is not None:
                                    target_gene_ids.append( tokenizer.convert_tokens_to_ids(entrez_alias) )
                                else:
                                    # Update the json file with the missing target genes, its drug and the addressed disease
                                    missing.update(symbol, disease_abbrev, drug_name)
                        else:
                            log.warning(
                                "  Target gene '%s' not found in cohort var (may have "
                                "been dropped by HVG filtering) — skipping for drug %s.",
                                symbol, drug_name,
                            )
                            # Update the json file with the missing target genes, its drug and the addressed disease
                            missing.update(symbol, disease_abbrev, drug_name)
                    else:
                        target_gene_ids.append( tokenizer.convert_tokens_to_ids(entrez) )

                if not target_gene_ids:
                    log.warning(
                        "  No target genes resolved for drug %s in disease %s — "
                        "skipping this drug-disease pair.",
                        drug_name, disease_abbrev,
                    )
                    results.append({
                        "drug_name":         drug_name,
                        "disease_abbrev":    disease_abbrev,
                        "median_score":      float("nan"),
                        "expected_efficacy": expected,
                    })
                    continue

                log.info("  Drug: %-30s  targets: %s", drug_name, target_gene_ids)

                # Perturb each disease sample individually (one backward per sample)
                perturbed_x_list: list[torch.Tensor] = []

                for i in tqdm(range(n_disease), desc=f"{drug_name}", leave=False):
                    # Slice precomputed gene embeddings for this sample: (S, H) → (1, S, H)
                    gene_embs_1 = disease_gene_embs[i].unsqueeze(0)  # (1, S, H), on CPU

                    # gene_ids are still needed by perturbation_loss to locate
                    # target-gene positions; build the (1, n_genes) tensor as before.
                    gene_ids_1, _ = make_batch_tensors(
                        disease_expr, token_ids_tensor, [i], device
                    )

                    _, x_prime = perturb_one_sample(
                        model=model,
                        gene_embeddings_1=gene_embs_1,
                        gene_ids_1=gene_ids_1,
                        target_gene_ids=target_gene_ids,
                        device=device,
                    )                                  # (1, S)

                    perturbed_x_list.append(x_prime)

                perturbed_x = torch.cat(perturbed_x_list, dim=0)   # (n_disease, S)

                # Save per-patient perturbed expression profiles
                out_path = OUTPUT_DIR / f"{disease_abbrev}_{drug_name}_perturbed_expr.npy"
                np.save(out_path, perturbed_x.numpy())
                log.info("  Saved perturbed expressions → %s", out_path)

                # Score: shift of x' toward healthy centroid
                # NOTE: compute_shift_score raises NotImplementedError until the
                # scoring formulation in expression space is finalised.
                try:
                    scores = compute_shift_score(
                        original_x=disease_decoded_orig,
                        perturbed_x=perturbed_x,
                        healthy_centroid=healthy_centroid,
                    )
                    median_score = float(scores.median().item())
                except NotImplementedError:
                    log.warning(
                        "  compute_shift_score not yet implemented — "
                        "median_score set to NaN. Implement scoring.py to proceed."
                    )
                    median_score = float("nan")

                results.append({
                    "drug_name":         drug_name,
                    "disease_abbrev":    disease_abbrev,
                    "median_score":      median_score,
                    "expected_efficacy": expected,
                })

    results_df = pd.DataFrame(results)
    csv_path   = OUTPUT_DIR / "perturbation_results.csv"
    results_df.to_csv(csv_path, index=False)
    log.info("Summary saved → %s", csv_path)
    return results_df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Using device: %s", device)

    log.info("Loading EVA-RNA model...")
    model = AutoModel.from_pretrained("ScientaLab/eva-rna", trust_remote_code=True)
    model = model.to(device)
    model.eval()

    # Freeze all model parameters — gradients flow only through intermediate
    # activations (gene_embeddings), not through any weight tensor
    for param in model.parameters():
        param.requires_grad_(False)

    tokenizer = AutoTokenizer.from_pretrained(
        "ScientaLab/eva-rna", trust_remote_code=True
    )

    benchmark = pd.read_csv(BENCHMARK_PATH)
    log.info("Benchmark: %d drug-disease pairs", len(benchmark))

    results = run_perturbation_pipeline(model, tokenizer, benchmark, device)
    log.info("Done.")