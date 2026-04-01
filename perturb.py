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
       a. Encode → gene_embeddings z  (shape 1, S, H)
       b. Decode → predicted expression x  (shape 1, S)
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

from encode_and_save import load_cohort_data
from gradient_flow_pert_loss import perturbation_loss
from scoring import compute_healthy_centroid, compute_shift_score

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BENCHMARK_PATH = "data/benchmark_drug_target_disease_matrix.csv"
OUTPUT_DIR     = Path("data/perturbation_scores")
BATCH_SIZE     = 16      # samples per forward pass (backward is always per-sample)
N_TOP_GENES    = 4000    # HVG subset, matching encode_and_save.py
EPS            = 1e-8    # numerical stability for gradient L2 normalisation
PERTURBATION_DIR = -1    # δ = -1: knockdown for all targets (mission statement)
_TARGET_SUM    = 1e4     # library-size normalisation target, matching utils.py

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
    gene_ids_1: torch.Tensor,
    expression_values_1: torch.Tensor,
    target_gene_ids: list[int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply latent-space perturbation to a single expression profile.

    Implements EVA report equations (19)–(20):
        z'  = z + ∇_z L_pert    (∇_z L normalised to unit L2 norm, eq. 25)
        x'  = f_dec(z')

    where z = gene_embeddings (shape 1, S, H) — the gene-level hidden states
    that are the direct input to the expression decoder.

    Parameters
    ----------
    model : EvaRnaModel
        EVA-RNA model (weights frozen via ``requires_grad_(False)``).
    gene_ids_1 : torch.Tensor
        Token IDs for this sample, shape ``(1, n_genes)``.
    expression_values_1 : torch.Tensor
        Log-normalised expression for this sample, shape ``(1, n_genes)``.
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

    # Forward pass — model weights are frozen (no_grad not used here so that
    # autograd tracks the graph from gene_embeddings → decode → loss)
    out       = model.encode(gene_ids_1, expression_values_1)
    z         = out.gene_embeddings                      # (1, S, H) — on graph
    pred_expr = model.decode(z)                          # (1, S) — on graph

    directions = [PERTURBATION_DIR] * len(target_gene_ids)

    try:
        loss = perturbation_loss(
            predicted_expression=pred_expr,
            gene_ids=gene_ids_1,
            target_gene_ids=target_gene_ids,
            perturbation_directions=directions,
        )                                                # (1,)
    except KeyError as exc:
        log.warning("Skipping perturbation (target gene absent): %s", exc)
        with torch.no_grad():
            x_orig = model.decode(z)
        return z.detach().cpu(), x_orig.detach().cpu()

    # z is not a leaf tensor (it is an intermediate output of the encoder),
    # so we must call retain_grad() to keep its gradient after backward().
    z.retain_grad()
    loss.sum().backward()

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

    for disease_abbrev, disease_group in benchmark.groupby("disease_abbrev"):
        log.info("=== Disease: %s ===", disease_abbrev)

        # ---- Load cohort and prepare shared tokenisation ------------------
        adata = load_cohort_data(disease_abbrev, N_TOP_GENES)
        X_filtered, token_ids_tensor, _ = prepare_tokenisation(
            adata, tokenizer, device
        )

        # Build symbol → Entrez ID map from adata.var.
        # adata.var_names contains Entrez IDs as strings (e.g. "7157");
        # adata.var["gene_symbols"] contains HGNC symbols (e.g. "TP53").
        # The benchmark uses HGNC symbols, so we need this lookup to convert
        # target gene symbols to the integer token IDs expected by the model.
        symbol_to_entrez: dict[str, int] = {
            symbol: int(entrez)
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
        log.info("  Encoding disease samples...")
        _, disease_decoded_orig = encode_and_decode_samples(
            model, disease_expr, token_ids_tensor, device, desc="Disease"
        )                                          # (n_disease, S)

        # ---- Per-drug perturbation loop -----------------------------------
        for _, row in disease_group.iterrows():
            drug_name        = row["drug_name"]
            target_genes_raw = row["target_genes"]   # semicolon-separated HGNC symbols
            expected         = row["expected_efficacy"]

            # Resolve HGNC symbols → Entrez integer IDs via adata.var.
            # Symbols absent from the HVG-filtered cohort (e.g. low-variance
            # genes dropped by highly_variable_genes) are skipped with a warning.
            target_gene_ids: list[int] = []
            for symbol in str(target_genes_raw).split(";"):
                symbol = symbol.strip()
                entrez = symbol_to_entrez.get(symbol)
                if entrez is None:
                    log.warning(
                        "  Target gene '%s' not found in cohort var (may have "
                        "been dropped by HVG filtering) — skipping for drug %s.",
                        symbol, drug_name,
                    )
                else:
                    target_gene_ids.append(entrez)

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
                gene_ids_1, expr_vals_1 = make_batch_tensors(
                    disease_expr, token_ids_tensor, [i], device
                )                                  # (1, n_genes) each

                _, x_prime = perturb_one_sample(
                    model=model,
                    gene_ids_1=gene_ids_1,
                    expression_values_1=expr_vals_1,
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