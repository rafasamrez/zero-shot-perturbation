"""Perturbation scoring utilities for EVA-RNA zero-shot efficacy prediction.

This module is intentionally kept separate from the perturbation loop so that
alternative scoring formulations can be swapped in without touching the main
pipeline.

Two public functions are exposed:

    compute_healthy_centroid(healthy_predicted_expression) -> torch.Tensor
        Computes the reference centroid from healthy control decoded expression
        profiles (i.e. the output of model.decode() on healthy samples).

    compute_shift_score(original_x, perturbed_x, healthy_centroid) -> torch.Tensor
        Scores how much each perturbed sample shifted toward the healthy centroid
        in decoded expression space. Currently raises NotImplementedError —
        the scoring formulation in expression space is under active design.

Design note
-----------
The latent-space perturbation (EVA report eq. 19–20) perturbs gene_embeddings z
and decodes the result to a predicted expression profile x' = f_dec(z').
Scoring therefore operates in *decoded expression space*, not in CLS embedding
space.  Simple cosine similarity between CLS embeddings is no longer appropriate.
The correct formulation (e.g. correlation-based distance, Wasserstein distance,
or a normalised expression-space cosine) will be determined separately.
"""

from __future__ import annotations

import torch


def compute_healthy_centroid(healthy_predicted_expression: torch.Tensor) -> torch.Tensor:
    """Compute the healthy reference centroid in decoded expression space.

    Parameters
    ----------
    healthy_predicted_expression : torch.Tensor
        Decoded expression profiles of healthy control samples, shape
        ``(n_healthy, seq_len)``, produced by ``model.decode()`` on healthy
        gene embeddings with frozen weights.

    Returns
    -------
    torch.Tensor
        Mean centroid vector of shape ``(seq_len,)``, representing the average
        healthy decoded expression profile across all control samples.
        Not L2-normalised — the appropriate normalisation depends on the
        scoring formulation chosen in ``compute_shift_score``.
    """
    if healthy_predicted_expression.ndim != 2:
        raise ValueError(
            f"healthy_predicted_expression must be 2D (n_healthy, seq_len), "
            f"got shape {tuple(healthy_predicted_expression.shape)}"
        )
    if healthy_predicted_expression.shape[0] == 0:
        raise ValueError(
            "healthy_predicted_expression must contain at least one sample."
        )

    centroid = healthy_predicted_expression.mean(dim=0)   # (seq_len,)
    return centroid


def compute_shift_score(
    original_x: torch.Tensor,
    perturbed_x: torch.Tensor,
    healthy_centroid: torch.Tensor,
) -> torch.Tensor:
    """Score how much each perturbation shifted a sample toward the healthy centroid.

    .. note::
        This function is not yet implemented.  The scoring formulation in decoded
        expression space (x' vs healthy centroid) is under active design and will
        be added once a suitable metric is identified.

        Candidate formulations include:
        - Pearson correlation between x' and the healthy centroid
        - Normalised Euclidean distance improvement (original → perturbed)
        - Rank-based or Spearman correlation
        - Wasserstein distance in expression space

    Parameters
    ----------
    original_x : torch.Tensor
        Decoded expression profiles of disease samples before perturbation,
        shape ``(batch, seq_len)``.  Produced by ``model.decode()`` on
        unperturbed gene embeddings.
    perturbed_x : torch.Tensor
        Decoded expression profiles after perturbation,
        shape ``(batch, seq_len)``.  Produced by ``model.decode(z')`` where
        z' = z + ∇_z L (EVA report eq. 19–20).
    healthy_centroid : torch.Tensor
        Mean healthy decoded expression profile from ``compute_healthy_centroid``,
        shape ``(seq_len,)``.

    Returns
    -------
    torch.Tensor
        Per-sample shift scores of shape ``(batch,)``.

    Raises
    ------
    NotImplementedError
        Always. Scoring in decoded expression space is pending design.
    """
    raise NotImplementedError(
        "compute_shift_score is not yet implemented.  Scoring in decoded "
        "expression space (x' vs healthy centroid) requires a metric that is "
        "appropriate for log-normalised bulk RNA-seq profiles.  Candidates: "
        "Pearson correlation, normalised Euclidean distance improvement, or "
        "Spearman correlation.  Implement and test before running the pipeline."
    )