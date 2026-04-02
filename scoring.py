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
        in decoded expression space.

Design note
-----------
The latent-space perturbation (EVA report eq. 19–20) perturbs gene_embeddings z
and decodes the result to a predicted expression profile x' = f_dec(z').
Scoring therefore operates in *decoded expression space*, not in CLS embedding
space.  Simple cosine similarity between CLS embeddings is not appropriate.
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
    """
    Score = Pearson correlation improvement toward healthy centroid.
    
    For each disease sample i:
        score_i = pearson(x'_i, c_healthy) - pearson(x_i, c_healthy)
    
    A positive score means the perturbation pushed expression toward healthy.
    The drug-disease score is the median across patients.
    """
    def pearson_rows_vs_vector(mat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        # mat: (B, S), vec: (S,)
        # Returns per-row Pearson correlation with vec, shape (B,)
        mat_centered = mat - mat.mean(dim=1, keepdim=True)
        vec_centered = vec - vec.mean()
        
        numerator = (mat_centered * vec_centered).sum(dim=1)
        denom = mat_centered.norm(dim=1) * vec_centered.norm() + 1e-8
        return numerator / denom
    
    r_original  = pearson_rows_vs_vector(original_x,  healthy_centroid)  # (B,)
    r_perturbed = pearson_rows_vs_vector(perturbed_x, healthy_centroid)  # (B,)
    
    return r_perturbed - r_original  # (B,) — positive = shift toward healthy