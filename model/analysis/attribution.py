"""Attribution analysis for classifier and generator models.

Provides logit-impact (gradient-based) and attention-based attribution
for identifying discriminative transcription factor programmes.

Migrated from the demo analysis notebooks.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def compute_logit_impact(
    model: torch.nn.Module,
    gene_ids: torch.Tensor,
    gene_vals: torch.Tensor,
    pad_mask: torch.Tensor,
    target_class: int = 1,
    n_genes: int | None = None,
) -> np.ndarray:
    """Compute gene-level logit-impact via gradient of target class logit.

    Returns
    -------
    impacts : np.ndarray [n_genes] average absolute gradient aggregated by gene ID
    """
    model.eval()
    gene_vals_param = gene_vals.clone().detach().requires_grad_(True)

    logits = model(gene_ids, gene_vals_param, pad_mask)
    target_logit = logits[:, target_class].sum()
    target_logit.backward()

    grad = gene_vals_param.grad
    if grad is None:
        raise RuntimeError("Gradient is None; check requires_grad on input.")

    valid_mask = (gene_ids > 0) & (~pad_mask.bool())
    if n_genes is None:
        n_genes = int(gene_ids[valid_mask].max().item()) if valid_mask.any() else 0
    if n_genes <= 0:
        return np.zeros(0, dtype=np.float32)

    gene_index = (gene_ids - 1).clamp(min=0, max=n_genes - 1)
    scores = torch.zeros(n_genes, device=gene_ids.device, dtype=grad.dtype)
    counts = torch.zeros(n_genes, device=gene_ids.device, dtype=grad.dtype)

    scores.scatter_add_(0, gene_index[valid_mask], grad.abs()[valid_mask])
    counts.scatter_add_(
        0,
        gene_index[valid_mask],
        torch.ones_like(grad.abs()[valid_mask]),
    )
    impacts = scores / counts.clamp(min=1.0)
    return impacts.detach().cpu().numpy()


def compute_attention_attribution(
    model: torch.nn.Module,
    gene_ids: torch.Tensor,
    gene_vals: torch.Tensor,
    pad_mask: torch.Tensor,
) -> np.ndarray:
    """Extract attention weights from the classifier's last encoder block.

    Returns
    -------
    attn_weights : np.ndarray [batch, n_genes] averaged across heads
    """
    model.eval()
    with torch.no_grad():
        model(gene_ids, gene_vals, pad_mask)

    weights = model.last_attn_weights
    if weights is None:
        raise RuntimeError(
            "No attention weights stored. Ensure model was run before calling this function."
        )

    # weights shape: [batch, nhead, 1, seq_len] —— average over heads
    if weights.ndim == 4:
        avg_weights = weights.mean(dim=1).squeeze(1)  # [batch, seq_len]
    else:
        avg_weights = weights.squeeze(1)

    return avg_weights.detach().cpu().numpy()


def rank_genes_by_impact(
    impacts: np.ndarray,
    gene_names: list[str] | None = None,
    top_k: int = 20,
) -> list[dict[str, Any]]:
    """Rank genes by logit-impact score (descending)."""
    order = np.argsort(impacts)[::-1]
    results: list[dict[str, Any]] = []
    for rank, idx in enumerate(order[:top_k]):
        entry: dict[str, Any] = {
            "rank": rank + 1,
            "gene_index": int(idx),
            "impact_score": float(impacts[idx]),
        }
        if gene_names is not None and idx < len(gene_names):
            entry["gene_name"] = gene_names[idx]
        results.append(entry)
    return results


def compute_generator_attribution(
    model: torch.nn.Module,
    gene_ids: torch.Tensor,
    gene_vals: torch.Tensor,
    source_time: torch.Tensor,
    target_time: torch.Tensor,
    padding_mask: torch.Tensor | None = None,
) -> dict[str, np.ndarray]:
    """Extract encoder self-attention and cross-attention from generator.

    Returns
    -------
    dict with keys: encoder_attn, cross_attn (or empty arrays if not requested)
    """
    model.eval()
    with torch.no_grad():
        _, enc_attn, cross_attn = model(
            gene_ids,
            gene_vals,
            source_time,
            target_time,
            padding_mask=padding_mask,
            need_weights=True,
        )

    result: dict[str, np.ndarray] = {}
    if enc_attn is not None:
        if enc_attn.ndim == 4:
            result["encoder_attn"] = enc_attn.mean(dim=1).detach().cpu().numpy()
        else:
            result["encoder_attn"] = enc_attn.detach().cpu().numpy()
    if cross_attn is not None:
        if cross_attn.ndim == 4:
            result["cross_attn"] = cross_attn.mean(dim=1).detach().cpu().numpy()
        else:
            result["cross_attn"] = cross_attn.detach().cpu().numpy()

    return result
