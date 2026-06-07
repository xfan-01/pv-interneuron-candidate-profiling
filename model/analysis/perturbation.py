"""In silico perturbation analysis for TF candidate screening.

Provides perturbation specification, application, and single-step
perturbation comparison against baseline.

Migrated from ``demo/perturbation.ipynb``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def perturb_expression(
    expression: np.ndarray,
    gene_indices: dict[str, int],
    perturbation_spec: dict[str, str],
    gene_mean_log1p: np.ndarray | None = None,
    gene_std_log1p: np.ndarray | None = None,
    oe_zscore: float = 2.0,
    kd_fraction: float = 0.5,
) -> np.ndarray:
    """Apply TF perturbation to an expression matrix in-place.

    Parameters
    ----------
    expression : [N, G] float32
    gene_indices : gene_name → column index
    perturbation_spec : {gene_name: "OE"|"KD"|"KO"}
    gene_mean_log1p : [G] optional baseline mean for OE z-scoring
    gene_std_log1p : [G] optional baseline std for OE z-scoring
    oe_zscore : float
    kd_fraction : float

    Returns
    -------
    perturbed : [N, G] modified expression (copy)
    """
    perturbed = expression.copy()
    n_genes = perturbed.shape[1]

    for gene_name, action in perturbation_spec.items():
        if gene_name not in gene_indices:
            continue
        g_idx = gene_indices[gene_name]
        action_upper = action.upper()

        if action_upper == "OE":
            if gene_mean_log1p is not None and gene_std_log1p is not None:
                target_val = gene_mean_log1p[g_idx] + oe_zscore * max(
                    gene_std_log1p[g_idx], 1e-6
                )
            else:
                target_val = np.percentile(perturbed[:, g_idx], 95) * 2.0
            perturbed[:, g_idx] = np.maximum(perturbed[:, g_idx], target_val)

        elif action_upper == "KD":
            perturbed[:, g_idx] = perturbed[:, g_idx] * kd_fraction

        elif action_upper == "KO":
            perturbed[:, g_idx] = 0.0

    return perturbed


def compute_gene_log1p_stats(
    expression: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-gene mean and std in log1p space."""
    log_expr = np.log1p(expression.astype(np.float32))
    mean = log_expr.mean(axis=0).astype(np.float32)
    std = log_expr.std(axis=0).astype(np.float32)
    std[std < 1e-8] = 1.0
    return mean, std


def perturb_token_values(
    gene_ids: np.ndarray,
    gene_vals: np.ndarray,
    gene_indices: dict[str, int],
    perturbation_spec: dict[str, str],
    gene_mean_log1p: np.ndarray | None = None,
    gene_std_log1p: np.ndarray | None = None,
    oe_zscore: float = 2.0,
    kd_fraction: float = 0.5,
) -> np.ndarray:
    """Apply perturbations to token-level generator inputs.

    Generator inputs store expression values by token position, while
    ``gene_indices`` stores full gene-column indices. Therefore perturbations
    must be applied where ``gene_ids == gene_index + 1``.
    """
    perturbed = gene_vals.copy()

    for gene_name, action in perturbation_spec.items():
        if gene_name not in gene_indices:
            continue

        full_gene_idx = gene_indices[gene_name]
        token_id = full_gene_idx + 1
        mask = gene_ids == token_id
        if not mask.any():
            continue

        action_upper = action.upper()
        if action_upper == "OE":
            if gene_mean_log1p is not None and gene_std_log1p is not None:
                target_val = gene_mean_log1p[full_gene_idx] + oe_zscore * max(
                    gene_std_log1p[full_gene_idx], 1e-6
                )
            else:
                target_val = np.percentile(perturbed[mask], 95) * 2.0
            perturbed[mask] = np.maximum(perturbed[mask], target_val)
        elif action_upper == "KD":
            perturbed[mask] = perturbed[mask] * kd_fraction
        elif action_upper == "KO":
            perturbed[mask] = 0.0

    return perturbed


def run_single_step_perturbation(
    generator: torch.nn.Module,
    gene_ids: torch.Tensor,
    gene_vals: torch.Tensor,
    source_time: torch.Tensor,
    target_time: torch.Tensor,
    padding_mask: torch.Tensor,
    perturbation_spec: dict[str, str],
    gene_indices: dict[str, int],
    gene_mean_log1p: np.ndarray | None = None,
    gene_std_log1p: np.ndarray | None = None,
    oe_zscore: float = 2.0,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Run a single-step perturbation comparison.

    Returns
    -------
    dict with baseline_pred, perturbed_pred, perturbation_spec
    """
    if isinstance(device, str):
        device = torch.device(device)

    generator.eval()

    # Baseline prediction (original expression)
    with torch.no_grad():
        baseline_pred, _, _ = generator(
            gene_ids.to(device),
            gene_vals.to(device),
            source_time.to(device),
            target_time.to(device),
            padding_mask=padding_mask.to(device),
        )

    # Apply perturbation to token-level values and predict.
    perturbed_expr = perturb_token_values(
        gene_ids=gene_ids.cpu().numpy(),
        gene_vals=gene_vals.cpu().numpy(),
        gene_indices=gene_indices,
        perturbation_spec=perturbation_spec,
        gene_mean_log1p=gene_mean_log1p,
        gene_std_log1p=gene_std_log1p,
        oe_zscore=oe_zscore,
    )
    perturbed_vals = torch.tensor(perturbed_expr, dtype=torch.float32)

    with torch.no_grad():
        perturbed_pred, _, _ = generator(
            gene_ids.to(device),
            perturbed_vals.to(device),
            source_time.to(device),
            target_time.to(device),
            padding_mask=padding_mask.to(device),
        )

    return {
        "baseline_pred": baseline_pred.cpu(),
        "perturbed_pred": perturbed_pred.cpu(),
        "perturbation_spec": perturbation_spec,
    }
