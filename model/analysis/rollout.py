"""Sustained multi-step rollout for perturbation persistence analysis.

Migrated from ``demo/perturbation.ipynb``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch


def run_sustained_rollout(
    generator: torch.nn.Module,
    gene_ids: torch.Tensor,
    gene_vals: torch.Tensor,
    source_time: torch.Tensor,
    perturbation_spec: dict[str, str],
    gene_indices: dict[str, int],
    n_steps: int = 5,
    step_size: float = 0.05,
    oe_zscore: float = 2.0,
    gene_mean_log1p: np.ndarray | None = None,
    gene_std_log1p: np.ndarray | None = None,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Run sustained multi-step perturbation rollout.

    At each step, apply perturbation, predict forward, and feed the
    predicted state as the next source.

    Parameters
    ----------
    generator : nn.Module
    gene_ids : [B, L] long
    gene_vals : [B, L] float, original source expression
    source_time : [B] float, initial time
    perturbation_spec : dict
    gene_indices : dict
    n_steps : int, number of forward steps
    step_size : float, time increment per step
    oe_zscore : float
    gene_mean_log1p : optional
    gene_std_log1p : optional
    device : torch.device

    Returns
    -------
    dict with:
        steps: list of dicts (step, pred, time)
        baseline_trajectory: list of dicts (step, pred, time)
        perturbation_spec
    """
    if isinstance(device, str):
        device = torch.device(device)

    generator.eval()
    n_genes = gene_vals.shape[1]
    b = gene_ids.size(0)

    # --- baseline trajectory (no perturbation) ---
    baseline_trajectory: list[dict[str, Any]] = []
    current_time = source_time.clone()
    current_gene_vals = gene_vals.clone()

    with torch.no_grad():
        for step in range(n_steps):
            target_time_bl = current_time + step_size
            pred_bl, _, _ = generator(
                gene_ids.to(device),
                current_gene_vals.to(device),
                current_time.to(device),
                target_time_bl.to(device),
            )
            pred_bl_np = pred_bl.cpu().numpy()
            baseline_trajectory.append({
                "step": step,
                "pred": pred_bl_np.copy(),
                "time": target_time_bl.cpu().numpy(),
            })
            # Feed prediction as next source
            current_time = target_time_bl
            # Convert per-gene prediction back to token format
            current_gene_vals = _pred_to_token_vals(
                pred_bl_np, gene_ids.cpu().numpy()
            )

    # --- perturbed trajectory ---
    perturbed_trajectory: list[dict[str, Any]] = []
    current_time = source_time.clone()
    current_gene_vals_pt = gene_vals.clone()

    # Apply initial perturbation
    from .perturbation import perturb_token_values

    perturbed_expr = perturb_token_values(
        gene_ids=gene_ids.cpu().numpy(),
        gene_vals=current_gene_vals_pt.cpu().numpy(),
        gene_indices=gene_indices,
        perturbation_spec=perturbation_spec,
        gene_mean_log1p=gene_mean_log1p,
        gene_std_log1p=gene_std_log1p,
        oe_zscore=oe_zscore,
    )
    current_gene_vals_pt = torch.tensor(perturbed_expr, dtype=torch.float32)

    with torch.no_grad():
        for step in range(n_steps):
            target_time_pt = current_time + step_size
            pred_pt, _, _ = generator(
                gene_ids.to(device),
                current_gene_vals_pt.to(device),
                current_time.to(device),
                target_time_pt.to(device),
            )
            pred_pt_np = pred_pt.cpu().numpy()
            perturbed_trajectory.append({
                "step": step,
                "pred": pred_pt_np.copy(),
                "time": target_time_pt.cpu().numpy(),
            })
            current_time = target_time_pt
            current_gene_vals_pt = _pred_to_token_vals(
                pred_pt_np, gene_ids.cpu().numpy()
            )

    return {
        "baseline_trajectory": baseline_trajectory,
        "perturbed_trajectory": perturbed_trajectory,
        "perturbation_spec": perturbation_spec,
        "n_steps": n_steps,
        "step_size": step_size,
    }


def _pred_to_token_vals(
    pred: np.ndarray, gene_ids: np.ndarray
) -> torch.Tensor:
    """Convert per-gene predictions back to token-level values.

    pred: [B, G] full gene predictions
    gene_ids: [B, L] 1-based gene token IDs
    Returns: [B, L] token values
    """
    b, g = pred.shape
    _, l = gene_ids.shape
    token_vals = np.zeros((b, l), dtype=np.float32)
    for i in range(b):
        for j in range(l):
            gid = gene_ids[i, j]
            if gid > 0 and gid <= g:
                token_vals[i, j] = pred[i, gid - 1]
    return torch.tensor(token_vals, dtype=torch.float32)


# ---------------------------------------------------------------------------
#  Multi-seed aggregation for rollout persistence summaries
#  (migrated from demo/perturbation.ipynb cells 57 & 46)
# ---------------------------------------------------------------------------

def aggregate_stepwise_across_seeds(
    combined_summary_df: pd.DataFrame,
    repeat_boot_n: int = 1000,
    repeat_boot_ci: float = 0.95,
    repeat_boot_seed: int = 42,
) -> pd.DataFrame:
    """Aggregate stepwise rollout summary across repeat seeds.

    Input row unit: label × metric × step × repeat_seed
    Output row unit: label × metric × step

    For each (label, metric, step) group, bootstraps the mean of
    per-repeat ``point_estimate`` values.
    """
    import pandas as pd

    if combined_summary_df.empty:
        return pd.DataFrame(
            columns=[
                "label", "metric", "step", "n_repeats",
                "point_estimate", "ci_low", "ci_high",
            ]
        )

    required_cols = {"label", "metric", "step", "point_estimate"}
    missing = [c for c in required_cols if c not in combined_summary_df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in combined_summary_df: {missing}"
        )

    from .candidate_panel import bootstrap_metric

    rows = []
    for (label, metric, step), grp in combined_summary_df.groupby(
        ["label", "metric", "step"], observed=True
    ):
        vals = pd.to_numeric(
            grp["point_estimate"], errors="coerce"
        ).to_numpy(dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue

        boot = bootstrap_metric(
            values=vals,
            stat="mean",
            n_boot=repeat_boot_n,
            ci=repeat_boot_ci,
            seed=repeat_boot_seed + int(step),
        )
        rows.append({
            "label": label,
            "metric": metric,
            "step": int(step),
            "n_repeats": int(vals.size),
            "point_estimate": boot["point_estimate"],
            "ci_low": boot["ci_low"],
            "ci_high": boot["ci_high"],
        })

    return pd.DataFrame(rows)


def calculate_persistence_scores(
    summary_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute persistence scores from a stepwise bootstrap summary.

    Scores (per label × metric):
    - Sign_Consistency: fraction of steps with point_estimate > 0
    - Strict_Consistency: fraction of steps with ci_low > 0
    - Duration: consecutive positive steps from step 1
    - Strict_Duration: consecutive steps from step 1 with ci_low > 0
    - EndMinusStart: last step minus first step point estimate
    - AUC: sum of positive point estimates across steps (area under curve)

    For metrics where lower values are favourable (delta_path_deviation,
    delta_target_distance), scores use negated values.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Output of ``aggregate_stepwise_across_seeds`` or similar.
        Must have columns: label, metric, step, point_estimate, ci_low, ci_high.

    Returns
    -------
    pd.DataFrame with columns: label, metric, {scores}
    """
    if summary_df.empty:
        return pd.DataFrame()

    score_rows = []
    for (label, metric), df in summary_df.groupby(
        ["label", "metric"], observed=True
    ):
        df = df.sort_values("step").reset_index(drop=True)
        vals = df["point_estimate"].to_numpy(dtype=np.float32)
        ci_lows = df["ci_low"].to_numpy(dtype=np.float32)
        ci_highs = df["ci_high"].to_numpy(dtype=np.float32)

        # Flip sign for distance/deviation readouts
        if metric in {"delta_path_deviation", "delta_target_distance"}:
            vals_for_score = -vals
            ci_lows_for_score = -ci_highs
        else:
            vals_for_score = vals
            ci_lows_for_score = ci_lows

        sign_consistency = float(np.mean(vals_for_score > 0))
        strict_consistency = float(np.mean(ci_lows_for_score > 0))

        duration = 0
        for v in vals_for_score:
            if v > 0:
                duration += 1
            else:
                break

        strict_duration = 0
        for lo in ci_lows_for_score:
            if lo > 0:
                strict_duration += 1
            else:
                break

        end_minus_start = float(vals_for_score[-1] - vals_for_score[0])
        auc = float(np.sum(np.maximum(vals_for_score, 0)))

        score_rows.append({
            "label": label,
            "metric": metric,
            "Sign_Consistency": sign_consistency,
            "Strict_Consistency": strict_consistency,
            "Duration": int(duration),
            "Strict_Duration": int(strict_duration),
            "EndMinusStart": end_minus_start,
            "AUC": auc,
            "n_steps": int(len(vals)),
        })

    return pd.DataFrame(score_rows)


def aggregate_persistence_panel_scores(
    panel_score_df: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate persistence scores across repeat seeds within a panel.

    For each (label, metric), bootstraps the distribution of per-repeat
    ``Strict_Duration`` values to produce a summary with CIs.
    """
    if panel_score_df.empty:
        return pd.DataFrame()

    from .candidate_panel import bootstrap_metric

    rows = []
    for (label, metric), grp in panel_score_df.groupby(
        ["label", "metric"], observed=True
    ):
        dur_vals = grp["Strict_Duration"].to_numpy(dtype=np.float32)
        dur_vals = dur_vals[np.isfinite(dur_vals)]
        if dur_vals.size == 0:
            continue

        boot = bootstrap_metric(dur_vals, stat="mean", n_boot=500, ci=0.95, seed=42)
        rows.append({
            "label": label,
            "metric": metric,
            "n_repeats": int(dur_vals.size),
            "mean_strict_duration": boot["point_estimate"],
            "ci_low_strict_duration": boot["ci_low"],
            "ci_high_strict_duration": boot["ci_high"],
        })

    return pd.DataFrame(rows)
