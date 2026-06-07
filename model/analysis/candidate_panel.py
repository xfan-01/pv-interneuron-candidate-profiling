"""Candidate panel hit-calling and statistical summary for perturbation screening.

Provides the core analytical pipeline (no plotting) migrated from
``demo/perturbation.ipynb``:

- Metric configuration and canonical naming
- Exact sign-flip test on repeat means
- Effect-size estimation (Cohen's d, Hedges' g)
- Rule-based hit classification (A/B/C/non-hit)
- Candidate panel summary tables

Basic bootstrap and BH-FDR are in ``model.analysis.stats``.
All functions operate on pandas DataFrames and numpy arrays; they do not
depend on PyTorch, AnnData, or any model objects.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from model.utils.constants import (
    get_diagnostic_metrics as _get_diagnostic_metrics,
    get_legacy_to_canonical_metric_map as _get_legacy_to_canonical_metric_map,
    get_main_geometry_metrics as _get_main_geometry_metrics,
    get_main_identity_metrics as _get_main_identity_metrics,
    get_main_metrics as _get_main_metrics,
    get_readout_orientation as _get_readout_orientation,
)
from .stats import bh_fdr, bootstrap_metric, bootstrap_metric_panel

# ---------------------------------------------------------------------------
#  Metric configuration (canonical source of truth)
# ---------------------------------------------------------------------------

MAIN_GEOMETRY_METRICS = _get_main_geometry_metrics()
MAIN_IDENTITY_METRICS = _get_main_identity_metrics()
MAIN_METRICS = _get_main_metrics()
DIAGNOSTIC_METRICS = _get_diagnostic_metrics()
LEGACY_TO_CANONICAL_METRIC = _get_legacy_to_canonical_metric_map()
READOUT_ORIENTATION = _get_readout_orientation()

ROLLOUT_PRIMARY_METRICS = list(MAIN_METRICS)
ROLLOUT_SUPPORTING_METRICS = [
    "delta_forward_margin",
    "delta_pseudotime",
    "delta_target_distance",
]

HIT_CLASS_ORDER = [
    "A_productive",
    "B_identity_only",
    "C_geometry_only",
    "D_non_significant",
]


def canonical_metric_name(metric_name: str) -> str:
    return LEGACY_TO_CANONICAL_METRIC.get(str(metric_name), str(metric_name))


def is_control_label(label: str) -> bool:
    s = str(label)
    return s.startswith("RANDOM_CTRL_") or s.startswith("SHAM_")


# ---------------------------------------------------------------------------
#  Bootstrap and BH-FDR helpers → see ``model.analysis.stats``
# ---------------------------------------------------------------------------
# (bootstrap_metric, bootstrap_metric_panel, bh_fdr are re-exported via stats)

# ---------------------------------------------------------------------------
#  Exact sign-flip test
# ---------------------------------------------------------------------------

def exact_sign_flip_p_value(values: np.ndarray | list[float]) -> float:
    """Exact sign-flip test p-value (one-sample, two-sided).

    For n ≤ 15: exact enumeration of all 2^n sign configurations.
    For n > 15: approximate via 10,000 random sign flips.
    """
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    n = int(values.size)
    if n == 0:
        return float("nan")
    observed = float(np.mean(values))
    if n == 1:
        return 1.0

    if n <= 15:
        flips = np.arange(1 << n, dtype=np.uint32)[:, None]
        bit_positions = np.arange(n, dtype=np.uint32)
        signs = 1.0 - 2.0 * (((flips >> bit_positions) & 1).astype(np.float64))
        perm_stats = (signs * values[None, :]).mean(axis=1)
    else:
        rng = np.random.default_rng(0)
        n_perm = 10000
        signs = rng.choice([-1.0, 1.0], size=(n_perm, n))
        perm_stats = (signs * values[None, :]).mean(axis=1)

    return float(np.mean(np.abs(perm_stats) >= abs(observed)))


# ---------------------------------------------------------------------------
#  Effect-size helpers
# ---------------------------------------------------------------------------

def sample_cohens_d(values: np.ndarray | list[float]) -> float:
    """Cohen's d for a one-sample test (mean / SD)."""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    n = int(values.size)
    if n < 2:
        return float("nan")
    sd = float(np.std(values, ddof=1))
    if np.isclose(sd, 0.0):
        return float("nan")
    return float(np.mean(values) / sd)


def hedges_g(values: np.ndarray | list[float]) -> float:
    """Hedges' g (bias-corrected Cohen's d)."""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    n = int(values.size)
    if n < 3:
        return float("nan")
    d = sample_cohens_d(values)
    if not np.isfinite(d):
        return float("nan")
    correction = 1.0 - (3.0 / (4.0 * n - 9.0))
    return float(d * correction)


# ---------------------------------------------------------------------------
#  Priority scoring for repeat-mean summary rows
# ---------------------------------------------------------------------------

def compute_priority_scores(
    mean_effect: float,
    repeat_sd: float,
    positive_fraction: float,
    hedges_g_val: float,
) -> tuple[float, float]:
    """Composite stability and directional priority scores.

    Stability score = positive_fraction * (1 - cv), clamped to [0, 1].
    Directional score = sign(mean_effect) * min(|hedges_g| / 2, 1).
    """
    cv = repeat_sd / (abs(mean_effect) + 1e-8)
    stability_score = float(positive_fraction * max(0.0, 1.0 - cv))
    stability_score = max(0.0, min(1.0, stability_score))

    sign = 1.0 if mean_effect >= 0 else -1.0
    hedges_directional_score = float(sign * min(abs(hedges_g_val) / 2.0, 1.0))
    return stability_score, hedges_directional_score


# ---------------------------------------------------------------------------
#  Aggregate repeated-experiment summary
# ---------------------------------------------------------------------------

def aggregate_repeated_experiment_summary(
    per_cell_dfs: dict[str, pd.DataFrame],
    metrics: list[str] | None = None,
) -> pd.DataFrame:
    """Aggregate per-cell delta tables from multiple repeat seeds.

    Parameters
    ----------
    per_cell_dfs : dict  {label: per_cell_df}
        Each per_cell_df is a long-format DataFrame with at least columns
        ``repeat_seed`` and one column per metric.
    metrics : list[str] | None
        Metrics to summarise. Defaults to MAIN_METRICS + DIAGNOSTIC_METRICS.

    Returns
    -------
    pd.DataFrame with columns: label, metric, n_repeats,
    repeat_mean_of_means, repeat_std_of_means, repeat_positive_fraction,
    repeat_mean_hedges_g, repeat_mean_exact_sign_flip_p,
    repeat_mean_reject_zero_0p05, priority_stability_score,
    priority_hedges_directional_score
    """
    if metrics is None:
        metrics = list(MAIN_METRICS) + list(DIAGNOSTIC_METRICS)

    rows = []
    for label, per_cell_df in per_cell_dfs.items():
        if per_cell_df.empty:
            continue
        if "repeat_seed" not in per_cell_df.columns:
            continue

        for metric in metrics:
            metric = canonical_metric_name(metric)
            if metric not in per_cell_df.columns:
                continue

            repeat_group = (
                per_cell_df.groupby("repeat_seed", observed=True)[metric]
                .agg(["mean", "median"])
                .reset_index()
            )
            mean_vals = repeat_group["mean"].to_numpy(dtype=np.float64)
            mean_vals = mean_vals[np.isfinite(mean_vals)]
            if len(mean_vals) == 0:
                continue

            repeat_mean_of_means = float(np.mean(mean_vals))
            repeat_std_of_means = (
                float(np.std(mean_vals, ddof=1))
                if len(mean_vals) > 1
                else float("nan")
            )
            orientation = float(READOUT_ORIENTATION.get(metric, 1.0))
            oriented_mean_vals = orientation * mean_vals
            repeat_oriented_mean_of_means = float(np.mean(oriented_mean_vals))
            repeat_positive_fraction = float(np.mean(oriented_mean_vals > 0))
            repeat_mean_hedges_g = float(hedges_g(oriented_mean_vals))
            sign_flip_p = float(exact_sign_flip_p_value(mean_vals))

            stability, hedges_dir = compute_priority_scores(
                mean_effect=repeat_oriented_mean_of_means,
                repeat_sd=repeat_std_of_means,
                positive_fraction=repeat_positive_fraction,
                hedges_g_val=repeat_mean_hedges_g,
            )

            rows.append({
                "label": label,
                "metric": metric,
                "is_main_metric": bool(metric in MAIN_METRICS),
                "metric_axis": (
                    "geometry"
                    if metric in MAIN_GEOMETRY_METRICS
                    else ("identity" if metric in MAIN_IDENTITY_METRICS else "diagnostic")
                ),
                "n_repeats": int(len(mean_vals)),
                "repeat_mean_of_means": repeat_mean_of_means,
                "readout_orientation": orientation,
                "repeat_oriented_mean_of_means": repeat_oriented_mean_of_means,
                "repeat_std_of_means": repeat_std_of_means,
                "repeat_positive_fraction": repeat_positive_fraction,
                "repeat_mean_cohens_d": float(sample_cohens_d(mean_vals)),
                "repeat_mean_hedges_g": repeat_mean_hedges_g,
                "repeat_mean_exact_sign_flip_p": sign_flip_p,
                "repeat_mean_reject_zero_0p05": bool(sign_flip_p < 0.05),
                "priority_stability_score": (
                    float(stability) if np.isfinite(stability) else float("nan")
                ),
                "priority_hedges_directional_score": (
                    float(hedges_dir) if np.isfinite(hedges_dir) else float("nan")
                ),
                "repeat_mean_test": "exact sign-flip on repeat means",
                "stat_unit": "repeat_mean",
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
#  Rule-based hit classification
# ---------------------------------------------------------------------------

def build_rule_based_hit_classes(
    panel_summary_df: pd.DataFrame,
    fdr_threshold: float = 0.05,
    main_metrics: list[str] | None = None,
) -> pd.DataFrame:
    """Classify perturbation candidates into A/B/C/non-hit.

    Rules (applied per candidate label across the four MAIN_METRICS):
    - A_productive: at least one geometry AND one identity metric pass BH-FDR.
    - B_identity_only: at least one identity metric passes, but no geometry.
    - C_geometry_only: at least one geometry metric passes, but no identity.
    - D_non_significant: none pass.

    A metric passes when ``repeat_mean_exact_sign_flip_p < fdr_threshold``
    after BH-FDR correction across all labels for that metric.

    Parameters
    ----------
    panel_summary_df : pd.DataFrame
        Output of ``aggregate_repeated_experiment_summary``.
    fdr_threshold : float
    main_metrics : list[str] | None
        Default: MAIN_METRICS.

    Returns
    -------
    pd.DataFrame with columns: label, hit_class, geometry_hit_count,
    identity_hit_count, geometry_hit_metrics, identity_hit_metrics,
    total_hit_count, pareto_front
    """
    if main_metrics is None:
        main_metrics = list(MAIN_METRICS)

    df = panel_summary_df.copy()
    df = df[df["metric"].isin(main_metrics)].copy()

    if df.empty:
        return pd.DataFrame(columns=["label", "hit_class"])

    # BH-FDR correction per metric across all labels
    df["fdr_q"] = float("nan")
    for metric in df["metric"].unique():
        mask = df["metric"] == metric
        p_vals = df.loc[mask, "repeat_mean_exact_sign_flip_p"].to_numpy(
            dtype=np.float64
        )
        q_vals = bh_fdr(p_vals)
        df.loc[mask, "fdr_q"] = q_vals

    # Classify per label
    geom_metrics = [m for m in main_metrics if m in MAIN_GEOMETRY_METRICS]
    id_metrics = [m for m in main_metrics if m in MAIN_IDENTITY_METRICS]

    hit_rows = []
    for label, grp in df.groupby("label", observed=True):
        grp = grp.copy()
        if "repeat_oriented_mean_of_means" not in grp.columns:
            grp["repeat_oriented_mean_of_means"] = grp.apply(
                lambda r: READOUT_ORIENTATION.get(str(r["metric"]), 1.0)
                * float(r["repeat_mean_of_means"]),
                axis=1,
            )
        grp["pass_fdr"] = (
            (grp["fdr_q"] < fdr_threshold)
            & (grp["repeat_oriented_mean_of_means"] > 0)
        )

        geom_pass = grp[grp["metric"].isin(geom_metrics) & grp["pass_fdr"]]
        id_pass = grp[grp["metric"].isin(id_metrics) & grp["pass_fdr"]]

        geom_hit_metrics = sorted(geom_pass["metric"].unique().tolist())
        id_hit_metrics = sorted(id_pass["metric"].unique().tolist())
        geom_n = len(geom_hit_metrics)
        id_n = len(id_hit_metrics)
        total_n = geom_n + id_n

        if geom_n >= 1 and id_n >= 1:
            hit_class = "A_productive"
        elif id_n >= 1 and geom_n == 0:
            hit_class = "B_identity_only"
        elif geom_n >= 1 and id_n == 0:
            hit_class = "C_geometry_only"
        else:
            hit_class = "D_non_significant"

        hit_rows.append({
            "label": label,
            "hit_class": hit_class,
            "geometry_hit_count": geom_n,
            "identity_hit_count": id_n,
            "geometry_hit_metrics": ",".join(geom_hit_metrics),
            "identity_hit_metrics": ",".join(id_hit_metrics),
            "total_hit_count": total_n,
        })

    hit_df = pd.DataFrame(hit_rows)

    # Pareto front: A_productive hits that are not dominated by another
    # A_productive hit in both geometry and identity counts
    if not hit_df.empty:
        hit_df["pareto_front"] = False
        a_mask = hit_df["hit_class"] == "A_productive"
        a_df = hit_df.loc[a_mask].copy()
        if len(a_df) >= 1:
            dominated = np.zeros(len(a_df), dtype=bool)
            gc = a_df["geometry_hit_count"].to_numpy()
            ic = a_df["identity_hit_count"].to_numpy()
            for i in range(len(a_df)):
                for j in range(len(a_df)):
                    if i == j:
                        continue
                    if gc[j] >= gc[i] and ic[j] >= ic[i] and (
                        gc[j] > gc[i] or ic[j] > ic[i]
                    ):
                        dominated[i] = True
                        break
            pareto_indices = a_df.index[~dominated]
            hit_df.loc[pareto_indices, "pareto_front"] = True

    return hit_df


# ---------------------------------------------------------------------------
#  Panel summary table helpers
# ---------------------------------------------------------------------------

def subset_panel_summary_metrics(
    panel_summary_df: pd.DataFrame,
    include_diagnostic: bool = False,
) -> pd.DataFrame:
    """Filter panel summary to canonical main (and optionally diagnostic) metrics."""
    metric_keep = list(MAIN_METRICS)
    if include_diagnostic:
        metric_keep = metric_keep + list(DIAGNOSTIC_METRICS)

    out = panel_summary_df.copy()
    if "metric" in out.columns:
        out["metric"] = out["metric"].map(canonical_metric_name)
        out = out[out["metric"].isin(metric_keep)].copy()
    return out


def split_panel_summary_tables(
    panel_summary_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split panel summary into main-metric and diagnostic-metric tables."""
    canonical_df = panel_summary_df.copy()
    if "metric" in canonical_df.columns:
        canonical_df["metric"] = canonical_df["metric"].map(canonical_metric_name)
    panel_summary_main = canonical_df[
        canonical_df["metric"].isin(MAIN_METRICS)
    ].copy()
    panel_summary_diag = canonical_df[
        canonical_df["metric"].isin(DIAGNOSTIC_METRICS)
    ].copy()
    return panel_summary_main, panel_summary_diag


# ---------------------------------------------------------------------------
#  Rollout candidate selection helpers
# ---------------------------------------------------------------------------

def select_rollout_labels_from_hits(
    hit_df: pd.DataFrame,
    classes: tuple[str, ...] = ("A_productive",),
    pareto_only: bool = False,
) -> list[str]:
    """Select candidate labels from hit-classification DataFrame."""
    df = hit_df.copy()
    df = df.loc[~df["label"].astype(str).map(is_control_label)].copy()
    df = df[df["hit_class"].isin(classes)].copy()
    if pareto_only and "pareto_front" in df.columns:
        df = df[df["pareto_front"]]
    return sorted(df["label"].dropna().unique().tolist())


def auto_select_rollout_candidates(
    hit_df: pd.DataFrame,
    include_random_control: bool = True,
    pareto_only: bool = False,
    n_a: int = 2,
    n_b: int = 2,
    n_c: int = 1,
) -> list[str]:
    """Automatic selection of rollout candidates from hit classification.

    Picks top *n_a* A_productive, *n_b* B_identity_only, *n_c*
    C_geometry_only by composite rank score, plus one random control.
    """
    if hit_df is None or len(hit_df) == 0:
        return []

    df_all = hit_df.copy()
    random_pool = df_all[
        df_all["label"].astype(str).str.startswith("RANDOM_CTRL_")
    ].copy()

    df = df_all.loc[
        ~df_all["label"].astype(str).map(is_control_label)
    ].copy()
    if pareto_only and "pareto_front" in df.columns:
        df = df[df["pareto_front"]].copy()

    # Composite rank score
    score_cols = [c for c in ["geom_score", "identity_score"] if c in df.columns]
    if len(score_cols) >= 1:
        df["_rank_score"] = pd.to_numeric(
            df[score_cols].sum(axis=1), errors="coerce"
        ).fillna(0)
    else:
        df["_rank_score"] = 0.0

    picked: list[str] = []
    for cls, n_pick in [
        ("A_productive", n_a),
        ("B_identity_only", n_b),
        ("C_geometry_only", n_c),
    ]:
        sub = df[df["hit_class"] == cls].sort_values(
            "_rank_score", ascending=False
        )
        picked.extend(
            sub["label"].head(max(0, int(n_pick))).tolist()
        )

    if include_random_control and not random_pool.empty:
        if "_rank_score" not in random_pool.columns:
            random_pool["_rank_score"] = 0.0
        picked.append(
            str(
                random_pool.sort_values("_rank_score", ascending=False)
                .iloc[0]["label"]
            )
        )

    # Deduplicate while preserving order
    seen: set[str] = set()
    result = []
    for x in picked:
        if x not in seen:
            seen.add(x)
            result.append(x)
    return result


def build_candidate_random_calibration_df(
    panel_summary_main_df: pd.DataFrame,
    random_prefix: str = "RANDOM_CTRL_",
) -> pd.DataFrame:
    """Build calibration table contrasting candidates against random controls.

    Output is one row per candidate-label × metric with candidate effect and
    random-control baseline statistics for quick scatter/threshold plots.
    """
    if panel_summary_main_df is None or panel_summary_main_df.empty:
        return pd.DataFrame()

    df = panel_summary_main_df.copy()
    required = {"label", "metric", "repeat_mean_of_means"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"panel_summary_main_df missing required columns: {missing}"
        )

    random_mask = df["label"].astype(str).str.startswith(random_prefix)
    random_df = df[random_mask].copy()
    cand_df = df[~random_mask].copy()
    if random_df.empty or cand_df.empty:
        return pd.DataFrame()

    random_stats = (
        random_df.groupby("metric", observed=True)["repeat_mean_of_means"]
        .agg(["mean", "std", "median"])
        .reset_index()
        .rename(
            columns={
                "mean": "random_mean",
                "std": "random_std",
                "median": "random_median",
            }
        )
    )
    random_q = (
        random_df.groupby("metric", observed=True)["repeat_mean_of_means"]
        .quantile([0.05, 0.95])
        .unstack()
        .reset_index()
        .rename(columns={0.05: "random_q05", 0.95: "random_q95"})
    )
    random_stats = random_stats.merge(random_q, on="metric", how="left")

    out = cand_df.merge(random_stats, on="metric", how="left")
    out["z_vs_random"] = (
        (out["repeat_mean_of_means"] - out["random_mean"])
        / (out["random_std"].replace(0, np.nan))
    )
    out["readout_orientation"] = out["metric"].map(
        lambda m: float(READOUT_ORIENTATION.get(str(m), 1.0))
    )
    out["oriented_z_vs_random"] = out["readout_orientation"] * out["z_vs_random"]
    out["outside_random_q05_q95"] = (
        (out["repeat_mean_of_means"] < out["random_q05"])
        | (out["repeat_mean_of_means"] > out["random_q95"])
    )
    return out
