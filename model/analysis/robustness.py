"""Source-ranking robustness metrics, baseline comparisons, and statistical summaries.

Provides pure-dataframe functions migrated from
``demo/benchmark_source_ranking_robustness.ipynb``:

- Multi-readout scoring, response-mode classification
- Baseline ranking evaluation against perturbation panels
- Random-TF null-distribution generation
- Seed-to-seed rank correlations and top-k overlaps
- Bootstrap CI aggregation for primary readouts

All functions operate on pandas DataFrames and numpy arrays; they do not
depend on PyTorch, AnnData, or model objects (except for optional AnnData-based
scoring functions that accept pre-extracted dense matrices).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from model.utils.constants import (
    get_cluster_column,
    get_readout_labels,
    get_readout_orientation,
    get_time_column,
)


# ---------------------------------------------------------------------------
#  Readout configuration (canonical)
# ---------------------------------------------------------------------------

DEFAULT_READOUT_ORIENTATION: dict[str, float] = get_readout_orientation()
DEFAULT_READOUT_LABELS: dict[str, str] = get_readout_labels()


def _tensor_scalar(x: Any) -> float:
    """Convert scalar-like tensor/value to float without importing torch."""
    if hasattr(x, "detach") and hasattr(x, "cpu") and hasattr(x, "item"):
        return float(x.detach().cpu().item())
    return float(x)


def _tensor_int(x: Any) -> int:
    """Convert scalar-like tensor/value to int without importing torch."""
    if hasattr(x, "detach") and hasattr(x, "cpu") and hasattr(x, "item"):
        return int(x.detach().cpu().item())
    return int(x)


def ensure_time_bin_series(
    adata: Any,
    n_bins: int = 120,
    time_col: str | None = None,
) -> pd.Series:
    """Return time-bin series aligned to ``adata.obs``.

    Notebook source: ``benchmark_source_ranking_robustness.ipynb`` Figure 1D cell.
    """
    from model.utils.constants import get_time_column

    obs = adata.obs
    if "time_bin" in obs.columns:
        return obs["time_bin"].astype(int).reset_index(drop=True)
    _time_col = time_col or get_time_column()
    if _time_col in obs.columns:
        return pd.qcut(
            obs[_time_col], q=n_bins, labels=False, duplicates="drop"
        ).astype(int).reset_index(drop=True)
    if "norm_time" in obs.columns:
        return pd.qcut(
            obs["norm_time"], q=n_bins, labels=False, duplicates="drop"
        ).astype(int).reset_index(drop=True)
    raise ValueError(
        f"adata.obs must contain one of: time_bin, {_time_col}, norm_time"
    )


def transition_pair_metadata(
    adata: Any,
    items: list[dict[str, Any]],
    cluster_col: str | None = None,
    time_col: str | None = None,
    n_bins: int = 120,
) -> pd.DataFrame:
    """Build metadata for already-constructed transition pairs.

    Notebook source: ``benchmark_source_ranking_robustness.ipynb`` Figure 1D cell.
    """
    from model.utils.constants import get_cluster_column

    obs = adata.obs.reset_index(drop=True)
    _cluster_col = cluster_col or get_cluster_column()
    if _cluster_col not in obs.columns:
        raise ValueError(f"cluster_col not found in adata.obs: {_cluster_col}")

    time_bins = ensure_time_bin_series(adata, n_bins=n_bins, time_col=time_col)
    cluster_values = obs[_cluster_col].astype(str).to_numpy()
    rows: list[dict[str, Any]] = []
    for item_i, item in enumerate(items):
        c_idx = _tensor_int(item["c_idx"])
        target_idx = _tensor_int(item["target_idx"])
        rows.append(
            {
                "item_i": item_i,
                "c_idx": c_idx,
                "target_idx": target_idx,
                "source_cluster": cluster_values[c_idx],
                "target_cluster": cluster_values[target_idx],
                "source_bin": int(time_bins.iloc[c_idx]),
                "target_bin": int(time_bins.iloc[target_idx]),
                "source_time": _tensor_scalar(item["time"]),
                "target_time": _tensor_scalar(item["target_time"]),
                "offset_bins": int(time_bins.iloc[target_idx] - time_bins.iloc[c_idx]),
            }
        )
    return pd.DataFrame(rows)


def choose_same_target_bin_pairs(
    adata: Any,
    items: list[dict[str, Any]],
    min_cells_per_cluster: int = 10,
    min_source_states: int = 2,
    n_per_cluster: int = 10,
    seed: int = 42,
    allow_illustrative: bool = True,
    cluster_col: str | None = None,
    time_col: str | None = None,
    n_bins: int = 120,
) -> tuple[list[dict[str, Any]], pd.DataFrame, pd.DataFrame]:
    """Choose legal pairs sharing one target bin but from different source states.

    Notebook source: ``benchmark_source_ranking_robustness.ipynb`` Figure 1D cell.
    """
    rng = np.random.default_rng(seed)
    meta = transition_pair_metadata(
        adata=adata,
        items=items,
        cluster_col=cluster_col,
        time_col=time_col,
        n_bins=n_bins,
    )
    if meta.empty:
        raise ValueError("No transition pairs are available for same-target-bin selection.")

    search_settings = [(min_cells_per_cluster, min_source_states, "strict")]
    if allow_illustrative:
        search_settings.append((2, 2, "illustrative"))

    for min_count, min_states, evidence_level in search_settings:
        candidates: list[dict[str, Any]] = []
        for target_bin, sub in meta.groupby("target_bin"):
            counts = sub["source_cluster"].value_counts()
            usable_clusters = counts[counts >= min_count].index.tolist()
            if len(usable_clusters) >= min_states:
                candidates.append(
                    {
                        "target_bin": int(target_bin),
                        "n_source_states": int(len(usable_clusters)),
                        "n_pairs": int(sub[sub["source_cluster"].isin(usable_clusters)].shape[0]),
                        "min_count": int(min_count),
                        "usable_clusters": usable_clusters,
                        "evidence_level": evidence_level,
                    }
                )
        if not candidates:
            continue

        candidate_bins = pd.DataFrame(candidates).sort_values(
            ["n_source_states", "n_pairs", "target_bin"],
            ascending=[False, False, False],
        )
        cand = candidate_bins.iloc[0]
        target_bin = int(cand["target_bin"])
        clusters = list(cand["usable_clusters"][:5])
        target_sub = meta[meta["target_bin"].eq(target_bin)]
        selected: list[int] = []
        for cl in clusters:
            idx = target_sub[target_sub["source_cluster"].eq(cl)]["item_i"].to_numpy(dtype=int)
            chosen = rng.choice(idx, size=min(n_per_cluster, len(idx)), replace=False)
            selected.extend(chosen.tolist())
        selected_meta = meta.set_index("item_i").loc[selected].reset_index()
        selected_meta["evidence_level"] = str(cand["evidence_level"])
        selected_meta["min_pairs_per_source_state"] = int(cand["min_count"])
        selected_items = [items[i] for i in selected]
        return selected_items, selected_meta, candidate_bins

    top = (
        meta.groupby("target_bin")["source_cluster"]
        .nunique()
        .sort_values(ascending=False)
        .head(10)
    )
    raise ValueError(
        "Could not find a legal target bin with multiple source states. "
        f"Top target-bin source-state counts were: {top.to_dict()}"
    )


# ---------------------------------------------------------------------------
#  Label parsing
# ---------------------------------------------------------------------------

def parse_label(label: str) -> tuple[str, str]:
    """Split a label like 'MYT1L_OE' into (gene, mode)."""
    parts = str(label).rsplit("_", 1)
    if len(parts) == 2 and parts[1] in {"OE", "KD", "KO"}:
        return parts[0].replace("RANDOM_CTRL_", ""), parts[1]
    return str(label), "unknown"


# ---------------------------------------------------------------------------
#  Multi-readout scoring
# ---------------------------------------------------------------------------

def compute_multi_readout_scores(
    panel_df: pd.DataFrame,
    readout_orientation: dict[str, float] | None = None,
    readout_metrics: list[str] | None = None,
) -> pd.DataFrame:
    """Add z-scored multi-readout columns to a candidate panel DataFrame.

    Expects columns matching *readout_metrics* and an ``is_control`` boolean
    column. Computes oriented values, per-readout z-scores, a composite
    ``multi_readout_score``, and ``dominant_readout``.

    Returns a new DataFrame with added columns:
    ``oriented_{m}``, ``z_{m}``, ``multi_readout_score``, ``dominant_readout``.
    """
    if readout_orientation is None:
        readout_orientation = DEFAULT_READOUT_ORIENTATION
    if readout_metrics is None:
        readout_metrics = list(readout_orientation.keys())

    df = panel_df.copy()

    for metric, sign in readout_orientation.items():
        if metric not in df.columns:
            continue
        df[f"oriented_{metric}"] = sign * df[metric]

    non_ctrl = df.loc[~df["is_control"]]
    for metric in readout_metrics:
        z_col = f"z_{metric}"
        if f"oriented_{metric}" not in df.columns:
            df[z_col] = 0.0
            continue
        mu = non_ctrl[f"oriented_{metric}"].mean()
        sd = non_ctrl[f"oriented_{metric}"].std(ddof=0)
        df[z_col] = (df[f"oriented_{metric}"] - mu) / (sd + 1e-12)

    z_cols = [f"z_{m}" for m in readout_metrics if f"z_{m}" in df.columns]
    if z_cols:
        df["multi_readout_score"] = df[z_cols].mean(axis=1)
        df["dominant_readout"] = (
            df[z_cols].idxmax(axis=1).str.replace("z_", "", regex=False)
        )
    else:
        df["multi_readout_score"] = 0.0
        df["dominant_readout"] = "unknown"

    return df


def classify_response_modes(
    scores_df: pd.DataFrame,
    readout_metrics: list[str] | None = None,
    z_threshold: float = 0.5,
) -> pd.DataFrame:
    """Classify each row into a response mode based on z-score patterns.

    Modes: ``concordant``, ``progression-biased``, ``identity-biased``,
    ``geometry-biased``, ``mixed/weak``.

    Returns a new DataFrame with an added ``response_mode`` column.
    """
    if readout_metrics is None:
        readout_metrics = list(DEFAULT_READOUT_ORIENTATION.keys())

    df = scores_df.copy()
    z_cols = [f"z_{m}" for m in readout_metrics if f"z_{m}" in df.columns]
    if not z_cols:
        df["response_mode"] = "mixed/weak"
        return df

    dominant = df.get("dominant_readout", pd.Series("unknown", index=df.index))

    n_above = (df[z_cols] > z_threshold).sum(axis=1)

    conditions = [
        n_above >= 3,
        dominant.eq("delta_path_progress"),
        dominant.eq("delta_logit_cluster15")
        | dominant.eq("delta_path_index_expectation"),
        dominant.eq("delta_path_deviation"),
    ]
    choices = [
        "concordant",
        "progression-biased",
        "identity-biased",
        "geometry-biased",
    ]

    df["response_mode"] = np.select(conditions, choices, default="mixed/weak")
    return df


# ---------------------------------------------------------------------------
#  Baseline ranking evaluation
# ---------------------------------------------------------------------------

def evaluate_gene_ranking_against_panel(
    ranking_df: pd.DataFrame,
    baseline_name: str,
    panel_scores: pd.DataFrame,
    top_k: int = 8,
    gene_col: str = "Gene",
    score_col: str = "score",
) -> pd.DataFrame:
    """Score a gene ranking baseline against multi-readout perturbation panel.

    For each gene selected by the baseline ranking, picks the best-observed
    perturbation mode (highest ``multi_readout_score``) and returns the
    selected rows augmented with baseline metadata.

    Parameters
    ----------
    ranking_df : DataFrame
        At minimum a column of gene names and a score column (higher = better).
    baseline_name : str
        Label for this baseline strategy.
    panel_scores : DataFrame
        Must contain ``is_control``, ``gene``, ``multi_readout_score`` columns.
    top_k : int
        Number of top-ranked genes to evaluate.
    gene_col : str
        Column name for gene identifiers in *ranking_df*.
    score_col : str
        Column name for ranking scores in *ranking_df*.

    Returns
    -------
    DataFrame with selected perturbation rows plus ``baseline`` and
    ``baseline_rank`` columns.
    """
    tested = panel_scores.loc[~panel_scores["is_control"]].copy()
    tested_genes = set(tested["gene"].astype(str))

    ranking = ranking_df.copy()
    ranking[gene_col] = ranking[gene_col].astype(str)
    ranking = ranking[
        ranking[gene_col].isin(tested_genes)
    ].drop_duplicates(gene_col, keep="first")

    if ranking.empty:
        return pd.DataFrame()

    chosen_genes = ranking.head(top_k)[gene_col].tolist()
    chosen = tested[tested["gene"].isin(chosen_genes)].copy()
    chosen = (
        chosen.sort_values("multi_readout_score", ascending=False)
        .drop_duplicates("gene", keep="first")
    )
    chosen["baseline"] = baseline_name
    chosen["baseline_rank"] = chosen["gene"].map(
        {g: i + 1 for i, g in enumerate(chosen_genes)}
    )
    return chosen


def random_tf_baseline(
    panel_scores: pd.DataFrame,
    n_sets: int = 500,
    top_k: int = 8,
    seed: int = 42,
    readout_metrics: list[str] | None = None,
) -> pd.DataFrame:
    """Generate a random-TF null distribution for multi-readout scores.

    Repeatedly samples *top_k* genes uniformly from the tested (non-control)
    panel, picks each gene's best mode, and records mean readout scores.

    Returns a DataFrame with columns:
    ``baseline``, ``set_id``, ``multi_readout_score``, and one column per
    oriented metric.
    """
    if readout_metrics is None:
        readout_metrics = list(DEFAULT_READOUT_ORIENTATION.keys())

    rng = np.random.default_rng(seed)
    tested = panel_scores.loc[~panel_scores["is_control"]].copy()
    genes = np.array(sorted(tested["gene"].unique()))
    if len(genes) == 0:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    effective_k = min(top_k, len(genes))
    for i in range(n_sets):
        chosen_genes = rng.choice(genes, size=effective_k, replace=False)
        chosen = (
            tested[tested["gene"].isin(chosen_genes)]
            .sort_values("multi_readout_score", ascending=False)
            .drop_duplicates("gene", keep="first")
        )
        row: dict[str, Any] = {
            "baseline": "random TF ranking",
            "set_id": i,
            "multi_readout_score": chosen["multi_readout_score"].mean(),
        }
        for m in readout_metrics:
            col = f"oriented_{m}"
            row[col] = chosen[col].mean() if col in chosen.columns else np.nan
        rows.append(row)

    return pd.DataFrame(rows)


def build_baseline_comparison_summary(
    selected_rows_df: pd.DataFrame,
    random_sets_df: pd.DataFrame,
    readout_labels: dict[str, str] | None = None,
    readout_metrics: list[str] | None = None,
) -> pd.DataFrame:
    """Combine baseline-selected and random-baseline summaries into one table.

    Returns a DataFrame with columns: ``baseline``, ``n_genes``,
    ``multi_readout_score``, and one column per labelled readout.
    """
    if readout_metrics is None:
        readout_metrics = list(DEFAULT_READOUT_ORIENTATION.keys())
    if readout_labels is None:
        readout_labels = DEFAULT_READOUT_LABELS

    if selected_rows_df.empty:
        baseline_summary = pd.DataFrame()
    else:
        agg: dict[str, str | Any] = {
            "n_genes": ("gene", "nunique"),
            "multi_readout_score": ("multi_readout_score", "mean"),
        }
        for m in readout_metrics:
            label = readout_labels.get(m, m)
            agg[label] = (f"oriented_{m}", "mean")

        baseline_summary = (
            selected_rows_df.groupby("baseline")
            .agg(**agg)
            .reset_index()
        )

    random_summary = pd.DataFrame()
    if not random_sets_df.empty:
        rand_row: dict[str, Any] = {
            "baseline": "random TF ranking",
            "n_genes": 8,
            "multi_readout_score": random_sets_df["multi_readout_score"].mean(),
        }
        for m in readout_metrics:
            col = f"oriented_{m}"
            label = readout_labels.get(m, m)
            rand_row[label] = (
                random_sets_df[col].mean() if col in random_sets_df.columns else np.nan
            )
        random_summary = pd.DataFrame([rand_row])

    return pd.concat([baseline_summary, random_summary], ignore_index=True)


# ---------------------------------------------------------------------------
#  Seed-to-seed robustness
# ---------------------------------------------------------------------------

def ranking_correlation_table(
    df: pd.DataFrame,
    score_col: str = "multi_readout_score",
    setting_col: str = "repeat_seed",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute pairwise Spearman correlations of candidate rankings across settings.

    Returns
    -------
    corr_df : DataFrame
        Columns: ``setting_a``, ``setting_b``, ``spearman_rho``, ``n_labels``.
    piv : DataFrame
        Pivot table (label × setting) of scores for inspection.
    """
    from scipy.stats import spearmanr

    piv = df.loc[~df["is_control"]].pivot_table(
        index="label", columns=setting_col, values=score_col
    )
    cols = list(piv.columns)
    rows: list[dict[str, Any]] = []
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            tmp = piv[[a, b]].dropna()
            rho = spearmanr(tmp[a], tmp[b]).correlation if len(tmp) >= 3 else np.nan
            rows.append(
                {
                    "setting_a": a,
                    "setting_b": b,
                    "spearman_rho": rho,
                    "n_labels": len(tmp),
                }
            )
    return pd.DataFrame(rows), piv


def topk_overlap_table(
    df: pd.DataFrame,
    score_col: str = "multi_readout_score",
    setting_col: str = "repeat_seed",
    k: int = 8,
) -> pd.DataFrame:
    """Compute pairwise Jaccard overlap of top-k labels across settings.

    Returns
    -------
    DataFrame with columns: ``setting_a``, ``setting_b``, ``top_k``,
    ``jaccard``, ``overlap_n``.
    """
    top: dict[Any, set[str]] = {}
    for setting, sub in df.loc[~df["is_control"]].groupby(setting_col):
        top[setting] = set(
            sub.sort_values(score_col, ascending=False).head(k)["label"]
        )

    rows: list[dict[str, Any]] = []
    keys = list(top)
    for i, a in enumerate(keys):
        for b in keys[i + 1 :]:
            inter = len(top[a] & top[b])
            union = len(top[a] | top[b])
            rows.append(
                {
                    "setting_a": a,
                    "setting_b": b,
                    "top_k": k,
                    "jaccard": inter / union if union else np.nan,
                    "overlap_n": inter,
                }
            )
    return pd.DataFrame(rows)


def aggregate_robustness_summary(
    seed_corr_df: pd.DataFrame,
    topk_df: pd.DataFrame,
) -> dict[str, Any]:
    """Return scalar robustness summary statistics.

    Returns a dict with: ``median_seed_spearman``,
    ``median_topk_jaccard`` (*k* taken from *topk_df* if consistent),
    ``n_seed_pairs``, ``n_topk_pairs``.
    """
    summary: dict[str, Any] = {
        "median_seed_spearman": np.nan,
        "median_topk_jaccard": np.nan,
        "n_seed_pairs": len(seed_corr_df),
        "n_topk_pairs": len(topk_df),
    }
    if not seed_corr_df.empty and "spearman_rho" in seed_corr_df.columns:
        summary["median_seed_spearman"] = float(
            seed_corr_df["spearman_rho"].median()
        )
    if not topk_df.empty and "jaccard" in topk_df.columns:
        summary["median_topk_jaccard"] = float(topk_df["jaccard"].median())
    return summary


# ---------------------------------------------------------------------------
#  Bootstrap CI helpers
# ---------------------------------------------------------------------------

def bootstrap_ci_readout_table(
    bootstrap_df: pd.DataFrame,
    main_metrics: list[str] | None = None,
    readout_orientation: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Aggregate bootstrap CIs across seeds for primary readout metrics.

    Expects *bootstrap_df* with columns: ``label``, ``metric``, ``stat``,
    ``point_estimate``, ``ci_low``, ``ci_high``. Filters to stat="mean"
    and metrics in *main_metrics*.

    Returns a DataFrame with per-label/per-metric mean point estimates
    and oriented CI bounds: ``oriented_point``, ``oriented_ci_low``,
    ``oriented_ci_high``.
    """
    if main_metrics is None:
        main_metrics = list(DEFAULT_READOUT_ORIENTATION.keys())
    if readout_orientation is None:
        readout_orientation = DEFAULT_READOUT_ORIENTATION

    if bootstrap_df.empty or "metric" not in bootstrap_df.columns:
        return pd.DataFrame()

    mask = (
        bootstrap_df["metric"].isin(main_metrics)
        & bootstrap_df["stat"].eq("mean")
    )
    if not mask.any():
        return pd.DataFrame()

    boot_mean = (
        bootstrap_df.loc[mask]
        .groupby(["label", "metric"])
        .agg(
            point_estimate=("point_estimate", "mean"),
            ci_low=("ci_low", "mean"),
            ci_high=("ci_high", "mean"),
        )
        .reset_index()
    )

    def _oriented(row: pd.Series, col: str) -> float:
        sign = readout_orientation.get(row["metric"], 1.0)
        return sign * float(row[col])

    boot_mean["oriented_point"] = boot_mean.apply(
        lambda r: _oriented(r, "point_estimate"), axis=1
    )
    boot_mean["oriented_ci_low"] = boot_mean.apply(
        lambda r: min(
            _oriented(r, "ci_low"),
            _oriented(r, "ci_high"),
        ),
        axis=1,
    )
    boot_mean["oriented_ci_high"] = boot_mean.apply(
        lambda r: max(
            _oriented(r, "ci_low"),
            _oriented(r, "ci_high"),
        ),
        axis=1,
    )
    return boot_mean


# ---------------------------------------------------------------------------
#  Repeat-seed aggregation
# ---------------------------------------------------------------------------

def aggregate_repeat_seed_readout_summary(
    per_cell_df: pd.DataFrame,
    main_metrics: list[str] | None = None,
    readout_orientation: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Aggregate per-cell delta values into per-repeat-seed mean readouts.

    Expects *per_cell_df* with columns: ``repeat_seed``, ``label``,
    ``is_control``, plus metric columns.

    Returns a DataFrame with addded oriented and multi-readout-score columns.
    """
    if main_metrics is None:
        main_metrics = list(DEFAULT_READOUT_ORIENTATION.keys())
    if readout_orientation is None:
        readout_orientation = DEFAULT_READOUT_ORIENTATION

    grp_cols = [c for c in ["repeat_seed", "label", "is_control"] if c in per_cell_df.columns]
    if not grp_cols:
        raise ValueError("per_cell_df must have repeat_seed, label, or is_control columns")

    repeat_summary = (
        per_cell_df.groupby(grp_cols, dropna=False)[[m for m in main_metrics if m in per_cell_df.columns]]
        .mean()
        .reset_index()
    )

    for metric, sign in readout_orientation.items():
        if metric in repeat_summary.columns:
            repeat_summary[f"oriented_{metric}"] = sign * repeat_summary[metric]

    non_ctrl = repeat_summary.loc[~repeat_summary["is_control"]]
    z_cols = []
    for m in main_metrics:
        ori_col = f"oriented_{m}"
        z_col = f"z_{m}"
        if ori_col not in repeat_summary.columns:
            continue
        mu = non_ctrl[ori_col].mean()
        sd = non_ctrl[ori_col].std(ddof=0)
        repeat_summary[z_col] = (repeat_summary[ori_col] - mu) / (sd + 1e-12)
        z_cols.append(z_col)

    if z_cols:
        repeat_summary["multi_readout_score"] = repeat_summary[z_cols].mean(axis=1)
    else:
        repeat_summary["multi_readout_score"] = 0.0

    return repeat_summary


# ---------------------------------------------------------------------------
#  Optional: AnnData-based baseline scorers (legacy/optional helpers)
# ---------------------------------------------------------------------------

def score_latent_time_correlation(
    adata: Any,
    X: np.ndarray,
    time_col: str | None = None,
) -> pd.DataFrame:
    """Compute |Pearson r| of each gene vs latent time.

    *Notebook-scoped helper.* Requires AnnData and dense expression matrix.
    This is provided for reproducibility of the conventional-baseline
    comparison; it is not part of the core thesis analysis pipeline.
    """
    from scipy.stats import pearsonr

    t = pd.to_numeric(adata.obs[time_col], errors="coerce").to_numpy(dtype=np.float32)
    mask = np.isfinite(t)
    rows: list[dict[str, Any]] = []
    n_genes = X.shape[1]
    for j in range(n_genes):
        x = X[mask, j]
        if np.std(x) < 1e-12 or np.std(t[mask]) < 1e-12:
            r: float = np.nan
        else:
            r = pearsonr(x, t[mask])[0]
        gene = str(adata.var_names[j]) if hasattr(adata, "var_names") else str(j)
        rows.append({"Gene": gene, "score": abs(float(r)) if np.isfinite(r) else 0.0})
    return pd.DataFrame(rows).sort_values("score", ascending=False)


def score_expression_abundance_variance(
    X: np.ndarray,
    gene_names: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return abundance and variance rankings for expression matrix *X*.

    *Notebook-scoped helper.* Requires dense expression matrix.
    """
    n_genes = X.shape[1]
    if gene_names is None:
        gene_names = [str(i) for i in range(n_genes)]
    abund = pd.DataFrame(
        {"Gene": gene_names[:n_genes], "score": X.mean(axis=0), "baseline": "expression abundance"}
    ).sort_values("score", ascending=False)
    var_df = pd.DataFrame(
        {"Gene": gene_names[:n_genes], "score": X.var(axis=0), "baseline": "expression variance"}
    ).sort_values("score", ascending=False)
    return abund, var_df
    if time_col is None:
        time_col = get_time_column()

    if cluster_col is None:
        cluster_col = get_cluster_column()
    if time_col is None:
        time_col = get_time_column()

    if time_col is None:
        time_col = get_time_column()
