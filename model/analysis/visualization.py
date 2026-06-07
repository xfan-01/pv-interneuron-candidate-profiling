"""Plotting helpers for thesis figures.

Notebook mappings (traceability):
  - classifier_analysis.ipynb cells 7, 16-19, 24
  - classifier_analysis_multi.ipynb cells 8, 12
  - generator_analysis_3.ipynb cells 6, 7, 17, 18, 20
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from model.utils.constants import get_pv_identity_markers


def _lazy_plot_libs() -> tuple[Any, Any]:
    import matplotlib.pyplot as plt
    import seaborn as sns

    return plt, sns


def _plot_main_metric_heatmap(
    panel_summary_main_df: pd.DataFrame,
    value_col: str = "repeat_mean_of_means",
    figsize: tuple[float, float] = (10, 6),
    center: float = 0.0,
    cmap: str = "RdBu_r",
):
    """Plot label × metric heatmap for main panel summary values."""
    plt, sns = _lazy_plot_libs()
    if panel_summary_main_df.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title("Main Metric Heatmap (empty)")
        return fig, ax

    pivot = panel_summary_main_df.pivot_table(
        index="label", columns="metric", values=value_col, aggfunc="mean"
    )
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        pivot,
        cmap=cmap,
        center=center,
        linewidths=0.5,
        linecolor="white",
        ax=ax,
    )
    ax.set_title("Main Metric Heatmap")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Candidate")
    fig.tight_layout()
    return fig, ax


def plot_persistence_duration_bars(
    persistence_scores_df: pd.DataFrame,
    label_col: str = "label",
    value_col: str = "Strict_Duration",
    metric_col: str = "metric",
    figsize: tuple[float, float] = (10, 5),
):
    """Bar chart of strict-duration persistence by candidate/metric."""
    plt, sns = _lazy_plot_libs()
    fig, ax = plt.subplots(figsize=figsize)
    if persistence_scores_df.empty:
        ax.set_title("Persistence Duration (empty)")
        return fig, ax

    plot_df = persistence_scores_df.copy()
    sns.barplot(
        data=plot_df,
        x=label_col,
        y=value_col,
        hue=metric_col,
        ax=ax,
    )
    ax.set_title("Persistence Strict Duration")
    ax.set_xlabel("Candidate")
    ax.set_ylabel(value_col)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return fig, ax


def plot_persistence_ribbon(
    stepwise_summary_df: pd.DataFrame,
    label: str,
    metric: str,
    step_col: str = "step",
    value_col: str = "point_estimate",
    low_col: str = "ci_low",
    high_col: str = "ci_high",
    figsize: tuple[float, float] = (8, 4),
):
    """Plot one candidate/metric persistence ribbon across rollout steps."""
    plt, _ = _lazy_plot_libs()
    fig, ax = plt.subplots(figsize=figsize)

    df = stepwise_summary_df.copy()
    df = df[(df["label"] == label) & (df["metric"] == metric)].copy()
    if df.empty:
        ax.set_title(f"Persistence Ribbon (empty): {label} / {metric}")
        return fig, ax

    df = df.sort_values(step_col)
    x = df[step_col].to_numpy(dtype=float)
    y = df[value_col].to_numpy(dtype=float)
    y0 = df[low_col].to_numpy(dtype=float)
    y1 = df[high_col].to_numpy(dtype=float)

    ax.plot(x, y, marker="o", linewidth=2)
    ax.fill_between(x, y0, y1, alpha=0.25)
    ax.axhline(0.0, color="black", linewidth=1, linestyle="--")
    ax.set_title(f"Persistence Ribbon: {label} / {metric}")
    ax.set_xlabel("Step")
    ax.set_ylabel("Effect")
    fig.tight_layout()
    return fig, ax


def plot_programme_persistence_ribbon(
    stepwise_summary_df: pd.DataFrame,
    label: str,
    metrics: list[str],
    step_col: str = "step",
    value_col: str = "point_estimate",
    low_col: str = "ci_low",
    high_col: str = "ci_high",
    figsize: tuple[float, float] = (9, 5),
):
    """Plot multi-metric persistence ribbons for one candidate."""
    plt, sns = _lazy_plot_libs()
    fig, ax = plt.subplots(figsize=figsize)

    df = stepwise_summary_df.copy()
    df = df[(df["label"] == label) & (df["metric"].isin(metrics))].copy()
    if df.empty:
        ax.set_title(f"Programme Persistence (empty): {label}")
        return fig, ax

    palette = sns.color_palette(n_colors=len(metrics))
    for i, metric in enumerate(metrics):
        sub = df[df["metric"] == metric].sort_values(step_col)
        if sub.empty:
            continue
        x = sub[step_col].to_numpy(dtype=float)
        y = sub[value_col].to_numpy(dtype=float)
        y0 = sub[low_col].to_numpy(dtype=float)
        y1 = sub[high_col].to_numpy(dtype=float)
        ax.plot(x, y, marker="o", linewidth=2, color=palette[i], label=metric)
        ax.fill_between(x, y0, y1, alpha=0.18, color=palette[i])

    ax.axhline(0.0, color="black", linewidth=1, linestyle="--")
    ax.set_title(f"Programme Persistence: {label}")
    ax.set_xlabel("Step")
    ax.set_ylabel("Effect")
    ax.legend(loc="best")
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
#  Robustness visualizations (from benchmark_source_ranking_robustness.ipynb)
# ---------------------------------------------------------------------------


def plot_bootstrap_ci_forest(
    boot_mean_df: pd.DataFrame,
    focus_labels: list[str] | None = None,
    readout_labels: dict[str, str] | None = None,
    figsize: tuple[float, float] = (6, 5),
) -> Any:
    """Forest plot of bootstrap CIs for primary readouts per label.

    Parameters
    ----------
    boot_mean_df : DataFrame
        Must have columns: ``label``, ``metric``, ``oriented_point``,
        ``oriented_ci_low``, ``oriented_ci_high``.
    focus_labels : list[str] or None
        Subset of labels to show. If None, uses all labels in *boot_mean_df*.
    readout_labels : dict or None
        Mapping from metric key to display label.
    figsize : tuple
    """
    plt, _ = _lazy_plot_libs()
    if readout_labels is None:
        readout_labels = {}

    df = boot_mean_df.copy()
    if focus_labels:
        df = df[df["label"].isin(focus_labels)]

    if df.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title("Bootstrap CI Forest (empty)")
        return fig, ax

    df = df.sort_values(["label", "metric"])
    y_labels: list[str] = []
    y_pos: list[int] = []
    for i, row in enumerate(df.itertuples(index=False)):
        rl = readout_labels.get(row.metric, row.metric)
        y_labels.append(f"{row.label}\n{rl}")
        y_pos.append(i)

    fig, ax = plt.subplots(figsize=figsize)
    for i, row in enumerate(df.itertuples(index=False)):
        ax.plot(
            [row.oriented_ci_low, row.oriented_ci_high],
            [i, i],
            color="#4A5568",
            linewidth=1.3,
        )
        ax.scatter(row.oriented_point, i, color="#D95F02", s=22, zorder=3)

    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels, fontsize=7)
    ax.set_title("Bootstrap CI for primary readouts")
    ax.set_xlabel("oriented effect")
    fig.tight_layout()
    return fig, ax


def plot_seed_rank_correlation(
    seed_corr_df: pd.DataFrame,
    value_col: str = "spearman_rho",
    figsize: tuple[float, float] = (4, 4),
) -> Any:
    """Boxplot + stripplot of seed-to-seed ranking correlations."""
    plt, sns = _lazy_plot_libs()
    fig, ax = plt.subplots(figsize=figsize)

    if seed_corr_df.empty:
        ax.set_title("Seed Rank Correlation (empty)")
        return fig, ax

    sns.boxplot(
        data=seed_corr_df,
        y=value_col,
        ax=ax,
        color="#8DA0CB",
        width=0.45,
    )
    sns.stripplot(
        data=seed_corr_df,
        y=value_col,
        ax=ax,
        color="black",
        size=4,
        alpha=0.75,
    )
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Candidate ranking correlation across seeds")
    ax.set_ylabel("Spearman rho")
    ax.set_xlabel("")
    fig.tight_layout()
    return fig, ax


def plot_topk_overlap(
    topk_df: pd.DataFrame,
    value_col: str = "jaccard",
    figsize: tuple[float, float] = (4, 4),
) -> Any:
    """Boxplot + stripplot of top-k candidate Jaccard overlaps across seeds."""
    plt, sns = _lazy_plot_libs()
    fig, ax = plt.subplots(figsize=figsize)

    if topk_df.empty:
        ax.set_title("Top-k Overlap (empty)")
        return fig, ax

    sns.boxplot(
        data=topk_df,
        y=value_col,
        ax=ax,
        color="#66C2A5",
        width=0.45,
    )
    sns.stripplot(
        data=topk_df,
        y=value_col,
        ax=ax,
        color="black",
        size=4,
        alpha=0.75,
    )
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Top-k candidate overlap across seeds")
    ax.set_ylabel("Jaccard index")
    ax.set_xlabel("")
    fig.tight_layout()
    return fig, ax


def plot_baseline_readout_heatmap(
    baseline_summary_df: pd.DataFrame,
    readout_labels: dict[str, str] | None = None,
    readout_metrics: list[str] | None = None,
    figsize: tuple[float, float] = (10, 6),
) -> Any:
    """Z-scored heatmap of baseline strategies vs readout dimensions."""
    plt, sns = _lazy_plot_libs()

    if readout_labels is None:
        readout_labels = {}
    if readout_metrics is None:
        readout_metrics = []

    if baseline_summary_df.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title("Baseline Readout Heatmap (empty)")
        return fig, ax

    heat_cols = [readout_labels.get(m, m) for m in readout_metrics]
    heat_cols = [c for c in heat_cols if c in baseline_summary_df.columns]
    if not heat_cols:
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title("Baseline Readout Heatmap (no columns)")
        return fig, ax

    heat = baseline_summary_df.set_index("baseline")[heat_cols]
    heat_z = (heat - heat.mean(axis=0)) / (heat.std(axis=0, ddof=0) + 1e-12)

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        heat_z,
        cmap="vlag",
        center=0,
        annot=True,
        fmt=".2f",
        linewidths=0.4,
        ax=ax,
        cbar_kws={"label": "relative readout score"},
    )
    ax.set_title("Top candidates selected by each strategy")
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
#  Classifier analysis figures
#   (classifier_analysis.ipynb cells 7, 16-19, 24)
# ---------------------------------------------------------------------------


def plot_confusion_matrix(
    cm: "np.ndarray",
    class_names: list[str],
    title: str = "",
    figsize: tuple[float, float] = (5, 4),
    cmap: str = "Blues",
    annot_fmt: str = "d",
) -> Any:
    """Heatmap of a confusion matrix (binary or multi-class).

    Notebook: ``classifier_analysis.ipynb`` cell 7,
    ``classifier_analysis_multi.ipynb`` cell 8.
    """
    plt, sns = _lazy_plot_libs()
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt=annot_fmt,
        cmap=cmap,
        cbar=False,
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_title(title or "Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    return fig, ax


def plot_per_class_metrics(
    metrics_df: pd.DataFrame,
    metric_cols: list[str] | None = None,
    class_col: str = "class",
    figsize: tuple[float, float] = (10, 4),
) -> Any:
    """Grouped bar chart of per-class precision / recall / F1.

    Notebook: ``classifier_analysis_multi.ipynb`` cell 8 (classification_report).
    """
    plt, sns = _lazy_plot_libs()
    if metrics_df.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title("Per-Class Metrics (empty)")
        return fig, ax

    if metric_cols is None:
        metric_cols = [c for c in ("precision", "recall", "f1-score") if c in metrics_df.columns]

    melted = metrics_df.melt(
        id_vars=[class_col], value_vars=metric_cols, var_name="metric", value_name="value"
    )
    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(data=melted, x=class_col, y="value", hue="metric", ax=ax)
    ax.set_title("Per-Class Metrics")
    ax.set_xlabel("Class")
    ax.set_ylabel("Score")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig, ax


def plot_gene_impact_bar(
    impact_df: pd.DataFrame,
    gene_col: str = "Gene",
    score_col: str = "Abs_Mean_IG_to_PV",
    top_k: int = 15,
    figsize: tuple[float, float] = (5, 6),
) -> Any:
    """Horizontal bar chart of top gene impact scores.

    Notebook: ``classifier_analysis.ipynb`` cell 17-18 (global logit impact).
    """
    plt, sns = _lazy_plot_libs()
    df = impact_df.dropna(subset=[score_col]).copy()
    if df.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title("Gene Impact (empty)")
        return fig, ax

    top = df.sort_values(score_col, ascending=True).tail(top_k)
    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(top[gene_col].astype(str), top[score_col], color="#5B8DEF")
    ax.set_title("Top Gene Impact Scores")
    ax.set_xlabel(score_col)
    sns.despine(ax=ax)
    fig.tight_layout()
    return fig, ax


def plot_ig_bubble(
    cluster_gene_df: pd.DataFrame,
    cluster_order: list[str] | None = None,
    gene_col: str = "Gene",
    cluster_col: str = "Cluster",
    value_col: str = "Abs_Mean_IG_to_PV",
    freq_col: str = "Frequency",
    top_k: int = 12,
    min_freq: float = 0.0,
    title: str = "TF contributions towards PV fate across branch progression",
    figsize: tuple[float, float] | None = None,
) -> Any:
    """Bubble plot of gene impact per cluster (IG bubble).

    Notebook: ``classifier_analysis.ipynb`` cell 16-18
    (ImpactVisualizer.plot_ig_bubble).
    """
    plt, sns = _lazy_plot_libs()
    df = cluster_gene_df.copy()
    df[cluster_col] = df[cluster_col].astype(str)
    df[gene_col] = df[gene_col].astype(str)

    if min_freq > 0 and freq_col in df.columns:
        df = df[df[freq_col] >= min_freq]

    if cluster_order is None:
        cluster_order = sorted(df[cluster_col].unique())
    else:
        cluster_order = [str(c) for c in cluster_order]

    top_genes = (
        df.groupby(gene_col)[value_col]
        .max()
        .sort_values(ascending=False)
        .head(top_k)
        .index
        .tolist()
    )
    plot_df = df[df[gene_col].isin(top_genes)].copy()

    x_map = {c: i for i, c in enumerate(cluster_order)}
    y_map = {g: i for i, g in enumerate(reversed(top_genes))}
    plot_df["_x"] = plot_df[cluster_col].map(x_map)
    plot_df["_y"] = plot_df[gene_col].map(y_map)
    plot_df = plot_df.dropna(subset=["_x", "_y"])

    if plot_df.empty:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title(f"{title} (empty)")
        return fig, ax

    size_factor = 200
    if freq_col in plot_df.columns:
        sizes = plot_df[freq_col] * size_factor
    else:
        sizes = 80

    if figsize is None:
        figsize = (max(6, len(cluster_order) * 1.0), max(5, len(top_genes) * 0.35))

    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    sc = ax.scatter(
        plot_df["_x"],
        plot_df["_y"],
        s=sizes,
        c=plot_df[value_col],
        cmap="viridis",
        edgecolors="white",
        linewidth=0.5,
        zorder=3,
    )
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(value_col)

    ax.set_xticks(range(len(cluster_order)))
    ax.set_xticklabels(cluster_order, rotation=45)
    ax.set_yticks(range(len(top_genes)))
    ax.set_yticklabels(reversed(top_genes), fontsize=9)
    ax.set_title(title)
    ax.set_xlabel("Cluster (branch progression)")
    ax.set_ylabel("")
    fig.tight_layout()
    return fig, ax


def plot_attention_heatmap(
    cluster_gene_df: pd.DataFrame,
    cluster_order: list[str] | None = None,
    genes: list[str] | None = None,
    gene_col: str = "Gene",
    cluster_col: str = "Cluster",
    value_col: str = "Mean_Attn",
    top_k: int = 15,
    min_freq: float = 0.0,
    freq_col: str = "Frequency",
    title: str = "",
    figsize: tuple[float, float] | None = None,
) -> Any:
    """Attention heatmap of gene × cluster.

    Notebook: ``classifier_analysis.ipynb`` cell 18
    (ImpactVisualizer.plot_attention_heatmap).
    """
    plt, sns = _lazy_plot_libs()
    df = cluster_gene_df.copy()
    df[cluster_col] = df[cluster_col].astype(str)
    df[gene_col] = df[gene_col].astype(str)

    if min_freq > 0 and freq_col in df.columns:
        df = df[df[freq_col] >= min_freq]

    if cluster_order is None:
        cluster_order = sorted(df[cluster_col].unique())
    else:
        cluster_order = [str(c) for c in cluster_order]

    if genes is None:
        genes = (
            df.groupby(gene_col)[value_col]
            .max()
            .sort_values(ascending=False)
            .head(top_k)
            .index
            .tolist()
        )

    piv = df.pivot_table(
        index=gene_col, columns=cluster_col, values=value_col, aggfunc="mean"
    )
    piv = piv.reindex(
        index=[g for g in genes if g in piv.index],
        columns=[c for c in cluster_order if c in piv.columns],
    )

    if piv.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.set_title("Attention Heatmap (empty)")
        return fig, ax

    if figsize is None:
        figsize = (max(6, len(cluster_order) * 0.9), max(5, len(genes) * 0.35))

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        piv,
        cmap="YlOrRd",
        annot=True,
        fmt=".3f",
        linewidths=0.3,
        ax=ax,
        cbar_kws={"label": value_col},
    )
    ax.set_title(title or "Attention Heatmap")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Gene")
    fig.tight_layout()
    return fig, ax


def plot_ig_flip_bar(
    flip_df: pd.DataFrame,
    gene_col: str = "Gene",
    score_col: str = "IGShiftScore",
    sign_col: str = "SignShift_NegToPos",
    top_n: int = 15,
    figsize: tuple[float, float] = (6, 8),
    title: str = "Top NPV-to-PV IG Shifts",
) -> Any:
    """Horizontal bar chart of NPV-to-PV IG shift scores.

    Notebook: ``classifier_analysis.ipynb`` cell 19
    (ImpactVisualizer.plot_flip_bar).
    """
    plt, sns = _lazy_plot_libs()
    df = flip_df.dropna(subset=[score_col]).copy()
    if df.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title(f"{title} (empty)")
        return fig, ax

    pdf = df.head(top_n).iloc[::-1]
    colors = []
    for _, row in pdf.iterrows():
        if sign_col in pdf.columns and row[sign_col]:
            colors.append("#e74c3c")
        else:
            colors.append("#3498db")

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(pdf[gene_col].astype(str), pdf[score_col], color=colors)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_title(title, fontsize=14, pad=15)
    ax.set_xlabel("FlipScore (PV Mean - NPV Mean)")
    sns.despine()
    fig.tight_layout()
    return fig, ax


def plot_cluster_enrichment_dotplot(
    cluster_enrich_df: pd.DataFrame,
    cluster_col: str = "Cluster",
    term_col: str = "Term_clean",
    pval_col: str = "Adjusted P-value",
    overlap_col: str = "Overlap_Ratio",
    neglog10_col: str = "NegLog10_Adjusted_P",
    top_n_pathways: int = 20,
    cluster_order: list[str] | None = None,
    figsize: tuple[float, float] = (9, 10),
    title: str = "Cluster-wise pathway enrichment",
    cmap: str = "viridis",
    size_scale: float = 4,
) -> Any:
    """Matrix-style dotplot for cluster-wise enrichment results.

    Notebook: ``classifier_analysis.ipynb`` cell 24
    (``plot_cluster_enrichment_dotplot``).

    Returns (fig, ax, plot_df).
    """
    plt, _ = _lazy_plot_libs()
    if cluster_enrich_df is None or cluster_enrich_df.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title(f"{title} (empty)")
        return fig, ax, pd.DataFrame()

    df = cluster_enrich_df.copy()
    required = [cluster_col, term_col, pval_col]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df[cluster_col] = df[cluster_col].astype(str).str.strip()
    df[term_col] = df[term_col].astype(str).str.strip()
    for col in [pval_col, overlap_col, neglog10_col]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=[cluster_col, term_col, pval_col])
    if df.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title(f"{title} (no valid rows)")
        return fig, ax, pd.DataFrame()

    pathway_order = (
        df.groupby(term_col)[pval_col]
        .min()
        .sort_values(ascending=True)
        .head(top_n_pathways)
        .index
        .tolist()
    )
    if not pathway_order:
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title(f"{title} (no pathways)")
        return fig, ax, pd.DataFrame()

    plot_df = df[df[term_col].isin(pathway_order)].copy()
    available = plot_df[cluster_col].dropna().astype(str).unique().tolist()
    if cluster_order is None:
        cluster_order = sorted(available)
    else:
        cluster_order = [str(x).strip() for x in cluster_order if str(x).strip() in available]

    y_labels = pathway_order[::-1]
    x_map = {c: i for i, c in enumerate(cluster_order)}
    y_map = {p: i for i, p in enumerate(y_labels)}

    plot_df["_x"] = plot_df[cluster_col].map(x_map)
    plot_df["_y"] = plot_df[term_col].map(y_map)
    plot_df = plot_df.dropna(subset=["_x", "_y"])
    if plot_df.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title(f"{title} (no matching rows)")
        return fig, ax, pd.DataFrame()

    sizes = (
        plot_df[overlap_col] * size_scale * 80
        if overlap_col in plot_df.columns
        else 80
    )
    color_val = (
        plot_df[neglog10_col]
        if neglog10_col in plot_df.columns
        else -np.log10(plot_df[pval_col].clip(lower=1e-300))
    )

    fig, ax = plt.subplots(figsize=figsize)
    sc = ax.scatter(
        plot_df["_x"],
        plot_df["_y"],
        s=sizes,
        c=color_val,
        cmap=cmap,
        edgecolors="white",
        linewidth=0.5,
        zorder=3,
    )
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("-log10(adjusted P)")

    ax.set_xticks(range(len(cluster_order)))
    ax.set_xticklabels(cluster_order, rotation=45, ha="right")
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.set_title(title)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("")
    fig.tight_layout()
    return fig, ax, plot_df


# ---------------------------------------------------------------------------
#  Multi-class classifier figures
#   (classifier_analysis_multi.ipynb cells 8, 12)
# ---------------------------------------------------------------------------


def plot_low_recall_rows(
    cm_row_norm: "np.ndarray",
    class_names: list[str],
    lowest_idx: "np.ndarray",
    lowest_names: list[str],
    figsize: tuple[float, float] = (12, 5),
) -> Any:
    """Heatmap of row-normalised confusion rows for lowest-recall classes.

    Notebook: ``classifier_analysis_multi.ipynb`` cell 12.
    """
    plt, sns = _lazy_plot_libs()
    submatrix = cm_row_norm[np.ix_(lowest_idx, np.arange(len(class_names)))]
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        submatrix,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        vmin=0,
        vmax=1,
        cbar_kws={"label": "Row-normalised recall / confusion rate"},
        xticklabels=class_names,
        yticklabels=lowest_names,
        ax=ax,
    )
    ax.set_title("Classes with the lowest held-out recall")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    fig.tight_layout()
    return fig, ax


def plot_uncertainty_diagnostics(
    comparison_df: pd.DataFrame,
    metric_specs: list[tuple[str, str, str]] | None = None,
    figsize: tuple[float, float] = (15, 5),
) -> Any:
    """Box + strip plots of confidence / entropy / margin by outcome.

    Notebook: ``classifier_analysis_multi.ipynb`` cell 12.

    Parameters
    ----------
    comparison_df : DataFrame
        Must have columns ``outcome`` and each metric column.
    metric_specs : list of (column, ylabel, color), optional
    """
    plt, sns = _lazy_plot_libs()
    if metric_specs is None:
        metric_specs = [
            ("confidence", "Max softmax confidence", "#4c78a8"),
            ("entropy", "Normalized entropy", "#59a14f"),
            ("margin", "Top-1 vs top-2 probability gap", "#f28e2b"),
        ]

    n = len(metric_specs)
    fig, axes = plt.subplots(1, n, figsize=figsize)

    if n == 1:
        axes = [axes]

    for ax, (metric, ylabel, _color) in zip(axes, metric_specs):
        if metric not in comparison_df.columns:
            ax.set_title(f"{ylabel} (missing)")
            continue
        sns.boxplot(
            data=comparison_df,
            x="outcome",
            y=metric,
            palette=["#4c78a8", "#d62728"],
            ax=ax,
        )
        sns.stripplot(
            data=comparison_df,
            x="outcome",
            y=metric,
            color="black",
            alpha=0.18,
            size=2.0,
            ax=ax,
        )
        ax.set_title(ylabel)
        ax.set_xlabel("")
        ax.set_ylabel(ylabel)
        sns.despine(ax=ax)

    fig.tight_layout()
    return fig, axes


# ---------------------------------------------------------------------------
#  Generator analysis figures
#   (generator_analysis_3.ipynb cells 6, 7, 17, 18, 20)
# ---------------------------------------------------------------------------


def plot_forecast_scatter(
    pred: "np.ndarray",
    true: "np.ndarray",
    nz_mask: "np.ndarray | None" = None,
    alpha: float = 0.25,
    figsize: tuple[float, float] = (6, 6),
) -> Any:
    """Scatter plot of predicted vs true expression values.

    Notebook: ``generator_analysis_3.ipynb`` cell 6 (safe_pearsonr scatter).
    """
    plt, _ = _lazy_plot_libs()
    p = np.asarray(pred, dtype=np.float64).ravel()
    t = np.asarray(true, dtype=np.float64).ravel()

    if nz_mask is not None:
        p = p[nz_mask]
        t = t[nz_mask]
    else:
        mask = (p > 0) & (t > 0)
        p = p[mask]
        t = t[mask]

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(p, t, alpha=alpha, s=2, color="#404040", edgecolors="none")
    lims = [min(p.min(), t.min()), max(p.max(), t.max())]
    ax.plot(lims, lims, "r--", linewidth=1, alpha=0.7)
    ax.set_xlabel("Predicted expression")
    ax.set_ylabel("True expression")
    ax.set_title("Forecasting: predicted vs true")
    fig.tight_layout()
    return fig, ax


def plot_zscore_ranking(
    permutation_results: pd.DataFrame,
    tf_col: str = "Source_TF",
    marker_col: str = "Target_Marker",
    zscore_col: str = "Z_Score_Vs_Null",
    top_n_per_marker: int = 10,
    facet_order: list[str] | None = None,
    n_cols: int = 4,
    save_path: str | None = None,
) -> Any:
    """Faceted horizontal bar chart of Z-score rankings per target marker.

    Notebook: ``generator_analysis_3.ipynb`` cell 17
    (``plot_zscore_ranking``).
    """
    plt, sns = _lazy_plot_libs()
    df = permutation_results.copy()

    if facet_order is None:
        present = set(df[marker_col].dropna().astype(str).tolist())
        default_order = get_pv_identity_markers()
        facet_order = [m for m in default_order if m in present]
        if not facet_order:
            facet_order = sorted(present)
    else:
        present = set(df[marker_col].dropna().astype(str).tolist())
        facet_order = [m for m in facet_order if m in present]

    plot_df = (
        df.sort_values([marker_col, zscore_col], ascending=[True, False])
        .groupby(marker_col)
        .head(top_n_per_marker)
        .copy()
    )
    plot_df[marker_col] = pd.Categorical(
        plot_df[marker_col], categories=facet_order, ordered=True
    )
    plot_df = plot_df.sort_values([marker_col, zscore_col], ascending=[True, False])

    if plot_df.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.set_title("Z-Score Ranking (empty)")
        return fig, ax

    g = sns.FacetGrid(
        plot_df,
        col=marker_col,
        col_order=facet_order,
        col_wrap=min(n_cols, len(facet_order)),
        sharey=False,
        sharex=False,
        height=4,
        aspect=1.2,
    )

    def _draw_bar(data: pd.DataFrame, **kwargs: Any) -> None:
        data = data.sort_values(zscore_col, ascending=True)
        ax = plt.gca()
        colors = sns.color_palette("Blues_d", n_colors=len(data))
        ax.barh(data[tf_col], data[zscore_col], color=colors, edgecolor="none", alpha=0.9)
        ax.axvline(1.96, color="#d62728", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.grid(axis="x", linestyle=":", alpha=0.4)
        ax.tick_params(axis="both", which="both", length=0)
        sns.despine(ax=ax, left=True, bottom=False)

    g.map_dataframe(_draw_bar)
    g.set_axis_labels("Z-Score", "")
    g.set_titles(col_template="{col_name}", size=13, weight="bold", pad=15)
    plt.subplots_adjust(top=0.9, wspace=0.4, hspace=0.5)

    if save_path:
        g.savefig(save_path, dpi=300, bbox_inches="tight")

    return g.fig, g.axes.flatten()


def plot_selection_frequency_distribution(
    freq_df: pd.DataFrame,
    freq_col: str = "Selection_Freq",
    threshold: float = 0.8,
    figsize: tuple[float, float] = (6, 4),
) -> Any:
    """Histogram of bootstrap selection frequencies with stability threshold.

    Notebook: ``generator_analysis_3.ipynb`` cell 20
    (``plot_selection_frequency_distribution``).
    """
    plt, sns = _lazy_plot_libs()
    vals = pd.to_numeric(freq_df[freq_col], errors="coerce").to_numpy()
    vals = vals[np.isfinite(vals)]

    if len(vals) == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title("Selection Frequency Distribution (empty)")
        return fig, ax

    fig, ax = plt.subplots(figsize=figsize)
    sns.histplot(vals, bins=np.linspace(0, 1, 21), color="gray", edgecolor="white", ax=ax)
    ax.axvline(threshold, color="red", linestyle="--", alpha=0.7, label=f"Stable-edge cut-off ({threshold})")
    ax.set_xlabel("Bootstrap selection frequency")
    ax.set_ylabel("Number of TF–target pairs")
    ax.legend(frameon=False)
    sns.despine()
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
#  AnnData-dependent helpers (optional)
#   - generator_analysis_3.ipynb (trajectory profiles)
#   - classifier_analysis_multi.ipynb (UMAP overlay)
#   - analysis DEG delta cell(s) (delta scatter)
# ---------------------------------------------------------------------------


def plot_trajectories(
    adata: Any,
    pseudotime_col: str = "latent_time",
    value_col: str = "trajectory_score",
    group_col: str | None = None,
    figsize: tuple[float, float] = (7, 4),
) -> Any:
    """Trajectory profile helper from ``generator_analysis_3.ipynb``.

    Requires an AnnData-like object with ``obs`` DataFrame.
    """
    plt, sns = _lazy_plot_libs()
    obs = getattr(adata, "obs", None)
    if obs is None:
        raise ValueError("adata must provide an .obs DataFrame")
    if pseudotime_col not in obs.columns or value_col not in obs.columns:
        raise ValueError(f"Missing required columns: {pseudotime_col}, {value_col}")

    df = obs[[pseudotime_col, value_col] + ([group_col] if group_col else [])].dropna()
    fig, ax = plt.subplots(figsize=figsize)
    if df.empty:
        ax.set_title("Trajectory Profiles (empty)")
        return fig, ax

    if group_col and group_col in df.columns:
        sns.lineplot(
            data=df.sort_values(pseudotime_col),
            x=pseudotime_col,
            y=value_col,
            hue=group_col,
            estimator="mean",
            errorbar=None,
            ax=ax,
        )
    else:
        sns.scatterplot(
            data=df.sort_values(pseudotime_col),
            x=pseudotime_col,
            y=value_col,
            s=8,
            alpha=0.35,
            color="#4C78A8",
            ax=ax,
        )
    ax.set_title("Trajectory Profiles")
    ax.set_xlabel(pseudotime_col)
    ax.set_ylabel(value_col)
    fig.tight_layout()
    return fig, ax


def plot_umap_overlay(
    adata: Any,
    color_col: str = "predicted_class",
    basis_key: str = "X_umap",
    figsize: tuple[float, float] = (6, 5),
) -> Any:
    """UMAP overlay helper from ``classifier_analysis_multi.ipynb``.

    Requires an AnnData-like object with ``obsm[basis_key]`` and ``obs[color_col]``.
    """
    plt, sns = _lazy_plot_libs()
    obsm = getattr(adata, "obsm", None)
    obs = getattr(adata, "obs", None)
    if obsm is None or obs is None:
        raise ValueError("adata must provide .obsm and .obs")
    if basis_key not in obsm:
        raise ValueError(f"Missing embedding key in adata.obsm: {basis_key}")
    if color_col not in obs.columns:
        raise ValueError(f"Missing column in adata.obs: {color_col}")

    emb = np.asarray(obsm[basis_key])
    if emb.ndim != 2 or emb.shape[1] < 2:
        raise ValueError(f"Embedding {basis_key} must have shape (n_cells, >=2)")

    plot_df = pd.DataFrame(
        {"umap1": emb[:, 0], "umap2": emb[:, 1], color_col: obs[color_col].astype(str).values}
    ).dropna()

    fig, ax = plt.subplots(figsize=figsize)
    if plot_df.empty:
        ax.set_title("UMAP Overlay (empty)")
        return fig, ax

    sns.scatterplot(
        data=plot_df,
        x="umap1",
        y="umap2",
        hue=color_col,
        s=8,
        alpha=0.7,
        linewidth=0,
        ax=ax,
    )
    ax.set_title(f"UMAP Overlay: {color_col}")
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.legend(loc="best", markerscale=1.2, frameon=False)
    fig.tight_layout()
    return fig, ax


def plot_deg_delta_scatter(
    deg_df: pd.DataFrame,
    x_col: str = "delta_npv",
    y_col: str = "delta_pv",
    gene_col: str = "gene",
    label_top_n: int = 10,
    figsize: tuple[float, float] = (6, 6),
) -> Any:
    """DEG delta scatter helper from analysis DEG-delta notebook cell(s)."""
    plt, _ = _lazy_plot_libs()
    if x_col not in deg_df.columns or y_col not in deg_df.columns:
        raise ValueError(f"Missing required columns: {x_col}, {y_col}")

    df = deg_df[[x_col, y_col] + ([gene_col] if gene_col in deg_df.columns else [])].dropna()
    fig, ax = plt.subplots(figsize=figsize)
    if df.empty:
        ax.set_title("DEG Delta Scatter (empty)")
        return fig, ax

    x = pd.to_numeric(df[x_col], errors="coerce").to_numpy()
    y = pd.to_numeric(df[y_col], errors="coerce").to_numpy()
    ax.scatter(x, y, s=12, alpha=0.6, color="#4C78A8", edgecolors="none")
    ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    ax.axvline(0.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title("DEG Delta Scatter")

    if gene_col in df.columns and label_top_n > 0:
        score = np.abs(x) + np.abs(y)
        idx = np.argsort(score)[-label_top_n:]
        for i in idx:
            g = str(df.iloc[i][gene_col])
            ax.text(x[i], y[i], g, fontsize=7, alpha=0.8)

    fig.tight_layout()
    return fig, ax


def plot_benchmark1_source_dependence_metrics(
    source_metrics_df: pd.DataFrame,
    metric_specs: list[tuple[str, str, str]] | None = None,
    condition_order: list[str] | None = None,
    condition_labels: dict[str, str] | None = None,
    figsize: tuple[float, float] = (30, 3.7),
) -> Any:
    """Benchmark 1A-C composition from robustness notebook.

    Notebook source: ``benchmark_source_ranking_robustness.ipynb`` figure Benchmark 1A-C cell.
    """
    plt, sns = _lazy_plot_libs()
    if metric_specs is None:
        metric_specs = [
            ("raw_mse", "Raw MSE", "lower"),
            ("raw_pearson_nonzero", "Raw Pearson, nonzero targets", "higher"),
            ("pseudo_r2_vs_train_bin_mean", "Pseudo-R2 vs train-estimated bin mean", "higher"),
            ("pseudo_r2_vs_heldout_oracle_bin_mean", "Pseudo-R2 vs held-out oracle bin mean", "higher"),
            ("residual_pearson_after_train_bin_mean", "Residual Pearson after train-bin mean", "higher"),
            ("pred_change_l2_mean", "Prediction change after source alteration", "higher"),
            ("sensitivity_ratio", "Prediction/source change ratio", "higher"),
        ]
    if condition_order is None:
        condition_order = [
            "train_estimated_target_bin_mean",
            "heldout_oracle_target_bin_mean",
            "full_model",
            "source_values_zeroed",
            "source_expr_shuffled_global",
            "within_target_bin_source_shuffle",
        ]
    if condition_labels is None:
        condition_labels = {
            "train_estimated_target_bin_mean": "Train-bin mean",
            "heldout_oracle_target_bin_mean": "Held-out oracle bin mean",
            "full_model": "Full model",
            "source_values_zeroed": "Source values zeroed",
            "source_expr_shuffled_global": "Global source shuffle",
            "within_target_bin_source_shuffle": "Within-target-bin shuffle",
        }

    if source_metrics_df.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.set_title("Benchmark 1A-C (empty)")
        return fig, [ax]

    plot_df = source_metrics_df.copy()
    plot_df["condition_label"] = plot_df["condition"].map(condition_labels).fillna(plot_df["condition"])
    plot_df["condition_label"] = pd.Categorical(
        plot_df["condition_label"],
        [condition_labels.get(x, x) for x in condition_order],
        ordered=True,
    )

    fig, axes = plt.subplots(1, len(metric_specs), figsize=figsize, constrained_layout=True)
    if len(metric_specs) == 1:
        axes = [axes]
    for ax, (metric, title, _) in zip(axes, metric_specs):
        if metric not in plot_df.columns:
            ax.set_title(f"{title} (missing)")
            continue
        sns.barplot(data=plot_df, x="condition_label", y=metric, errorbar="sd", ax=ax, color="#6A9FB5")
        sns.stripplot(data=plot_df, x="condition_label", y=metric, ax=ax, color="black", size=3.0, alpha=0.65)
        if metric.startswith("pseudo_r2"):
            ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=45)
    fig.suptitle(" ", y=1.06, fontsize=13)
    return fig, axes


def plot_benchmark1_same_target_bin_vectors(
    same_bin_df: pd.DataFrame,
    figsize: tuple[float, float] = (5.4, 4.5),
) -> Any:
    """Benchmark 1D vector composition from robustness notebook.

    Notebook source: ``benchmark_source_ranking_robustness.ipynb`` figure Benchmark 1D cell.
    """
    plt, sns = _lazy_plot_libs()
    fig, ax = plt.subplots(figsize=figsize)
    if same_bin_df.empty:
        ax.set_title("Benchmark 1D (empty)")
        return fig, ax

    required = ["source_pc1", "source_pc2", "pred_pc1", "pred_pc2", "source_cluster"]
    missing = [c for c in required if c not in same_bin_df.columns]
    if missing:
        ax.set_title(f"Benchmark 1D (missing columns: {missing})")
        return fig, ax

    palette = sns.color_palette("tab10", n_colors=len(same_bin_df))
    for color, row in zip(palette, same_bin_df.itertuples(index=False)):
        ax.scatter(row.source_pc1, row.source_pc2, color=color, s=40)
        ax.arrow(
            row.source_pc1,
            row.source_pc2,
            row.pred_pc1 - row.source_pc1,
            row.pred_pc2 - row.source_pc2,
            color=color,
            width=0.01,
            length_includes_head=True,
            alpha=0.85,
        )
        ax.text(row.pred_pc1, row.pred_pc2, str(row.source_cluster), color=color, fontsize=8)
    ax.set_title("")
    ax.set_xlabel("PC1 of source/predicted expression")
    ax.set_ylabel("PC2")
    fig.tight_layout()
    return fig, ax


def plot_benchmark3_response_mode_robustness(
    boot_mean_df: pd.DataFrame,
    seed_corr_df: pd.DataFrame,
    topk_df: pd.DataFrame,
    focus_labels: list[str] | None = None,
    readout_labels: dict[str, str] | None = None,
    figsize: tuple[float, float] = (15.5, 4.6),
) -> Any:
    """Benchmark 3 triptych composition (CI forest + seed corr + top-k overlap).

    Notebook source: ``benchmark_source_ranking_robustness.ipynb`` figure Benchmark 3 cell.
    """
    plt, sns = _lazy_plot_libs()
    if focus_labels is None:
        focus_labels = ["MYT1L_OE", "ZEB2_OE", "SOX6_OE", "PBX3_OE"]
    if readout_labels is None:
        readout_labels = {}

    fig, axes = plt.subplots(1, 3, figsize=figsize, constrained_layout=True)

    focus_boot = boot_mean_df.copy()
    if not focus_boot.empty:
        focus_boot = focus_boot[focus_boot["label"].isin(focus_labels)]
        focus_boot = focus_boot.sort_values(["label", "metric"])
        y_labels: list[str] = []
        y_pos: list[int] = []
        for i, row in enumerate(focus_boot.itertuples(index=False)):
            metric_label = readout_labels.get(row.metric, row.metric)
            y_labels.append(f"{row.label}\n{metric_label}")
            y_pos.append(i)
            axes[0].plot([row.oriented_ci_low, row.oriented_ci_high], [i, i], color="#4A5568", linewidth=1.3)
            axes[0].scatter(row.oriented_point, i, color="#D95F02", s=22, zorder=3)
    axes[0].axvline(0, color="black", linewidth=0.8)
    axes[0].set_yticks(y_pos if not focus_boot.empty else [])
    axes[0].set_yticklabels(y_labels if not focus_boot.empty else [], fontsize=7)
    axes[0].set_title("Bootstrap CI for primary readouts")
    axes[0].set_xlabel("oriented effect")

    if seed_corr_df.empty or "spearman_rho" not in seed_corr_df.columns:
        axes[1].set_title("Candidate ranking correlation across seeds (empty)")
    else:
        sns.boxplot(data=seed_corr_df, y="spearman_rho", ax=axes[1], color="#8DA0CB", width=0.45)
        sns.stripplot(data=seed_corr_df, y="spearman_rho", ax=axes[1], color="black", size=4, alpha=0.75)
        axes[1].set_ylim(-0.05, 1.05)
        axes[1].set_title("Candidate ranking correlation across seeds")
        axes[1].set_ylabel("Spearman rho")
        axes[1].set_xlabel("")

    if topk_df.empty or "jaccard" not in topk_df.columns:
        axes[2].set_title("Top-k candidate overlap across seeds (empty)")
    else:
        sns.boxplot(data=topk_df, y="jaccard", ax=axes[2], color="#66C2A5", width=0.45)
        sns.stripplot(data=topk_df, y="jaccard", ax=axes[2], color="black", size=4, alpha=0.75)
        axes[2].set_ylim(-0.05, 1.05)
        axes[2].set_title("Top-k candidate overlap across seeds")
        axes[2].set_ylabel("Jaccard index")
        axes[2].set_xlabel("")

    fig.suptitle(" ", y=1.03, fontsize=13)
    return fig, axes
