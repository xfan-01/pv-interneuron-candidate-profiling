"""Centralised access to thesis-wide biological/experimental constants.

All cluster labels, trajectory definitions, marker gene sets, column-name
conventions, and perturbation readout metrics are resolved from
``configs/thesis_constants.yaml`` through a lazy-loading singleton.

Usage::

    from model.utils.constants import (
        get_cluster_labels,
        get_pv_path_nodes,
        get_pv_markers,
        get_readout_orientation,
        get_column_defaults,
    )

    clusters = get_cluster_labels()
    pv_nodes = get_pv_path_nodes()

When adding a new constant that is referenced in 2+ files, add it to
``configs/thesis_constants.yaml`` first, then expose it here.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from .config import load_yaml_config


# ---------------------------------------------------------------------------
#  Internal singleton
# ---------------------------------------------------------------------------

_CONSTANTS_PATH = (
    Path(__file__).resolve().parent.parent.parent / "configs" / "thesis_constants.yaml"
)


@lru_cache(maxsize=1)
def _load_constants() -> dict[str, Any]:
    """Load the thesis-constants YAML (cached after first call)."""
    return load_yaml_config(_CONSTANTS_PATH)


def reload_constants() -> None:
    """Force re-read of the YAML file (useful for testing / notebook reload)."""
    _load_constants.cache_clear()


# ---------------------------------------------------------------------------
#  Cluster / trajectory
# ---------------------------------------------------------------------------


def get_cluster_column() -> str:
    """Default cluster column name in ``adata.obs``."""
    return _load_constants()["cluster"]["column"]


def get_cluster_labels() -> list[str]:
    """Canonical cluster label order for display."""
    return list(_load_constants()["cluster"]["labels"])


def get_cluster_normalize_rules() -> dict[str, str]:
    """Label normalisation map (e.g. ``6,0`` → ``6.0``)."""
    return dict(_load_constants()["cluster"]["normalize"])


def get_pv_path_nodes() -> list[str]:
    """PV-path trajectory progression order (source → … → terminal)."""
    return list(_load_constants()["cluster"]["pv_path"])


def get_npv_clusters() -> list[str]:
    """Cluster labels assigned to the NPV branch."""
    return list(_load_constants()["cluster"]["npv_clusters"])


def get_pv_clusters() -> list[str]:
    """Cluster labels assigned to the PV branch."""
    return list(_load_constants()["cluster"]["pv_clusters"])


# ---------------------------------------------------------------------------
#  Class labels
# ---------------------------------------------------------------------------


def get_exclude_class() -> str:
    """Cell type excluded from binary classification."""
    return _load_constants()["class_labels"]["exclude"]


def get_binary_class_map() -> dict[str, int]:
    """Binary label → integer mapping (e.g. ``{"NPV": 0, "PV": 1}``)."""
    return dict(_load_constants()["class_labels"]["binary"])


def get_multi_class_order() -> list[str]:
    """Multi-class classifier display order."""
    return list(_load_constants()["class_labels"]["multi_order"])


def get_branch_map() -> dict[str, int]:
    """Generator trajectory-variant → integer mapping."""
    return dict(_load_constants()["class_labels"]["branch_map"])


# ---------------------------------------------------------------------------
#  Time / pseudotime
# ---------------------------------------------------------------------------


def get_time_column() -> str:
    """Primary pseudotime column name."""
    return _load_constants()["time"]["primary_column"]


def get_time_fallback_columns() -> list[str]:
    """Ordered fallback pseudotime column names."""
    return list(_load_constants()["time"]["fallback_columns"])


def get_time_n_bins() -> int:
    """Default number of time bins for trajectory discretisation."""
    return int(_load_constants()["time"]["n_bins"])


def get_column_defaults() -> dict[str, str]:
    """Convenience: cluster_col and time_col defaults in one dict."""
    return {
        "cluster_col": get_cluster_column(),
        "time_col": get_time_column(),
    }


# ---------------------------------------------------------------------------
#  Marker gene sets
# ---------------------------------------------------------------------------


def get_pv_identity_markers() -> list[str]:
    """PV-terminal identity marker genes."""
    return list(_load_constants()["markers"]["pv_identity"])


def get_tf_marker_pairs_top1() -> list[tuple[str, str]]:
    """Top TF→marker pairs for trajectory profile figures."""
    pairs = _load_constants()["markers"]["tf_marker_pairs_top1"]
    return [tuple(p) for p in pairs]


def get_tf_marker_pairs_control() -> list[tuple[str, str]]:
    """Control TF→marker pairs for comparison."""
    pairs = _load_constants()["markers"]["tf_marker_pairs_control"]
    return [tuple(p) for p in pairs]


# ---------------------------------------------------------------------------
#  Perturbation readouts
# ---------------------------------------------------------------------------


def get_readout_orientation() -> dict[str, float]:
    """Metric → sign orientation (+1 or −1)."""
    return dict(_load_constants()["perturbation"]["readout_orientation"])


def get_readout_labels() -> dict[str, str]:
    """Metric key → display label."""
    return dict(_load_constants()["perturbation"]["readout_labels"])


def get_main_geometry_metrics() -> list[str]:
    return list(_load_constants()["perturbation"]["main_geometry_metrics"])


def get_main_identity_metrics() -> list[str]:
    return list(_load_constants()["perturbation"]["main_identity_metrics"])


def get_main_metrics() -> list[str]:
    """All primary perturbation readout metrics."""
    return get_main_geometry_metrics() + get_main_identity_metrics()


def get_diagnostic_metrics() -> list[str]:
    """Supporting diagnostic readout metrics."""
    return list(_load_constants()["perturbation"]["diagnostic_metrics"])


def get_legacy_to_canonical_metric_map() -> dict[str, str]:
    """Legacy metric name → canonical name."""
    return dict(_load_constants()["perturbation"]["legacy_to_canonical"])


# ---------------------------------------------------------------------------
#  Highlight sets (thesis storytelling)
# ---------------------------------------------------------------------------


def get_focus_candidates() -> list[str]:
    """Focus candidate labels for robustness/benchmark figures."""
    return list(_load_constants()["highlights"]["focus_candidates"])


def get_spotlight_tfs() -> list[str]:
    """Spotlight TF gene names for UMAP / summary displays."""
    return list(_load_constants()["highlights"]["spotlight_tfs"])


def get_low_recall_spotlight_classes() -> list[str]:
    """Classes highlighted in low-recall diagnostics."""
    return list(_load_constants()["highlights"]["low_recall_spotlight_classes"])
