"""Data preparation for classifier training, migrated from demo notebooks."""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse
import torch

from .datasets import DictTensorDataset
from model.utils.constants import get_cluster_column, get_exclude_class


def prepare_classifier_data(
    h5ad_path: str,
    max_len: int,
    exclude_class: str | None = None,
    label_mode: str = "binary",
    cluster_col: str | None = None,
    verbose: bool = True,
) -> tuple[DictTensorDataset, dict[str, Any]]:
    """Load .h5ad and build tokenised tensors for classifier training.

    Parameters
    ----------
    h5ad_path : str
        Path to the AnnData file.
    max_len : int
        Maximum number of non-zero genes per cell.
    exclude_class : str
        Trajectory class to exclude (e.g. ``"hGPC"``).
    label_mode : {"binary", "multi"}
        - ``"binary"`` : PV vs NPV binary labels from ``trajectory_class``.
        - ``"multi"``  : Cluster labels from *cluster_col* (only PV/NPV cells).
    cluster_col : str
        Column in ``.obs`` used for multi-class labels.
    verbose : bool
        Print dataset statistics.

    Returns
    -------
    dataset : DictTensorDataset
    stats : dict
    """
    try:
        import scanpy as sc
    except ImportError:
        raise ImportError("scanpy is required for data loading. Install with: pip install scanpy")

    if exclude_class is None:
        exclude_class = get_exclude_class()
    if cluster_col is None:
        cluster_col = get_cluster_column()

    adata = sc.read_h5ad(h5ad_path)

    # --- filter excluded class ---
    if exclude_class in adata.obs["trajectory_class"].unique():
        adata = adata[adata.obs["trajectory_class"] != exclude_class].copy()

    # --- label construction ---
    if label_mode == "multi":
        if "trajectory_class" in adata.obs:
            adata = adata[adata.obs["trajectory_class"].isin(["PV", "NPV"])].copy()
            if verbose:
                print(f"Filtered to PV/NPV, remaining cells: {adata.n_obs}")

        if cluster_col not in adata.obs:
            raise ValueError(f"Cluster column '{cluster_col}' not found in adata.obs.")

        # Normalise cluster labels (e.g. "6,0" → "6.0")
        adata = prepare_clusters(adata, cluster_col)

        label_col = adata.obs[cluster_col].astype("category").cat.remove_unused_categories()
        adata.obs[cluster_col] = label_col
        labels = label_col.cat.codes.values.astype(int)
        class_names = label_col.cat.categories.tolist()
        num_classes = len(class_names)
        class_counts = np.bincount(labels)
        class_weights = len(labels) / (num_classes * class_counts + 1e-6)

        if verbose:
            print(f"Detected {num_classes} clusters: {class_names}")
            print(f"Class counts: {class_counts}")
            print(f"Class weights: {np.round(class_weights, 4)}")
    else:
        if "trajectory_class" in adata.obs:
            labels = (adata.obs["trajectory_class"] == "PV").values.astype(int)
        elif "label" in adata.obs:
            labels = adata.obs["label"].values.astype(int)
        else:
            raise ValueError("No label column found in adata.obs.")
        class_names = ["NPV", "PV"]
        num_classes = 2
        class_weights = None

    # --- extract expression matrix ---
    data_matrix = adata.X
    if scipy.sparse.issparse(data_matrix):
        data_matrix = data_matrix.toarray()

    num_cells = data_matrix.shape[0]
    n_vars = data_matrix.shape[1]

    # --- tensor allocation ---
    all_gene_ids = torch.zeros((num_cells, max_len), dtype=torch.long)
    all_gene_vals = torch.zeros((num_cells, max_len), dtype=torch.float32)
    all_valid_masks = torch.zeros((num_cells, max_len), dtype=torch.bool)

    kept_labels: list[int] = []
    write_pos = 0
    truncated_cells = 0
    max_gene_id_seen = 0
    max_nonzero_genes = 0
    max_used_len = 0

    for i in range(num_cells):
        expr_values = data_matrix[i]

        non_zero_indices = np.where(expr_values > 0)[0]
        non_zero_values = expr_values[non_zero_indices]

        if len(non_zero_indices) == 0:
            continue

        mean_val = np.mean(non_zero_values)
        scaled_values = non_zero_values / (mean_val + 1e-6)

        actual_len = min(len(non_zero_indices), max_len)

        max_nonzero_genes = max(max_nonzero_genes, len(non_zero_indices))
        max_used_len = max(max_used_len, actual_len)
        if len(non_zero_indices) > max_len:
            truncated_cells += 1
        max_gene_id_seen = max(max_gene_id_seen, int(non_zero_indices.max()) + 1)

        all_gene_ids[write_pos, :actual_len] = torch.tensor(
            non_zero_indices[:actual_len] + 1, dtype=torch.long
        )
        all_gene_vals[write_pos, :actual_len] = torch.tensor(
            scaled_values[:actual_len], dtype=torch.float32
        )
        all_valid_masks[write_pos, :actual_len] = True

        kept_labels.append(int(labels[i]))
        write_pos += 1

    if write_pos == 0:
        raise ValueError("All cells are empty after filtering; cannot build dataset.")

    all_labels = torch.tensor(kept_labels, dtype=torch.long)
    all_valid_masks = all_valid_masks[:write_pos]
    all_pad_masks = ~all_valid_masks

    dataset = DictTensorDataset(
        {
            "gene_id": all_gene_ids[:write_pos],
            "gene_val": all_gene_vals[:write_pos],
            "valid_mask": all_valid_masks,
            "pad_mask": all_pad_masks,
            "label": all_labels,
        }
    )

    stats: dict[str, Any] = {
        "num_cells_kept": write_pos,
        "n_vars": int(n_vars),
        "vocab_size": int(n_vars) + 1,
        "max_nonzero_genes_per_cell": int(max_nonzero_genes),
        "max_used_len_after_truncation": int(max_used_len),
        "truncated_cells": int(truncated_cells),
        "max_gene_id_seen": int(max_gene_id_seen),
        "num_classes": int(num_classes),
        "class_names": class_names,
    }
    if class_weights is not None:
        stats["class_weights"] = class_weights.tolist()

    if verbose:
        print(f"Cells kept: {write_pos}, n_vars: {n_vars}, num_classes: {num_classes}")
        print(
            f"Max nonzero genes: {max_nonzero_genes}, "
            f"Max used length: {max_used_len}, "
            f"Truncated cells: {truncated_cells}"
        )

    return dataset, stats


def prepare_clusters(
    adata: Any,
    cluster_col: str | None = None,
) -> Any:
    """Normalise cluster labels in-place and return *adata*.

    If *cluster_col* is not found, the first ``leiden``-prefixed column is
    used as fallback.  Cluster labels are stripped and commas are replaced
    with dots (``6,0`` → ``6.0``).

    Notebook: ``classifier_analysis.ipynb`` cell 5,
    ``classifier_analysis_multi.ipynb`` cell 10,
    ``generator_analysis_3.ipynb`` cell 6.
    """
    if cluster_col is None:
        cluster_col = get_cluster_column()

    if cluster_col not in adata.obs.columns:
        leiden_cols = [c for c in adata.obs.columns if str(c).startswith("leiden")]
        if not leiden_cols:
            raise ValueError("No cluster column found in adata.obs.")
        cluster_col = leiden_cols[0]

    adata.obs[cluster_col] = (
        adata.obs[cluster_col]
        .astype(str)
        .str.strip()
        .str.replace(",", ".", regex=False)
    )
    return adata
