"""Trajectory pair construction for generator training.

Migrated from ``demo/generator_3.ipynb``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset
from model.utils.constants import get_cluster_column, get_time_column
from model.data.preprocessing import prepare_clusters


class TrajectoryDataset(Dataset):
    """Minimal list-backed dataset for trajectory pair items."""

    def __init__(self, data_list: list[dict[str, Any]]):
        self.data_list = data_list

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, i: int) -> dict[str, Any]:
        return self.data_list[i]


class PrepareTrajectoryData:
    """Build source-target trajectory pairs across multi-horizon time bins.

    This class reads an AnnData file, normalises pseudotime, extracts PCA
    coordinates, bins cells along time, and constructs per-bin-pair
    source-disjoint train/val/held-out splits.
    """

    def __init__(
        self,
        h5ad_path: str,
        config: dict[str, Any],
        subset_col: str = "trajectory_class",
        subset_values: tuple[str, ...] = ("PV",),
        time_col: str | None = None,
        cluster_col: str | None = None,
        n_bins: int = 120,
        allowed_offsets: tuple[int, ...] = (1, 2, 3, 4, 5, 6),
        base_max_dist: float = 12.0,
        dist_alpha: float = 1.0,
        allowed_cross_steps: tuple[int, ...] = (1, 2),
        k_intra: int = 1,
        k_cross: int = 2,
        val_split: float = 0.2,
        heldout_split: float = 0.1,
        pair_diagnostics: bool = True,        random_state: int = 42,    ):
        try:
            import scanpy as sc
            import pandas as pd
            import scipy.sparse
        except ImportError as e:
            raise ImportError(f"scanpy/pandas/scipy required: {e}")

        self.random_state = random_state
        self.adata = sc.read_h5ad(h5ad_path)
        if time_col is None:
            time_col = get_time_column()
        if cluster_col is None:
            cluster_col = get_cluster_column()
        model_params = config.get("model_params", config)
        self.cfg = model_params
        self.max_len = int(self.cfg.get("max_len", 1000))
        self.n_genes = int(self.cfg.get("n_genes", self.adata.n_vars))

        # --- 1. Filter & time normalisation ---
        if subset_col is not None and subset_values is not None:
            self.adata = self.adata[
                self.adata.obs[subset_col].isin(subset_values)
            ].copy()

        if cluster_col not in self.adata.obs.columns:
            leiden_cols = [
                c for c in self.adata.obs.columns if str(c).startswith("leiden")
            ]
            cluster_col = leiden_cols[0] if leiden_cols else subset_col

        # Normalise cluster labels (e.g. "6,0" → "6.0")
        self.adata = prepare_clusters(self.adata, cluster_col)

        times = pd.to_numeric(
            self.adata.obs[time_col], errors="coerce"
        ).to_numpy(dtype=np.float32)
        valid_time_mask = ~np.isnan(times)
        self.adata = self.adata[valid_time_mask].copy()
        times = times[valid_time_mask]

        t_min, t_max = np.min(times), np.max(times)
        if not np.isclose(t_min, t_max):
            norm_times = (times - t_min) / (t_max - t_min + 1e-8)
        else:
            norm_times = np.zeros_like(times)
        self.adata.obs["norm_time"] = norm_times

        raw_data = self.adata.X
        if scipy.sparse.issparse(raw_data):
            raw_data = raw_data.toarray()
        raw_data = np.asarray(raw_data, dtype=np.float32)
        if raw_data.shape[1] < self.n_genes:
            self.n_genes = raw_data.shape[1]

        # --- 2. PCA coordinates ---
        if "X_pca" in self.adata.obsm and self.adata.obsm["X_pca"] is not None:
            self.coords = np.asarray(self.adata.obsm["X_pca"], dtype=np.float32).copy()
        else:
            adata_for_pca = self.adata.copy()
            try:
                sc.pp.highly_variable_genes(
                    adata_for_pca, min_mean=0.0125, max_mean=3, min_disp=0.5
                )
                if (
                    "highly_variable" in adata_for_pca.var.columns
                    and adata_for_pca.var["highly_variable"].sum() >= 2
                ):
                    adata_for_pca = adata_for_pca[
                        :, adata_for_pca.var["highly_variable"]
                    ].copy()
            except Exception:
                pass

            if scipy.sparse.issparse(adata_for_pca.X):
                adata_for_pca.X = adata_for_pca.X.toarray()
            sc.pp.scale(adata_for_pca, max_value=10)
            n_comps = min(
                50, min(adata_for_pca.n_obs, adata_for_pca.n_vars) - 1
            )
            try:
                sc.tl.pca(adata_for_pca, svd_solver="arpack", n_comps=n_comps)
                self.coords = np.asarray(
                    adata_for_pca.obsm["X_pca"], dtype=np.float32
                ).copy()
            except Exception:
                self.coords = np.asarray(adata_for_pca.X, dtype=np.float32)
            del adata_for_pca

        coord_mean = self.coords.mean(axis=0, keepdims=True)
        coord_std = self.coords.std(axis=0, keepdims=True)
        coord_std[coord_std < 1e-8] = 1.0
        self.coords = (self.coords - coord_mean) / coord_std
        self.coords = np.asarray(self.coords, dtype=np.float32)

        # --- 3. Cluster rank & time bins ---
        obs_df = self.adata.obs.copy()
        cluster_mean_times = (
            obs_df.groupby(cluster_col, observed=True)[time_col]
            .mean()
            .sort_values()
        )
        cluster_rank_dict = {
            c: rank for rank, c in enumerate(cluster_mean_times.index)
        }
        cell_ranks = np.array(
            [cluster_rank_dict[c] for c in obs_df[cluster_col]], dtype=np.int32
        )

        obs_df["time_bin"] = pd.qcut(
            obs_df[time_col], q=n_bins, labels=False, duplicates="drop"
        )
        valid_bin_mask = ~obs_df["time_bin"].isna().to_numpy()

        self.adata = self.adata[valid_bin_mask].copy()
        raw_data = raw_data[valid_bin_mask]
        norm_times = norm_times[valid_bin_mask]
        self.coords = self.coords[valid_bin_mask]
        cell_ranks = cell_ranks[valid_bin_mask]

        time_bin_np = obs_df.loc[valid_bin_mask, "time_bin"].astype(int).to_numpy()
        actual_bins = np.sort(np.unique(time_bin_np))
        self.adata.obs["time_bin"] = time_bin_np

        # --- 4. Per-bin-pair source-disjoint pair construction ---
        self.train_data: list[dict[str, Any]] = []
        self.val_data: list[dict[str, Any]] = []
        self.heldout_data: list[dict[str, Any]] = []

        intra_count = 0
        cross_count = 0
        horizon_stats: dict[int, int] = {offset: 0 for offset in allowed_offsets}
        cluster_source_counts: dict[object, int] = {
            c: 0 for c in cluster_mean_times.index
        }

        rng = np.random.default_rng(self.random_state)

        for i, curr_bin in enumerate(actual_bins):
            idx_curr = np.where(time_bin_np == curr_bin)[0]
            if len(idx_curr) == 0:
                continue

            for offset in allowed_offsets:
                target_bin_idx = i + offset
                if target_bin_idx >= len(actual_bins):
                    continue

                next_bin = actual_bins[target_bin_idx]
                idx_next = np.where(time_bin_np == next_bin)[0]
                if len(idx_next) == 0:
                    continue

                current_max_dist = base_max_dist + dist_alpha * (offset - 1)

                transition_candidates: dict[int, list[tuple[str, dict[str, Any]]]] = {}

                for c_idx in idx_curr:
                    curr_coord = self.coords[c_idx].reshape(1, -1)
                    next_coords = self.coords[idx_next]
                    dists = np.linalg.norm(next_coords - curr_coord, axis=1)

                    safe_mask = dists <= current_max_dist
                    valid_idx_next = idx_next[safe_mask]
                    valid_dists = dists[safe_mask]

                    if len(valid_idx_next) == 0:
                        continue

                    rank_s = cell_ranks[c_idx]
                    rank_t = cell_ranks[valid_idx_next]

                    mask_intra = rank_t == rank_s
                    mask_cross = np.isin(rank_t - rank_s, allowed_cross_steps)

                    source_pairs: list[tuple[str, dict[str, Any]]] = []

                    # Intra-cluster
                    idx_intra = valid_idx_next[mask_intra]
                    dist_intra = valid_dists[mask_intra]
                    if len(idx_intra) > 0:
                        k_i = min(k_intra, len(idx_intra))
                        top_intra = idx_intra[np.argsort(dist_intra)[:k_i]]
                        for t_idx in top_intra:
                            source_pairs.append(
                                (
                                    "intra",
                                    self._create_item(
                                        raw_data, norm_times, int(c_idx), int(t_idx)
                                    ),
                                )
                            )

                    # Boundary-crossing
                    idx_cross = valid_idx_next[mask_cross]
                    dist_cross = valid_dists[mask_cross]
                    if len(idx_cross) > 0:
                        k_c = min(k_cross, len(idx_cross))
                        top_cross = idx_cross[np.argsort(dist_cross)[:k_c]]
                        for t_idx in top_cross:
                            source_pairs.append(
                                (
                                    "cross",
                                    self._create_item(
                                        raw_data, norm_times, int(c_idx), int(t_idx)
                                    ),
                                )
                            )

                    if source_pairs:
                        transition_candidates[int(c_idx)] = source_pairs

                if not transition_candidates:
                    continue

                unique_sources = list(transition_candidates.keys())
                for s_idx in unique_sources:
                    cluster_name = obs_df[cluster_col].iloc[s_idx]
                    cluster_source_counts[cluster_name] += 1

                rng.shuffle(unique_sources)
                n_sources = len(unique_sources)

                if n_sources >= 3:
                    n_heldout = int(np.floor(heldout_split * n_sources))
                    n_heldout = min(max(n_heldout, 1), n_sources - 2)
                else:
                    n_heldout = 0

                remaining_sources = unique_sources[n_heldout:]
                n_remaining = len(remaining_sources)

                if n_remaining >= 2:
                    n_val = int(np.floor(val_split * n_remaining))
                    n_val = min(max(n_val, 1), n_remaining - 1)
                else:
                    n_val = 0

                heldout_sources = set(unique_sources[:n_heldout])
                val_sources = set(remaining_sources[:n_val])

                for c_idx, pairs in transition_candidates.items():
                    if c_idx in heldout_sources:
                        target_list = self.heldout_data
                    elif c_idx in val_sources:
                        target_list = self.val_data
                    else:
                        target_list = self.train_data

                    for p_type, item in pairs:
                        target_list.append(item)
                        horizon_stats[offset] += 1
                        if p_type == "intra":
                            intra_count += 1
                        else:
                            cross_count += 1

        if pair_diagnostics:
            total_pairs = (
                len(self.train_data) + len(self.val_data) + len(self.heldout_data)
            )
            print("=" * 50)
            print(
                "Dataset Pipeline Finished (Multi-Horizon & Per-Bin-Pair Split)"
            )
            print(
                f"Total pairs: {total_pairs} "
                f"(Train: {len(self.train_data)}, "
                f"Val: {len(self.val_data)}, "
                f"Held-out: {len(self.heldout_data)})"
            )
            print(f"A Class (Intra-cluster) pairs: {intra_count}")
            print(f"B Class (Boundary-cross) pairs: {cross_count}")
            print("\nPairs per Horizon (Offset):")
            for offset in allowed_offsets:
                print(f"  Offset {offset}: {horizon_stats[offset]} pairs")
            print("\nTop 5 Source Clusters by Contribution:")
            sorted_clusters = sorted(
                cluster_source_counts.items(), key=lambda x: x[1], reverse=True
            )
            for c_name, count in sorted_clusters[:5]:
                print(f"  {c_name}: {count} unique source participations")
            print("=" * 50)

    def _create_item(
        self,
        data: np.ndarray,
        times: np.ndarray,
        i: int,
        j: int,
    ) -> dict[str, Any]:
        gene_id, gene_val, padding_mask = self._preprocess_numpy(data[i])
        return {
            "gene_id": gene_id,
            "gene_val": gene_val,
            "padding_mask": padding_mask,
            "full_input_val": torch.tensor(
                data[i][: self.n_genes], dtype=torch.float32
            ),
            "time": torch.tensor(times[i], dtype=torch.float32),
            "target_time": torch.tensor(times[j], dtype=torch.float32),
            "target_val": torch.tensor(
                data[j][: self.n_genes], dtype=torch.float32
            ),
            "c_idx": i,
            "target_idx": j,
        }

    def _preprocess_numpy(
        self, expr: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        expr = np.asarray(expr, dtype=np.float32)
        nz = np.where(expr > 0)[0]
        nz = nz[nz < self.n_genes]

        max_len = self.max_len
        if len(nz) == 0:
            return (
                torch.zeros(max_len, dtype=torch.long),
                torch.zeros(max_len, dtype=torch.float32),
                torch.ones(max_len, dtype=torch.bool),
            )

        idx = nz + 1
        val = expr[nz]

        if len(idx) > max_len:
            idx = idx[:max_len]
            val = val[:max_len]
            pad_mask = np.zeros(max_len, dtype=bool)
        else:
            pad_len = max_len - len(idx)
            idx = np.pad(idx, (0, pad_len), constant_values=0)
            val = np.pad(val, (0, pad_len), constant_values=0.0)
            pad_mask = np.pad(
                np.zeros(len(nz), dtype=bool),
                (0, pad_len),
                constant_values=True,
            )

        return (
            torch.tensor(idx, dtype=torch.long),
            torch.tensor(val, dtype=torch.float32),
            torch.tensor(pad_mask, dtype=torch.bool),
        )
