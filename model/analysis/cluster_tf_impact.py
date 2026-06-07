"""Cluster-level TF impact engine (IG + attention) for classifier analysis.

Notebook source:
  - demo/classifier_analysis.ipynb (cell 14, ClusterTFImpactEngine)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch


def _tqdm_iter(iterable: Any, desc: str, leave: bool = False, enabled: bool = True) -> Any:
    if not enabled:
        return iterable
    try:
        from tqdm import tqdm
    except Exception:
        return iterable
    return tqdm(iterable, desc=desc, leave=leave)


class ClusterTFImpactEngine:
    """Compute cluster-level TF importance using IG and attention readouts."""

    def __init__(
        self,
        model: torch.nn.Module,
        adata: Any,
        device: torch.device | str,
        tf_file: str = "data/Homo_sapiens_TF.html",
        max_len: int = 2134,
        output_dir: str = "TFresults",
        attn_reduce: str = "mean_key",
        label_col: str = "trajectory_class",
        target_class_index: int = 1,
    ) -> None:
        self.model = model.to(device)
        self.model.eval()
        self.adata = adata
        self.device = torch.device(device)
        self.max_len = int(max_len)
        self.output_dir = Path(output_dir)
        self.attn_reduce = str(attn_reduce)
        self.label_col = str(label_col)
        self.target_class_index = int(target_class_index)

        self.gene_names = np.asarray(adata.var_names).astype(str)
        self.tf_symbols = self._load_tf_set(tf_file)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.label_col not in self.adata.obs.columns:
            raise ValueError(f"adata.obs must contain '{self.label_col}'")

    @staticmethod
    def _normalise_cluster_label(x: Any) -> str:
        s = str(x).strip()
        if s.endswith(".0"):
            return s[:-2]
        return s

    @staticmethod
    def _load_tf_set(tf_file: str) -> set[str]:
        p = Path(tf_file)
        if not p.exists():
            raise FileNotFoundError(f"TF file not found: {tf_file}")
        tf_df = None
        try:
            tf_df = pd.read_csv(p, sep="\t")
        except Exception:
            tf_df = None
        if tf_df is None or "Symbol" not in tf_df.columns:
            tables = pd.read_html(str(p))
            if not tables:
                raise ValueError(f"No table found in TF file: {tf_file}")
            tf_df = tables[0]
        symbols = tf_df["Symbol"] if "Symbol" in tf_df.columns else tf_df.iloc[:, 0]
        return set(symbols.dropna().astype(str).str.upper())

    def _get_expression_matrix(self) -> np.ndarray:
        x = self.adata.X
        return x.toarray() if sp.issparse(x) else np.asarray(x)

    def _prepare_inputs(
        self, raw_expr: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        non_zero_idx = np.where(raw_expr > 0)[0]
        g_id = torch.zeros((1, self.max_len), dtype=torch.long, device=self.device)
        g_val = torch.zeros((1, self.max_len), dtype=torch.float32, device=self.device)
        pad_mask = torch.ones((1, self.max_len), dtype=torch.bool, device=self.device)
        if len(non_zero_idx) == 0:
            return g_id, g_val, pad_mask

        non_zero_val = raw_expr[non_zero_idx]
        scaled_val = non_zero_val / (float(np.mean(non_zero_val)) + 1e-6)
        n = min(len(non_zero_idx), self.max_len)
        g_id[0, :n] = torch.as_tensor(non_zero_idx[:n] + 1, dtype=torch.long, device=self.device)
        g_val[0, :n] = torch.as_tensor(scaled_val[:n], dtype=torch.float32, device=self.device)
        pad_mask[0, :n] = False
        return g_id, g_val, pad_mask

    def _compute_integrated_gradients(
        self,
        g_id: torch.Tensor,
        g_val: torch.Tensor,
        pad_mask: torch.Tensor,
        steps: int = 50,
    ) -> np.ndarray:
        self.model.eval()
        baseline = torch.zeros_like(g_val)
        alphas = torch.linspace(0.0, 1.0, steps + 1, device=self.device).view(-1, 1, 1)
        interpolated = baseline + alphas * (g_val - baseline)
        grads: list[torch.Tensor] = []
        for i in range(steps + 1):
            step_input = interpolated[i].clone().detach().requires_grad_(True)
            logits = self.model(g_id, step_input, pad_mask)
            score = logits[0, self.target_class_index]
            self.model.zero_grad(set_to_none=True)
            score.backward()
            grads.append(step_input.grad.detach() if step_input.grad is not None else torch.zeros_like(step_input))
        avg_grads = torch.mean(torch.stack(grads), dim=0)
        ig = (g_val - baseline) * avg_grads
        return ig.squeeze().detach().cpu().numpy()

    def _extract_token_attention(self) -> np.ndarray | None:
        attn = getattr(self.model, "last_attn_weights", None)
        if attn is None:
            return None
        attn = attn.detach()
        if attn.dim() == 4:
            attn = attn.mean(dim=1)[0]
        elif attn.dim() == 3:
            attn = attn.mean(dim=0)
        elif attn.dim() == 2:
            pass
        elif attn.dim() == 1:
            return attn.cpu().numpy()
        else:
            return None

        token_attn = attn.mean(dim=1) if self.attn_reduce == "mean_query" else attn.mean(dim=0)
        return token_attn.cpu().numpy()

    def _compute_metrics(
        self, g_id: torch.Tensor, g_val: torch.Tensor, pad_mask: torch.Tensor, ig_steps: int
    ) -> tuple[np.ndarray, np.ndarray]:
        ig = self._compute_integrated_gradients(g_id=g_id, g_val=g_val, pad_mask=pad_mask, steps=ig_steps)
        with torch.no_grad():
            _ = self.model(g_id, g_val, pad_mask)
        attn = self._extract_token_attention()
        if attn is None:
            attn = np.zeros_like(ig)
        if len(attn) != len(ig):
            m = min(len(attn), len(ig))
            ig = ig[:m]
            attn = attn[:m]
        return ig, attn

    def _get_cluster_indices(self, cluster_col: str, cluster_id: str) -> np.ndarray:
        obs_cluster = (
            self.adata.obs[cluster_col]
            .astype(str)
            .str.strip()
            .map(self._normalise_cluster_label)
        )
        obs_traj = self.adata.obs[self.label_col].astype(str).str.strip()
        cid = self._normalise_cluster_label(cluster_id)
        mask = (obs_cluster == cid) & (obs_traj.isin(["PV", "NPV"]))
        return np.where(mask.to_numpy(dtype=bool))[0]

    @staticmethod
    def _sample_cluster_indices(
        indices: np.ndarray,
        n_cells: int | None = None,
        replace: bool = False,
        random_state: int = 42,
    ) -> np.ndarray:
        if len(indices) == 0:
            return np.array([], dtype=int)
        rng = np.random.default_rng(random_state)
        n_take = len(indices) if n_cells is None else (n_cells if replace else min(len(indices), n_cells))
        return rng.choice(indices, size=n_take, replace=replace).astype(int)

    def _run_one_cluster_repeat(
        self,
        x: np.ndarray,
        sampled_idx: np.ndarray,
        cluster_id: str,
        path_label: str,
        repeat_id: int,
        ig_steps: int = 50,
        verbose: bool = True,
    ) -> pd.DataFrame:
        gene_stats: dict[str, dict[str, Any]] = {}
        iterator = _tqdm_iter(
            sampled_idx,
            desc=f"Cluster {cluster_id} | Repeat {repeat_id}",
            leave=False,
            enabled=verbose,
        )

        for idx in iterator:
            raw_expr = x[int(idx)]
            try:
                g_id, g_val, pad_mask = self._prepare_inputs(raw_expr)
                if int((~pad_mask).sum().item()) == 0:
                    continue
                ig, attn = self._compute_metrics(
                    g_id=g_id, g_val=g_val, pad_mask=pad_mask, ig_steps=ig_steps
                )
                tids = g_id.detach().cpu().numpy().squeeze()
                valid_pos = np.where((~pad_mask).detach().cpu().numpy().squeeze())[0]
                seen_genes_in_cell: set[str] = set()

                for pos in valid_pos:
                    gid_p1 = int(tids[pos])
                    if gid_p1 == 0:
                        continue
                    gene_name = self.gene_names[gid_p1 - 1]
                    if gene_name.upper() not in self.tf_symbols:
                        continue
                    if gene_name not in gene_stats:
                        gene_stats[gene_name] = {
                            "ig": [],
                            "attn": [],
                            "cells_present": 0,
                        }
                    if pos < len(ig):
                        gene_stats[gene_name]["ig"].append(float(ig[pos]))
                    if pos < len(attn):
                        gene_stats[gene_name]["attn"].append(float(attn[pos]))
                    if gene_name not in seen_genes_in_cell:
                        gene_stats[gene_name]["cells_present"] += 1
                        seen_genes_in_cell.add(gene_name)
            except Exception as e:
                if verbose:
                    print(f"[WARN] Cell {idx} failed in cluster {cluster_id}, repeat {repeat_id}: {e}")
                continue

        rows: list[dict[str, Any]] = []
        n_cells = int(len(sampled_idx))
        for gene_name, stats in gene_stats.items():
            rows.append(
                {
                    "Cluster": str(cluster_id),
                    "Path": str(path_label),
                    "Repeat": int(repeat_id),
                    "Gene": gene_name,
                    "Mean_IG_to_PV": float(np.mean(stats["ig"])) if stats["ig"] else 0.0,
                    "Abs_Mean_IG_to_PV": float(np.mean(np.abs(stats["ig"]))) if stats["ig"] else 0.0,
                    "Mean_Attn": float(np.mean(stats["attn"])) if stats["attn"] else 0.0,
                    "Frequency": float(stats["cells_present"] / n_cells) if n_cells > 0 else 0.0,
                    "Cells_Present": int(stats["cells_present"]),
                    "N_Cells": n_cells,
                    "TargetClass": "PV",
                }
            )
        return pd.DataFrame(rows)

    def run_repeated_subsampling(
        self,
        cluster_path_dict: dict[str, str],
        cluster_col: str | None = None,
        n_repeats: int = 10,
        n_cells_per_cluster: int = 200,
        replace: bool = False,
        ig_steps: int = 50,
        file_prefix: str = "branch_tf_impact",
        verbose: bool = True,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if cluster_col is None:
            from model.utils.constants import get_cluster_column
            cluster_col = get_cluster_column()
        if cluster_col not in self.adata.obs.columns:
            raise ValueError(f"Missing column in adata.obs: {cluster_col}")
        x = self._get_expression_matrix()
        repeat_tables: list[pd.DataFrame] = []
        meta_rows: list[dict[str, Any]] = []

        for cid, path_label in cluster_path_dict.items():
            cid_norm = self._normalise_cluster_label(cid)
            indices = self._get_cluster_indices(cluster_col=cluster_col, cluster_id=cid_norm)
            if len(indices) == 0:
                if verbose:
                    print(f"Skip cluster {cid_norm}: 0 valid cells")
                continue

            for r in range(int(n_repeats)):
                seed = 42 + r
                sampled_idx = self._sample_cluster_indices(
                    indices=indices,
                    n_cells=int(n_cells_per_cluster),
                    replace=bool(replace),
                    random_state=seed,
                )
                meta_rows.append(
                    {
                        "Cluster": cid_norm,
                        "Path": path_label,
                        "Repeat": r,
                        "ClusterTotalCells": int(len(indices)),
                        "SampledCells": int(len(sampled_idx)),
                        "Replace": bool(replace),
                    }
                )
                rep_df = self._run_one_cluster_repeat(
                    x=x,
                    sampled_idx=sampled_idx,
                    cluster_id=cid_norm,
                    path_label=path_label,
                    repeat_id=r,
                    ig_steps=int(ig_steps),
                    verbose=verbose,
                )
                if not rep_df.empty:
                    repeat_tables.append(rep_df)

        if not repeat_tables:
            raise ValueError("No TF results generated.")

        repeat_df = pd.concat(repeat_tables, axis=0, ignore_index=True)
        meta_df = pd.DataFrame(meta_rows)
        cluster_summary_df = (
            repeat_df.groupby(["Cluster", "Path", "Gene"])
            .agg(
                Mean_IG_to_PV=("Mean_IG_to_PV", "mean"),
                SD_IG_to_PV=("Mean_IG_to_PV", "std"),
                Abs_Mean_IG_to_PV=("Abs_Mean_IG_to_PV", "mean"),
                SD_Abs_IG=("Abs_Mean_IG_to_PV", "std"),
                Mean_Attn=("Mean_Attn", "mean"),
                SD_Attn=("Mean_Attn", "std"),
                Frequency=("Frequency", "mean"),
                SD_Frequency=("Frequency", "std"),
                N_Repeats=("Repeat", "nunique"),
            )
            .reset_index()
        )
        balanced_tf_summary_df = (
            cluster_summary_df.groupby("Gene")
            .agg(
                Balanced_Mean_IG_to_PV=("Mean_IG_to_PV", "mean"),
                Balanced_Abs_Mean_IG_to_PV=("Abs_Mean_IG_to_PV", "mean"),
                Balanced_Mean_Attn=("Mean_Attn", "mean"),
                Balanced_Frequency=("Frequency", "mean"),
                N_Clusters=("Cluster", "nunique"),
            )
            .reset_index()
            .sort_values("Balanced_Abs_Mean_IG_to_PV", ascending=False)
        )

        repeat_path = self.output_dir / f"{file_prefix}_repeat_level.csv"
        cluster_summary_path = self.output_dir / f"{file_prefix}_cluster_summary.csv"
        balanced_path = self.output_dir / f"{file_prefix}_balanced_summary.csv"
        meta_path = self.output_dir / f"{file_prefix}_meta.csv"
        repeat_df.to_csv(repeat_path, index=False)
        cluster_summary_df.to_csv(cluster_summary_path, index=False)
        balanced_tf_summary_df.to_csv(balanced_path, index=False)
        meta_df.to_csv(meta_path, index=False)
        return repeat_df, cluster_summary_df, balanced_tf_summary_df, meta_df

