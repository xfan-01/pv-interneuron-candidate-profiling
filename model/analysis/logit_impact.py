"""Global logit-impact engine for classifier-level perturbation attribution.

Notebook source:
  - demo/classifier_analysis.ipynb (cell 9, GlobalLogitImpactEngine)
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


class GlobalLogitImpactEngine:
    """Estimate global gene importance via decision-logit perturbation.

    Decision score for binary classifier is defined as:
      ``logit(target_class) - logit(reference_class)``

    For each non-zero gene token in each sampled cell, this engine zeroes
    the token value, recomputes the decision score, and accumulates:
      ``delta = original_score - perturbed_score``.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        adata: Any,
        device: torch.device | str,
        label_col: str = "trajectory_class",
        max_len: int = 2134,
        output_dir: str = "LogitImpactResults",
        target_label: str = "PV",
        reference_label: str = "NPV",
        target_class_index: int = 1,
        reference_class_index: int = 0,
    ) -> None:
        self.model = model.to(device)
        self.model.eval()

        self.adata = adata
        self.device = torch.device(device)
        self.label_col = label_col
        self.max_len = int(max_len)
        self.output_dir = Path(output_dir)
        self.target_label = str(target_label)
        self.reference_label = str(reference_label)
        self.target_class_index = int(target_class_index)
        self.reference_class_index = int(reference_class_index)

        self.gene_names = np.asarray(adata.var_names).astype(str)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if self.label_col not in self.adata.obs.columns:
            raise ValueError(f"adata.obs must contain '{self.label_col}'")

    def _get_expression_matrix(self) -> np.ndarray:
        x = self.adata.X
        return x.toarray() if sp.issparse(x) else np.asarray(x)

    def _get_valid_indices(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        labels = self.adata.obs[self.label_col].astype(str).str.strip()
        valid_mask = labels.isin([self.target_label, self.reference_label])
        target_idx = np.where((labels == self.target_label).values)[0]
        ref_idx = np.where((labels == self.reference_label).values)[0]
        if len(target_idx) == 0 or len(ref_idx) == 0:
            raise ValueError(
                f"Need both classes present in '{self.label_col}'. "
                f"Found {self.target_label}={len(target_idx)}, {self.reference_label}={len(ref_idx)}"
            )
        return target_idx, ref_idx, valid_mask.to_numpy(dtype=bool)

    def _prepare_inputs(
        self, raw_expr: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare (gene_id, gene_val, padding_mask) for classifier forward."""
        non_zero_idx = np.where(raw_expr > 0)[0]
        g_id = torch.zeros((1, self.max_len), dtype=torch.long, device=self.device)
        g_val = torch.zeros((1, self.max_len), dtype=torch.float32, device=self.device)
        pad_mask = torch.ones((1, self.max_len), dtype=torch.bool, device=self.device)

        if len(non_zero_idx) == 0:
            return g_id, g_val, pad_mask

        non_zero_val = raw_expr[non_zero_idx]
        scaled_val = non_zero_val / (float(np.mean(non_zero_val)) + 1e-6)
        n = min(len(non_zero_idx), self.max_len)

        g_id[0, :n] = torch.as_tensor(
            non_zero_idx[:n] + 1, dtype=torch.long, device=self.device
        )
        g_val[0, :n] = torch.as_tensor(
            scaled_val[:n], dtype=torch.float32, device=self.device
        )
        pad_mask[0, :n] = False
        return g_id, g_val, pad_mask

    @torch.no_grad()
    def _decision_score(
        self, g_id: torch.Tensor, g_val: torch.Tensor, pad_mask: torch.Tensor
    ) -> float:
        logits = self.model(g_id, g_val, pad_mask)
        return float(
            logits[0, self.target_class_index] - logits[0, self.reference_class_index]
        )

    @torch.no_grad()
    def _compute_cell_gene_deltas(
        self, g_id: torch.Tensor, g_val: torch.Tensor, pad_mask: torch.Tensor
    ) -> list[tuple[str, float]]:
        original_score = self._decision_score(g_id, g_val, pad_mask)
        tids = g_id.detach().cpu().numpy().squeeze()
        valid_pos = np.where((~pad_mask).detach().cpu().numpy().squeeze())[0]
        rows: list[tuple[str, float]] = []

        for pos in valid_pos:
            gid_p1 = int(tids[pos])
            if gid_p1 <= 0:
                continue
            perturbed_g_val = g_val.clone()
            perturbed_g_val[0, pos] = 0.0
            perturbed_score = self._decision_score(g_id, perturbed_g_val, pad_mask)
            delta = original_score - perturbed_score
            rows.append((self.gene_names[gid_p1 - 1], float(delta)))
        return rows

    def _sample_indices(
        self,
        target_idx: np.ndarray,
        ref_idx: np.ndarray,
        n_per_class: int = 300,
        replace: bool = False,
        random_state: int = 42,
    ) -> np.ndarray:
        rng = np.random.default_rng(random_state)
        n_t = n_per_class if replace else min(len(target_idx), n_per_class)
        n_r = n_per_class if replace else min(len(ref_idx), n_per_class)
        sampled_t = rng.choice(target_idx, size=n_t, replace=replace)
        sampled_r = rng.choice(ref_idx, size=n_r, replace=replace)
        sampled = np.concatenate([sampled_t, sampled_r])
        rng.shuffle(sampled)
        return sampled.astype(int)

    def _run_one_repeat(
        self,
        x: np.ndarray,
        sampled_idx: np.ndarray,
        repeat_id: int,
        verbose: bool = True,
    ) -> pd.DataFrame:
        gene_stats: dict[str, dict[str, Any]] = {}
        iterator = _tqdm_iter(sampled_idx, desc=f"Repeat {repeat_id}", leave=False, enabled=verbose)

        for idx in iterator:
            raw_expr = x[int(idx)]
            g_id, g_val, pad_mask = self._prepare_inputs(raw_expr)
            if int((~pad_mask).sum().item()) == 0:
                continue

            try:
                rows = self._compute_cell_gene_deltas(g_id, g_val, pad_mask)
            except Exception as e:
                if verbose:
                    print(f"[WARN] Cell {idx} failed in repeat {repeat_id}: {e}")
                continue

            seen_genes: set[str] = set()
            for gene_name, delta in rows:
                if gene_name not in gene_stats:
                    gene_stats[gene_name] = {
                        "delta": [],
                        "abs_delta": [],
                        "cells_present": 0,
                    }
                gene_stats[gene_name]["delta"].append(delta)
                gene_stats[gene_name]["abs_delta"].append(abs(delta))
                if gene_name not in seen_genes:
                    gene_stats[gene_name]["cells_present"] += 1
                    seen_genes.add(gene_name)

        rows_out: list[dict[str, Any]] = []
        n_sampled = int(len(sampled_idx))
        for gene_name, stats in gene_stats.items():
            rows_out.append(
                {
                    "Repeat": int(repeat_id),
                    "Gene": gene_name,
                    "MeanSignedDelta": float(np.mean(stats["delta"])) if stats["delta"] else 0.0,
                    "MeanAbsDelta": float(np.mean(stats["abs_delta"])) if stats["abs_delta"] else 0.0,
                    "CellsPresent": int(stats["cells_present"]),
                    "SampledCells": n_sampled,
                    "Frequency": float(stats["cells_present"] / max(1, n_sampled)),
                }
            )
        return pd.DataFrame(rows_out)

    def run_repeated_subsampling(
        self,
        n_repeats: int = 10,
        n_per_class: int = 300,
        replace: bool = False,
        random_state: int = 42,
        save_prefix: str = "global_logit_impact",
        verbose: bool = True,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        x = self._get_expression_matrix()
        target_idx, ref_idx, _ = self._get_valid_indices()
        repeat_tables: list[pd.DataFrame] = []
        meta_rows: list[dict[str, Any]] = []

        for r in range(int(n_repeats)):
            seed = int(random_state + r)
            sampled_idx = self._sample_indices(
                target_idx=target_idx,
                ref_idx=ref_idx,
                n_per_class=int(n_per_class),
                replace=bool(replace),
                random_state=seed,
            )
            sampled_labels = (
                self.adata.obs.iloc[sampled_idx][self.label_col].astype(str).str.strip()
            )
            meta_rows.append(
                {
                    "Repeat": r,
                    "RandomState": seed,
                    "SampledCells": int(len(sampled_idx)),
                    "SampledTarget": int((sampled_labels == self.target_label).sum()),
                    "SampledReference": int((sampled_labels == self.reference_label).sum()),
                    "Replace": bool(replace),
                }
            )
            rep_df = self._run_one_repeat(x=x, sampled_idx=sampled_idx, repeat_id=r, verbose=verbose)
            if not rep_df.empty:
                repeat_tables.append(rep_df)

        if not repeat_tables:
            raise ValueError("No results generated across repeats.")

        repeat_df = pd.concat(repeat_tables, axis=0, ignore_index=True)
        meta_df = pd.DataFrame(meta_rows)
        summary_df = (
            repeat_df.groupby("Gene")
            .agg(
                MeanSignedDelta=("MeanSignedDelta", "mean"),
                SD_SignedDelta=("MeanSignedDelta", "std"),
                MeanAbsDelta=("MeanAbsDelta", "mean"),
                SD_MeanAbsDelta=("MeanAbsDelta", "std"),
                MeanFrequency=("Frequency", "mean"),
                SD_Frequency=("Frequency", "std"),
                N_Repeats=("Repeat", "nunique"),
            )
            .reset_index()
            .sort_values("MeanAbsDelta", ascending=False)
        )

        repeat_path = self.output_dir / f"{save_prefix}_repeat_level.csv"
        summary_path = self.output_dir / f"{save_prefix}_summary.csv"
        meta_path = self.output_dir / f"{save_prefix}_repeat_meta.csv"
        repeat_df.to_csv(repeat_path, index=False)
        summary_df.to_csv(summary_path, index=False)
        meta_df.to_csv(meta_path, index=False)
        return summary_df, repeat_df, meta_df

    def run_single_subsample(
        self,
        n_per_class: int = 300,
        replace: bool = False,
        random_state: int = 42,
        save_prefix: str = "global_logit_impact_quick",
        verbose: bool = True,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        x = self._get_expression_matrix()
        target_idx, ref_idx, _ = self._get_valid_indices()
        sampled_idx = self._sample_indices(
            target_idx=target_idx,
            ref_idx=ref_idx,
            n_per_class=int(n_per_class),
            replace=bool(replace),
            random_state=int(random_state),
        )
        rep_df = self._run_one_repeat(
            x=x, sampled_idx=sampled_idx, repeat_id=0, verbose=verbose
        )
        if rep_df.empty:
            raise ValueError("No results generated in quick run.")

        summary_df = (
            rep_df.groupby("Gene")
            .agg(
                MeanSignedDelta=("MeanSignedDelta", "mean"),
                MeanAbsDelta=("MeanAbsDelta", "mean"),
                MeanFrequency=("Frequency", "mean"),
            )
            .reset_index()
            .sort_values("MeanAbsDelta", ascending=False)
        )

        rep_path = self.output_dir / f"{save_prefix}_repeat_level.csv"
        summary_path = self.output_dir / f"{save_prefix}_summary.csv"
        rep_df.to_csv(rep_path, index=False)
        summary_df.to_csv(summary_path, index=False)
        return summary_df, rep_df

