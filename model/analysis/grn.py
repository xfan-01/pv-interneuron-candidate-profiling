"""GRN extraction, bootstrap robustness, enrichment, and permutation testing.

Notebook source:
  - demo/generator_analysis_3.ipynb
    - load_tf_set (cell 9)
    - extract_tf_marker_network / run_heatmap_pipeline (cell 10)
    - GRNBootstrapper (cell 12)
    - GRNEvaluator (cell 14)
    - GRNPermutationTester (cell 16)
"""

from __future__ import annotations

import re
import textwrap
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


def load_tf_set(file_path: str) -> set[str]:
    """Load TF symbols from TSV/HTML file."""
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"TF file not found: {file_path}")
    tf_df = None
    try:
        tf_df = pd.read_csv(p, sep="\t")
    except Exception:
        tf_df = None
    if tf_df is None or "Symbol" not in tf_df.columns:
        tables = pd.read_html(str(p))
        if not tables:
            raise ValueError(f"No table found in TF file: {file_path}")
        tf_df = tables[0]
    symbols = tf_df["Symbol"] if "Symbol" in tf_df.columns else tf_df.iloc[:, 0]
    return set(symbols.dropna().astype(str).str.strip().tolist())


def _tqdm_iter(iterable: Any, desc: str, leave: bool = False, enabled: bool = True) -> Any:
    if not enabled:
        return iterable
    try:
        from tqdm import tqdm
    except Exception:
        return iterable
    return tqdm(iterable, desc=desc, leave=leave)


def extract_tf_marker_network(
    model: torch.nn.Module,
    dataset: Any,
    target_markers: list[str],
    known_tfs: set[str],
    adata: Any,
    device: str | torch.device = "cpu",
    batch_size: int = 32,
    threshold: float = 0.005,
    verbose: bool = True,
) -> pd.DataFrame:
    """Extract TF->marker interactions from generator cross-attention."""
    device = torch.device(device)
    gene_names = list(adata.var_names)
    gene_to_idx = {name: i for i, name in enumerate(gene_names)}
    valid_markers = [m for m in target_markers if m in gene_to_idx]
    if verbose:
        print(f"Valid markers: {valid_markers}")

    model.eval()
    interaction_sums: dict[tuple[str, str], float] = {}
    interaction_counts: dict[tuple[str, str], int] = {}

    indices = range(len(dataset))
    num_batches = int(np.ceil(len(indices) / max(1, batch_size)))
    for i in _tqdm_iter(range(num_batches), desc="Scanning Attention", enabled=verbose):
        batch_idx = list(indices)[i * batch_size : (i + 1) * batch_size]
        g_ids, g_vals, times, target_times, masks = [], [], [], [], []
        for idx in batch_idx:
            item = dataset[idx]
            g_ids.append(item["gene_id"].detach())
            g_vals.append(item["gene_val"].detach())
            times.append(item["time"].detach())
            target_times.append(item["target_time"].detach())
            masks.append(item["padding_mask"].detach())

        g_id_b = torch.stack(g_ids).to(device)
        g_val_b = torch.stack(g_vals).to(device)
        time_b = torch.stack(times).to(device)
        target_time_b = torch.stack(target_times).to(device)
        mask_b = torch.stack(masks).to(device)

        with torch.no_grad():
            _, _, cross_attn = model(
                g_id_b,
                g_val_b,
                time_b,
                target_time_b,
                padding_mask=mask_b,
                need_weights=True,
            )
        if cross_attn is None:
            continue

        cross_attn_np = cross_attn.detach().cpu().numpy()
        batch_input_ids = g_id_b.detach().cpu().numpy()
        for b in range(len(batch_idx)):
            curr_attn = cross_attn_np[b]
            curr_input_seq = batch_input_ids[b]
            for marker in valid_markers:
                query_idx = gene_to_idx[marker]
                if query_idx >= curr_attn.shape[0]:
                    continue
                attn_to_inputs = curr_attn[query_idx, :]
                for seq_pos, input_id in enumerate(curr_input_seq):
                    if input_id == 0:
                        continue
                    source_gene = gene_names[int(input_id) - 1]
                    if source_gene in known_tfs:
                        weight = float(attn_to_inputs[seq_pos])
                        if weight > threshold:
                            pair = (source_gene, marker)
                            interaction_sums[pair] = interaction_sums.get(pair, 0.0) + weight
                            interaction_counts[pair] = interaction_counts.get(pair, 0) + 1

    edges = []
    total_cells = len(dataset)
    for pair, total_weight in interaction_sums.items():
        count = interaction_counts[pair]
        mean_weight = total_weight / count
        freq = count / max(1, total_cells)
        score = mean_weight * freq * 100.0
        edges.append(
            {
                "Source_TF": pair[0],
                "Target_Marker": pair[1],
                "Weight": mean_weight,
                "Frequency": freq,
                "Score": score,
            }
        )
    df_edges = pd.DataFrame(edges)
    if not df_edges.empty:
        df_edges = df_edges.sort_values("Score", ascending=False).reset_index(drop=True)
    return df_edges


def run_heatmap_pipeline(df_stable: pd.DataFrame, top_n: int = 30) -> pd.DataFrame:
    """Prepare standardized edge table used by heatmap visualization."""
    subset = df_stable.head(top_n).copy()
    name_map = {"Mean_Score": "Score", "Mean_Weight": "Weight"}
    return subset.rename(columns={k: v for k, v in name_map.items() if k in subset.columns})


class GRNBootstrapper:
    def __init__(self, model: torch.nn.Module, dataset: Any, markers: list[str], tfs: set[str], adata: Any, k_mad: float = 3.0):
        self.model = model
        self.dataset = dataset
        self.markers = markers
        self.tfs = tfs
        self.adata = adata
        self.k_mad = float(k_mad)

    def run_bootstrap(
        self,
        n_iterations: int = 20,
        sample_frac: float = 0.8,
        device: str | torch.device = "cpu",
        batch_size: int = 32,
        threshold: float = 0.005,
        verbose: bool = True,
    ) -> pd.DataFrame:
        all_iterations_results = []
        n_total = len(self.dataset)

        for i in range(int(n_iterations)):
            if verbose:
                print(f"\n--- Bootstrap Iteration {i + 1}/{n_iterations} ---")
            indices = np.random.choice(n_total, size=max(1, int(n_total * sample_frac)), replace=True)

            class _Subset:
                def __init__(self, ds, idxs):
                    self.ds = ds
                    self.idxs = list(map(int, idxs))

                def __len__(self):
                    return len(self.idxs)

                def __getitem__(self, j):
                    return self.ds[self.idxs[j]]

            subset_ds = _Subset(self.dataset, indices)
            df_iter = extract_tf_marker_network(
                self.model,
                subset_ds,
                self.markers,
                self.tfs,
                self.adata,
                device=device,
                batch_size=batch_size,
                threshold=threshold,
                verbose=verbose,
            )
            if df_iter.empty:
                continue
            df_iter["is_sig"] = 0
            for marker in self.markers:
                mask = df_iter["Target_Marker"] == marker
                if not mask.any():
                    continue
                scores = pd.to_numeric(df_iter.loc[mask, "Score"], errors="coerce")
                med = scores.median()
                mad = (scores - med).abs().median()
                thresh = med + self.k_mad * mad
                df_iter.loc[mask & (scores > thresh), "is_sig"] = 1
            all_iterations_results.append(df_iter)

        if not all_iterations_results:
            return pd.DataFrame()
        master_df = pd.concat(all_iterations_results, ignore_index=True)
        return (
            master_df.groupby(["Source_TF", "Target_Marker"])
            .agg(
                Mean_Score=("Score", "mean"),
                Std_Score=("Score", "std"),
                Selection_Freq=("is_sig", "mean"),
                Appearance_Count=("is_sig", "count"),
            )
            .reset_index()
            .sort_values(["Selection_Freq", "Mean_Score"], ascending=False)
        )


class GRNEvaluator:
    def __init__(self, csv_path: str, k_multiplier: float = 3.5, target_markers: list[str] | None = None):
        self.csv_path = csv_path
        self.k_multiplier = float(k_multiplier)
        self.target_markers = target_markers or ["PVALB", "FGF13", "LHX6", "GAD1", "SYN3", "CACNA1A", "NLGN1"]
        self.db = "Reactome_2022"
        self.raw_df: pd.DataFrame | None = None
        self.filtered_tf_summary: dict[str, Any] = {}
        self.enrichment_summary: dict[str, pd.DataFrame] = {}
        self.plot_df: pd.DataFrame | None = None

    @staticmethod
    def _parse_overlap_to_ratio(overlap_str: str) -> float:
        try:
            a, b = overlap_str.split("/")
            return float(a) / float(b)
        except Exception:
            return np.nan

    @staticmethod
    def _remove_reactome_id(term: Any) -> Any:
        if not isinstance(term, str):
            return term
        cleaned = re.sub(r"\bR-HSA-\d+\b", "", term)
        cleaned = re.sub(r"\(\s*\)", "", cleaned)
        cleaned = re.sub(r"\[\s*\]", "", cleaned)
        cleaned = re.sub(r"\s{2,}", " ", cleaned)
        cleaned = re.sub(r"\s*[-–|:]\s*$", "", cleaned)
        return cleaned.strip()

    @staticmethod
    def _wrap_term(term: Any, width: int = 40) -> Any:
        if not isinstance(term, str):
            return term
        return "\n".join(textwrap.wrap(term, width=width))

    def load_data(self) -> pd.DataFrame:
        self.raw_df = pd.read_csv(self.csv_path)
        return self.raw_df

    def process_data(self) -> dict[str, Any]:
        if self.raw_df is None:
            self.load_data()
        assert self.raw_df is not None
        if "Score" not in self.raw_df.columns and "Mean_Score" in self.raw_df.columns:
            self.raw_df = self.raw_df.rename(columns={"Mean_Score": "Score"})
        required = {"Target_Marker", "Source_TF", "Score"}
        miss = required - set(self.raw_df.columns)
        if miss:
            raise ValueError(f"Missing required columns in input CSV: {miss}")
        out: dict[str, Any] = {}
        for marker in self.target_markers:
            df_marker = self.raw_df[self.raw_df["Target_Marker"] == marker].copy()
            if df_marker.empty:
                continue
            scores = pd.to_numeric(df_marker["Score"], errors="coerce")
            med = float(scores.median())
            mad = float((scores - med).abs().median())
            if mad == 0:
                mad = 1e-6
            threshold = med + self.k_multiplier * mad
            sig_tfs_df = df_marker[scores > threshold].copy()
            sig_tfs = (
                sig_tfs_df["Source_TF"]
                .dropna()
                .astype(str)
                .str.strip()
                .str.upper()
                .unique()
                .tolist()
            )
            out[marker] = {
                "median": med,
                "mad": mad,
                "threshold": threshold,
                "n_total_tfs": int(len(df_marker)),
                "n_selected_tfs": int(len(sig_tfs)),
                "selected_tfs": sig_tfs,
                "selected_tfs_df": sig_tfs_df,
            }
        self.filtered_tf_summary = out
        return out

    def run_enrichment(self, adjusted_p_cutoff: float = 0.05, max_retries: int = 3, retry_wait_sec: int = 5) -> dict[str, pd.DataFrame]:
        if not self.filtered_tf_summary:
            self.process_data()
        try:
            import gseapy as gp
        except ImportError as e:
            raise ImportError("gseapy is required for GRNEvaluator enrichment.") from e

        out: dict[str, pd.DataFrame] = {}
        for marker, info in self.filtered_tf_summary.items():
            sig_tfs = info["selected_tfs"]
            if len(sig_tfs) < 3:
                out[marker] = pd.DataFrame()
                continue
            success = False
            enr = None
            err: Exception | None = None
            for attempt in range(1, max_retries + 1):
                try:
                    enr = gp.enrichr(
                        gene_list=sig_tfs,
                        gene_sets=[self.db],
                        outdir=None,
                        no_plot=True,
                        cutoff=adjusted_p_cutoff,
                    )
                    success = True
                    break
                except Exception as e:
                    err = e
                    msg = str(e).lower()
                    if (("429" in msg) or ("rate" in msg)) and attempt < max_retries:
                        time.sleep(retry_wait_sec * attempt)
                        continue
                    break
            if not success:
                if err is not None:
                    print(f"[{marker}] enrichment failed: {err}")
                out[marker] = pd.DataFrame()
                continue
            res_df = enr.results.copy()
            sig_path = res_df[res_df["Adjusted P-value"] < adjusted_p_cutoff].copy() if not res_df.empty else pd.DataFrame()
            if not sig_path.empty:
                sig_path["Marker"] = marker
                sig_path["Overlap_Ratio"] = sig_path["Overlap"].map(self._parse_overlap_to_ratio)
                sig_path["NegLog10_Adjusted_P"] = -np.log10(sig_path["Adjusted P-value"].clip(lower=1e-10))
            out[marker] = sig_path
        self.enrichment_summary = out
        return out

    def extract_core_pathways(self, keywords: list[str] | None = None, top_n_per_marker: int | None = None) -> pd.DataFrame:
        if not self.enrichment_summary:
            self.run_enrichment()
        if keywords is None:
            keywords = ["Wnt", "FOXO", "stem cell", "negative regulation", "inhibition", "silencing", "suppression", "cell fate", "differentiation", "neuron", "neurogenesis", "synapse", "axon", "Notch", "TGF", "MAPK", "PI3K", "AKT"]
        all_core = []
        pattern = "|".join(keywords)
        for marker, df in self.enrichment_summary.items():
            if df is None or df.empty:
                continue
            core_df = df[df["Term"].str.contains(pattern, case=False, na=False)].copy()
            if top_n_per_marker is not None and not core_df.empty:
                core_df = core_df.sort_values("Adjusted P-value", ascending=True).head(top_n_per_marker)
            all_core.append(core_df)
        if not all_core:
            self.plot_df = pd.DataFrame()
            return self.plot_df
        plot_df = pd.concat(all_core, ignore_index=True)
        plot_df = (
            plot_df.sort_values(["Marker", "Adjusted P-value"], ascending=[True, True])
            .drop_duplicates(subset=["Marker", "Term"], keep="first")
            .copy()
        )
        plot_df["Display_Term"] = plot_df["Term"].map(self._remove_reactome_id).map(lambda x: self._wrap_term(x, width=40))
        self.plot_df = plot_df
        return plot_df


class GRNPermutationTester:
    def __init__(self, model: torch.nn.Module, dataset: Any, markers: list[str], tfs: set[str], adata: Any, real_results_df: pd.DataFrame):
        self.model = model
        self.dataset = dataset
        self.markers = markers
        self.tfs = tfs
        self.adata = adata
        self.real_results = real_results_df
        self.gene_names = list(adata.var_names)
        self.gene_to_idx = {name: i for i, name in enumerate(self.gene_names)}
        self.valid_markers = [m for m in self.markers if m in self.gene_to_idx]

    def _format_edges(self, interaction_sums: dict[tuple[str, str], float], interaction_counts: dict[tuple[str, str], int], total_samples: int) -> pd.DataFrame:
        edges = []
        for pair, total_weight in interaction_sums.items():
            count = interaction_counts[pair]
            mean_weight = total_weight / count
            freq = count / max(1, total_samples)
            score = mean_weight * freq * 100
            edges.append({"Source_TF": pair[0], "Target_Marker": pair[1], "Weight": mean_weight, "Frequency": freq, "Score": score})
        return pd.DataFrame(edges)

    def _extract_shuffled_network(
        self,
        device: str | torch.device = "cpu",
        batch_size: int = 32,
        threshold: float = 0.005,
        verbose: bool = True,
    ) -> pd.DataFrame:
        device = torch.device(device)
        interaction_sums: dict[tuple[str, str], float] = {}
        interaction_counts: dict[tuple[str, str], int] = {}
        indices = range(len(self.dataset))
        num_batches = int(np.ceil(len(indices) / max(1, batch_size)))
        self.model.eval()
        for i in _tqdm_iter(range(num_batches), desc="Shuffled Inference", leave=False, enabled=verbose):
            batch_idx = list(indices)[i * batch_size : (i + 1) * batch_size]
            g_ids, g_vals, times, t_times, masks = [], [], [], [], []
            for idx in batch_idx:
                item = self.dataset[idx]
                g_id = item["gene_id"].detach().clone()
                g_val = item["gene_val"].detach().clone()
                mask = item["padding_mask"].detach().clone()
                valid_indices = torch.where(~mask)[0]
                if len(valid_indices) > 1:
                    shuffled_vals = g_val[valid_indices][torch.randperm(len(valid_indices))]
                    g_val[valid_indices] = shuffled_vals
                g_ids.append(g_id)
                g_vals.append(g_val)
                times.append(item["time"].detach())
                t_times.append(item["target_time"].detach())
                masks.append(mask)
            g_id_b = torch.stack(g_ids).to(device)
            g_val_b = torch.stack(g_vals).to(device)
            time_b = torch.stack(times).to(device)
            t_time_b = torch.stack(t_times).to(device)
            mask_b = torch.stack(masks).to(device)
            with torch.no_grad():
                _, _, cross_attn = self.model(
                    g_id_b, g_val_b, time_b, t_time_b, padding_mask=mask_b, need_weights=True
                )
            if cross_attn is None:
                continue
            cross_attn_np = cross_attn.detach().cpu().numpy()
            batch_input_ids = g_id_b.detach().cpu().numpy()
            for b in range(len(batch_idx)):
                curr_attn = cross_attn_np[b]
                curr_input_seq = batch_input_ids[b]
                for marker in self.valid_markers:
                    query_idx = self.gene_to_idx[marker]
                    if query_idx >= curr_attn.shape[0]:
                        continue
                    attn_to_inputs = curr_attn[query_idx, :]
                    for seq_pos, input_id in enumerate(curr_input_seq):
                        if input_id == 0:
                            continue
                        source_gene = self.gene_names[int(input_id) - 1]
                        if source_gene in self.tfs:
                            weight = float(attn_to_inputs[seq_pos])
                            if weight > threshold:
                                pair = (source_gene, marker)
                                interaction_sums[pair] = interaction_sums.get(pair, 0.0) + weight
                                interaction_counts[pair] = interaction_counts.get(pair, 0) + 1
        return self._format_edges(interaction_sums, interaction_counts, len(self.dataset))

    def run_permutation_test(
        self,
        n_permutations: int = 20,
        device: str | torch.device = "cpu",
        batch_size: int = 32,
        threshold: float = 0.005,
        verbose: bool = True,
    ) -> pd.DataFrame:
        all_perm_scores = []
        for i in range(int(n_permutations)):
            if verbose:
                print(f"\n--- Permutation Round {i + 1}/{n_permutations} ---")
            df_perm = self._extract_shuffled_network(
                device=device, batch_size=batch_size, threshold=threshold, verbose=verbose
            )
            if not df_perm.empty:
                all_perm_scores.append(df_perm)
        if not all_perm_scores:
            return pd.DataFrame()
        perm_master_df = pd.concat(all_perm_scores, ignore_index=True)
        perm_stats = (
            perm_master_df.groupby(["Source_TF", "Target_Marker"])
            .agg(Perm_Mean_Score=("Score", "mean"), Perm_Std_Score=("Score", "std"))
            .reset_index()
        )
        comparison = pd.merge(
            self.real_results,
            perm_stats,
            on=["Source_TF", "Target_Marker"],
            how="left",
        ).fillna(0)
        comparison["Z_Score_Vs_Null"] = (
            (comparison["Mean_Score"] - comparison["Perm_Mean_Score"])
            / (comparison["Perm_Std_Score"] + 1e-8)
        )
        return comparison.sort_values("Z_Score_Vs_Null", ascending=False).reset_index(drop=True)
