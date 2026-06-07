"""Notebook orchestration helpers migrated into reusable analysis classes.

These classes intentionally provide a compact, script-friendly subset of the
original notebook orchestration. Plotting-heavy details remain notebook-scoped.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse
import torch
from model.data.preprocessing import prepare_clusters
from model.utils.constants import (
    get_cluster_column,
    get_pv_path_nodes,
    get_time_column,
    get_time_fallback_columns,
)


@dataclass
class ProcessorConfig:
    """Minimal processor config shared by generator/classifier adapters."""

    n_genes: int
    gen_max_len: int
    cls_max_len: int
    cluster_col: str = get_cluster_column()
    time_col: str = get_time_column()
    class_names: tuple[str, ...] = ()


class CellDataProcessor:
    """Bridge AnnData + model tokenization for analysis-time inference."""

    def __init__(
        self,
        adata: Any,
        generator: torch.nn.Module,
        classifier: torch.nn.Module | None,
        config: ProcessorConfig,
        device: str | torch.device = "cpu",
    ) -> None:
        self.adata = adata
        self.generator = generator
        self.classifier = classifier
        self.cfg = config
        self.device = torch.device(device)
        self.gene_names = list(self.adata.var_names)
        self.gene_to_index = {g: i for i, g in enumerate(self.gene_names)}
        self._ensure_time_column()
        self.adata = prepare_clusters(self.adata, self.cfg.cluster_col)

    def _ensure_time_column(self) -> None:
        if self.cfg.time_col not in self.adata.obs.columns:
            # Try primary constant, then cfg default, then fallbacks
            candidates: list[str] = [get_time_column()]
            if self.cfg.time_col not in candidates:
                candidates.append(self.cfg.time_col)
            candidates.extend(get_time_fallback_columns())
            seen: set[str] = set()
            fallback_cols: list[str] = []
            for c in candidates:
                if c not in seen:
                    seen.add(c)
                    fallback_cols.append(c)
            for col in fallback_cols:
                if col in self.adata.obs.columns:
                    values = pd.to_numeric(
                        self.adata.obs[col], errors="coerce"
                    ).to_numpy(dtype=np.float32)
                    mask = ~np.isnan(values)
                    self.adata = self.adata[mask].copy()
                    values = values[mask]
                    vmin, vmax = float(values.min()), float(values.max())
                    if np.isclose(vmin, vmax):
                        scaled = np.zeros_like(values, dtype=np.float32)
                    else:
                        scaled = (values - vmin) / (vmax - vmin + 1e-8)
                    self.adata.obs[self.cfg.time_col] = scaled
                    return
            raise ValueError("No valid pseudotime column found in adata.obs")

    def has_gene(self, gene_name: str) -> bool:
        return gene_name in self.gene_to_index

    def get_gene_index(self, gene_name: str) -> int:
        if gene_name not in self.gene_to_index:
            raise ValueError(f"Gene not found: {gene_name}")
        return int(self.gene_to_index[gene_name])

    def get_cell_expression_vector(self, cell_idx: int) -> np.ndarray:
        vec = self.adata.X[cell_idx]
        if scipy.sparse.issparse(vec):
            vec = vec.toarray().reshape(-1)
        else:
            vec = np.asarray(vec).reshape(-1)
        vec = vec.astype(np.float32)
        return np.clip(vec[: self.cfg.n_genes], a_min=0.0, a_max=None)

    def sample_cells(
        self,
        index_label: str = "all",
        index_mode: str = "cluster",
        n_samples: int = 100,
        seed: int = 42,
    ) -> list[dict[str, Any]]:
        rng = np.random.default_rng(seed)
        if str(index_label).lower() == "all" or str(index_mode).lower() == "all":
            indices = np.arange(self.adata.n_obs)
        else:
            labels = self.adata.obs[self.cfg.cluster_col].astype(str).to_numpy()
            indices = np.where(labels == str(index_label))[0]
            if len(indices) == 0:
                raise ValueError(f"No cells found for cluster: {index_label}")
        if len(indices) > n_samples:
            indices = rng.choice(indices, size=n_samples, replace=False)
        out: list[dict[str, Any]] = []
        for idx in indices:
            out.append(
                {
                    "cell_idx": int(idx),
                    "obs_name": str(self.adata.obs_names[idx]),
                    "expr": self.get_cell_expression_vector(int(idx)),
                    "source_time": float(self.adata.obs[self.cfg.time_col].iloc[idx]),
                    "cluster": (
                        str(self.adata.obs[self.cfg.cluster_col].iloc[idx])
                        if self.cfg.cluster_col in self.adata.obs.columns
                        else None
                    ),
                }
            )
        return out

    def vector_to_generator_tokens(
        self, expr_vec: np.ndarray, eps: float = 0.0
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        expr_vec = np.asarray(expr_vec, dtype=np.float32)
        nz = np.where(expr_vec > eps)[0]
        nz = nz[nz < self.cfg.n_genes]
        vals = expr_vec[nz]
        ids = nz + 1
        if len(ids) > self.cfg.gen_max_len:
            ids = ids[: self.cfg.gen_max_len]
            vals = vals[: self.cfg.gen_max_len]
        g_id = np.zeros(self.cfg.gen_max_len, dtype=np.int64)
        g_val = np.zeros(self.cfg.gen_max_len, dtype=np.float32)
        pad = np.ones(self.cfg.gen_max_len, dtype=bool)
        if len(ids) > 0:
            g_id[: len(ids)] = ids
            g_val[: len(ids)] = vals
            pad[: len(ids)] = False
        return (
            torch.from_numpy(g_id),
            torch.from_numpy(g_val),
            torch.from_numpy(pad),
        )

    def vector_to_classifier_tokens(
        self, expr_vec: np.ndarray, eps: float = 0.0
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        expr_vec = np.asarray(expr_vec, dtype=np.float32)
        nz = np.where(expr_vec > eps)[0]
        nz = nz[nz < self.cfg.n_genes]
        vals = expr_vec[nz]
        if len(vals) > 0:
            vals = vals / (float(np.mean(vals)) + 1e-8)
        ids = nz + 1
        if len(ids) > self.cfg.cls_max_len:
            ids = ids[: self.cfg.cls_max_len]
            vals = vals[: self.cfg.cls_max_len]
        c_id = np.zeros(self.cfg.cls_max_len, dtype=np.int64)
        c_val = np.zeros(self.cfg.cls_max_len, dtype=np.float32)
        valid = np.zeros(self.cfg.cls_max_len, dtype=bool)
        if len(ids) > 0:
            c_id[: len(ids)] = ids
            c_val[: len(ids)] = vals
            valid[: len(ids)] = True
        return (
            torch.from_numpy(c_id),
            torch.from_numpy(c_val),
            torch.from_numpy(valid),
        )

    def build_generator_batch(
        self, sampled_cells: list[dict[str, Any]], target_times: np.ndarray
    ) -> dict[str, torch.Tensor]:
        ids, vals, pads, src_t = [], [], [], []
        for rec in sampled_cells:
            g_id, g_val, pad = self.vector_to_generator_tokens(rec["expr"])
            ids.append(g_id)
            vals.append(g_val)
            pads.append(pad)
            src_t.append(float(rec["source_time"]))
        return {
            "g_id": torch.stack(ids).to(self.device),
            "g_val": torch.stack(vals).to(self.device),
            "padding_mask": torch.stack(pads).to(self.device),
            "source_time": torch.tensor(src_t, dtype=torch.float32, device=self.device),
            "target_time": torch.tensor(
                target_times, dtype=torch.float32, device=self.device
            ),
        }

    @torch.no_grad()
    def run_generator_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
        self.generator.eval()
        pred, _, _ = self.generator(
            batch["g_id"],
            batch["g_val"],
            batch["source_time"],
            batch["target_time"],
            padding_mask=batch["padding_mask"],
        )
        return {"pred_expr": pred.detach().cpu().numpy().astype(np.float32)}

    @torch.no_grad()
    def run_classifier_expr_matrix(self, expr_matrix: np.ndarray) -> dict[str, np.ndarray]:
        if self.classifier is None:
            raise ValueError("Classifier is not provided.")
        self.classifier.eval()
        ids, vals, pads = [], [], []
        for row in np.asarray(expr_matrix, dtype=np.float32):
            i, v, valid = self.vector_to_classifier_tokens(row)
            ids.append(i)
            vals.append(v)
            pads.append(~valid)
        logits = self.classifier(
            torch.stack(ids).to(self.device),
            torch.stack(vals).to(self.device),
            torch.stack(pads).to(self.device),
        )
        probs = torch.softmax(logits, dim=1)
        return {
            "logits": logits.detach().cpu().numpy(),
            "probs": probs.detach().cpu().numpy(),
        }


class PerturbationRunner:
    """Single-step generator perturbation runner."""

    def __init__(
        self,
        processor: CellDataProcessor,
        infer_batch_size: int = 64,
        target_time_horizon: float = 0.0,
    ) -> None:
        self.processor = processor
        self.generator = processor.generator
        self.infer_batch_size = int(infer_batch_size)
        self.target_time_horizon = float(target_time_horizon)

    def apply_perturbation(
        self, expr_vec: np.ndarray, perturbation_spec: dict[str, str]
    ) -> np.ndarray:
        out = np.asarray(expr_vec, dtype=np.float32).copy()
        for gene_name, action in perturbation_spec.items():
            if not self.processor.has_gene(gene_name):
                continue
            idx = self.processor.get_gene_index(gene_name)
            act = action.upper()
            if act == "KO":
                out[idx] = 0.0
            elif act == "KD":
                out[idx] = max(0.0, float(out[idx]) * 0.5)
            elif act == "OE":
                out[idx] = max(float(out[idx]), float(np.percentile(out, 95)))
        return out

    def run_generator(
        self,
        sampled_cells: list[dict[str, Any]],
        perturbation_spec: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        eval_cells = sampled_cells
        if perturbation_spec is not None:
            eval_cells = []
            for rec in sampled_cells:
                new_rec = dict(rec)
                new_rec["expr"] = self.apply_perturbation(rec["expr"], perturbation_spec)
                eval_cells.append(new_rec)

        src = np.asarray([r["source_time"] for r in eval_cells], dtype=np.float32)
        tgt = np.clip(src + self.target_time_horizon, 0.0, 1.0)
        pred_chunks = []
        for i in range(0, len(eval_cells), self.infer_batch_size):
            chunk = eval_cells[i : i + self.infer_batch_size]
            t_chunk = tgt[i : i + self.infer_batch_size]
            batch = self.processor.build_generator_batch(chunk, t_chunk)
            pred_chunks.append(self.processor.run_generator_batch(batch)["pred_expr"])

        return {
            "pred_expr": np.concatenate(pred_chunks, axis=0) if pred_chunks else np.zeros((0, self.processor.cfg.n_genes), dtype=np.float32),
            "target_times": tgt,
            "cell_meta": pd.DataFrame(
                [{k: v for k, v in rec.items() if k != "expr"} for rec in eval_cells]
            ),
            "perturbation_spec": perturbation_spec,
        }

    def compare_generator_outputs(
        self, baseline_out: dict[str, Any], perturbed_out: dict[str, Any]
    ) -> dict[str, Any]:
        delta = np.asarray(perturbed_out["pred_expr"]) - np.asarray(
            baseline_out["pred_expr"]
        )
        l2 = np.linalg.norm(delta, axis=1)
        return {
            "delta_expr": delta,
            "per_cell_l2": l2,
            "summary": {
                "n_cells": int(delta.shape[0]),
                "n_genes": int(delta.shape[1]) if delta.ndim == 2 else 0,
                "mean_l2_shift": float(np.mean(l2)) if len(l2) else 0.0,
            },
        }


class PathSpecificReferenceProjector:
    """Lightweight path-space projector for geometry comparison."""

    def __init__(
        self,
        adata: Any,
        n_genes: int,
        path_clusters: list[str],
        cluster_col: str | None = None,
        time_col: str | None = None,
        n_pca_components: int = 20,
    ) -> None:
        from sklearn.decomposition import PCA

        self.n_genes = int(n_genes)
        self.cluster_col = cluster_col or get_cluster_column()
        self.time_col = time_col or get_time_column()
        self.path_clusters = [str(x) for x in path_clusters]
        if self.cluster_col not in adata.obs.columns:
            raise KeyError(f"Missing cluster column: {self.cluster_col}")
        mask = adata.obs[self.cluster_col].astype(str).isin(self.path_clusters).to_numpy()
        ref = adata[mask].copy()
        X = ref.X.toarray() if scipy.sparse.issparse(ref.X) else np.asarray(ref.X)
        X = np.asarray(X[:, : self.n_genes], dtype=np.float32)
        self.pca = PCA(n_components=min(n_pca_components, X.shape[1], max(2, X.shape[0] - 1)))
        self.ref_pca = self.pca.fit_transform(X)
        self.ref_time = (
            pd.to_numeric(ref.obs[self.time_col], errors="coerce").to_numpy(dtype=np.float32)
            if self.time_col in ref.obs.columns
            else None
        )
        self.progress_sign = self._resolve_progress_sign(ref.obs, self.ref_pca)

    def _resolve_progress_sign(self, obs: pd.DataFrame, ref_pca: np.ndarray) -> float:
        """Orient PC1 so positive progress follows pseudotime/path order."""
        pc1 = np.asarray(ref_pca[:, 0], dtype=np.float32)
        if self.ref_time is not None and np.isfinite(self.ref_time).sum() >= 3:
            corr = np.corrcoef(pc1[np.isfinite(self.ref_time)], self.ref_time[np.isfinite(self.ref_time)])[0, 1]
            if np.isfinite(corr) and not np.isclose(corr, 0.0):
                return 1.0 if corr > 0 else -1.0

        clusters = obs[self.cluster_col].astype(str).to_numpy()
        centroid_pc1 = []
        for cl in self.path_clusters:
            mask = clusters == str(cl)
            if mask.any():
                centroid_pc1.append(float(pc1[mask].mean()))
        if len(centroid_pc1) >= 2 and centroid_pc1[-1] < centroid_pc1[0]:
            return -1.0
        return 1.0

    def project(self, expr_matrix: np.ndarray) -> np.ndarray:
        return self.pca.transform(np.asarray(expr_matrix, dtype=np.float32))

    def compare_runs(self, baseline_out: dict[str, Any], perturbed_out: dict[str, Any]) -> dict[str, np.ndarray]:
        b = self.project(np.asarray(baseline_out["pred_expr"], dtype=np.float32))
        p = self.project(np.asarray(perturbed_out["pred_expr"], dtype=np.float32))
        disp = p - b
        delta_dist = np.linalg.norm(disp, axis=1).astype(np.float32)
        return {
            "pca_displacement": delta_dist,
            "delta_path_progress": (self.progress_sign * disp[:, 0]).astype(np.float32),
            "delta_target_distance": delta_dist,
        }


class FateClassifierEvaluator:
    """Classifier-based run comparison helper."""

    def __init__(
        self,
        processor: CellDataProcessor,
        terminal_cluster_label: str = "15",
        path_cluster_sequence: list[str] | tuple[str, ...] | None = None,
    ) -> None:
        if processor.classifier is None:
            raise ValueError("Classifier is required for FateClassifierEvaluator.")
        self.processor = processor
        self.class_names = tuple(str(x).replace(",", ".") for x in processor.cfg.class_names)
        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}
        self.terminal_cluster_label = str(terminal_cluster_label).replace(",", ".")
        self.terminal_cluster_idx = self._resolve_class_index(self.terminal_cluster_label)
        if path_cluster_sequence is None:
            path_cluster_sequence = get_pv_path_nodes()
        self.path_cluster_sequence = [str(x).replace(",", ".") for x in path_cluster_sequence]
        self.path_index_map = {cl: i for i, cl in enumerate(self.path_cluster_sequence)}

    def _resolve_class_index(self, label: str) -> int | None:
        norm = str(label).replace(",", ".")
        if norm in self.class_to_idx:
            return int(self.class_to_idx[norm])
        try:
            idx = int(norm)
        except ValueError:
            return None
        if 0 <= idx < len(self.class_names):
            return idx
        return None

    def evaluate_expr_matrix(self, expr_matrix: np.ndarray) -> dict[str, np.ndarray]:
        return self.processor.run_classifier_expr_matrix(expr_matrix)

    def compare_runs(self, baseline_out: dict[str, Any], perturbed_out: dict[str, Any]) -> dict[str, np.ndarray]:
        b = self.evaluate_expr_matrix(np.asarray(baseline_out["pred_expr"], dtype=np.float32))
        p = self.evaluate_expr_matrix(np.asarray(perturbed_out["pred_expr"], dtype=np.float32))
        b_logits = np.asarray(b["logits"], dtype=np.float32)
        p_logits = np.asarray(p["logits"], dtype=np.float32)
        b_probs = np.asarray(b["probs"], dtype=np.float32)
        p_probs = np.asarray(p["probs"], dtype=np.float32)
        if self.terminal_cluster_idx is None:
            terminal_delta = np.full(p_logits.shape[0], np.nan, dtype=np.float32)
        else:
            terminal_delta = (
                p_logits[:, self.terminal_cluster_idx] - b_logits[:, self.terminal_cluster_idx]
            ).astype(np.float32)

        path_weights = np.zeros(p_probs.shape[1], dtype=np.float32)
        for cl, path_i in self.path_index_map.items():
            cls_idx = self._resolve_class_index(cl)
            if cls_idx is not None and 0 <= cls_idx < len(path_weights):
                path_weights[cls_idx] = float(path_i)
        return {
            "delta_logit_cluster15": terminal_delta,
            "delta_path_index_expectation": (
                (p_probs * path_weights).sum(axis=1)
                - (b_probs * path_weights).sum(axis=1)
            ).astype(np.float32),
        }


class ReprogrammingPipeline:
    """Combine generator perturbation, geometry, and optional classifier readouts."""

    def __init__(
        self,
        processor: CellDataProcessor,
        perturb_runner: PerturbationRunner,
        projector: PathSpecificReferenceProjector,
        classifier_evaluator: FateClassifierEvaluator | None = None,
    ) -> None:
        self.processor = processor
        self.runner = perturb_runner
        self.projector = projector
        self.classifier_evaluator = classifier_evaluator

    def compare_generator_experiment(
        self,
        sampled_cells: list[dict[str, Any]],
        perturbation_spec: dict[str, str],
        run_classifier: bool = True,
    ) -> dict[str, Any]:
        baseline = self.runner.run_generator(sampled_cells, perturbation_spec=None)
        perturbed = self.runner.run_generator(sampled_cells, perturbation_spec=perturbation_spec)
        result = {
            "baseline": baseline,
            "perturbed": perturbed,
            "expression_comparison": self.runner.compare_generator_outputs(
                baseline, perturbed
            ),
            "geometry_comparison": self.projector.compare_runs(baseline, perturbed),
        }
        if run_classifier and self.classifier_evaluator is not None:
            result["classifier_comparison"] = self.classifier_evaluator.compare_runs(
                baseline, perturbed
            )
        return result


class RolloutPersistenceManager:
    """Compute step-wise persistence summaries from rollout outputs."""

    def __init__(
        self,
        projector: PathSpecificReferenceProjector,
        classifier_evaluator: FateClassifierEvaluator | None = None,
    ) -> None:
        self.projector = projector
        self.classifier_evaluator = classifier_evaluator

    def extract_per_cell_long_df(
        self,
        rollout_out: dict[str, Any],
        label: str,
        repeat_seed: int | None = None,
    ) -> pd.DataFrame:
        rows = []
        for b_step, p_step in zip(
            rollout_out.get("baseline_steps", []),
            rollout_out.get("perturbed_steps", []),
        ):
            geom = self.projector.compare_runs(b_step, p_step)
            cls = (
                self.classifier_evaluator.compare_runs(b_step, p_step)
                if self.classifier_evaluator is not None
                else {}
            )
            n = len(geom["delta_target_distance"])
            for i in range(n):
                row: dict[str, Any] = {
                    "label": label,
                    "step": int(b_step["step"]),
                    "cell_row": int(i),
                    "delta_target_distance": float(geom["delta_target_distance"][i]),
                    "delta_path_progress": float(geom["delta_path_progress"][i]),
                }
                if "delta_logit_cluster15" in cls:
                    row["delta_logit_cluster15"] = float(cls["delta_logit_cluster15"][i])
                if "delta_path_index_expectation" in cls:
                    row["delta_path_index_expectation"] = float(
                        cls["delta_path_index_expectation"][i]
                    )
                if repeat_seed is not None:
                    row["repeat_seed"] = int(repeat_seed)
                rows.append(row)
        return pd.DataFrame(rows)

    def summarize_persistence(self, long_df: pd.DataFrame) -> pd.DataFrame:
        if long_df.empty:
            return pd.DataFrame()
        metric_cols = [
            c
            for c in [
                "delta_target_distance",
                "delta_path_progress",
                "delta_logit_cluster15",
                "delta_path_index_expectation",
            ]
            if c in long_df.columns
        ]
        grouped = long_df.groupby(["label", "step"], as_index=False)[metric_cols].mean()
        return grouped
