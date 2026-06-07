"""Source-dependence benchmark helpers from robustness notebook cells 7-10.

Notebook source:
  - demo/benchmark_source_ranking_robustness.ipynb
    (Source-dependence evaluation helpers + run/load benchmark section)
"""

from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd
from model.utils.constants import get_time_column


def _is_torch_tensor(x: Any) -> bool:
    return hasattr(x, "detach") and hasattr(x, "cpu") and hasattr(x, "clone")


def _clone_item(item: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in item.items():
        out[k] = v.clone() if _is_torch_tensor(v) else deepcopy(v)
    return out


def _copy_source_from_donor(
    item: dict[str, Any],
    donor: dict[str, Any],
    keep_source_time: bool = True,
) -> dict[str, Any]:
    out = _clone_item(item)
    for key in ("gene_id", "gene_val", "padding_mask", "full_input_val"):
        dv = donor[key]
        out[key] = dv.clone() if _is_torch_tensor(dv) else deepcopy(dv)
    if keep_source_time:
        tv = item["time"]
        out["time"] = tv.clone() if _is_torch_tensor(tv) else deepcopy(tv)
    return out


def make_zero_value_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep token identities/time but zero-out source expression magnitudes."""
    out: list[dict[str, Any]] = []
    for item in items:
        x = _clone_item(item)
        if _is_torch_tensor(x["gene_val"]):
            import torch

            x["gene_val"] = torch.zeros_like(x["gene_val"])
            x["full_input_val"] = torch.zeros_like(x["full_input_val"])
        else:
            x["gene_val"] = np.zeros_like(np.asarray(x["gene_val"]))
            x["full_input_val"] = np.zeros_like(np.asarray(x["full_input_val"]))
        out.append(x)
    return out


def ensure_time_bin_series_for_source_benchmark(
    adata: Any,
    n_bins: int = 120,
    time_col: str | None = None,
) -> pd.Series:
    if time_col is None:
        time_col = get_time_column()

    obs = adata.obs
    if "time_bin" in obs.columns:
        return obs["time_bin"].astype(int).reset_index(drop=True)
    if time_col in obs.columns:
        return pd.qcut(
            obs[time_col], q=n_bins, labels=False, duplicates="drop"
        ).astype(int).reset_index(drop=True)
    if "norm_time" in obs.columns:
        return pd.qcut(
            obs["norm_time"], q=n_bins, labels=False, duplicates="drop"
        ).astype(int).reset_index(drop=True)
    raise ValueError("adata.obs must contain one of: time_bin, refined_pseudotime, norm_time")


def target_bins_for_items(
    items: list[dict[str, Any]],
    adata: Any,
    n_bins: int = 120,
    time_col: str | None = None,
) -> np.ndarray:
    time_bins = ensure_time_bin_series_for_source_benchmark(adata, n_bins=n_bins, time_col=time_col)
    target_idx = np.array([int(x["target_idx"]) for x in items], dtype=int)
    return time_bins.iloc[target_idx].to_numpy(dtype=int)


def target_bins_from_meta(
    meta: dict[str, list[Any]],
    adata: Any,
    n_bins: int = 120,
    time_col: str | None = None,
) -> np.ndarray:
    time_bins = ensure_time_bin_series_for_source_benchmark(adata, n_bins=n_bins, time_col=time_col)
    target_idx = np.array(meta["target_idx"], dtype=int)
    return time_bins.iloc[target_idx].to_numpy(dtype=int)


def make_global_source_shuffle_items(
    items: list[dict[str, Any]],
    seed: int = 0,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    donors = rng.permutation(len(items))
    return [
        _copy_source_from_donor(item, items[int(donors[i])], keep_source_time=True)
        for i, item in enumerate(items)
    ]


def make_within_target_bin_shuffle_items(
    items: list[dict[str, Any]],
    adata: Any,
    seed: int = 0,
    n_bins: int = 120,
    time_col: str | None = None,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    bins = target_bins_for_items(items, adata=adata, n_bins=n_bins, time_col=time_col)
    out = [_clone_item(x) for x in items]
    for b in np.unique(bins):
        idx = np.where(bins == b)[0]
        if len(idx) <= 1:
            continue
        donors = idx.copy()
        rng.shuffle(donors)
        if len(donors) > 1 and np.any(donors == idx):
            donors = np.roll(donors, 1)
        for row_i, donor_i in zip(idx, donors):
            out[int(row_i)] = _copy_source_from_donor(
                items[int(row_i)], items[int(donor_i)], keep_source_time=True
            )
    return out


def _amp_autocast_ctx(device: Any) -> Any:
    try:
        import torch
    except Exception:
        return nullcontext()
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        return torch.amp.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def _make_loader_from_items(items: list[dict[str, Any]], batch_size: int = 128) -> Any:
    from torch.utils.data import DataLoader

    from model.data.trajectory_pairs import TrajectoryDataset

    return DataLoader(TrajectoryDataset(items), batch_size=batch_size, shuffle=False)


def predict_items(
    model: Any,
    items: list[dict[str, Any]],
    device: Any = "cpu",
    batch_size: int = 128,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, list[Any]]]:
    """Run generator prediction on pre-built item list."""
    import torch

    preds, trues, input_sources = [], [], []
    meta: dict[str, list[Any]] = {"c_idx": [], "target_idx": [], "time": [], "target_time": []}
    loader = _make_loader_from_items(items, batch_size=batch_size)
    model.eval()
    with torch.no_grad():
        for batch in loader:
            g_id = batch["gene_id"].to(device, non_blocking=True)
            g_val = batch["gene_val"].to(device, non_blocking=True)
            source_time = batch["time"].to(device, non_blocking=True)
            target_time = batch["target_time"].to(device, non_blocking=True)
            padding_mask = batch["padding_mask"].to(device, non_blocking=True)
            with _amp_autocast_ctx(device):
                pred, _, _ = model(
                    g_id,
                    g_val,
                    source_time,
                    target_time,
                    padding_mask=padding_mask,
                    need_weights=False,
                )
                pred = torch.clamp(pred, min=0.0, max=50.0)
            preds.append(pred.float().cpu().numpy())
            trues.append(batch["target_val"].numpy())
            input_sources.append(batch["full_input_val"].numpy())
            for k in meta:
                meta[k].extend(batch[k].numpy().tolist())
    return np.vstack(preds), np.vstack(trues), np.vstack(input_sources), meta


def safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    from scipy.stats import pearsonr

    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(pearsonr(x, y)[0])


def cosine_rows(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return np.sum(a * b, axis=1) / np.maximum(denom, eps)


def target_values_from_items(items: list[dict[str, Any]]) -> np.ndarray:
    return np.vstack(
        [
            x["target_val"].detach().cpu().numpy()
            if _is_torch_tensor(x["target_val"])
            else np.asarray(x["target_val"])
            for x in items
        ]
    ).astype(np.float32)


def target_bin_mean_matrix(trues: np.ndarray, target_bins: np.ndarray) -> np.ndarray:
    means: dict[int, np.ndarray] = {}
    for b in np.unique(target_bins):
        means[int(b)] = trues[target_bins == b].mean(axis=0)
    return np.vstack([means[int(b)] for b in target_bins])


def target_bin_mean_from_reference(
    ref_trues: np.ndarray,
    ref_target_bins: np.ndarray,
    query_target_bins: np.ndarray,
) -> np.ndarray:
    means: dict[int, np.ndarray] = {}
    global_mean = ref_trues.mean(axis=0)
    for b in np.unique(ref_target_bins):
        means[int(b)] = ref_trues[ref_target_bins == b].mean(axis=0)
    return np.vstack([means.get(int(b), global_mean) for b in query_target_bins])


def prediction_change_metrics(
    full_preds: np.ndarray | None,
    altered_preds: np.ndarray | None,
    original_sources: np.ndarray | None,
    altered_sources: np.ndarray | None,
) -> dict[str, float]:
    if full_preds is None or altered_preds is None or altered_sources is None or original_sources is None:
        return {
            "pred_change_l2_mean": float("nan"),
            "pred_change_l2_median": float("nan"),
            "source_change_l2_mean": float("nan"),
            "sensitivity_ratio": float("nan"),
        }
    pred_change = np.linalg.norm(np.asarray(altered_preds) - np.asarray(full_preds), axis=1)
    source_change = np.linalg.norm(np.asarray(altered_sources) - np.asarray(original_sources), axis=1)
    return {
        "pred_change_l2_mean": float(pred_change.mean()),
        "pred_change_l2_median": float(np.median(pred_change)),
        "source_change_l2_mean": float(source_change.mean()),
        "sensitivity_ratio": float(pred_change.mean() / (source_change.mean() + 1e-12)),
    }


def evaluate_prediction_table(
    preds: np.ndarray,
    trues: np.ndarray,
    original_sources: np.ndarray,
    meta: dict[str, list[Any]],
    adata: Any,
    condition: str,
    model_seed: int | None = None,
    shuffle_seed: int | None = None,
    train_bin_mean: np.ndarray | None = None,
    oracle_bin_mean: np.ndarray | None = None,
    full_preds: np.ndarray | None = None,
    altered_sources: np.ndarray | None = None,
    n_bins: int = 120,
    time_col: str | None = None,
) -> dict[str, Any]:
    from sklearn.metrics import mean_squared_error

    target_bins = target_bins_from_meta(meta, adata=adata, n_bins=n_bins, time_col=time_col)
    if oracle_bin_mean is None:
        oracle_bin_mean = target_bin_mean_matrix(trues, target_bins)

    raw_mse = float(mean_squared_error(trues, preds))
    oracle_bin_mse = float(mean_squared_error(trues, oracle_bin_mean))
    if train_bin_mean is not None:
        train_bin_mse = float(mean_squared_error(trues, train_bin_mean))
        pseudo_r2_train = float(1.0 - raw_mse / (train_bin_mse + 1e-12))
        relative_mse_train = float(raw_mse / (train_bin_mse + 1e-12))
        excess_mse_train = float(raw_mse - train_bin_mse)
        resid_corr_train = safe_pearson(preds - train_bin_mean, trues - train_bin_mean)
    else:
        train_bin_mse = pseudo_r2_train = relative_mse_train = excess_mse_train = resid_corr_train = float("nan")

    nz = trues > 0
    true_delta = trues - original_sources
    pred_delta = preds - original_sources
    cos = cosine_rows(pred_delta, true_delta)
    row: dict[str, Any] = {
        "condition": condition,
        "model_seed": model_seed,
        "shuffle_seed": shuffle_seed,
        "n_pairs": int(trues.shape[0]),
        "raw_mse": raw_mse,
        "raw_pearson_nonzero": safe_pearson(preds[nz], trues[nz]),
        "heldout_oracle_bin_mse": oracle_bin_mse,
        "train_estimated_bin_mse": train_bin_mse,
        "excess_mse_vs_heldout_oracle_bin_mean": float(raw_mse - oracle_bin_mse),
        "relative_mse_vs_heldout_oracle_bin_mean": float(raw_mse / (oracle_bin_mse + 1e-12)),
        "pseudo_r2_vs_heldout_oracle_bin_mean": float(1.0 - raw_mse / (oracle_bin_mse + 1e-12)),
        "excess_mse_vs_train_bin_mean": excess_mse_train,
        "relative_mse_vs_train_bin_mean": relative_mse_train,
        "pseudo_r2_vs_train_bin_mean": pseudo_r2_train,
        "residual_pearson_after_heldout_oracle_bin_mean": safe_pearson(
            preds - oracle_bin_mean, trues - oracle_bin_mean
        ),
        "residual_pearson_after_train_bin_mean": resid_corr_train,
        "transition_direction_cosine_mean": float(np.nanmean(cos)),
        "transition_direction_cosine_median": float(np.nanmedian(cos)),
    }
    row.update(
        prediction_change_metrics(
            full_preds=full_preds,
            altered_preds=preds,
            original_sources=original_sources,
            altered_sources=altered_sources,
        )
    )
    return row
