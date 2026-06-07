"""Forecasting evaluation utilities migrated from generator analysis notebooks."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    adjusted_rand_score,
    mean_squared_error,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader


def safe_pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation that returns NaN for degenerate inputs."""
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if x.size <= 1 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(pearsonr(x, y)[0])


def _to_scalar(x: Any) -> Any:
    if hasattr(x, "detach"):
        return x.detach().cpu().item()
    return np.asarray(x).item()


def stack_pair_arrays(data_list: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    """Convert list-backed trajectory-pair items to dense arrays."""
    source_expr = np.stack(
        [item["full_input_val"].detach().cpu().numpy() for item in data_list]
    ).astype(np.float32)
    target_expr = np.stack(
        [item["target_val"].detach().cpu().numpy() for item in data_list]
    ).astype(np.float32)
    source_time = np.asarray([float(_to_scalar(i["time"])) for i in data_list], dtype=np.float32)
    target_time = np.asarray(
        [float(_to_scalar(i["target_time"])) for i in data_list], dtype=np.float32
    )
    return {
        "source_expr": source_expr,
        "target_expr": target_expr,
        "source_time": source_time,
        "target_time": target_time,
        "delta_time": target_time - source_time,
        "source_idx": np.asarray([int(_to_scalar(i["c_idx"])) for i in data_list], dtype=np.int64),
        "target_idx": np.asarray(
            [int(_to_scalar(i["target_idx"])) for i in data_list], dtype=np.int64
        ),
    }


def _ridge_features(pair_arrays: dict[str, np.ndarray]) -> np.ndarray:
    return np.column_stack(
        [
            pair_arrays["source_expr"],
            pair_arrays["source_time"][:, None],
            pair_arrays["target_time"][:, None],
            pair_arrays["delta_time"][:, None],
        ]
    ).astype(np.float32)


def _clone_pair_item(item: dict[str, Any]) -> dict[str, Any]:
    cloned = {}
    for key, value in item.items():
        cloned[key] = value.detach().clone() if hasattr(value, "detach") else copy.deepcopy(value)
    return cloned


def make_source_shuffled_data_list(
    eval_data_list: list[dict[str, Any]],
    adata: Any,
    time_bin_col: str = "time_bin",
    seed: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Shuffle source expression within target time bins while keeping targets fixed."""
    if time_bin_col not in adata.obs.columns:
        raise KeyError(f"{time_bin_col!r} is required in adata.obs.")

    rng = np.random.default_rng(seed)
    target_idx = np.asarray([int(_to_scalar(item["target_idx"])) for item in eval_data_list])
    target_bins = adata.obs[time_bin_col].iloc[target_idx].to_numpy()

    bin_to_rows: dict[Any, list[int]] = {}
    for row, bin_value in enumerate(target_bins):
        key = None if pd.isna(bin_value) else int(bin_value)
        bin_to_rows.setdefault(key, []).append(row)

    shuffled: list[dict[str, Any]] = []
    donor_indices: list[int] = []
    singleton_count = 0
    for row, item in enumerate(eval_data_list):
        key = None if pd.isna(target_bins[row]) else int(target_bins[row])
        candidates = bin_to_rows.get(key, [row])
        if len(candidates) > 1:
            donor_idx = int(rng.choice([i for i in candidates if i != row]))
        else:
            donor_idx = row
            singleton_count += 1

        donor = eval_data_list[donor_idx]
        new_item = _clone_pair_item(item)
        for source_key in ("gene_id", "gene_val", "padding_mask", "full_input_val"):
            new_item[source_key] = donor[source_key].detach().clone()
        shuffled.append(new_item)
        donor_indices.append(donor_idx)

    return shuffled, {
        "n_pairs": len(eval_data_list),
        "n_target_bins": len(bin_to_rows),
        "singleton_count": singleton_count,
        "donor_indices": np.asarray(donor_indices, dtype=np.int64),
    }


@torch.no_grad()
def get_forecaster_predictions(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, list[Any]]]:
    """Run forecasting inference and return predictions, targets, masks, metadata."""
    model.eval()
    preds_list: list[np.ndarray] = []
    trues_list: list[np.ndarray] = []
    meta = {"c_idx": [], "target_idx": [], "time": [], "target_time": [], "source_val": []}

    for batch in dataloader:
        gene_id = batch["gene_id"].to(device)
        gene_val = batch["gene_val"].to(device)
        source_time = batch["time"].to(device)
        target_time = batch["target_time"].to(device)
        padding_mask = batch["padding_mask"].to(device)

        preds, _, _ = model(
            gene_id,
            gene_val,
            source_time,
            target_time,
            padding_mask=padding_mask,
            need_weights=False,
        )

        preds_list.append(preds.detach().cpu().numpy())
        trues_list.append(batch["target_val"].detach().cpu().numpy())
        meta["c_idx"].extend(batch["c_idx"].detach().cpu().numpy().astype(int).tolist())
        meta["target_idx"].extend(batch["target_idx"].detach().cpu().numpy().astype(int).tolist())
        meta["time"].extend(source_time.detach().cpu().numpy().tolist())
        meta["target_time"].extend(target_time.detach().cpu().numpy().tolist())
        meta["source_val"].append(batch["full_input_val"].detach().cpu().numpy())

    preds = np.vstack(preds_list).astype(np.float32)
    trues = np.vstack(trues_list).astype(np.float32)
    nz_masks = trues > 0
    meta["source_val"] = np.vstack(meta["source_val"]).astype(np.float32)
    return preds, trues, nz_masks, meta


def predict_target_bin_mean_baseline(
    train_data_list: list[dict[str, Any]],
    eval_data_list: list[dict[str, Any]],
    adata: Any,
    time_bin_col: str = "time_bin",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Predict held-out targets from training target means in the same target bin."""
    if time_bin_col not in adata.obs.columns:
        raise KeyError(f"{time_bin_col!r} is required in adata.obs.")

    train_arrays = stack_pair_arrays(train_data_list)
    eval_arrays = stack_pair_arrays(eval_data_list)
    global_mean = train_arrays["target_expr"].mean(axis=0)

    train_bins = adata.obs[time_bin_col].iloc[train_arrays["target_idx"]].to_numpy()
    bin_means = {}
    for bin_value in np.unique(train_bins[~pd.isna(train_bins)]):
        mask = train_bins == bin_value
        bin_means[int(bin_value)] = train_arrays["target_expr"][mask].mean(axis=0)

    eval_bins = adata.obs[time_bin_col].iloc[eval_arrays["target_idx"]].to_numpy()
    out = []
    fallback_count = 0
    for bin_value in eval_bins:
        if pd.isna(bin_value) or int(bin_value) not in bin_means:
            out.append(global_mean)
            fallback_count += 1
        else:
            out.append(bin_means[int(bin_value)])

    return np.vstack(out).astype(np.float32), {
        "n_bins": len(bin_means),
        "fallback_count": fallback_count,
    }


def fit_predict_ridge_baseline(
    train_data_list: list[dict[str, Any]],
    val_data_list: list[dict[str, Any]],
    eval_data_list: list[dict[str, Any]],
    alphas: tuple[float, ...] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit deterministic multi-output ridge on source expression and time features."""
    if alphas is None:
        alphas = (0.1, 1.0, 10.0, 100.0)

    train_arrays = stack_pair_arrays(train_data_list)
    val_arrays = stack_pair_arrays(val_data_list) if val_data_list else None
    eval_arrays = stack_pair_arrays(eval_data_list)

    x_train = _ridge_features(train_arrays)
    y_train = train_arrays["target_expr"]
    x_eval = _ridge_features(eval_arrays)

    best_alpha = float(alphas[0])
    best_mse = float("nan")
    if val_arrays is not None and len(val_data_list) > 0:
        x_val = _ridge_features(val_arrays)
        y_val = val_arrays["target_expr"]
        best_mse = float("inf")
        for alpha in alphas:
            candidate = make_pipeline(
                StandardScaler(with_mean=True, with_std=True),
                Ridge(alpha=float(alpha), fit_intercept=True),
            )
            candidate.fit(x_train, y_train)
            val_mse = mean_squared_error(y_val, candidate.predict(x_val))
            if val_mse < best_mse:
                best_mse = float(val_mse)
                best_alpha = float(alpha)

    fit_data = train_data_list + val_data_list
    fit_arrays = stack_pair_arrays(fit_data)
    final_model = make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
        Ridge(alpha=best_alpha, fit_intercept=True),
    )
    final_model.fit(_ridge_features(fit_arrays), fit_arrays["target_expr"])
    return final_model.predict(x_eval).astype(np.float32), {
        "alpha": best_alpha,
        "validation_mse": best_mse,
        "n_train": len(train_data_list),
        "n_val": len(val_data_list),
    }


def compute_forecasting_accuracy(
    preds: np.ndarray,
    trues: np.ndarray,
    nz_masks: np.ndarray,
    target_indices: np.ndarray,
    adata: Any,
) -> dict[str, Any]:
    """Global MSE, non-zero Pearson, and target-bin metacell Pearson."""
    mse = float(mean_squared_error(trues, preds))
    global_corr = safe_pearsonr(preds[nz_masks], trues[nz_masks])

    if "time_bin" not in adata.obs.columns:
        return {"mse": mse, "global_corr": global_corr, "meta_corr": float("nan"), "n_meta_bins": 0}

    pair_bins = adata.obs["time_bin"].iloc[target_indices].to_numpy()
    valid_mask = ~pd.isna(pair_bins)
    valid_bins = pair_bins[valid_mask].astype(int)
    meta_pred, meta_true = [], []
    for bin_value in np.sort(np.unique(valid_bins)):
        mask = valid_bins == bin_value
        meta_pred.append(preds[valid_mask][mask].mean(axis=0))
        meta_true.append(trues[valid_mask][mask].mean(axis=0))

    if len(meta_true) >= 2:
        meta_pred_arr = np.vstack(meta_pred)
        meta_true_arr = np.vstack(meta_true)
        meta_nz = meta_true_arr > 0
        meta_corr = safe_pearsonr(meta_pred_arr[meta_nz], meta_true_arr[meta_nz])
    else:
        meta_corr = float("nan")

    return {
        "mse": mse,
        "global_corr": global_corr,
        "meta_corr": meta_corr,
        "n_meta_bins": len(meta_true),
    }


def normalize_cluster_label_for_eval(x: Any) -> str:
    return str(x).strip().replace(",", ".")


def compute_transition_dynamic_gene_evaluation(
    preds: np.ndarray,
    trues: np.ndarray,
    source_expr: np.ndarray,
    meta: dict[str, Any],
    adata: Any,
    baseline_preds: dict[str, np.ndarray] | None = None,
    cluster_key: str = "leiden_0.2_c6",
    top_n: int = 50,
    min_pairs: int = 3,
) -> pd.DataFrame:
    """Evaluate transition-level dynamic gene programme fidelity."""
    source_idx = np.asarray(meta["c_idx"]).astype(int)
    target_idx = np.asarray(meta["target_idx"]).astype(int)
    source_clusters = np.asarray(
        [normalize_cluster_label_for_eval(x) for x in adata.obs[cluster_key].values[source_idx]]
    )
    target_clusters = np.asarray(
        [normalize_cluster_label_for_eval(x) for x in adata.obs[cluster_key].values[target_idx]]
    )
    transition_labels = np.asarray([f"{s}->{t}" for s, t in zip(source_clusters, target_clusters)])

    pred_map = {"Transformer": preds}
    if baseline_preds:
        pred_map.update(baseline_preds)

    rows = []
    for transition in sorted(np.unique(transition_labels), key=str):
        source_label, target_label = transition.split("->", 1)
        if source_label == target_label:
            continue
        mask = transition_labels == transition
        n_pairs = int(mask.sum())
        if n_pairs < min_pairs:
            continue

        obs_delta = trues[mask] - source_expr[mask]
        mean_abs_obs_delta = np.abs(obs_delta.mean(axis=0))
        if np.all(mean_abs_obs_delta == 0):
            continue

        n_genes = min(top_n, obs_delta.shape[1])
        top_idx = np.argsort(mean_abs_obs_delta)[::-1][:n_genes]
        obs_program = obs_delta[:, top_idx].mean(axis=0)

        for model_name, model_pred in pred_map.items():
            model_delta = np.asarray(model_pred)[mask] - source_expr[mask]
            model_program = model_delta[:, top_idx].mean(axis=0)
            rows.append(
                {
                    "Transition": transition,
                    "Model": model_name,
                    "N pairs": n_pairs,
                    "Top dynamic genes": n_genes,
                    "Dynamic Pearson": safe_pearsonr(model_program, obs_program),
                }
            )

    return pd.DataFrame(rows)


def compute_target_bin_residual_diagnostic(
    preds: np.ndarray,
    trues: np.ndarray,
    target_bin_mean_preds: np.ndarray,
    baseline_preds: dict[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    """Measure cell-specific residuals after removing target-bin means."""
    true_resid = trues - target_bin_mean_preds
    pred_map = {"Transformer": preds}
    if baseline_preds:
        pred_map.update(baseline_preds)

    rows = []
    for model_name, model_pred in pred_map.items():
        pred_resid = np.asarray(model_pred) - target_bin_mean_preds
        rows.append(
            {
                "Model": model_name,
                "Residual MSE": float(mean_squared_error(true_resid, pred_resid)),
                "Residual Pearson": safe_pearsonr(pred_resid, true_resid),
            }
        )
    return pd.DataFrame(rows)


def compute_source_shuffle_diagnostic(
    preds: np.ndarray,
    source_shuffle_preds: np.ndarray,
    trues: np.ndarray,
    target_bin_mean_preds: np.ndarray,
) -> pd.DataFrame:
    """Compare original and source-shuffled Transformer residual predictions."""
    true_resid = trues - target_bin_mean_preds
    original_resid = preds - target_bin_mean_preds
    shuffled_resid = source_shuffle_preds - target_bin_mean_preds
    original_resid_pearson = safe_pearsonr(original_resid, true_resid)
    shuffled_resid_pearson = safe_pearsonr(shuffled_resid, true_resid)
    return pd.DataFrame(
        [
            {
                "Original residual Pearson": original_resid_pearson,
                "Shuffled residual Pearson": shuffled_resid_pearson,
                "Residual Pearson drop": original_resid_pearson - shuffled_resid_pearson,
                "Original residual MSE": float(mean_squared_error(true_resid, original_resid)),
                "Shuffled residual MSE": float(mean_squared_error(true_resid, shuffled_resid)),
            }
        ]
    )


def compute_forecasting_manifold_metrics(
    preds: np.ndarray,
    target_indices: np.ndarray,
    adata: Any,
    cluster_key: str,
    seed: int = 42,
    n_pca: int = 50,
    batch_key: str | None = None,
) -> dict[str, Any]:
    """PCA/KMeans geometry metrics: silhouette, ARI, NMI, and optional batch ASW.

    Notebook: ``generator_analysis_3.ipynb`` cell 6 (``run_full_sanity_check`` sections 5-6).
    """
    clusters = adata.obs[cluster_key].values[target_indices].astype(str)
    unique_clusters = np.unique(clusters)
    n_components = min(n_pca, preds.shape[0] - 1, preds.shape[1])
    base = {
        "silhouette_raw": float("nan"),
        "silhouette_scaled": float("nan"),
        "ari": float("nan"),
        "nmi": float("nan"),
        "avg_bio": float("nan"),
        "n_clusters": int(len(unique_clusters)),
        "batch_asw": float("nan"),
    }
    if n_components < 2 or len(unique_clusters) < 2:
        return base

    preds_pca = PCA(n_components=n_components, random_state=seed).fit_transform(preds)
    silhouette_raw = float(silhouette_score(preds_pca, clusters))
    silhouette_scaled = float((silhouette_raw + 1.0) / 2.0)
    kmeans = KMeans(n_clusters=len(unique_clusters), random_state=seed, n_init=10)
    pred_clusters = kmeans.fit_predict(preds_pca)
    ari = float(adjusted_rand_score(clusters, pred_clusters))
    nmi = float(normalized_mutual_info_score(clusters, pred_clusters))

    # Batch mixing (original demo section 6)
    batch_asw = float("nan")
    if batch_key is not None and batch_key in adata.obs.columns:
        batches = adata.obs[batch_key].values[target_indices].astype(str)
        if len(np.unique(batches)) > 1:
            sil_batch_raw = silhouette_samples(preds_pca, batches)
            batch_scores = 1.0 - np.abs(sil_batch_raw)
            ct_batch_scores = []
            for c in unique_clusters:
                c_mask = clusters == c
                if np.sum(c_mask) > 0:
                    ct_batch_scores.append(float(np.mean(batch_scores[c_mask])))
            if ct_batch_scores:
                batch_asw = float(np.mean(ct_batch_scores))

    return {
        "silhouette_raw": silhouette_raw,
        "silhouette_scaled": silhouette_scaled,
        "ari": ari,
        "nmi": nmi,
        "avg_bio": float(np.mean([silhouette_scaled, ari, nmi])),
        "n_clusters": int(len(unique_clusters)),
        "batch_asw": batch_asw,
    }


def compute_umap_projection(
    preds: np.ndarray,
    trues: np.ndarray,
    seed: int = 42,
    max_points: int | None = 2000,
    save_path: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit UMAP on true targets and transform predictions into the same space."""
    import pickle
    import umap

    n = trues.shape[0] if max_points is None else min(trues.shape[0], max_points)
    rng = np.random.default_rng(seed)
    idx = np.arange(trues.shape[0]) if n == trues.shape[0] else rng.choice(trues.shape[0], n, replace=False)
    reducer = umap.UMAP(n_components=2, random_state=seed)
    umap_true = reducer.fit_transform(trues[idx])
    umap_pred = reducer.transform(preds[idx])
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with save_path.open("wb") as f:
            pickle.dump(reducer, f)
    return umap_true.astype(np.float32), umap_pred.astype(np.float32)
