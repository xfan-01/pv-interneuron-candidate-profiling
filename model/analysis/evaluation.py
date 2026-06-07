"""Evaluation metrics for classifier and generator models.

Migrated from the demo analysis notebooks:
- classifier_analysis.ipynb cells 7-8
- classifier_analysis_multi.ipynb cell 8
- generator_analysis_3.ipynb cells 5-7
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from model.utils.constants import get_cluster_column


def compute_accuracy(
    logits: torch.Tensor, labels: torch.Tensor
) -> float:
    """Top-1 accuracy from logits."""
    preds = torch.argmax(logits, dim=1)
    return (preds == labels).float().mean().item()


def compute_confusion_matrix(
    logits: torch.Tensor, labels: torch.Tensor, num_classes: int
) -> np.ndarray:
    """Confusion matrix as numpy array (rows=true, cols=pred)."""
    preds = torch.argmax(logits, dim=1)
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for t, p in zip(labels.cpu(), preds.cpu()):
        cm[int(t), int(p)] += 1
    return cm.numpy()


def per_class_accuracy(
    logits: torch.Tensor, labels: torch.Tensor, num_classes: int
) -> dict[int, float]:
    """Per-class accuracy."""
    preds = torch.argmax(logits, dim=1)
    result: dict[int, float] = {}
    labels_np = labels.cpu().numpy()
    preds_np = preds.cpu().numpy()
    for c in range(num_classes):
        mask = labels_np == c
        if mask.sum() > 0:
            result[c] = (preds_np[mask] == c).mean().item()
        else:
            result[c] = float("nan")
    return result


@torch.no_grad()
def evaluate_classifier(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    num_classes: int = 2,
) -> dict[str, Any]:
    """Run a full classifier evaluation pass.

    Returns
    -------
    dict with keys: accuracy, confusion_matrix, per_class_accuracy,
                    all_logits, all_labels, all_preds
    """
    model.eval()
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    for batch in loader:
        ids = batch["gene_id"].to(device)
        vals = batch["gene_val"].to(device)
        pad_mask = batch["pad_mask"].to(device)
        labels = batch["label"].to(device)

        logits = model(ids, vals, pad_mask)
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())

    logits_cat = torch.cat(all_logits, dim=0)
    labels_cat = torch.cat(all_labels, dim=0)
    preds = torch.argmax(logits_cat, dim=1)

    return {
        "accuracy": (preds == labels_cat).float().mean().item(),
        "confusion_matrix": compute_confusion_matrix(
            logits_cat, labels_cat, num_classes
        ),
        "per_class_accuracy": per_class_accuracy(
            logits_cat, labels_cat, num_classes
        ),
        "all_logits": logits_cat,
        "all_labels": labels_cat,
        "all_preds": preds,
    }


@torch.no_grad()
def evaluate_classifier_binary(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Binary held-out evaluation with probability outputs.

    Notebook mapping:
      - classifier_analysis.ipynb cell 7 (`evaluate_classifier_on_testset`)
    """
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    model.eval()
    probs: list[float] = []
    preds: list[int] = []
    y_true: list[int] = []

    for batch in loader:
        ids = batch["gene_id"].to(device)
        vals = batch["gene_val"].to(device)
        # Support both pad_mask and legacy valid_mask.
        if "pad_mask" in batch:
            pad_mask = batch["pad_mask"].to(device)
        elif "valid_mask" in batch:
            pad_mask = ~batch["valid_mask"].bool().to(device)
        else:
            raise ValueError("Batch missing pad_mask/valid_mask.")
        labels = batch["label"].to(device)

        logits = model(ids, vals, pad_mask)
        pv_prob = torch.softmax(logits, dim=-1)[:, 1]
        probs.extend(pv_prob.detach().cpu().numpy().tolist())
        preds.extend((pv_prob >= threshold).long().detach().cpu().numpy().tolist())
        y_true.extend(labels.detach().cpu().numpy().tolist())

    probs_np = np.asarray(probs, dtype=np.float64)
    preds_np = np.asarray(preds, dtype=np.int64)
    y_np = np.asarray(y_true, dtype=np.int64)

    metrics = {
        "n_samples": int(len(y_np)),
        "accuracy": float(accuracy_score(y_np, preds_np)),
        "precision": float(precision_score(y_np, preds_np, zero_division=0)),
        "recall": float(recall_score(y_np, preds_np, zero_division=0)),
        "f1": float(f1_score(y_np, preds_np, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_np, probs_np)) if len(np.unique(y_np)) > 1 else float("nan"),
        "pr_auc": float(average_precision_score(y_np, probs_np)) if len(np.unique(y_np)) > 1 else float("nan"),
    }
    cm = confusion_matrix(y_np, preds_np)
    report = classification_report(
        y_np, preds_np, target_names=["NPV(0)", "PV(1)"], zero_division=0
    )
    return {
        "metrics": metrics,
        "y_true": y_np,
        "y_pred": preds_np,
        "y_prob": probs_np,
        "confusion_matrix": cm,
        "classification_report": report,
    }


def compute_cosine_similarity(
    pred: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Per-sample cosine similarity between predicted and target expression."""
    pred_nz = pred * (target > 0).float()
    target_nz = target * (target > 0).float()
    cos = torch.nn.functional.cosine_similarity(
        pred_nz, target_nz, dim=1, eps=1e-8
    )
    return cos


def compute_expression_correlation(
    pred: torch.Tensor, target: torch.Tensor
) -> np.ndarray:
    """Per-gene Pearson correlation (numpy)."""
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    n_genes = pred_np.shape[1]
    corrs = np.zeros(n_genes, dtype=np.float32)
    for g in range(n_genes):
        p = pred_np[:, g]
        t = target_np[:, g]
        std_p, std_t = np.std(p), np.std(t)
        if std_p > 1e-8 and std_t > 1e-8:
            corrs[g] = np.corrcoef(p, t)[0, 1]
    return corrs


def compute_top_deg_delta(
    preds: "np.ndarray",
    trues: "np.ndarray",
    meta: dict[str, list[Any]],
    adata: Any,
    source_cluster: str,
    target_cluster: str,
    cluster_col: str | None = None,
    top_n: int = 50,
) -> dict[str, Any] | None:
    """Compute top DEG delta between source→target cluster transition.

    For a given cluster transition, filters cell pairs, computes per-cell
    expression delta (target − source), aggregates to cluster-level mean
    delta, identifies top *top_n* genes by absolute true delta, and returns
    a structured result dict plus a ``plot_df`` DataFrame suitable for
    ``plot_deg_delta_scatter``.

    Notebook: ``generator_analysis_3.ipynb`` cell 7
    (``evaluate_top_deg_delta``, computation portion only).

    Returns
    -------
    dict or None
        Keys: ``plot_df`` (DataFrame with columns Gene, True_Delta, Predicted_Delta),
        ``r``, ``p``, ``valid_count``, ``gene_names``, ``source_cluster``,
        ``target_cluster``.
        Returns None if no cell pairs match the transition.
    """
    import scipy.sparse
    from scipy.stats import pearsonr

    if cluster_col is None:
        cluster_col = get_cluster_column()

    c_idxs = np.asarray(meta["c_idx"]).astype(int)
    target_idxs = np.asarray(meta["target_idx"]).astype(int)

    source_clusters = adata.obs[cluster_col].values[c_idxs]
    target_clusters = adata.obs[cluster_col].values[target_idxs]

    pair_mask = (
        (source_clusters == str(source_cluster))
        & (target_clusters == str(target_cluster))
    )
    valid_count = int(pair_mask.sum())
    if valid_count == 0:
        return None

    X_src = adata.X[c_idxs[pair_mask]]
    if scipy.sparse.issparse(X_src):
        X_src = X_src.toarray()

    Y_true = trues[pair_mask]
    Y_pred = preds[pair_mask]

    delta_true_cells = Y_true - X_src
    delta_pred_cells = Y_pred - X_src

    delta_true_mean = delta_true_cells.mean(axis=0)
    delta_pred_mean = delta_pred_cells.mean(axis=0)

    top_deg_indices = np.argsort(np.abs(delta_true_mean))[-top_n:]
    gene_names = [str(adata.var_names[i]) for i in top_deg_indices]

    true_degs_delta = delta_true_mean[top_deg_indices]
    pred_degs_delta = delta_pred_mean[top_deg_indices]

    r, p = pearsonr(true_degs_delta, pred_degs_delta)

    import pandas as pd
    plot_df = pd.DataFrame(
        {
            "Gene": gene_names,
            "True_Delta": true_degs_delta,
            "Predicted_Delta": pred_degs_delta,
        }
    )

    return {
        "plot_df": plot_df,
        "r": float(r),
        "p": float(p),
        "valid_count": valid_count,
        "gene_names": gene_names,
        "source_cluster": str(source_cluster),
        "target_cluster": str(target_cluster),
    }


def compute_transition_deg_delta_panel(
    preds: "np.ndarray",
    trues: "np.ndarray",
    meta: dict[str, list[Any]],
    adata: Any,
    trajectory_nodes: list[str],
    cluster_col: str | None = None,
    top_n: int = 50,
) -> "pd.DataFrame":
    """Run ``compute_top_deg_delta`` across consecutive trajectory node pairs.

    Notebook: ``generator_analysis_3.ipynb`` cell 7 (trajectory loop).

    Returns a combined DataFrame of all transition DEG deltas.
    """
    import pandas as pd

    if cluster_col is None:
        cluster_col = get_cluster_column()

    frames: list[pd.DataFrame] = []
    for i in range(len(trajectory_nodes) - 1):
        src = trajectory_nodes[i]
        tgt = trajectory_nodes[i + 1]
        result = compute_top_deg_delta(
            preds=preds,
            trues=trues,
            meta=meta,
            adata=adata,
            source_cluster=src,
            target_cluster=tgt,
            cluster_col=cluster_col,
            top_n=top_n,
        )
        if result is None:
            continue
        df = result["plot_df"].copy()
        df["source_cluster"] = src
        df["target_cluster"] = tgt
        df["pearson_r"] = result["r"]
        df["pearson_p"] = result["p"]
        df["valid_count"] = result["valid_count"]
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    return combined
