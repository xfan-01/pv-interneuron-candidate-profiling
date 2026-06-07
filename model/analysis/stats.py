"""Statistical helper utilities for perturbation analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd


def bootstrap_metric(
    values: np.ndarray | list[float],
    stat: str = "mean",
    n_boot: int = 200,
    ci: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    """Bootstrap a 1D metric and return point estimate + CI."""
    rng = np.random.default_rng(seed)
    x = np.asarray(values, dtype=np.float32)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return {
            "n": 0.0,
            "point_estimate": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "boot_mean": float("nan"),
            "boot_std": float("nan"),
        }

    def _calc(arr: np.ndarray) -> float:
        if stat == "mean":
            return float(np.mean(arr))
        if stat == "median":
            return float(np.median(arr))
        if stat == "positive_fraction":
            return float(np.mean(arr > 0))
        raise ValueError(f"Unsupported stat: {stat}")

    point = _calc(x)
    n = len(x)
    boot = np.empty(n_boot, dtype=np.float32)
    for i in range(n_boot):
        sample = rng.choice(x, size=n, replace=True)
        boot[i] = _calc(sample)
    alpha = 1.0 - ci
    low = float(np.quantile(boot, alpha / 2))
    high = float(np.quantile(boot, 1 - alpha / 2))
    return {
        "n": float(n),
        "point_estimate": float(point),
        "ci_low": low,
        "ci_high": high,
        "boot_mean": float(np.mean(boot)),
        "boot_std": float(np.std(boot)),
    }


def bootstrap_metric_panel(
    values: np.ndarray | list[float],
    metric_name: str,
    n_boot: int = 200,
    ci: float = 0.95,
    seed: int = 42,
) -> pd.DataFrame:
    """Return bootstrap summaries for mean/median/positive_fraction."""
    rows = []
    for stat in ("mean", "median", "positive_fraction"):
        out = bootstrap_metric(values, stat=stat, n_boot=n_boot, ci=ci, seed=seed)
        rows.append({"metric": metric_name, "stat": stat, **out})
    return pd.DataFrame(rows)


def bh_fdr(p_values: np.ndarray | list[float]) -> np.ndarray:
    """Benjamini-Hochberg FDR adjustment."""
    p_values = np.asarray(p_values, dtype=np.float64)
    q_values = np.full(len(p_values), np.nan, dtype=np.float64)
    valid = np.isfinite(p_values)
    if not np.any(valid):
        return q_values
    p = p_values[valid]
    order = np.argsort(p)
    ranked = p[order]
    n = float(len(ranked))
    adjusted = ranked * n / (np.arange(1, len(ranked) + 1, dtype=np.float64))
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    inv = np.empty_like(adjusted)
    inv[order] = adjusted
    q_values[np.flatnonzero(valid)] = inv
    return q_values

