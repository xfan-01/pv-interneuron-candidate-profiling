"""Checkpoint and model-inspection utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def _remap_legacy_state_dict_keys(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Remap known legacy parameter prefixes to current module names.

    Current compatibility rules:
    - ``blks.`` -> ``blocks.`` (legacy Classifier checkpoints)
    """
    remapped: dict[str, Any] = {}
    for k, v in state_dict.items():
        if k.startswith("blks."):
            remapped["blocks." + k[len("blks."):]] = v
        else:
            remapped[k] = v
    return remapped


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count model parameters."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    metrics: dict[str, Any],
    path: str | Path,
    **extra: Any,
) -> None:
    """Save model state, optional optimizer state, metrics, and metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    state: dict[str, Any] = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "metrics": metrics,
        **extra,
    }
    if optimizer is not None:
        state["optimizer_state_dict"] = optimizer.state_dict()

    torch.save(state, path)


def load_checkpoint(
    model: nn.Module,
    path: str | Path,
    optimizer: torch.optim.Optimizer | None = None,
    device: str | torch.device = "cpu",
) -> tuple[nn.Module, dict[str, Any]]:
    """Load model weights and optionally restore an optimizer."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No checkpoint found at {path}")

    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint["model_state_dict"]
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        # Retry with legacy key remapping for backward compatibility.
        remapped = _remap_legacy_state_dict_keys(state_dict)
        model.load_state_dict(remapped)

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return model, checkpoint
