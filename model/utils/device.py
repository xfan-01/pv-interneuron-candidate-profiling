"""Device selection helpers."""

from __future__ import annotations

import torch


def get_device(index: int = 0) -> torch.device:
    """Return CUDA, Apple MPS, or CPU depending on local availability."""
    if torch.cuda.device_count() >= index + 1:
        return torch.device(f"cuda:{index}")

    try:
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return torch.device("mps")
    except AttributeError:
        pass

    return torch.device("cpu")

