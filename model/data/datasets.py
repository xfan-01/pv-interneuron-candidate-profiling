"""Dataset wrappers used by classifier and generator training."""

from __future__ import annotations

from collections.abc import Mapping

from torch.utils.data import Dataset


class DictTensorDataset(Dataset):
    """A generic dataset wrapper for dictionaries of aligned tensors."""

    def __init__(self, data: Mapping[str, object]):
        if not data:
            raise ValueError("DictTensorDataset requires at least one tensor.")

        self.keys = list(data.keys())
        self.data = dict(data)
        self.length = len(self.data[self.keys[0]])

        for key in self.keys:
            current_length = len(self.data[key])
            if current_length != self.length:
                raise ValueError(
                    f"Tensor size mismatch: {key!r} has length "
                    f"{current_length}, expected {self.length}."
                )

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, object]:
        return {key: self.data[key][index] for key in self.keys}

