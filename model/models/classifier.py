"""Fate classifier model migrated from the demo notebooks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from .layers import ClassifierEncoderBlock


@dataclass(frozen=True)
class ClassifierConfig:
    """Configuration for the Transformer fate classifier."""

    vocab_size: int
    d_model: int = 128
    nhead: int = 4
    dim_feedforward: int = 256
    dropout: float = 0.1
    num_layers: int = 2
    num_classes: int = 2

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be a positive integer.")
        if self.d_model <= 0:
            raise ValueError("d_model must be positive.")
        if self.nhead <= 0:
            raise ValueError("nhead must be positive.")
        if self.d_model % self.nhead != 0:
            raise ValueError("d_model must be divisible by nhead.")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive.")
        if self.num_classes <= 0:
            raise ValueError("num_classes must be positive.")

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "ClassifierConfig":
        return cls(
            vocab_size=int(config["vocab_size"]),
            d_model=int(config.get("d_model", 128)),
            nhead=int(config.get("nhead", 4)),
            dim_feedforward=int(config.get("dim_feedforward", 256)),
            dropout=float(config.get("dropout", 0.1)),
            num_layers=int(config.get("num_layers", 2)),
            num_classes=int(config.get("num_classes", 2)),
        )


class Classifier(nn.Module):
    """Transformer classifier for PV-associated fate prediction."""

    def __init__(self, config: ClassifierConfig | dict[str, Any]):
        super().__init__()
        if isinstance(config, dict):
            config = ClassifierConfig.from_dict(config)
        self.config = config

        self.gene_embedding = nn.Embedding(
            config.vocab_size,
            config.d_model,
            padding_idx=0,
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.d_model))

        self.blocks = nn.ModuleList(
            [
                ClassifierEncoderBlock(
                    d_model=config.d_model,
                    nhead=config.nhead,
                    dim_feedforward=config.dim_feedforward,
                    dropout=config.dropout,
                )
                for _ in range(config.num_layers)
            ]
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.num_classes),
        )
        self.last_attn_weights: torch.Tensor | None = None

    def forward(
        self,
        gene_ids: torch.Tensor,
        gene_vals: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = gene_ids.size(0)

        if torch.any(gene_ids < 0) or torch.any(gene_ids >= self.gene_embedding.num_embeddings):
            bad_min = int(gene_ids.min().item())
            bad_max = int(gene_ids.max().item())
            raise ValueError(
                f"gene_id out of range: min={bad_min}, max={bad_max}, "
                f"allowed=[0, {self.gene_embedding.num_embeddings - 1}]"
            )

        id_embedding = self.gene_embedding(gene_ids)
        gene_features = id_embedding * gene_vals.unsqueeze(-1)

        cell_query = self.cls_token.expand(batch_size, -1, -1)
        weights = None
        for block in self.blocks:
            cell_query, weights = block(cell_query, gene_features, padding_mask)

        self.last_attn_weights = weights
        return self.classifier(cell_query.squeeze(1))
