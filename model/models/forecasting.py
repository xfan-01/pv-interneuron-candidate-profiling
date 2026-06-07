"""Trajectory forecasting generator model migrated from demo/generator_3.ipynb."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import (
    GeneEncoder,
    QueryCrossAttentionBlock,
    QuerySelfAttentionBlock,
    SinusoidalTimeEmbedding,
    TransformerBlock,
)


@dataclass(frozen=True)
class ForecasterConfig:
    """Configuration for the trajectory forecasting generator."""

    n_genes: int
    max_len: int = 1000
    d_model: int = 256
    n_heads: int = 4
    n_encoder_layers: int = 4
    n_query_self_layers: int = 2
    n_query_cross_layers: int = 1
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.n_genes <= 0:
            raise ValueError("n_genes must be a positive integer.")
        if self.max_len <= 0:
            raise ValueError("max_len must be positive.")
        if self.d_model <= 0:
            raise ValueError("d_model must be positive.")
        if self.n_heads <= 0:
            raise ValueError("n_heads must be positive.")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")
        if self.n_encoder_layers <= 0:
            raise ValueError("n_encoder_layers must be positive.")
        if self.n_query_cross_layers <= 0:
            raise ValueError("n_query_cross_layers must be positive.")

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "ForecasterConfig":
        return cls(
            n_genes=int(config["n_genes"]),
            max_len=int(config.get("max_len", 1000)),
            d_model=int(config.get("d_model", 256)),
            n_heads=int(config.get("n_heads", 4)),
            n_encoder_layers=int(config.get("n_encoder_layers", 4)),
            n_query_self_layers=int(config.get("n_query_self_layers", 2)),
            n_query_cross_layers=int(config.get("n_query_cross_layers", 1)),
            dropout=float(config.get("dropout", 0.1)),
        )


class Forecaster(nn.Module):
    """Transformer-based trajectory forecasting model.

    Encodes a source cell state with time conditioning and predicts the
    future expression state at a target time via query-based decoding.
    """

    def __init__(self, config: ForecasterConfig | dict[str, Any]):
        super().__init__()
        if isinstance(config, dict):
            config = ForecasterConfig.from_dict(config)
        self.config = config

        self.d_model = config.d_model
        self.n_genes = config.n_genes

        self.gene_encoder = GeneEncoder(config.n_genes, config.d_model)

        self.source_time_encoder = SinusoidalTimeEmbedding(config.d_model)
        self.target_time_encoder = SinusoidalTimeEmbedding(config.d_model)
        self.delta_time_encoder = SinusoidalTimeEmbedding(config.d_model)

        self.encoder_layers = nn.ModuleList(
            [
                TransformerBlock(config.d_model, config.n_heads, config.dropout)
                for _ in range(config.n_encoder_layers)
            ]
        )

        self.gene_queries = nn.Parameter(torch.randn(config.n_genes, config.d_model))

        self.query_self_layers = nn.ModuleList(
            [
                QuerySelfAttentionBlock(config.d_model, config.n_heads, config.dropout)
                for _ in range(config.n_query_self_layers)
            ]
        )

        self.query_cross_layers = nn.ModuleList(
            [
                QueryCrossAttentionBlock(config.d_model, config.n_heads, config.dropout)
                for _ in range(config.n_query_cross_layers)
            ]
        )

        self.pred_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.SiLU(),
            nn.Linear(config.d_model // 2, 1),
        )

    def forward(
        self,
        g_id: torch.Tensor,
        g_val: torch.Tensor,
        source_time: torch.Tensor,
        target_time: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        need_weights: bool = False,
        epoch: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Forward pass.

        Parameters
        ----------
        g_id : [B, L] long
        g_val : [B, L] float
        source_time : [B] float
        target_time : [B] float
        padding_mask : [B, L] bool, True = padding
        need_weights : bool
        epoch : optional

        Returns
        -------
        pred : [B, G] non-negative expression predictions
        enc_attn_weights : optional
        cross_attn_weights : optional
        """
        delta_time = target_time - source_time

        src_t_emb = self.source_time_encoder(source_time)
        delta_t_emb = self.delta_time_encoder(delta_time)

        x = self.gene_encoder(g_id, g_val) + src_t_emb + delta_t_emb

        enc_attn_weights = None
        for layer in self.encoder_layers:
            x, enc_attn_weights = layer(
                x,
                key_padding_mask=padding_mask,
                need_weights=need_weights,
            )

        bsz = x.size(0)
        queries = self.gene_queries.unsqueeze(0).expand(bsz, -1, -1)

        tgt_t_emb = self.target_time_encoder(target_time)
        out = queries + tgt_t_emb + delta_t_emb

        for layer in self.query_self_layers:
            out = layer(out)

        cross_attn_weights = None
        for layer in self.query_cross_layers:
            if need_weights:
                out, cross_attn_weights = layer(
                    out,
                    memory=x,
                    memory_key_padding_mask=padding_mask,
                    need_weights=True,
                )
            else:
                out = layer(
                    out,
                    memory=x,
                    memory_key_padding_mask=padding_mask,
                    need_weights=False,
                )

        raw_pred = self.pred_head(out).squeeze(-1)
        final_pred = F.softplus(raw_pred)
        final_pred = torch.clamp(final_pred, min=0.0, max=1e5)

        if need_weights:
            return final_pred, enc_attn_weights, cross_attn_weights
        return final_pred, None, None


class AblationForecaster(nn.Module):
    """Forecaster with switchable query self/cross-attn and plain MLP decoder.

    Supports four ablation variants:
    - Full model: query self-attn + query cross-attn
    - w/o query self-attention: cross-attn only
    - w/o query cross-attention: self-attn only
    - Plain decoder: pooled encoder + MLP (no query attention)

    Migrated from ``demo/generator_ablation_3.ipynb``.
    """

    def __init__(
        self, config: ForecasterConfig | dict[str, Any], variant: dict[str, Any]
    ):
        super().__init__()
        if isinstance(config, dict):
            config = ForecasterConfig.from_dict(config)
        self.config = config
        self.variant = dict(variant)

        self.use_query_self_attention = bool(
            variant.get("use_query_self_attention", True)
        )
        self.use_query_cross_attention = bool(
            variant.get("use_query_cross_attention", True)
        )
        self.plain_decoder = bool(variant.get("plain_decoder", False))

        self.d_model = config.d_model
        self.n_genes = config.n_genes

        self.gene_encoder = GeneEncoder(self.n_genes, self.d_model)
        self.source_time_encoder = SinusoidalTimeEmbedding(self.d_model)
        self.target_time_encoder = SinusoidalTimeEmbedding(self.d_model)
        self.delta_time_encoder = SinusoidalTimeEmbedding(self.d_model)

        self.encoder_layers = nn.ModuleList(
            [
                TransformerBlock(self.d_model, config.n_heads, config.dropout)
                for _ in range(config.n_encoder_layers)
            ]
        )

        if self.plain_decoder:
            self.plain_decoder_head = nn.Sequential(
                nn.Linear(self.d_model * 3, self.d_model * 2),
                nn.SiLU(),
                nn.Dropout(config.dropout),
                nn.Linear(self.d_model * 2, self.n_genes),
            )
        else:
            self.gene_queries = nn.Parameter(
                torch.randn(self.n_genes, self.d_model)
            )
            self.query_self_layers = nn.ModuleList(
                [
                    QuerySelfAttentionBlock(
                        self.d_model, config.n_heads, config.dropout
                    )
                    for _ in range(
                        config.n_query_self_layers
                        if self.use_query_self_attention
                        else 0
                    )
                ]
            )
            self.query_cross_layers = nn.ModuleList(
                [
                    QueryCrossAttentionBlock(
                        self.d_model, config.n_heads, config.dropout
                    )
                    for _ in range(
                        config.n_query_cross_layers
                        if self.use_query_cross_attention
                        else 0
                    )
                ]
            )
            self.pred_head = nn.Sequential(
                nn.Linear(self.d_model, self.d_model // 2),
                nn.SiLU(),
                nn.Linear(self.d_model // 2, 1),
            )

    @staticmethod
    def masked_mean(
        x: torch.Tensor, padding_mask: torch.Tensor | None
    ) -> torch.Tensor:
        if padding_mask is None:
            return x.mean(dim=1)
        valid = (~padding_mask).float().unsqueeze(-1)
        denom = valid.sum(dim=1).clamp(min=1.0)
        return (x * valid).sum(dim=1) / denom

    def forward(
        self,
        g_id: torch.Tensor,
        g_val: torch.Tensor,
        source_time: torch.Tensor,
        target_time: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        need_weights: bool = False,
        epoch: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        delta_time = target_time - source_time
        src_t_emb = self.source_time_encoder(source_time)
        tgt_t_emb = self.target_time_encoder(target_time)
        delta_t_emb = self.delta_time_encoder(delta_time)

        x = self.gene_encoder(g_id, g_val) + src_t_emb + delta_t_emb

        enc_attn_weights = None
        for layer in self.encoder_layers:
            x, enc_attn_weights = layer(
                x,
                key_padding_mask=padding_mask,
                need_weights=need_weights,
            )

        if self.plain_decoder:
            pooled = self.masked_mean(x, padding_mask)
            cond = torch.cat(
                [pooled, tgt_t_emb.squeeze(1), delta_t_emb.squeeze(1)], dim=-1
            )
            raw_pred = self.plain_decoder_head(cond)
            final_pred = F.softplus(raw_pred)
            return (
                torch.clamp(final_pred, min=0.0, max=1e5),
                enc_attn_weights if need_weights else None,
                None,
            )

        bsz = x.size(0)
        queries = self.gene_queries.unsqueeze(0).expand(bsz, -1, -1)
        out = queries + tgt_t_emb + delta_t_emb

        for layer in self.query_self_layers:
            out = layer(out)

        cross_attn_weights = None
        for layer in self.query_cross_layers:
            if need_weights:
                out, cross_attn_weights = layer(
                    out,
                    memory=x,
                    memory_key_padding_mask=padding_mask,
                    need_weights=True,
                )
            else:
                out = layer(
                    out,
                    memory=x,
                    memory_key_padding_mask=padding_mask,
                    need_weights=False,
                )

        raw_pred = self.pred_head(out).squeeze(-1)
        final_pred = F.softplus(raw_pred)
        if need_weights:
            return (
                torch.clamp(final_pred, min=0.0, max=1e5),
                enc_attn_weights,
                cross_attn_weights,
            )
        return torch.clamp(final_pred, min=0.0, max=1e5), None, None


# Backward-compatible aliases (deprecated).
GeneratorConfig = ForecasterConfig
Generator = Forecaster
AblationGenerator = AblationForecaster

