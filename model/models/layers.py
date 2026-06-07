"""Shared neural-network layers migrated from the demo notebooks."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionWiseFFN(nn.Module):
    """Two-layer feed-forward block used by the classifier encoder."""

    def __init__(self, d_model: int, dim_feedforward: int):
        super().__init__()
        self.dense1 = nn.Linear(d_model, dim_feedforward)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(dim_feedforward, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dense2(self.relu(self.dense1(x)))


class AddNorm(nn.Module):
    """Residual add, dropout, and layer normalisation."""

    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.ln(self.dropout(y) + x)


class ClassifierEncoderBlock(nn.Module):
    """Cross-attention block used by the fate classifier."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
    ):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.addnorm1 = AddNorm(d_model, dropout)
        self.ffn = PositionWiseFFN(d_model, dim_feedforward)
        self.addnorm2 = AddNorm(d_model, dropout)

    def forward(
        self,
        cell_query: torch.Tensor,
        gene_features: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        padding_mask = padding_mask.bool()
        attention_output, attention_weights = self.attention(
            query=cell_query,
            key=gene_features,
            value=gene_features,
            key_padding_mask=padding_mask,
            need_weights=True,
        )

        hidden = self.addnorm1(cell_query, attention_output)
        return self.addnorm2(hidden, self.ffn(hidden)), attention_weights


class GeneEncoder(nn.Module):
    """Encode gene IDs and expression values into token embeddings."""

    def __init__(self, n_genes: int, d_model: int):
        super().__init__()
        self.n_genes = n_genes
        self.gene_emb = nn.Embedding(n_genes + 1, d_model, padding_idx=0)
        self.val_proj = nn.Linear(1, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, gene_ids: torch.Tensor, gene_vals: torch.Tensor) -> torch.Tensor:
        if torch.any(gene_ids < 0) or torch.any(gene_ids > self.n_genes):
            bad_min = int(gene_ids.min().item())
            bad_max = int(gene_ids.max().item())
            raise ValueError(
                f"gene_ids out of range. Expected ids in [0, {self.n_genes}], "
                f"but got min={bad_min}, max={bad_max}."
            )

        return self.norm(self.gene_emb(gene_ids) + self.val_proj(gene_vals.unsqueeze(-1)))


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal latent-time embedding followed by a small MLP."""

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_scaled = t * 100.0
        half_dim = self.d_model // 2
        freq_scale = math.log(10000.0) / max(1, half_dim - 1)
        freq = torch.exp(
            torch.arange(half_dim, device=t.device, dtype=torch.float32)
            * -freq_scale
        )

        emb = t_scaled.unsqueeze(-1).float() * freq.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

        if self.d_model % 2 == 1:
            emb = F.pad(emb, (0, 1))

        return self.mlp(emb).unsqueeze(1)


class TransformerBlock(nn.Module):
    """Self-attention encoder block used by the trajectory forecasting model."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model,
            n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(
        self,
        src: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        need_weights: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        attn_out, attn_weights = self.self_attn(
            src,
            src,
            src,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            average_attn_weights=True,
        )
        src = self.norm1(src + self.dropout1(attn_out))
        src = self.norm2(src + self.dropout2(self.ffn(src)))
        return src, attn_weights


class SwiGLU(nn.Module):
    """SwiGLU feed-forward block used in query attention layers."""

    def __init__(self, d_model: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, hidden_dim)
        self.w2 = nn.Linear(d_model, hidden_dim)
        self.w3 = nn.Linear(hidden_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w2(x)) * self.w1(x))


class QuerySelfAttentionBlock(nn.Module):
    """Self-attention block over learnable query tokens."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model,
            n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.feed_forward = SwiGLU(d_model, int(d_model * 8 / 3))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt: torch.Tensor) -> torch.Tensor:
        x, _ = self.self_attn(tgt, tgt, tgt)
        x = self.norm1(tgt + self.dropout(x))
        return self.norm2(x + self.dropout(self.feed_forward(x)))


class QueryCrossAttentionBlock(nn.Module):
    """Cross-attention block from query tokens to encoded source tokens."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            d_model,
            n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.feed_forward = SwiGLU(d_model, int(d_model * 8 / 3))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        memory_key_padding_mask: torch.Tensor | None = None,
        need_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        y, cross_attn_weights = self.cross_attn(
            tgt,
            memory,
            memory,
            key_padding_mask=memory_key_padding_mask,
            need_weights=need_weights,
            average_attn_weights=True,
        )
        x = self.norm1(tgt + self.dropout(y))
        out = self.norm2(x + self.dropout(self.feed_forward(x)))
        if need_weights:
            return out, cross_attn_weights
        return out


