"""Loss functions for classifier and generator training."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LDAMLoss(nn.Module):
    """Label-Distribution-Aware Margin loss for class-imbalanced multi-class tasks.

    Migrated from ``demo/classifier_multi.ipynb``.
    """

    def __init__(
        self,
        cls_num_list: list[int] | np.ndarray,
        max_m: float = 0.5,
        s: float = 30.0,
        weight: torch.Tensor | None = None,
    ):
        super().__init__()
        cls_num_list = np.asarray(cls_num_list, dtype=np.float32)
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list + 1e-12))
        m_list = m_list * (max_m / np.max(m_list))
        self.register_buffer("m_list", torch.tensor(m_list, dtype=torch.float32))
        self.s = s
        self.weight = weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 2:
            raise ValueError(f"logits must have shape [B, C], got {tuple(logits.shape)}")
        if target.ndim != 1:
            raise ValueError(f"target must have shape [B], got {tuple(target.shape)}")
        if logits.size(0) != target.size(0):
            raise ValueError("Batch size mismatch between logits and target.")

        m_list = self.m_list.to(logits.device)
        logits_adjusted = logits.clone()
        index = torch.arange(logits.size(0), device=logits.device)
        logits_adjusted[index, target] -= m_list[target]
        return F.cross_entropy(self.s * logits_adjusted, target, weight=self.weight)


class ManualHybridLoss(nn.Module):
    """Multi-component reconstruction loss for trajectory forecasting.

    Combines non-zero SmoothL1, zero-region MSE, zero-region L1, and
    non-zero cosine similarity loss.

    Migrated from ``demo/generator_3.ipynb``.
    """

    def __init__(
        self,
        lambda_nz: float = 1.0,
        lambda_z_l1: float = 0.2,
        lambda_z_l2: float = 0.2,
        lambda_nzcos: float = 0.2,
        min_genes_for_cos: int = 5,
        **kwargs: object,
    ):
        super().__init__()
        self.lambda_nz = float(lambda_nz)
        self.lambda_z_l1 = float(lambda_z_l1)
        self.lambda_z_l2 = float(lambda_z_l2)
        self.lambda_nzcos = float(lambda_nzcos)
        self.min_genes_for_cos = int(min_genes_for_cos)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        active_mask: torch.Tensor | None = None,
        dummy_gate: object = None,
    ) -> tuple[torch.Tensor, float, float, float, float]:
        """Compute the hybrid reconstruction loss.

        Parameters
        ----------
        pred : [B, G]
        target : [B, G]
        active_mask : [B, G] bool/float, optional

        Returns
        -------
        total_loss : Tensor
        loss_nz : float
        loss_nzcos : float
        loss_z : float
        loss_z_l1 : float
        """
        if not torch.isfinite(pred).all():
            raise RuntimeError("Non-finite prediction detected in loss input `pred`.")

        pred = pred.float()
        target = target.float()

        if active_mask is None:
            active_mask = torch.ones_like(target, dtype=pred.dtype, device=pred.device)
        else:
            active_mask = active_mask.float()

        nz_mask = (target > 0).float() * active_mask
        z_mask = (target <= 0).float() * active_mask

        # 1) non-zero SmoothL1
        smooth_l1 = F.smooth_l1_loss(pred, target, reduction="none")
        num_nz_per_cell = nz_mask.sum(dim=1).clamp(min=1.0)
        loss_nz_per_cell = (smooth_l1 * nz_mask).sum(dim=1) / num_nz_per_cell
        valid_nz_cells = nz_mask.sum(dim=1) > 0
        loss_nz = loss_nz_per_cell[valid_nz_cells].mean() if valid_nz_cells.any() else pred.new_tensor(0.0)

        # 2) zero-region MSE
        diff_sq = (pred - target).pow(2).clamp(max=1e4)
        num_z_per_cell = z_mask.sum(dim=1).clamp(min=1.0)
        loss_z_per_cell = (diff_sq * z_mask).sum(dim=1) / num_z_per_cell
        valid_z_cells = z_mask.sum(dim=1) > 0
        loss_z = loss_z_per_cell[valid_z_cells].mean() if valid_z_cells.any() else pred.new_tensor(0.0)

        # 3) zero-region L1
        loss_z_l1_per_cell = (pred.abs() * z_mask).sum(dim=1) / num_z_per_cell
        loss_z_l1 = loss_z_l1_per_cell[valid_z_cells].mean() if valid_z_cells.any() else pred.new_tensor(0.0)

        # 4) non-zero cosine similarity loss
        nz_count_per_cell = nz_mask.sum(dim=1)
        valid_cos_cells = nz_count_per_cell >= self.min_genes_for_cos
        if valid_cos_cells.any():
            pred_nz = pred * nz_mask
            target_nz = target * nz_mask
            cos_sim = F.cosine_similarity(
                pred_nz[valid_cos_cells], target_nz[valid_cos_cells], dim=1, eps=1e-8
            )
            loss_nzcos = 1.0 - cos_sim.mean()
        else:
            loss_nzcos = pred.new_tensor(0.0)

        total = (
            self.lambda_nz * loss_nz
            + self.lambda_z_l2 * loss_z
            + self.lambda_z_l1 * loss_z_l1
            + self.lambda_nzcos * loss_nzcos
        )

        return total, loss_nz.item(), loss_nzcos.item(), loss_z.item(), loss_z_l1.item()
