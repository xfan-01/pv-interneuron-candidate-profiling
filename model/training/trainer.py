"""ForecasterTrainer: Comprehensive trainer for trajectory forecasting models."""

from __future__ import annotations

import gc
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from model.training.losses import ManualHybridLoss


# Constants for masking
MASK_TOKEN_VAL = -1.0


class ForecasterTrainer:
    """
    Unified trainer for trajectory forecasting with support for:
    - Masked autoencoder (MAE) base training
    - Hot-start fine-tuning
    - Gradient accumulation and AMP (automatic mixed precision)
    - Time jittering and masking strategies
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        config: dict[str, Any],
        loss_weights: dict[str, float] | None = None,
    ):
        """
        Initialize the trainer.

        Parameters
        ----------
        model : nn.Module
            The forecasting model.
        device : torch.device
            Device for training.
        config : dict
            Training configuration (must include model_params, loss_weights, etc.)
        loss_weights : dict, optional
            Override loss weights. If None, uses config['loss_weights'].
        """
        self.model = model
        self.device = device
        self.config = config
        
        # Extract config parameters
        self.model_params = config.get("model_params", {})
        loss_cfg = loss_weights or config.get("loss_weights", {})
        
        self.criterion = ManualHybridLoss(**loss_cfg).to(device)
        
        # Training hyperparameters
        self.use_amp = bool(config.get("use_amp", True))
        self.use_accumulation = bool(config.get("use_accumulation", True))
        self.accumulation_steps = max(1, int(config.get("accumulation_steps", 1)))
        self.mask_prob = float(config.get("mask_prob", 0.30))
        self.base_mask_prob = float(config.get("base_mask_prob", self.mask_prob))
        self.hot_mask_prob = float(config.get("hot_mask_prob", self.mask_prob))
        self.time_jitter_std = float(config.get("time_jitter_std", 0.0))
        
        # Reconstruction-specific masking
        self.reconstruct_self_mask_prob_nz = float(
            config.get("reconstruct_self_mask_prob_nz", 0.30)
        )
        self.reconstruct_self_mask_prob_z = float(
            config.get("reconstruct_self_mask_prob_z", 0.05)
        )
        
        # Optimizer and scheduler will be initialized in setup_optimizer
        self.optimizer: optim.Optimizer | None = None
        self.scheduler: SequentialLR | None = None
        self.scaler: torch.amp.GradScaler | None = None
        
        self.warmup_epochs = 5

    def setup_optimizer(
        self,
        lr: float | None = None,
        weight_decay: float = 0.01,
        warmup_epochs: int | None = None,
        total_epochs: int | None = None,
    ) -> tuple[optim.Optimizer, SequentialLR]:
        """
        Setup AdamW optimizer and learning rate scheduler.

        Parameters
        ----------
        lr : float, optional
            Learning rate. If None, uses config['model_params']['lr'].
        weight_decay : float, default=0.01
            L2 regularization.
        warmup_epochs : int, optional
            Warmup duration. If None, uses self.warmup_epochs.
        total_epochs : int, optional
            Total training epochs. If None, uses config['model_params']['epochs'].

        Returns
        -------
        optimizer, scheduler : Tuple[Optimizer, SequentialLR]
        """
        if lr is None:
            lr = float(self.model_params.get("lr", 1e-4))
        if warmup_epochs is None:
            warmup_epochs = self.warmup_epochs
        if total_epochs is None:
            total_epochs = int(self.model_params.get("epochs", 100))

        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )

        scheduler_warmup = LinearLR(
            self.optimizer, start_factor=0.2, end_factor=1.0, total_iters=warmup_epochs
        )

        scheduler_cosine = CosineAnnealingLR(
            self.optimizer,
            T_max=max(1, total_epochs - warmup_epochs),
            eta_min=1e-6,
        )

        scheduler = SequentialLR(
            self.optimizer,
            schedulers=[scheduler_warmup, scheduler_cosine],
            milestones=[warmup_epochs],
        )
        self.scheduler = scheduler

        self.scaler = torch.amp.GradScaler(
            "cuda" if self.device.type == "cuda" else "cpu",
            enabled=(self.use_amp and self.device.type == "cuda"),
        )

        return self.optimizer, scheduler

    @staticmethod
    def _amp_autocast_ctx(device: torch.device) -> Any:
        """Return appropriate autocast context based on device."""
        if device.type == "cuda":
            return torch.amp.autocast("cuda", dtype=torch.bfloat16)
        return nullcontext()

    @staticmethod
    def apply_time_jitter(
        source_time: torch.Tensor, target_time: torch.Tensor, noise_std: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply Gaussian noise to time values, clamping to [0, 1].

        Parameters
        ----------
        source_time : [B, 1]
        target_time : [B, 1]
        noise_std : float
            Standard deviation of Gaussian noise.

        Returns
        -------
        source_time_noisy, target_time_noisy : Tuple of clamped tensors
        """
        if noise_std <= 0.0:
            return source_time, target_time

        source_noisy = torch.clamp(
            source_time + torch.randn_like(source_time) * noise_std, 0.0, 1.0
        )
        target_noisy = torch.clamp(
            target_time + torch.randn_like(target_time) * noise_std, 0.0, 1.0
        )
        return source_noisy, target_noisy

    def sample_mask_and_corrupt_values(
        self,
        s_val: torch.Tensor,
        padding_mask: torch.Tensor,
        mask_prob: float,
        reconstruct_self: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply masking and corruption to input values for training.

        Parameters
        ----------
        s_val : [B, L]
            Input values.
        padding_mask : [B, L]
            Boolean mask indicating padding positions (True = padding).
        mask_prob : float
            Base masking probability.
        reconstruct_self : bool, default=False
            If True, use separate masking rates for zero and non-zero genes.

        Returns
        -------
        s_val_masked, supervised_token_mask : Tuple[Tensor, Tensor]
            s_val_masked: Values with masking applied.
            supervised_token_mask: Positions that should be supervised (80% mask + 10% random).
        """
        if reconstruct_self:
            # Differential masking based on value
            token_mask_prob = torch.where(
                s_val > 0,
                torch.full_like(s_val, self.reconstruct_self_mask_prob_nz),
                torch.full_like(s_val, self.reconstruct_self_mask_prob_z),
            )
            random_mask = torch.rand(s_val.shape, device=s_val.device) < token_mask_prob
        else:
            random_mask = torch.rand(s_val.shape, device=s_val.device) < mask_prob

        valid_genes_mask = ~padding_mask
        actual_mask = random_mask & valid_genes_mask

        s_val_masked = s_val.clone()
        rand_selector = torch.rand(s_val.shape, device=s_val.device)

        mask_replace = actual_mask & (rand_selector < 0.8)
        mask_random = actual_mask & (rand_selector >= 0.8) & (rand_selector < 0.9)

        s_val_masked[mask_replace] = MASK_TOKEN_VAL
        s_val_masked[mask_random] = torch.rand_like(s_val_masked[mask_random]) * 5.0

        supervised_token_mask = mask_replace | mask_random
        return s_val_masked, supervised_token_mask

    @staticmethod
    def build_active_gene_mask_from_token_mask(
        s_id: torch.Tensor, token_mask: torch.Tensor, pred_dim: int
    ) -> torch.Tensor | None:
        """
        Build a per-gene supervision mask from token-level supervision.

        Parameters
        ----------
        s_id : [B, L]
            Gene IDs (1-based).
        token_mask : [B, L]
            Boolean tensor indicating which tokens should be supervised.
        pred_dim : int
            Number of genes (prediction dimension).

        Returns
        -------
        active_gene_mask : [B, pred_dim] bool or None
            Per-gene binary mask, or None if no valid tokens.
        """
        valid_token_mask = token_mask & (s_id > 0) & (s_id <= pred_dim)

        if not valid_token_mask.any().item():
            return None

        token_gene_idx = (s_id - 1).clamp(min=0, max=pred_dim - 1)
        active_gene_scores = torch.zeros(
            s_id.size(0), pred_dim, device=s_id.device, dtype=torch.float32
        )
        active_gene_scores.scatter_add_(1, token_gene_idx, valid_token_mask.float())
        return active_gene_scores > 0

    def train_one_epoch(
        self,
        train_loader,
        epoch: int,
        mask_prob_override: float | None = None,
        reconstruct_self: bool = False,
    ) -> tuple[float, float, float, float, float]:
        """
        Train for one epoch with gradient accumulation and AMP.

        Parameters
        ----------
        train_loader : DataLoader
            Training data loader.
        epoch : int
            Current epoch number.
        mask_prob_override : float, optional
            Override mask probability. If None, uses default mask_prob.
        reconstruct_self : bool, default=False
            If True, reconstruct masked input; else, predict future target.

        Returns
        -------
        (loss_total, loss_nz_smooth_l1, loss_nzcos, loss_z_mse, loss_z_l1) : Tuple[float, ...]
        """
        if self.optimizer is None or self.scaler is None:
            raise RuntimeError("Optimizer and scaler not initialized. Call setup_optimizer first.")

        self.model.train()
        totals = np.zeros(6, dtype=np.float64)

        effective_mask_prob = (
            mask_prob_override if mask_prob_override is not None else self.mask_prob
        )
        time_jitter_std = self.time_jitter_std
        accumulation_steps = (
            self.accumulation_steps if self.use_accumulation else 1
        )

        self.optimizer.zero_grad(set_to_none=True)

        with torch.enable_grad():
            for step, batch in enumerate(train_loader):
                s_id = batch["gene_id"].to(self.device, non_blocking=True)
                s_val = batch["gene_val"].to(self.device, non_blocking=True)
                padding_mask = batch["padding_mask"].to(self.device, non_blocking=True)
                s_time = batch["time"].to(self.device, non_blocking=True)
                target_time = batch["target_time"].to(self.device, non_blocking=True)
                
                if reconstruct_self:
                    source_full_val = batch["full_input_val"].to(
                        self.device, non_blocking=True
                    )
                    train_target = source_full_val
                    target_time_input = s_time
                else:
                    t_val = batch["target_val"].to(self.device, non_blocking=True)
                    train_target = t_val
                    target_time_input = target_time

                s_time_noisy, target_time_noisy = self.apply_time_jitter(
                    s_time, target_time, time_jitter_std
                )
                if reconstruct_self:
                    target_time_input = s_time_noisy
                else:
                    target_time_input = target_time_noisy

                s_val_masked, supervised_token_mask = self.sample_mask_and_corrupt_values(
                    s_val, padding_mask, effective_mask_prob, reconstruct_self
                )

                with self._amp_autocast_ctx(self.device):
                    preds, _, _ = self.model(
                        s_id,
                        s_val_masked,
                        s_time_noisy,
                        target_time_input,
                        padding_mask=padding_mask,
                    )
                    preds = torch.clamp(preds, min=0.0, max=50.0)

                    active_gene_mask = self.build_active_gene_mask_from_token_mask(
                        s_id, supervised_token_mask, preds.size(1)
                    )

                    loss_total, nz_smooth_l1, nzcosloss, z_mse, z_l1 = self.criterion(
                        preds, train_target, active_mask=active_gene_mask
                    )
                    loss = loss_total / accumulation_steps

                self.scaler.scale(loss).backward()

                should_step = ((step + 1) % accumulation_steps == 0) or (
                    (step + 1) == len(train_loader)
                )
                if should_step:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)

                bs = s_id.size(0)
                totals += np.array(
                    [loss_total.item(), nz_smooth_l1, nzcosloss, z_mse, z_l1, bs],
                    dtype=np.float64,
                )

                del s_id, s_val, padding_mask, s_time, target_time
                del s_val_masked, supervised_token_mask, active_gene_mask, preds, loss_total, loss
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()

        denom = max(1.0, totals[5])
        return (
            totals[0] / denom,
            totals[1] / denom,
            totals[2] / denom,
            totals[3] / denom,
            totals[4] / denom,
        )

    def validate_one_epoch(
        self,
        val_loader,
        mask_prob_override: float | None = None,
        reconstruct_self: bool = False,
    ) -> tuple[float, float, float, float, float]:
        """
        Validate for one epoch.

        Parameters
        ----------
        val_loader : DataLoader
            Validation data loader.
        mask_prob_override : float, optional
            Override mask probability.
        reconstruct_self : bool, default=False
            If True, reconstruct masked input; else, predict future target.

        Returns
        -------
        (loss_total, loss_nz_smooth_l1, loss_nzcos, loss_z_mse, loss_z_l1) : Tuple[float, ...]
        """
        self.model.eval()
        totals = np.zeros(6, dtype=np.float64)

        effective_mask_prob = (
            mask_prob_override if mask_prob_override is not None else self.mask_prob
        )

        with torch.no_grad():
            for batch in val_loader:
                s_id = batch["gene_id"].to(self.device, non_blocking=True)
                s_val = batch["gene_val"].to(self.device, non_blocking=True)
                padding_mask = batch["padding_mask"].to(self.device, non_blocking=True)
                s_time = batch["time"].to(self.device, non_blocking=True)
                target_time = batch["target_time"].to(self.device, non_blocking=True)

                if reconstruct_self:
                    source_full_val = batch["full_input_val"].to(
                        self.device, non_blocking=True
                    )
                    eval_target = source_full_val
                    target_time_eval = s_time
                else:
                    t_val = batch["target_val"].to(self.device, non_blocking=True)
                    eval_target = t_val
                    target_time_eval = target_time

                s_val_masked, supervised_token_mask = self.sample_mask_and_corrupt_values(
                    s_val, padding_mask, effective_mask_prob, reconstruct_self
                )

                with self._amp_autocast_ctx(self.device):
                    preds, _, _ = self.model(
                        s_id,
                        s_val_masked,
                        s_time,
                        target_time_eval,
                        padding_mask=padding_mask,
                    )
                    preds = torch.clamp(preds, min=0.0, max=50.0)

                    active_gene_mask = self.build_active_gene_mask_from_token_mask(
                        s_id, supervised_token_mask, preds.size(1)
                    )

                    loss_v, nz_smooth_l1_v, nzcosloss_v, z_mse_v, z_l1_v = self.criterion(
                        preds, eval_target, active_mask=active_gene_mask
                    )

                bs = s_id.size(0)
                totals += np.array(
                    [loss_v.item(), nz_smooth_l1_v, nzcosloss_v, z_mse_v, z_l1_v, bs],
                    dtype=np.float64,
                )

                del s_id, s_val, padding_mask, s_time, target_time
                del s_val_masked, supervised_token_mask, active_gene_mask, preds, loss_v
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()

        denom = max(1.0, totals[5])
        return (
            totals[0] / denom,
            totals[1] / denom,
            totals[2] / denom,
            totals[3] / denom,
            totals[4] / denom,
        )

    def save_checkpoint(
        self,
        path: str | Path,
        epoch: int,
        metrics: dict[str, Any],
        **extra: Any,
    ) -> None:
        """
        Save training checkpoint.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        model_to_save = (
            self.model.module
            if isinstance(self.model, nn.DataParallel)
            else self.model
        )

        state = {
            "epoch": epoch,
            "model_state_dict": model_to_save.state_dict(),
            "optimizer_state_dict": (
                self.optimizer.state_dict() if self.optimizer else None
            ),
            "scheduler_state_dict": (
                self.scheduler.state_dict() if self.scheduler else None
            ),
            "scaler_state_dict": (
                self.scaler.state_dict() if self.scaler else None
            ),
            "metrics": metrics,
            **extra,
        }
        torch.save(state, path)

    def load_checkpoint(
        self, path: str | Path, load_optimizer_state: bool = True
    ) -> dict[str, Any]:
        """
        Load training checkpoint.
        """
        path = Path(path)
        state = torch.load(path, map_location=self.device)

        model_state = state.get("model_state_dict", {})
        is_parallel = isinstance(self.model, nn.DataParallel)
        remapped_state = {}

        for k, v in model_state.items():
            if k.startswith("module."):
                if is_parallel:
                    remapped_state[k] = v
                else:
                    remapped_state[k[7:]] = v
            else:
                if is_parallel:
                    remapped_state[f"module.{k}"] = v
                else:
                    remapped_state[k] = v

        self.model.load_state_dict(remapped_state)

        if load_optimizer_state:
            if self.optimizer and "optimizer_state_dict" in state:
                self.optimizer.load_state_dict(state["optimizer_state_dict"])
            if self.scheduler and "scheduler_state_dict" in state:
                self.scheduler.load_state_dict(state["scheduler_state_dict"])
            if self.scaler and "scaler_state_dict" in state:
                self.scaler.load_state_dict(state["scaler_state_dict"])

        return state

    def cleanup_cuda(self) -> None:
        """Clean up CUDA memory."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
