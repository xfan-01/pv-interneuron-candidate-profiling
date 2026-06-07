"""ClassifierTrainer: Unified trainer for Binary and Multi-head Classifiers."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from model.training.losses import LDAMLoss


class ClassifierTrainer:
    """Trainer for Gene Expression classification models (Binary or Multi-class/LDAM)."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        config: dict[str, Any],
        train_class_counts: np.ndarray | None = None,
    ):
        self.model = model
        self.device = device
        self.config = config
        self.train_config = config.get("training", {})

        loss_type = self.train_config.get("loss", "cross_entropy")
        if loss_type == "ldam":
            if train_class_counts is None:
                raise ValueError("train_class_counts must be provided for LDAM loss.")
            cls_num_list = (
                train_class_counts.tolist()
                if isinstance(train_class_counts, np.ndarray)
                else train_class_counts
            )
            self.criterion = LDAMLoss(
                cls_num_list=cls_num_list,
                max_m=0.5,
                s=30.0,
                weight=None,
            ).to(device)
            self.is_ldam = True
        else:
            self.criterion = nn.CrossEntropyLoss().to(device)
            self.is_ldam = False

        self.optimizer = self._setup_optimizer()

    def _setup_optimizer(self) -> optim.Optimizer:
        # Multi-head models have separate LRs for backbone and head
        if "learning_rate_backbone" in self.train_config and "learning_rate_head" in self.train_config:
            lr_backbone = float(self.train_config["learning_rate_backbone"])
            lr_head = float(self.train_config["learning_rate_head"])
            param_groups = [
                {"params": self.model.gene_embedding.parameters(), "lr": lr_backbone},
                {"params": self.model.blocks.parameters(), "lr": lr_backbone},
                {"params": self.model.classifier.parameters(), "lr": lr_head},
            ]
            optimizer = optim.Adam(param_groups)
        else:
            lr = float(self.train_config.get("learning_rate", 1e-4))
            optimizer = optim.Adam(self.model.parameters(), lr=lr)

        return optimizer

    def train_one_epoch(self, train_loader) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            x_id = batch["gene_id"].to(self.device, non_blocking=True)
            x_val = batch["gene_val"].to(self.device, non_blocking=True)
            pad_mask = batch["pad_mask"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)

            self.optimizer.zero_grad()
            logits = self.model(x_id, x_val, padding_mask=pad_mask)

            if self.is_ldam:
                loss = self.criterion(logits, labels)
            else:
                loss = self.criterion(logits, labels)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(1, n_batches)

    def validate_one_epoch(self, val_loader) -> tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        n_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                x_id = batch["gene_id"].to(self.device, non_blocking=True)
                x_val = batch["gene_val"].to(self.device, non_blocking=True)
                pad_mask = batch["pad_mask"].to(self.device, non_blocking=True)
                labels = batch["label"].to(self.device, non_blocking=True)

                logits = self.model(x_id, x_val, padding_mask=pad_mask)
                loss = self.criterion(logits, labels)
                total_loss += loss.item()

                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                n_batches += 1

        avg_loss = total_loss / max(1, n_batches)
        acc = float(correct) / max(1, total)
        return avg_loss, acc

    def load_pretrained_backbone(self, checkpoint_path: str | Path):
        """Loads pretrained binary classification weights, keeping the backbone only."""
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        pretrained_dict = ckpt.get("model_state_dict", ckpt)

        model_dict = self.model.state_dict()
        filtered_dict = {
            self._normalise_checkpoint_key(k): v
            for k, v in pretrained_dict.items()
            if self._normalise_checkpoint_key(k) in model_dict
            and "classifier" not in self._normalise_checkpoint_key(k)
        }

        model_dict.update(filtered_dict)
        self.model.load_state_dict(model_dict)
        print(f"Loaded pretrained backbone from {checkpoint_path} (skipped classifier head)")

    @staticmethod
    def _normalise_checkpoint_key(key: str) -> str:
        """Map legacy/DataParallel checkpoint keys onto current classifier names."""
        if key.startswith("module."):
            key = key[len("module."):]
        if key.startswith("blks."):
            key = "blocks." + key[len("blks."):]
        return key

    def save_checkpoint(self, path: str | Path, epoch: int, val_acc: float):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_accuracy": val_acc,
            },
            path,
        )

    def cleanup(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
