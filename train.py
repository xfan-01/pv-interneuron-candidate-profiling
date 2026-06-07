import argparse
import sys
import os
import yaml
import torch
from pathlib import Path
import numpy as np

from torch.utils.data import DataLoader, random_split, WeightedRandomSampler

from model.utils.device import get_device
from model.utils.reproducibility import seed_everything
from model.data.preprocessing import prepare_classifier_data
from model.models.classifier import Classifier, ClassifierConfig
from model.training.trainer_classifier import ClassifierTrainer

from model.data.trajectory_pairs import PrepareTrajectoryData, TrajectoryDataset
from model.models.forecasting import Forecaster, ForecasterConfig
from model.training.trainer import ForecasterTrainer

def load_yaml_config(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def run_classifier_training(config: dict, device: torch.device):
    print("=" * 60)
    print("Starting Classifier Training")
    print("=" * 60)

    # 1. Infer Binary vs Multi-class
    num_classes = config['model'].get('num_classes', 2)
    label_mode = "multi" if num_classes > 2 else "binary"
    cluster_col = config['data'].get("cluster_col")
    
    # 2. Data Preparation
    print(f"Loading data (mode: {label_mode})...")
    dataset, stats = prepare_classifier_data(
        h5ad_path=config['data']['h5ad_path'],
        max_len=config['data'].get('max_len', 2134),
        exclude_class=config['data'].get('exclude_class', 'hGPC'),
        label_mode=label_mode,
        cluster_col=cluster_col,
    )
    
    # Update vocab_size dynamically based on data stats
    config['model']['vocab_size'] = stats['n_vars'] + 1
    
    # Splitting logic
    train_ratio = config['training'].get('train_ratio', config['training'].get('split_ratio', 0.8))
    val_ratio = config['training'].get('val_ratio', round((1.0 - train_ratio)/2, 2))
    
    train_size = int(train_ratio * len(dataset))
    val_size = int(val_ratio * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_set, val_set, test_set = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(config.get('seed', 42))
    )
    
    batch_size = config['training'].get('batch_size', 128)
    
    # Weighted sampler for multi-class LDAM
    sampler = None
    train_class_counts = None
    if label_mode == "multi" and 'label' in dataset.data:
        train_indices = train_set.indices
        train_labels = dataset.data['label'][train_indices].numpy()
        
        classes, counts = np.unique(train_labels, return_counts=True)
        train_class_counts = np.zeros(num_classes, dtype=np.int64)
        train_class_counts[classes] = counts
        print("Training class counts:", train_class_counts)
        
        weights = 1.0 / np.where(train_class_counts == 0, 1.0, train_class_counts)
        sample_weights = weights[train_labels]
        sampler = WeightedRandomSampler(
            weights=sample_weights, num_samples=len(sample_weights), replacement=True
        )
        train_loader = DataLoader(train_set, batch_size=batch_size, sampler=sampler)
    else:
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    # 3. Setup Model
    clf_config = ClassifierConfig.from_dict(config['model'])
    model = Classifier(clf_config).to(device)
    
    trainer = ClassifierTrainer(model, device, config, train_class_counts=train_class_counts)
    
    # Load pretrained backbone if specified
    pretrained_path = config['training'].get('pretrained_checkpoint_path')
    if pretrained_path:
        trainer.load_pretrained_backbone(pretrained_path)

    # 4. Training Loop
    epochs = config['training'].get('epochs', 100)
    best_val_acc = 0.0
    checkpoint_path = config['training'].get('checkpoint_path', 'best_classifier.pth')
    
    for epoch in range(1, epochs + 1):
        train_loss = trainer.train_one_epoch(train_loader)
        val_loss, val_acc = trainer.validate_one_epoch(val_loader)
        
        print(f"Epoch {epoch:03d}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            trainer.save_checkpoint(checkpoint_path, epoch, val_acc)
            print(f"  -> Saved new best checkpoint (Val Acc: {val_acc:.4f})")
            
    print(f"Done. Best Val Acc: {best_val_acc:.4f}")


def build_forecaster_trainer_config(config: dict) -> dict:
    """Flatten YAML config into the ForecasterTrainer notebook-style schema."""
    training = dict(config.get("training", {}))
    model_params = dict(config.get("model", {}))
    model_params["lr"] = training.get("learning_rate", model_params.get("lr", 1e-4))
    model_params["epochs"] = training.get("epochs", model_params.get("epochs", 100))

    trainer_config = {
        "model_params": model_params,
        "loss_weights": config.get("loss", {}),
    }
    trainer_config.update(training)
    return trainer_config


def get_warmup_epochs(config: dict) -> int:
    training = config.get("training", {})
    if "warmup_epochs" in training:
        return max(1, int(training["warmup_epochs"]))

    total_epochs = int(training.get("epochs", 100))
    ratio = float(training.get("warmup_epochs_ratio", 0.2))
    return max(1, int(total_epochs * ratio))


def get_forecaster_checkpoint_paths(config: dict) -> tuple[str, str]:
    training = config.get("training", {})
    hot_enabled = config.get("hot_start", {}).get("enabled", False)
    configured_path = str(training.get("checkpoint_path", "checkpoints/forecasting/best_forecaster.pth"))

    if not hot_enabled:
        return configured_path, configured_path

    hot_path = str(training.get("hot_checkpoint_path", configured_path))
    if "base_checkpoint_path" in training:
        base_path = str(training["base_checkpoint_path"])
    elif hot_path.endswith("_hot.pth"):
        base_path = hot_path[:-len("_hot.pth")] + ".pth"
    else:
        base_path = hot_path.replace(".pth", "_base.pth")
    return base_path, hot_path


def run_forecaster_training(config: dict, device: torch.device):
    print("=" * 60)
    print("Starting Forecaster Training (MAE Base + Hot-Start)")
    print("=" * 60)
    
    # 1. Prepare Data
    print("Preparing Trajectory Data...")
    processor = PrepareTrajectoryData(
        h5ad_path=config['data']['h5ad_path'],
        config=config['model'],
        subset_col=config['data'].get("subset_col", "trajectory_class"),
        subset_values=tuple(config['data'].get("subset_values", ["PV"])),
        time_col=config['data'].get("time_col"),
        cluster_col=config['data'].get("cluster_col"),
        n_bins=config['data'].get("n_bins", 120),
        allowed_offsets=tuple(config['data'].get("allowed_offsets", [1, 2, 3, 4, 5, 6])),
        base_max_dist=config['data'].get("base_max_dist", 12.0),
        dist_alpha=config['data'].get("dist_alpha", 1.0),
        allowed_cross_steps=tuple(config['data'].get("allowed_cross_steps", [1, 2])),
        k_intra=config['data'].get("k_intra", 1),
        k_cross=config['data'].get("k_cross", 2),
        val_split=config['data'].get("val_split", 0.2),
        heldout_split=config['data'].get("heldout_split", 0.1),
        random_state=config.get("seed", 42),
    )

    train_set = TrajectoryDataset(processor.train_data)
    val_set = TrajectoryDataset(processor.val_data)

    train_loader = DataLoader(
        train_set,
        batch_size=config['training'].get('batch_size', 128), 
        shuffle=True
    )
    val_loader = DataLoader(
        val_set,
        batch_size=config['training'].get('batch_size', 128), 
        shuffle=False
    )
    
    # 2. Setup Model
    model = Forecaster(ForecasterConfig.from_dict(config['model'])).to(device)
    
    # 3. Setup Trainer
    trainer_config = build_forecaster_trainer_config(config)
    trainer = ForecasterTrainer(model, device, trainer_config, loss_weights=config.get("loss"))
    trainer.scheduler = trainer.setup_optimizer(
        lr=config['training'].get('learning_rate'),
        weight_decay=config['training'].get('weight_decay', 0.01),
        warmup_epochs=get_warmup_epochs(config),
        total_epochs=config['training'].get('epochs', 100),
    )[1]
    
    epochs = config['training'].get('epochs', 100)
    base_checkpoint_path, hot_checkpoint_path = get_forecaster_checkpoint_paths(config)
    
    # === PHASE 1: Base Pretraining (MAE) ===
    print("--- Phase 1: Base Pretraining (Self-Reconstruction) ---")
    best_val_loss = float('inf')
    
    for epoch in range(1, epochs + 1):
        tr_loss, *_ = trainer.train_one_epoch(
            train_loader, epoch, mask_prob_override=trainer.base_mask_prob, reconstruct_self=True
        )
        val_loss, *_ = trainer.validate_one_epoch(
            val_loader, mask_prob_override=trainer.base_mask_prob, reconstruct_self=True
        )
        
        trainer.scheduler.step()
        
        lr_curr = trainer.optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:03d} | LR {lr_curr:.2e} | Tr Loss: {tr_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trainer.save_checkpoint(base_checkpoint_path, epoch, {"val_loss": val_loss})
    
    # === PHASE 2: Hot-Start Fine-tuning ===
    hot_cfg = config.get("hot_start", {})
    if hot_cfg.get("enabled", False):
        print(f"\n--- Phase 2: Hot-start Target Prediction (mask={trainer.hot_mask_prob}) ---")
        
        runs = hot_cfg.get("n_runs", 1)
        epochs_per_run = hot_cfg.get("epochs_per_run", 100)
        
        # Load best base model to start phase 2
        trainer.load_checkpoint(base_checkpoint_path, load_optimizer_state=False)
        
        from torch.optim.lr_scheduler import CosineAnnealingLR
        best_overall_val_loss = float('inf')
        
        for run_idx in range(1, runs + 1):
            print(f"  Hot-start Run {run_idx}/{runs}")
            
            # Reset optimizer for warm restart
            trainer.optimizer = torch.optim.AdamW(
                trainer.model.parameters(), 
                lr=hot_cfg.get("start_lr", 1e-4), 
                weight_decay=config['training'].get("weight_decay", 0.01)
            )
            trainer.scheduler = CosineAnnealingLR(
                trainer.optimizer, T_max=epochs_per_run, eta_min=hot_cfg.get("eta_min", 1e-5)
            )
            
            for epoch in range(1, epochs_per_run + 1):
                tr_loss, *_ = trainer.train_one_epoch(
                    train_loader, epoch, mask_prob_override=trainer.hot_mask_prob, reconstruct_self=False
                )
                val_loss, *_ = trainer.validate_one_epoch(
                    val_loader, mask_prob_override=trainer.hot_mask_prob, reconstruct_self=False
                )
                
                trainer.scheduler.step()
                
                if val_loss < best_overall_val_loss:
                    best_overall_val_loss = val_loss
                    trainer.save_checkpoint(hot_checkpoint_path, epoch, {"val_loss": val_loss})
                    print(f"  Epoch {epoch:03d} | Val Loss: {val_loss:.4f}  *** NEW BEST target prediction")
                    
            if hot_cfg.get("resume_from_best_each_run", True):
                trainer.load_checkpoint(hot_checkpoint_path, load_optimizer_state=False)

    print("Forecaster Training Done.")

def main():
    parser = argparse.ArgumentParser(description="Unified Training Orchestrator")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    seed = config.get("seed", 42)
    seed_everything(seed)
    
    device = get_device(0)
    print(f"Using device: {device}")

    # Determine Task Type
    if 'subset_col' in config.get('data', {}) or 'time_col' in config.get('data', {}):
        run_forecaster_training(config, device)
    else:
        run_classifier_training(config, device)

if __name__ == "__main__":
    main()
