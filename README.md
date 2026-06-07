# PV Reprogramming Model

Reusable research code for Transformer-based modelling of human glia-to-PV interneuron reprogramming.

This repository contains the migrated, reusable implementation of the three original demo notebook models:

- Binary fate classifier from `demo/classifier.ipynb`
- Multi-class cluster classifier from `demo/classifier_multi.ipynb`
- Trajectory forecasting generator from `demo/generator_3.ipynb`

The model architectures, training configurations, dataset path, and pretrained checkpoints have been consolidated into package modules, YAML configs, and a single command-line training entry point.

## Environment

Recommended Python:

```text
Python 3.10 or 3.11
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Reproducibility Assets

The reproducible dataset asset used by both configs and demo notebooks is:

```text
data/labelled_data.h5ad
```

Included checkpoints:

```text
checkpoints/classifier/best_classifier.pth
checkpoints/classifier/best_classifier_multi.pth
checkpoints/forecasting/best_forecaster.pth
checkpoints/forecasting/best_forecaster_hot.pth
```

Random seeds are configured in YAML files and applied through `model.utils.reproducibility.seed_everything`.

## Model Coverage

### Binary Classifier

Config:

```text
configs/classifier_binary.yaml
```

Architecture:

- Gene ID embedding scaled by expression values
- Learnable CLS token
- Cross-attention encoder blocks
- Position-wise FFN + residual LayerNorm
- `LayerNorm + Linear` binary classifier head

Training settings reproduce the original notebook:

- `max_len: 2134`
- `batch_size: 128`
- `epochs: 100`
- `learning_rate: 1e-5`
- `loss: cross_entropy`

Run:

```bash
python train.py --config configs/classifier_binary.yaml
```

### Multi-Class Classifier

Config:

```text
configs/classifier_multi.yaml
```

Architecture:

- Same classifier backbone as the binary model
- Seven-class classifier head

Training behaviour:

- Loads the binary checkpoint as a pretrained backbone
- Skips the classifier head during pretrained loading
- Uses `WeightedRandomSampler` for class imbalance
- Uses `LDAMLoss`
- Uses decoupled learning rates:
  - Backbone: `1e-6`
  - Classifier head: `1e-4`

Run:

```bash
python train.py --config configs/classifier_multi.yaml
```

### Trajectory Forecaster

Config:

```text
configs/forecasting.yaml
```

Primary API:

```python
from model.models import Forecaster, ForecasterConfig
```

Backward-compatible generator aliases are also preserved:

```python
from model.models import Generator, GeneratorConfig
```

Architecture:

- Gene ID/value encoder
- Source, target, and delta-time sinusoidal embeddings
- Transformer encoder over source cell tokens
- Learnable per-gene query tokens
- Query self-attention
- Query cross-attention over encoded source tokens
- SwiGLU feed-forward blocks
- `Softplus + clamp` non-negative expression output

Training behaviour:

- Phase 1: masked self-reconstruction with `base_mask_prob: 0.30`
- Phase 2: target-prediction hot start with `hot_mask_prob: 0.00`
- AdamW optimizer
- Warmup + cosine schedule for base training
- Cosine schedule for hot-start training
- Hybrid loss with non-zero SmoothL1, zero-region L1/L2, and non-zero cosine terms

Run:

```bash
python train.py --config configs/forecasting.yaml
```

## Notebook Entry Points

Unified notebooks call the package API directly:

```text
notebooks/00_quickstart.ipynb
notebooks/01_classifier_analysis.ipynb
notebooks/02_forecasting_analysis.ipynb
notebooks/03_model_performance.ipynb
notebooks/04_perturbation_analysis.ipynb
notebooks/05_benchmarking_alignment.ipynb
notebooks/06_ablation_analysis.ipynb
```

The original exploratory notebooks remain under `demo/` for traceability.

## Package Layout

```text
model/
  models/
    classifier.py
    forecasting.py
    generator.py
    layers.py
  data/
    preprocessing.py
    trajectory_pairs.py
  training/
    trainer_classifier.py
    trainer.py
    losses.py
    checkpointing.py
  analysis/
  utils/

configs/
  classifier_binary.yaml
  classifier_multi.yaml
  forecasting.yaml
  forecasting_ablation.yaml
  smoke_forecasting.yaml
  thesis_constants.yaml
  thesis_figures.yaml
```

## Shared Constants

Shared biological and experimental constants are centralized in:

```text
configs/thesis_constants.yaml
```

Access them through `model.utils.constants`.

## Notes

- `Forecaster*` is the primary API name for the migrated generator model.
- `Generator*` aliases remain available for backward compatibility.
- `data/` and `checkpoints/` are intentionally kept for local reproduction.
- This release does not currently include a separate `tests/` directory.
