# PV Reprogramming Model

Reusable research code for the Transformer-based framework developed in the thesis
*Transformer-Based Regulatory Candidate Profiling for Parvalbumin-Associated
Interneuron Reprogramming*.

This repository implements the thesis workflow for prioritising regulatory candidates in
human glia-to-PV-associated interneuron reprogramming. The codebase consolidates the
former exploratory notebook models into reusable package modules, YAML configs,
analysis utilities, lightweight experiment summaries, and a single command-line
training entry point. Large data files and trained checkpoint weights are kept out
of the repository to stay within GitHub size limits and to keep the source release
portable.

The main reusable components are:

- A binary Transformer fate classifier for distinguishing PV-associated and non-PV-associated cell states.
- An auxiliary seven-class Transformer state classifier for cluster-level identity readouts in perturbation analysis.
- A latent-time trajectory forecasting Transformer that predicts downstream expression states along the PV-associated branch.
- Attribution, candidate-panel, in silico perturbation, sustained rollout, benchmarking, and ablation utilities used to connect model readouts with regulatory hypotheses.

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

The reproducible dataset asset expected by both configs and demo notebooks is:

```text
data/labelled_data.h5ad
```

This file is not included in the GitHub repository. Place the local dataset at the
path above before running training or notebook workflows. The repository only keeps
small reference assets, such as:

```text
data/Homo_sapiens_TF.html
```

Trained checkpoint weights are also excluded from version control. The configs
expect or create local checkpoint files such as:

```text
checkpoints/classifier/best_classifier.pth
checkpoints/classifier/best_classifier_multi.pth
checkpoints/forecasting/best_forecaster.pth
checkpoints/forecasting/best_forecaster_hot.pth
```

Random seeds are configured in YAML files and applied through `model.utils.reproducibility.seed_everything`.

Tracked checkpoint outputs are limited to lightweight analysis artifacts, including
ablation `.csv` summaries and `.json` metric files under:

```text
checkpoints/forecasting_ablation/
```

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
- Large files in `data/` and checkpoint weights in `checkpoints/` are intentionally excluded from Git. Keep them locally for reproduction.
- This release does not currently include a separate `tests/` directory.
