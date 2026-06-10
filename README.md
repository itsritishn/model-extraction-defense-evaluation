# Threat-Aware Analysis of Model Extraction Attacks in Machine Learning Pipeline

**MSc Dissertation Project**

A comprehensive framework for evaluating model extraction attacks against
deep learning classifiers and the effectiveness of output-perturbation
defenses — all within a strict **black-box threat model**.

---

## Table of Contents

1. [Overview](#overview)
2. [System Requirements](#system-requirements)
3. [Installation](#installation)
4. [Project Structure](#project-structure)
5. [How to Run](#how-to-run)
6. [Modules](#modules)
7. [Attacks](#attacks)
8. [Defenses](#defenses)
9. [Evaluation Metrics](#evaluation-metrics)
10. [Results & Charts](#results--charts)

---

## Overview

| Component | Details |
|-----------|---------|
| **Victim model** | ResNet-50 (fine-tuned on CIFAR-10) |
| **Substitute models** | SmallCNN (3 conv layers), MobileNetV3-Small |
| **Dataset** | CIFAR-10 (60 000 images, 10 classes) |
| **Threat model** | Black-box only — attacker sees probabilities, never weights |
| **Attacks** | Random Query, Knockoff Nets, Active Learning |
| **Defenses** | Rounding, Gaussian Noise, Laplace Noise, Top-K, Throttling, Prediction Poisoning, Adaptive Noise |

---

## System Requirements

| Item | Specification |
|------|--------------|
| Python | **3.9.25** (strict — no 3.10+ features) |
| GPU | NVIDIA RTX 4070 (12 GB VRAM) |
| CUDA | 12.1 |
| OS | Linux |

---

## Installation

```bash
# 1. Create and activate a virtual environment (recommended)
python3.9 -m venv venv
source venv/bin/activate

# 2. Install PyTorch with CUDA 12.1
pip install torch==2.1.0 torchvision==0.16.0 \
    --index-url https://download.pytorch.org/whl/cu121

# 3. Install remaining dependencies
pip install numpy==1.24.0 pandas==2.0.0 \
    matplotlib==3.7.0 seaborn==0.12.0 \
    scikit-learn==1.3.0 flask==3.0.0 tqdm==4.65.0
```

---

## Project Structure

```
project/
├── victim.py            # Victim model training & black-box query API
├── threat_model.py      # Threat model, query budgets, knowledge levels
├── attack.py            # 3 attack strategies + substitute architectures
├── defenses.py          # 7 defense mechanisms (20 parameter variants)
├── evaluate.py          # Full evaluation pipeline, CSV, charts
├── data_loader.py       # CIFAR-10 data loading & transforms
├── requirements.txt     # Pinned dependencies
├── README.md            # This file
├── models/
│   └── victim_resnet50.pth
├── data/
│   └── cifar10/
└── experiments/
    ├── results/         # CSV output files
    │   └── *.csv
    └── charts/          # Generated plots
        └── *.png
```

---

## How to Run

### Step 1 — Train victim models

```bash
python victim.py
```

This fine-tunes ResNet-50 on CIFAR-10 and saves
checkpoints to `models/`.

### Step 2 — Run full evaluation

```bash
python evaluate.py
```

This iterates over every **(attack × defense × budget × substitute)**
combination, computes metrics, saves CSVs to `experiments/results/`,
and generates charts in `experiments/charts/`.

### Step 3 — Individual modules (optional)

```bash
# Sanity-check data pipeline
python data_loader.py

# Print all threat configurations
python threat_model.py

# Quick attack demo
python attack.py

# Smoke-test all defenses
python defenses.py
```

---

## Modules

### `data_loader.py`
- Downloads CIFAR-10 automatically.
- Resizes images to 224×224 for ImageNet-pretrained backbones.
- Applies ImageNet normalisation and data augmentation.

### `victim.py`
- Builds ResNet-50 with a frozen backbone and
  trainable final layer.
- Trains with automatic mixed precision (`torch.cuda.amp`).
- Exposes a `VictimModel.query()` method that returns only
  **probability vectors** — no weights, no gradients.

### `threat_model.py`
- Defines query budgets: 1 000 / 5 000 / 10 000 / 50 000.
- Defines attacker knowledge levels (random vs. in-domain queries).
- `QueryGenerator` creates the right query data.
- `QueryExecutor` enforces the budget limit.

### `attack.py`
- **Random Query Attack** — random noise images.
- **Knockoff Nets Attack** — in-domain images + knowledge distillation.
- **Active Learning Attack** — entropy-based uncertainty sampling
  over multiple rounds.
- Two substitute architectures: `SmallCNN`, `MobileNetV3-Small`.

### `defenses.py`
- 7 defense classes, each callable: `defense_fn(probs) → probs`.
- Stateful defenses (Throttling, Adaptive Noise) have `.reset()`.
- `get_all_defenses()` returns 20 named parameter variants.

### `evaluate.py`
- Orchestrates the full experiment matrix.
- Computes 8 metrics per combination.
- Saves results to CSV and generates 7 chart types.

---

## Attacks

| # | Attack | Strategy | Knowledge |
|---|--------|----------|-----------|
| 1 | Random Query | Query with random noise images | Level 1 |
| 2 | Knockoff Nets | Query with CIFAR-10 test images + distillation | Level 2 |
| 3 | Active Learning | Uncertainty sampling over multiple rounds | Level 2 |

---

## Defenses

| # | Defense | Parameters Tested |
|---|---------|-------------------|
| 1 | Rounding | k = 1, 2, 3 |
| 2 | Gaussian Noise | σ = 0.01, 0.05, 0.10 |
| 3 | Laplace Noise | scale = 0.05, 0.10 |
| 4 | Top-K Output | k = 1, 2, 3 |
| 5 | Throttling | max = 250, 500, 1 000 |
| 6 | Prediction Poisoning | rate = 0.10, 0.30, 0.50 |
| 7 | Adaptive Noise | base + aggressive config |

---

## Evaluation Metrics

| Metric | Formula / Description |
|--------|----------------------|
| `victim_accuracy` | Top-1 accuracy of the victim on CIFAR-10 test |
| `substitute_accuracy` | Top-1 accuracy of the substitute |
| `fidelity` | % agreement between victim and substitute predictions |
| `protection_score` | 1 − (substitute_acc / victim_acc) |
| `utility_cost` | Accuracy drop caused by the defense |
| `query_efficiency` | Fidelity per 1 000 queries |
| `attack_roi` | Substitute accuracy / query budget |
| `defense_latency_ms` | Wall-clock overhead of the defense (ms) |

---

## Results & Charts

After running `evaluate.py`, check:

- **`experiments/results/experiment_results.csv`** — raw numbers.
- **`experiments/charts/`** — publication-quality plots:
  - `fidelity_per_defense.png` — bar chart
  - `protection_per_defense.png` — bar chart
  - `security_vs_utility.png` — scatter plot
  - `fidelity_vs_budget.png` — line chart
  - `heatmap_fidelity.png` — heatmap
  - `heatmap_substitute_accuracy.png` — heatmap
  - `heatmap_protection_score.png` — heatmap

---

## Reproducibility

- Fixed random seed: `torch.manual_seed(42)` in every module.
- Deterministic cuDNN: `cudnn.deterministic = True`.
- Pinned library versions in `requirements.txt`.

---

## License

This project is for academic research purposes (MSc dissertation).
