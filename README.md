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
| **Defenses** | Throttling, Prediction Poisoning, Adaptive Noise |

---

## System Requirements

| Item | Specification |
|------|--------------|
| Python | 3.9+ (developed under 3.9; also runs on 3.13) |
| Compute | Device-agnostic: CUDA GPU → Apple Silicon (MPS) → CPU |
| Notes | Originally targeted RTX 4070 / CUDA 12.1; results here produced on Apple MPS |

---

## Installation

```bash
# 1. Create and activate a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# 2. Install PyTorch
#    CUDA machine:
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
#    macOS (Apple Silicon) / CPU: just `pip install torch torchvision`

# 3. Install remaining dependencies
pip install -r requirements.txt
```

> The interactive dashboard runs on **Streamlit** (`streamlit run dashboard.py`).

---

## Project Structure

```
project/
├── victim.py            # Victim model training & black-box query API
├── threat_model.py      # Threat model, query budgets, knowledge levels
├── attack.py            # 3 attack strategies + substitute architectures
├── defenses.py          # 3 defense mechanisms (6 active parameter variants)
├── evaluate.py          # Full evaluation pipeline, CSV, charts
├── dashboard.py         # Interactive Streamlit dashboard
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

### Step 1 — Train victim model

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

### Step 4 — Interactive dashboard (optional)

```bash
streamlit run dashboard.py
```

Opens a local web UI to explore the victim, attacks, defenses, and results.

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
- Defines query budget tiers: 1 000 / 5 000 / 10 000 (the evaluation uses 1 000 and 5 000).
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
- 3 defense classes, each callable: `defense_fn(probs) → probs`.
- Stateful defenses (Throttling, Adaptive Noise) have `.reset()`.
- `get_all_defenses()` returns 7 named entries (baseline `none` + 6 active variants).

### `evaluate.py`
- Orchestrates the full experiment matrix.
- Computes 8 metrics per combination.
- Saves results to CSV and generates 5 chart types.

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
| 1 | Throttling | max = 250, 500, 1 000 |
| 2 | Prediction Poisoning | rate = 0.30, 0.50 |
| 3 | Adaptive Noise | aggressive config (base σ = 0.02, ×10 escalation after 500 queries) |

> Weaker defenses (rounding, fixed Gaussian/Laplace noise, top-K, low-rate poisoning,
> mild adaptive noise) were trialled but removed after empirical testing — they reduced
> fidelity by ≤10 pp while the attacker still succeeded.

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

- **`experiments/results/*.csv`** — raw numbers
  (`results_no_defense_resnet50.csv`, `results_with_defense_resnet50.csv`).
- **`experiments/charts/`** — generated plots:
  - `fidelity_comparison.png` — fidelity per defense (bar chart)
  - `protection_scores.png` — protection score per defense (bar chart)
  - `security_utility_tradeoff.png` — security vs. utility (scatter)
  - `fidelity_vs_query_budget.png` — fidelity vs. budget (line chart)
  - `attack_defense_heatmap.png` — fidelity heatmap (attack × defense)

---

## Reproducibility

- Fixed random seed (`42`) across Python, NumPy, and PyTorch in every module.
- Deterministic settings enabled (cuDNN deterministic on CUDA; device-agnostic elsewhere).
- Pinned library versions in `requirements.txt`.

---

## License

This project is for academic research purposes (MSc dissertation).
