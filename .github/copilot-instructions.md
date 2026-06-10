# Copilot Instructions — Model Extraction Defense Evaluation

## Project Overview

MSc dissertation framework evaluating **model extraction attacks** against deep learning classifiers and **output-perturbation defenses**, under a strict **black-box threat model**. The attacker only ever sees probability vectors — never weights, gradients, or architecture.

**Pipeline:** `data_loader.py` → `victim.py` → `threat_model.py` → `attack.py` → `defenses.py` → `evaluate.py`

## Architecture & Data Flow

- **`victim.py`** — Trains ResNet-50 (frozen backbone, trainable final layer; the factory also supports EfficientNet-B0 but the single-victim evaluation uses ResNet-50). Exposes `VictimModel.query(images) → probability vectors` as the only interface. Never leak weights or gradients through this boundary.
- **`threat_model.py`** — `ThreatConfig` dataclass defines every experiment run (budget, knowledge level, substitute arch). `QueryExecutor` enforces the budget limit and applies defenses via `defense_fn(probs)` before the attacker sees outputs.
- **`attack.py`** — Three attacks (`random_query`, `knockoff_nets`, `active_learning`) registered in `ATTACK_REGISTRY` dict. All return `(nn.Module, metadata_dict)`. Substitute training uses `soft_cross_entropy` (KL-divergence distillation at temperature=3.0).
- **`defenses.py`** — Seven defense classes, all callable `__call__(probs) → probs`. Stateful defenses (`ThrottlingDefense`, `AdaptiveNoiseDefense`) have `.reset()` — must be called between experiment runs. `get_all_defenses()` returns 20 named variants.
- **`evaluate.py`** — Orchestrates the full 4-phase pipeline (victim check → attacks without defense → attacks with defense → charts). Computes 8 metrics per combination. Saves CSV to `experiments/results/`, charts to `experiments/charts/`.

## Critical Conventions

- **Python 3.9.25 strict** — no walrus operator in complex expressions, no `match/case`, no `type` aliases from 3.10+. Use `typing.Optional`, `typing.Tuple`, etc.
- **Reproducibility** — every module sets `torch.manual_seed(42)`, `cudnn.deterministic = True`, `cudnn.benchmark = False`. Preserve these when adding new modules.
- **AMP everywhere** — all forward passes use `torch.cuda.amp.autocast()` and `GradScaler`. OOM errors are caught and handled with `torch.cuda.empty_cache()` + batch skip.
- **Logging pattern** — every module uses `logging.getLogger(__name__)` with format `"%(asctime)s | %(name)s | %(levelname)s | %(message)s"`. No `print()` except in `print_final_conclusion`.

## Key Patterns

- **Defense as callable:** New defenses must be a class with `__call__(self, probs: torch.Tensor) -> torch.Tensor` that returns renormalised probabilities. Use `_renormalise()` from `defenses.py`. Add to `get_all_defenses()` registry.
- **Attack registration:** New attacks must follow signature `fn(victim, config, defense_fn, sub_epochs) -> (nn.Module, Dict)` and be added to `ATTACK_REGISTRY` in `attack.py`.
- **Metrics dict:** `compute_all_metrics()` returns a fixed 8-key dict. If adding metrics, update this function, `save_results_csv`, and chart functions.
- **Checkpoints:** Saved to `models/victim_<arch>.pth`. Training is skipped if checkpoint exists (`os.path.exists` check in `main()`).

## Running the Project

```bash
# Step 1: Train victims (skips if models/*.pth exist)
python victim.py
# Step 2: Full evaluation (attacks × defenses × budgets × substitutes)
python evaluate.py
# Smoke-test individual modules:
python data_loader.py    # verify CIFAR-10 pipeline
python threat_model.py   # print all threat configs
python attack.py         # quick attack demo (needs victim checkpoint)
python defenses.py       # test all 20 defense variants
```

## Dependencies & Environment

- PyTorch 2.1.0 + torchvision 0.16.0 with CUDA 12.1 — install separately before `pip install -r requirements.txt`
- Target GPU: RTX 4070 (12 GB VRAM), batch size 64 tuned to this
- All images resized to 224×224 with ImageNet normalisation (`IMAGENET_MEAN`, `IMAGENET_STD` in `data_loader.py`)
- CIFAR-10 auto-downloads to `./data/cifar10/` on first run
