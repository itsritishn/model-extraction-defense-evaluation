# Threat-Aware Analysis of Model Extraction Attacks in Machine Learning Pipeline

## Complete Project Documentation

**Author:** MSc Dissertation Project  
**Date:** March 2026  
**MSc Advanced Computer Science (Data Analytics)**

---

## Table of Contents

1. [Project Goal](#1-project-goal)
2. [Background — What is Model Extraction?](#2-background--what-is-model-extraction)
3. [Project Architecture Overview](#3-project-architecture-overview)
4. [File-by-File Explanation](#4-file-by-file-explanation)
   - 4.1 [data_loader.py — Data Loading](#41-data_loaderpy--data-loading)
   - 4.2 [victim.py — Victim Model Training & Black-Box API](#42-victimpy--victim-model-training--black-box-api)
   - 4.3 [threat_model.py — Threat Model Definition](#43-threat_modelpy--threat-model-definition)
   - 4.4 [attack.py — Attack Implementations](#44-attackpy--attack-implementations)
   - 4.5 [defenses.py — Defense Mechanisms](#45-defensespy--defense-mechanisms)
   - 4.6 [evaluate.py — Evaluation Pipeline](#46-evaluatepy--evaluation-pipeline)
5. [How the Complete Pipeline Works](#5-how-the-complete-pipeline-works)
6. [Metrics Explained](#6-metrics-explained)
7. [Charts Explained](#7-charts-explained)
8. [Key Concepts for Viva](#8-key-concepts-for-viva)
9. [Technical Setup](#9-technical-setup)

---

## 1. Project Goal

**In one sentence:** This project tests whether an attacker can copy a machine learning model by only asking it questions through an API, and whether defensive techniques can stop them.

**In more detail:**

Machine learning models are expensive to build. Companies like Google, Amazon, and Microsoft deploy them as paid APIs — you send an image, they return a prediction. The problem is: an attacker can send thousands of images, collect the predictions, and train their own "copycat" model that behaves almost identically to the original. This is called **model extraction** or **model stealing**.

This project:
1. Trains a high-quality **victim model** (ResNet-50) on CIFAR-10
2. Implements **3 different attack strategies** that try to steal the model
3. Implements **7 defense mechanisms** (20 configurations) that try to prevent stealing
4. Runs **all combinations** and measures how effective each attack and defense is
5. Produces **charts and tables** showing the results

---

## 2. Background — What is Model Extraction?

### The Real-World Scenario

```
┌──────────────────────────────────────────────────────┐
│                    MODEL OWNER                        │
│                                                      │
│   Spent £millions training a model (ResNet-50)       │
│   Deployed it as a paid API:                         │
│                                                      │
│   POST /predict  →  { "cat": 0.82, "dog": 0.04 }   │
│                                                      │
└──────────────────────┬───────────────────────────────┘
                       │  API (bl ack-box)
                       │  Only probabilities come out
                       │  No weights, no architecture
                       ▼
┌──────────────────────────────────────────────────────┐
│                     ATTACKER                          │
│                                                      │
│   Sends thousands of images through the API          │
│   Collects all the probability responses             │
│   Trains their own cheap model (SmallCNN) to mimic   │
│   the victim's answers                               │
│                                                      │
│   Result: A "copycat" model that gives similar       │
│   predictions WITHOUT paying for the original        │
│                                                      │
└──────────────────────────────────────────────────────┘
```

### Key Point — The Victim's Weights Are NEVER Stolen

The attacker **never** gets the actual model files, weights, or architecture. They only observe the input-output behaviour. It is like tasting a chef's dishes through a window and writing your own recipe — you never enter the kitchen.

### Why This Matters

| Concern | Explanation |
|---------|-------------|
| **Intellectual Property Theft** | Someone spent millions training a model. An attacker copies it with just API queries for a fraction of the cost |
| **Revenue Loss** | The attacker can now offer the same service without paying licensing fees |
| **Security Risk** | Once you have a local copy, you can find adversarial examples to attack the original model |
| **Privacy Risk** | The substitute model may reveal information about the victim's training data |

---

## 3. Project Architecture Overview

### File Structure

```
model-extraction-defense-evaluation/
├── data_loader.py      →  Loads CIFAR-10 images
├── victim.py           →  Trains & serves the victim model (black-box API)
├── threat_model.py     →  Defines attacker rules (budgets, knowledge levels)
├── attack.py           →  3 attack strategies + 2 substitute architectures
├── defenses.py         →  7 defense mechanisms (20 configurations)
├── evaluate.py         →  Runs all experiments, computes metrics, makes charts
├── requirements.txt    →  Python package dependencies
├── README.md           →  Project README
├── models/
│   └── victim_resnet50.pth       →  Trained ResNet-50 weights (82.65% accuracy)
├── data/cifar10/                 →  CIFAR-10 dataset (auto-downloaded)
├── experiments/
│   ├── results/
│   │   ├── results_no_defense.csv    →  Phase 2 results (attacks without defence)
│   │   └── results_with_defense.csv  →  Phase 3 results (attacks with defences)
│   └── charts/
│       ├── fidelity_comparison.png
│       ├── protection_scores.png
│       ├── security_utility_tradeoff.png
│       ├── fidelity_vs_query_budget.png
│       └── attack_defense_heatmap.png
└── venv/                         →  Python virtual environment
```

### How the Files Connect

```
data_loader.py  ──→  Provides images to ALL other modules
      │
      ▼
victim.py  ──→  Trains the real model, wraps it in a black-box API
      │
      ▼
threat_model.py  ──→  Defines rules: budgets, knowledge, generates queries
      │
      ▼
attack.py  ──→  Queries the victim, trains a copycat substitute model
      │
      ▼
defenses.py  ──→  Modifies victim outputs to corrupt attacker's data
      │
      ▼
evaluate.py  ──→  Runs all combinations, measures everything, makes charts
```

---

## 4. File-by-File Explanation

---

### 4.1 `data_loader.py` — Data Loading

**Purpose:** Downloads and prepares the CIFAR-10 dataset for use by all other modules.

#### What is CIFAR-10?

CIFAR-10 is a standard benchmark dataset containing 60,000 colour images (32×32 pixels) in 10 classes:

| Class | Examples |
|-------|----------|
| airplane | Photos of airplanes |
| automobile | Cars, trucks on roads |
| bird | Various bird species |
| cat | Domestic cats |
| deer | Deer in nature |
| dog | Dogs of various breeds |
| frog | Frogs |
| horse | Horses |
| ship | Ships, boats |
| truck | Trucks |

- **50,000 training images** — used to train the victim model
- **10,000 test images** — used to evaluate accuracy and used by the attacker as query images

#### What This File Does Step by Step

**Step 1 — Download:** If CIFAR-10 isn't already on disk, it downloads it from the internet and saves it into `./data/cifar10/`.

**Step 2 — Resize:** CIFAR-10 images are tiny (32×32 pixels), but our victim model (ResNet-50) was designed for 224×224 pixel images. So every image is resized from 32×32 → 224×224.

**Step 3 — Normalise:** Pixel values are adjusted using ImageNet statistics:
- Mean: (0.485, 0.456, 0.406) for the three colour channels (Red, Green, Blue)
- Standard Deviation: (0.229, 0.224, 0.225)

This normalisation makes the images "look right" to models that were pretrained on ImageNet.

**Step 4 — Data Augmentation (training only):**
- Random horizontal flip — randomly mirrors the image left-to-right (50% chance)
- Random crop with padding — shifts the image slightly in random directions

These variations help the model learn more robustly by seeing slightly different versions of each image.

**Step 5 — Create DataLoaders:** Packages the images into batches of 64 for efficient GPU processing.

#### Key Functions

| Function | What It Does |
|----------|--------------|
| `get_train_transforms()` | Creates the transformation pipeline for training: resize → flip → crop → normalise |
| `get_test_transforms()` | Creates the transformation pipeline for testing: resize → normalise (no augmentation) |
| `get_cifar10_datasets()` | Downloads CIFAR-10 and returns train + test datasets with transforms applied |
| `get_data_loaders()` | Wraps datasets in PyTorch DataLoaders (feeds batches of 64 images to the model) |
| `get_subset_loader()` | Creates a DataLoader with only a subset of images — used by the attacker to respect query budgets |

#### Why It Matters

Every other file depends on this. The victim trains on this data. The attacker queries with images from this data. The evaluator measures accuracy on this data.

---

### 4.2 `victim.py` — Victim Model Training & Black-Box API

**Purpose:** Trains a powerful image classifier and wraps it behind a black-box interface that only returns probability outputs.

#### The Victim Model

**ResNet-50:**
- A deep neural network with 50 layers (Residual Network)
- Originally trained on ImageNet (1.4 million images, 1000 classes)
- We use "transfer learning" — take the pretrained model and adapt it for CIFAR-10
- Total parameters: ~23.5 million (but we only train 20,490 of them)
- Final test accuracy: **82.65%**

*(The model factory also supports EfficientNet-B0, retained for reference, but the
single-victim evaluation uses ResNet-50 only.)*

#### How Training Works (Transfer Learning)

Transfer learning means we don't train from scratch. Instead:

1. **Start with a pretrained model** — ResNet-50 already knows how to recognise edges, textures, shapes, and objects from ImageNet training
2. **Freeze all layers** — Lock down all the learned knowledge so it doesn't get overwritten
3. **Replace the final layer** — Swap the 1000-class output head for a new 10-class head (since CIFAR-10 has 10 classes)
4. **Train only the final layer** — For 15 epochs, teach just the final layer to map the frozen features to CIFAR-10 classes

This is much faster than training from scratch and gives better results with limited data.

#### Training Details

| Setting | Value | Why |
|---------|-------|-----|
| Optimiser | Adam | Adaptive learning rates, good default |
| Learning rate | 0.001 | Standard starting point |
| Scheduler | Cosine Annealing | Gradually reduces learning rate over epochs |
| Epochs | 15 | Enough for fine-tuning (not training from scratch) |
| Batch size | 64 | Fits comfortably in RTX 4070 (12GB VRAM) |
| Mixed precision (AMP) | Enabled | Speeds up training by using 16-bit floats where possible |
| Dropout | 0.3 (30%) | Prevents overfitting by randomly disabling 30% of neurons |
| Checkpoint saving | Best test accuracy | Saves weights whenever test accuracy improves |

#### The VictimModel Class — The Black-Box API

This is the **most important class** in the entire project. It wraps the trained model and exposes ONLY a `.query()` method:

```
Attacker sends: batch of images (e.g., 64 cat/dog photos)
                        │
                        ▼
            ┌───────────────────────┐
            │   VictimModel.query() │
            │                       │
            │   1. Forward pass     │
            │   2. Softmax          │
            │   3. Return probs     │
            └───────────┬───────────┘
                        │
                        ▼
Attacker receives: probability vectors
   e.g., [airplane:0.02, car:0.01, bird:0.05, cat:0.82, ...]
```

**What the attacker CANNOT access:**
- ✗ Model weights (the learned parameters)
- ✗ Model architecture (how many layers, what types)
- ✗ Gradients (how the model would change with different training)
- ✗ Intermediate features (what the model "sees" at hidden layers)
- ✗ Training data (what images the model was trained on)
Q1 ✅ Perfect

No changes needed. Clean, concise, professional.
Q2 ✅ Very Strong

CoreDNS 5G justification is now excellent. 3GPP reference added. FRR justification solid.

One tiny fix — this line still says "32 MB" inconsistently:

    "FRR requires only 32 MB RAM"

Change to:

    "FRR requires only 32Mi RAM"

Keep units consistent throughout (Mi not MB).
Q3 ✅ Strong

Version mismatch acknowledged. FRR verification improved. Challenges documented well.

One tiny fix — this line is slightly awkward:

    "manually managing /etc/rancher/k3s/k3s.yaml permissions via chmod 644 for local kubectl access"

Change to:

    "manually copying /etc/rancher/k3s/k3s.yaml to ~/.kube/config and applying chmod 644 to enable local kubectl access without sudo."

This is more technically precise and shows deeper understanding.
Q4 ✅ Strong

Timing resolution explanation added correctly. Millisecond precision explained well.

One small structural issue — this part reads oddly because the timing explanation is buried inside the DNS test description:

Currently:

    "The total execution batch time was recorded using epoch timestamps. DNS batch timing used second-resolution epoch timestamps. Results of '0 seconds' represent..."

You say "epoch timestamps" twice in a row. Fix by merging:

    "The total execution batch time was recorded using second-resolution epoch timestamps (date +%s). Results of '0 seconds' represent sub-second completions under 1000ms, while FRR benchmarks used nanosecond-precision timestamps (date +%s%N) converted to milliseconds, providing significantly higher measurement fidelity."

Q5 ✅ Excellent

This is now your strongest section. All three missing pieces added correctly:

    ✅ Standard deviation analysis
    ✅ Missing cloud metrics explained
    ✅ CoreDNS vs FRR methodology difference explained

One small addition — your conclusion paragraph is good but could end stronger connecting to 6G. Change the last sentence from:

    "pushing lightweight infrastructure (like K3s) closer to the hardware metal ensures lower latencies for control plane functions compared to thickly virtualized, data-center-style cloud orchestration."

To:

    "pushing lightweight infrastructure (like K3s) closer to the hardware metal ensures lower latencies for control plane functions compared to thickly virtualized, data-center-style cloud orchestration. As networks evolve toward 6G with sub-millisecond latency requirements, the architectural efficiency demonstrated by K3s becomes not merely advantageous but essential."

This directly addresses the 6G mention in your brief and will impress the marker.
**What the attacker CAN access:**
- ✓ The probability vector for each image they send (10 numbers summing to 1.0)

#### Key Functions

| Function | What It Does |
|----------|--------------|
| `build_victim_model(architecture)` | Creates a ResNet-50 with a new 10-class head (factory also supports EfficientNet-B0) |
| `train_victim(model, ...)` | Fine-tunes the model for 15 epochs with AMP, saves best checkpoint |
| `evaluate_model(model, data_loader)` | Computes top-1 accuracy on a test set |
| `VictimModel(model)` | Wraps the model — only exposes `.query()` |
| `VictimModel.query(images)` | THE BLACK-BOX API: images → probabilities |
| `load_victim(arch, checkpoint)` | Loads a saved model and wraps it in VictimModel |
| `main()` | Trains both models (skips if checkpoints already exist) |

---

### 4.3 `threat_model.py` — Threat Model Definition

**Purpose:** Formally defines what the attacker is allowed to do. This ensures fair, reproducible experiments.

#### Query Budgets

The attacker has a limited number of API calls (like a credit limit):

| Budget Tier | Number of Queries | Real-World Analogy |
|-------------|-------------------|-------------------|
| LOW | 1,000 | Free trial — very limited |
| MEDIUM | 5,000 | Standard subscription |
| HIGH | 10,000 | Power user |
| VERY_HIGH | 50,000 | Determined attacker with resources |

More queries = more data for the attacker = potentially better copycat.

#### Attacker Knowledge Levels

How much does the attacker know about the victim?

| Level | Knowledge | How They Query | Effectiveness |
|-------|-----------|----------------|---------------|
| LEVEL_1 | **Nothing** — no domain knowledge | Sends random noise images (meaningless static) | Weak — like asking gibberish questions |
| LEVEL_2 | **Domain knowledge** — knows the victim classifies natural images | Sends real CIFAR-10 test images | Strong — like asking relevant questions |

#### ThreatConfig — Settings for One Experiment

Every experiment is defined by a `ThreatConfig` object:

```
ThreatConfig(
    budget = 5000,                        # Can ask 5000 questions
    knowledge = AttackerKnowledge.LEVEL_2, # Has real images
    substitute_arch = "small_cnn",         # Will build a SmallCNN
    victim_arch = "resnet50",              # Attacking a ResNet-50
    batch_size = 64,                       # Send 64 images at a time
    image_size = 224,                      # Images are 224×224
)
```

#### QueryGenerator — Creates the Attacker's Query Images

- **Level 1:** Generates random noise tensors. Each "image" is random static — no meaningful content. The victim still returns probabilities for these, but the information is less useful.
- **Level 2:** Takes real CIFAR-10 test images. If the budget is smaller than the test set (10,000), it randomly samples. If larger, it wraps around.

#### QueryExecutor — Sends Queries and Enforces Rules

This class ensures the attacker plays fair:
1. Takes images from the QueryGenerator
2. Sends them to `VictimModel.query()` one batch at a time
3. **Counts every query** — once the budget is exhausted, it stops
4. **Applies the defense** (if any) to the victim's output before the attacker sees it
5. Returns all (image, modified_probability) pairs

#### Key Functions

| Function | What It Does |
|----------|--------------|
| `QueryBudget` | Enum of 4 budget tiers (1K, 5K, 10K, 50K) |
| `AttackerKnowledge` | Enum of 2 knowledge levels |
| `ThreatConfig` | Dataclass holding all settings for one experiment |
| `QueryGenerator.get_query_loader()` | Returns a DataLoader of query images (random noise or real images) |
| `QueryExecutor.execute()` | Sends all queries, enforces budget, applies defense, collects results |
| `get_all_threat_configs()` | Generates the full cross-product of all possible configurations |

---

### 4.4 `attack.py` — Attack Implementations

**Purpose:** Implements 3 attack strategies and 2 substitute model architectures.

#### The Substitute Models (What the Attacker Builds)

**SmallCNN — A tiny 3-layer CNN:**

```
Input Image (3×224×224)
    │
    ▼
Conv Layer 1: 3→32 filters, BatchNorm, ReLU, MaxPool
    │  (learns edges and simple patterns)
    ▼
Conv Layer 2: 32→64 filters, BatchNorm, ReLU, MaxPool
    │  (learns textures and shapes)
    ▼
Conv Layer 3: 64→128 filters, BatchNorm, ReLU, AdaptivePool
    │  (learns object parts)
    ▼
Fully Connected: 128 → 10 classes
    │
    ▼
Output: [airplane: 0.05, car: 0.03, ..., cat: 0.78, ...]
```

- Very small (~150K parameters vs victim's ~23M)
- Fast to train
- The attacker doesn't need the same architecture as the victim

**MobileNetV3-Small:**
- A modern efficient architecture designed for mobile phones
- Bigger than SmallCNN but still much smaller than ResNet-50
- Tests whether a better substitute architecture helps the attacker

#### How the Attacker Trains the Substitute — Knowledge Distillation

Normal training: `model learns from true labels → "this is a cat"`

Attacker's training: `model learns from victim's probability outputs → "victim said [cat:0.82, dog:0.04, bird:0.05, ...]"`

This is called **knowledge distillation** and uses a special loss function called `soft_cross_entropy`:

1. Take the substitute's raw outputs (logits)
2. Divide by **temperature T=3.0** (this "softens" the probabilities)
3. Do the same to the victim's outputs
4. Compute KL-Divergence between them
5. Multiply by T² to keep gradients at the right scale

**Why temperature matters:** Without temperature, the victim might say `[cat: 0.99, dog: 0.01, ...]`. With temperature=3, it becomes `[cat: 0.60, dog: 0.15, bird: 0.10, ...]`. The softened version reveals **relationships between classes** — the model thinks this image is a bit like a dog too. This "dark knowledge" helps the substitute learn much better.

#### Attack 1: Random Query Attack

```
Step 1: Generate random noise images (Level 1 knowledge)
        → Images look like TV static, no meaningful content
Step 2: Send ALL of them to victim.query()
Step 3: Collect (noise_image, probability_output) pairs
Step 4: Train SmallCNN on these pairs for 30 epochs
```

**Effectiveness:** Weakest attack. Random noise gives the victim confusing inputs, and the probability outputs are less informative. Still, the substitute can learn something because even noise images produce structured probability patterns.

#### Attack 2: Knockoff Nets Attack

```
Step 1: Take real CIFAR-10 test images (Level 2 knowledge)
Step 2: Send ALL of them to victim.query()
Step 3: Collect (real_image, probability_output) pairs
Step 4: Train SmallCNN on these pairs for 30 epochs
```

**Effectiveness:** Much stronger. Real images produce confident, informative probability outputs from the victim. The substitute sees what the victim truly "thinks" about real-world images.

**Based on:** "Knockoff Nets: Stealing Functionality of Black-Box Models" by Orekondy, Schiele, Fritz (CVPR 2019)

#### Attack 3: Active Learning Attack

This is the **smartest** attack. Instead of querying blindly, it strategically picks the most informative images:

```
Round 0 (Seed): Use 30% of budget on random images
    → Train a preliminary substitute
    
Round 1-5 (Active Selection):
    → Use the substitute to score ALL remaining images
    → Find images where substitute is MOST UNCERTAIN (highest entropy)
    → These are the images where more information is needed
    → Query the victim with ONLY these uncertain images
    → Add new (image, probability) pairs to training set
    → Retrain the substitute
    → Repeat

Final: Train the substitute on ALL collected data for 30 epochs
```

**Why uncertainty sampling works:** If the substitute already confidently classifies an image, querying the victim for that image adds little new information. But if the substitute is uncertain (e.g., 50% cat, 50% dog), the victim's answer resolves that uncertainty. This makes each query maximally informative.

**Effectiveness:** Strongest attack — gets the most information per query.

#### Key Functions

| Function | What It Does |
|----------|--------------|
| `SmallCNN` | The small copycat model (3 conv layers) |
| `build_substitute_model(arch)` | Creates SmallCNN or MobileNetV3-Small |
| `soft_cross_entropy(logits, soft_targets, T)` | Knowledge distillation loss function |
| `train_substitute(model, images, labels)` | Train the copycat on stolen data for 30 epochs |
| `random_query_attack(victim, config)` | Attack 1 — random noise queries |
| `knockoff_attack(victim, config)` | Attack 2 — real image queries |
| `active_learning_attack(victim, config)` | Attack 3 — smart uncertainty-based queries |
| `run_attack(name, victim, config)` | Dispatcher — runs whichever attack you name |

---

### 4.5 `defenses.py` — Defense Mechanisms

**Purpose:** Implements 7 defense techniques that modify the victim's probability outputs before the attacker sees them, making the stolen data less useful.

All defenses are **callable objects** — they take a probability tensor in and return a modified probability tensor out. They slot into the pipeline via the `defense_fn` parameter.

#### Defense 1: Rounding

**Idea:** Round probabilities to fewer decimal places so the attacker gets less precise information.

| Configuration | Before | After | Information Loss |
|---------------|--------|-------|-----------------|
| k=1 (1 decimal) | 0.8234 | 0.8 | High — only 10 possible values per class |
| k=2 (2 decimals) | 0.8234 | 0.82 | Medium |
| k=3 (3 decimals) | 0.8234 | 0.823 | Low |

After rounding, probabilities are **renormalised** to sum to 1.0.

#### Defense 2: Gaussian Noise

**Idea:** Add random noise drawn from a Gaussian (bell curve) distribution to every probability value.

| Configuration | Noise Level | Effect |
|---------------|-------------|--------|
| sigma=0.01 | Very small | Barely noticeable — probabilities shift by ~1% |
| sigma=0.05 | Medium | Moderate corruption — probabilities shift by ~5% |
| sigma=0.10 | Large | Heavy corruption — probabilities shift by ~10% |

After adding noise, values are clamped to be non-negative and renormalised to sum to 1.0.

#### Defense 3: Laplace Noise

**Idea:** Same concept as Gaussian noise, but uses the Laplace distribution instead. The Laplace distribution has **heavier tails**, meaning occasional large noise spikes that can significantly distort individual probabilities.

| Configuration | Scale |
|---------------|-------|
| scale=0.05 | Moderate |
| scale=0.10 | Heavy |

#### Defense 4: Top-K Output

**Idea:** Only return the top K class probabilities; set all others to zero.

| Configuration | What the Attacker Sees | Information Lost |
|---------------|----------------------|-----------------|
| k=1 | Only the top class → `[0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0]` | All soft information — only hard label remains |
| k=2 | Top 2 classes visible | Most of the soft information |
| k=3 | Top 3 classes visible | Some soft information |

**Top-K=1 is one of the strongest defenses** because it eliminates all the "dark knowledge" from soft probabilities. The attacker only gets hard labels (equivalent to just knowing the predicted class), which carries much less information for knowledge distillation.

#### Defense 5: Throttling (Query-Rate Limit)

**Idea:** Hard-cut the attacker after N queries. Once the limit is reached, return a **uniform distribution** (every class equally likely = `[0.1, 0.1, 0.1, ...]`) which contains zero useful information.

| Configuration | Queries Allowed | After Limit |
|---------------|-----------------|-------------|
| max_queries=250 | Only 250 real answers | All subsequent queries return uniform noise |
| max_queries=500 | 500 real answers | Same |
| max_queries=1000 | 1000 real answers | Same |

This is very effective because it directly limits how much data the attacker can collect, regardless of their total budget.

#### Defense 6: Prediction Poisoning

**Idea:** Randomly give **wrong answers**. For a fraction of queries, the probability vector is randomly shuffled (the probabilities are correct but assigned to wrong classes).

| Configuration | Effect |
|---------------|--------|
| poison_rate=0.10 | 10% of responses are deliberately wrong |
| poison_rate=0.30 | 30% are wrong — nearly 1 in 3 |
| poison_rate=0.50 | 50% are wrong — half the data is corrupted |

The attacker cannot tell which responses are poisoned, so the noise corrupts their training data.

#### Defense 7: Adaptive Noise

**Idea:** A "smart" defense that monitors query patterns. Normal users ask occasional questions; attackers send thousands in rapid succession.

- **Normal mode:** Adds small Gaussian noise (sigma = 0.01)
- **Detection:** Tracks query timestamps over a sliding window
- **Escalation:** If queries arrive faster than `rate_threshold` per second, noise is multiplied by `escalation_factor` (5× or 10×)

| Configuration | Base Noise | Escalated Noise | Trigger |
|---------------|-----------|-----------------|---------|
| adaptive_base | sigma=0.01 | sigma=0.05 | >50 queries/sec |
| adaptive_aggressive | sigma=0.02 | sigma=0.20 | >30 queries/sec |

#### Complete Defense Registry (20 Configurations)

| # | Name | Type | Key Parameter |
|---|------|------|---------------|
| 1 | none | No defense | Baseline |
| 2 | rounding_k1 | Rounding | 1 decimal place |
| 3 | rounding_k2 | Rounding | 2 decimal places |
| 4 | rounding_k3 | Rounding | 3 decimal places |
| 5 | gaussian_0.01 | Gaussian Noise | sigma=0.01 |
| 6 | gaussian_0.05 | Gaussian Noise | sigma=0.05 |
| 7 | gaussian_0.10 | Gaussian Noise | sigma=0.10 |
| 8 | laplace_0.05 | Laplace Noise | scale=0.05 |
| 9 | laplace_0.10 | Laplace Noise | scale=0.10 |
| 10 | topk_1 | Top-K | k=1 (only top class) |
| 11 | topk_2 | Top-K | k=2 |
| 12 | topk_3 | Top-K | k=3 |
| 13 | throttle_250 | Throttling | 250 queries max |
| 14 | throttle_500 | Throttling | 500 queries max |
| 15 | throttle_1000 | Throttling | 1000 queries max |
| 16 | poison_0.10 | Poisoning | 10% wrong answers |
| 17 | poison_0.30 | Poisoning | 30% wrong answers |
| 18 | poison_0.50 | Poisoning | 50% wrong answers |
| 19 | adaptive_base | Adaptive Noise | Mild escalation |
| 20 | adaptive_aggressive | Adaptive Noise | Aggressive escalation |

---

### 4.6 `evaluate.py` — Evaluation Pipeline

**Purpose:** The main orchestrator. Runs every (attack × defense × budget × substitute) combination, computes 8 metrics, saves results to CSV, and generates 7 charts.

#### What It Does Step by Step

The pipeline runs in **4 clear phases:**

```
PHASE 1: Check that victim model is trained (victim.py must be run first)
PHASE 2: Run ALL attacks WITHOUT any defence → save to results_no_defense.csv
PHASE 3: Run ALL attacks WITH every defence → save to results_with_defense.csv
PHASE 4: Combine results, generate 5 charts, print final conclusion
```

Phase 2 runs: 3 attacks × 1 (no defence) × 3 budgets × 2 substitutes = **18 experiments**
Phase 3 runs: 3 attacks × 19 defences × 3 budgets × 2 substitutes = **342 experiments**
Total: **360 experiments**

#### The 8 Metrics (see Section 6 for full details)

| Metric | Measures |
|--------|----------|
| victim_accuracy | How good the original model is |
| substitute_accuracy | How good the copycat is |
| fidelity | How often victim and copycat agree |
| protection_score | How much the defense prevents copying |
| utility_cost | How much the defense hurts the original model |
| query_efficiency | Fidelity gained per 1000 queries |
| attack_roi | Accuracy gained per query |
| defense_latency_ms | Speed overhead of the defense |

#### The 5 Charts (see Section 7 for full details)

1. Attack Fidelity With and Without Defences (bar chart)
2. Protection Score Per Defence (bar chart)
3. Security vs Utility Tradeoff (scatter plot)
4. Attack Success vs Query Budget (line chart)
5. Fidelity Heatmap: Attack vs Defence (heatmap)

---

## 5. How the Complete Pipeline Works

### Visual Flow

```
                    ┌─────────────────────┐
                    │   CIFAR-10 Dataset   │
                    │  (data_loader.py)    │
                    └─────────┬───────────┘
                              │
                    ┌─────────▼───────────┐
                    │   Train Victim       │
                    │   (victim.py)        │
                    │                      │
                    │   ResNet-50: 82.65%  │
                    └─────────┬───────────┘
                              │
                    ┌─────────▼───────────┐
                    │   Black-Box API      │
                    │   victim.query()     │
                    │   Images → Probs     │
                    └─────────┬───────────┘
                              │
              ┌───────────────┼───────────────┐
              │               │               │
    ┌─────────▼────┐  ┌──────▼─────┐  ┌──────▼──────┐
    │Defense applied│  │   No       │  │Defense      │
    │Rounding/Noise│  │   Defense   │  │Throttling   │
    │Top-K/Poison  │  │   (none)   │  │Adaptive     │
    └─────────┬────┘  └──────┬─────┘  └──────┬──────┘
              │               │               │
              └───────────────┼───────────────┘
                              │
                    ┌─────────▼───────────┐
                    │   Attacker receives  │
                    │   (image, probs)     │
                    │   pairs              │
                    └─────────┬───────────┘
                              │
              ┌───────────────┼───────────────┐
              │               │               │
    ┌─────────▼────┐  ┌──────▼─────┐  ┌──────▼──────┐
    │Random Query  │  │Knockoff    │  │Active       │
    │Attack        │  │Nets Attack │  │Learning     │
    │(noise imgs)  │  │(real imgs) │  │(smart imgs) │
    └─────────┬────┘  └──────┬─────┘  └──────┬──────┘
              │               │               │
              └───────────────┼───────────────┘
                              │
                    ┌─────────▼───────────┐
                    │   Train Substitute   │
                    │   (Knowledge         │
                    │    Distillation)     │
                    │                      │
                    │   SmallCNN or        │
                    │   MobileNetV3        │
                    └─────────┬───────────┘
                              │
                    ┌─────────▼───────────┐
                    │   Evaluate           │
                    │   (evaluate.py)      │
                    │                      │
                    │   8 metrics          │
                    │   2 CSV results      │
                    │   5 charts           │
                    └─────────────────────┘
```

### Concrete Example — One Experiment

**Settings:** Knockoff attack, Gaussian noise (sigma=0.05), budget=5000, SmallCNN substitute

1. Load victim ResNet-50 from `models/victim_resnet50.pth` (82.65% accuracy)
2. Create GaussianNoiseDefense(sigma=0.05)
3. Attacker picks 5000 random CIFAR-10 test images
4. Sends them to victim → gets back 5000 probability vectors
5. **Defense modifies each probability vector** — adds Gaussian noise
6. Attacker receives 5000 (image, noisy_probability) pairs
7. Attacker creates a SmallCNN (~150K parameters)
8. Trains SmallCNN on those 5000 pairs for 30 epochs using knowledge distillation
9. **Measure results:**
   - Does the SmallCNN give the same predictions as the victim? (fidelity)
   - How accurate is the SmallCNN on its own? (substitute accuracy)
   - How well did the noise defense protect the model? (protection score)

---

## 6. Metrics Explained

### Metric 1: Victim Accuracy

**What:** The victim model's top-1 accuracy on the CIFAR-10 test set.

**Formula:** $\text{Victim Accuracy} = \frac{\text{Correctly classified test images}}{\text{Total test images}} \times 100$

**Value in our experiments:** 82.65% (ResNet-50)

**Interpretation:** This is the baseline. It tells us how good the model being protected is.

### Metric 2: Substitute Accuracy

**What:** The copycat model's top-1 accuracy on the CIFAR-10 test set (using true labels).

**Formula:** Same as victim accuracy, but for the substitute model.

**Interpretation:** Higher = the attacker built a more useful copy. If substitute accuracy is close to victim accuracy, the attack was very successful.

### Metric 3: Fidelity

**What:** The percentage of test images where the victim and substitute predict the **same class**.

**Formula:** $\text{Fidelity} = \frac{\text{Images where victim and substitute agree}}{\text{Total test images}} \times 100$

**Interpretation:** This is the key metric. 100% fidelity means the copycat perfectly mimics the victim. Even if both models are wrong on an image, as long as they agree, fidelity counts it.

### Metric 4: Protection Score

**What:** How much worse is the substitute compared to the victim (scaled 0–1).

**Formula:** $\text{Protection Score} = 1 - \frac{\text{Substitute Accuracy}}{\text{Victim Accuracy}}$

**Interpretation:**
- 1.0 = Perfect protection (substitute is useless)
- 0.0 = No protection (substitute matches the victim exactly)
- 0.5 = Substitute is half as accurate as the victim

### Metric 5: Utility Cost

**What:** How much accuracy the victim loses due to the defense.

**Formula:** $\text{Utility Cost} = \text{Victim Accuracy (no defense)} - \text{Victim Accuracy (with defense)}$

**Interpretation:** Defenses that add noise to outputs don't change the victim's actual accuracy, so utility cost is 0 for most defenses. But some defenses (like aggressive noise) could theoretically affect utility for legitimate users.

### Metric 6: Query Efficiency

**What:** How much fidelity the attacker gains per 1000 queries.

**Formula:** $\text{Query Efficiency} = \frac{\text{Fidelity}}{\text{Query Budget}} \times 1000$

**Interpretation:** Higher = the attacker is efficient. Measures how "cost-effective" each query is.

### Metric 7: Attack ROI (Return on Investment)

**What:** Substitute accuracy gained per query.

**Formula:** $\text{Attack ROI} = \frac{\text{Substitute Accuracy}}{\text{Query Budget}}$

**Interpretation:** Higher = better return per query for the attacker.

### Metric 8: Defense Latency

**What:** How long (in milliseconds) the defense function takes to process one batch.

**Interpretation:** If the defense is too slow, it degrades API response time for all users, not just attackers.

---

## 7. Charts Explained

### Chart 1: Attack Fidelity With and Without Defences (Bar Chart)
- **X-axis:** Defense name
- **Y-axis:** Average fidelity (%)
- **What it shows:** Which defenses are best at reducing fidelity (preventing the copycat from matching the victim)
- **Good defenses** will have LOW bars (low fidelity = poor copy)

### Chart 2: Protection Score Per Defence (Bar Chart)
- **X-axis:** Defense name
- **Y-axis:** Average protection score (0–1)
- **What it shows:** Which defenses protect the model best
- **Good defenses** will have HIGH bars (high protection = effective defense)

### Chart 3: Security vs Utility Tradeoff (Scatter Plot)
- **X-axis:** Utility cost (accuracy drop %)
- **Y-axis:** Protection score
- **Each dot:** One experiment
- **What it shows:** The fundamental tradeoff — better security usually costs more utility
- **Ideal defenses** are in the **top-left** (high protection, low utility cost)

### Chart 4: Attack Success vs Query Budget (Line Chart)
- **X-axis:** Query budget (1K, 5K, 10K) — log scale
- **Y-axis:** Average fidelity (%)
- **Lines:** One per attack strategy
- **What it shows:** More queries generally give the attacker higher fidelity
- **Steeper lines** = that attack benefits more from additional queries

### Chart 5: Fidelity Heatmap: Attack vs Defence
- **Rows:** Defense configurations
- **Columns:** Attack strategies
- **Colour intensity:** Fidelity value
- **What it shows:** Which (attack, defence) pairs are the most/least effective
- **Darker cells** = higher fidelity (worse for defender)

---

## 8. Key Concepts for Viva

### Q: "What is model extraction?"
**A:** Model extraction is an attack where an adversary creates a functionally equivalent copy of a machine learning model by only observing its input-output behaviour through a black-box API. The attacker never accesses the model's weights, architecture, or training data — they only send images and collect the returned probability predictions.

### Q: "How does the substitute model learn?"
**A:** Through knowledge distillation. The substitute is trained on (image, victim_probability) pairs using a soft cross-entropy loss with temperature scaling. The temperature softens the probability distribution, revealing inter-class relationships ("dark knowledge") that transfer more information than hard labels alone.

### Q: "Why does the attacker's model work even though it's smaller?"
**A:** The substitute doesn't need to replicate the victim's architecture. It only needs to approximate the victim's decision function — the mapping from images to class predictions. A smaller model can learn this mapping because it's being supervised by the victim's rich probability outputs, not just hard labels.

### Q: "Which attack is the strongest and why?"
**A:** Active learning is the strongest because it strategically selects the most informative queries. By focusing on images where the current substitute is most uncertain (highest entropy), each query maximally reduces the substitute's ignorance. This is more efficient than random or uniform querying.

### Q: "Which defense is the most effective and why?"
**A:** Top-K (k=1) and throttling are among the most effective. Top-K=1 eliminates all soft probability information, reducing victim outputs to hard labels that contain much less knowledge for distillation. Throttling directly limits the amount of data the attacker can collect.

### Q: "What is the security-utility tradeoff?"
**A:** Stronger defenses (heavy noise, aggressive throttling) better protect the model but degrade the API experience for legitimate users. For example, Top-K=1 prevents extraction but also removes useful confidence information that legitimate applications might need. The scatter plot visualises this tradeoff.

### Q: "Why did you use CIFAR-10 and not a larger dataset?"
**A:** CIFAR-10 is a standard benchmark that is computationally tractable for the scope of this dissertation while still demonstrating real model extraction dynamics. The 10-class classification task with 224×224 resized images provides a realistic evaluation setting. The methodology generalises to larger datasets and more complex models.

### Q: "How does this relate to Data Analytics?"
**A:** Model extraction attacks are fundamentally a data analytics problem. The attacker performs systematic data collection (querying the API), data analysis (identifying patterns in the victim's responses), and builds predictive models from that collected data. The defense side involves anomaly detection (adaptive noise detects suspicious query patterns) and statistical analysis (evaluating effectiveness through metrics). The entire evaluation pipeline is a data analytics workflow — collecting experimental results, computing statistical metrics, and visualising findings through charts.

### Q: "What is transfer learning and why did you use it?"
**A:** Transfer learning takes a model pretrained on a large dataset (ImageNet, 1.4M images) and fine-tunes it for a smaller target task (CIFAR-10). We freeze all pretrained layers and only train the final classification head. This achieves high accuracy (82%+) quickly because the pretrained features (edges, textures, shapes) transfer well across vision tasks.

---

## 9. Technical Setup

### Hardware
- **GPU:** NVIDIA RTX 4070 (12GB VRAM)
- **CUDA:** 12.1

### Software
- **Python:** 3.9.25
- **PyTorch:** 2.1.0+cu121
- **torchvision:** 0.16.0+cu121
- **Other:** numpy 1.24.0, pandas 2.0.0, matplotlib 3.7.0, seaborn 0.12.0, scikit-learn 1.3.0, flask 3.0.0, tqdm 4.65.0

### Reproducibility
- Random seed: 42 (set in every file)
- `torch.backends.cudnn.deterministic = True`
- `torch.backends.cudnn.benchmark = False`

### How to Run

```bash
# 1. Activate environment
source venv/bin/activate

# 2. Train victim model (Phase 1 — skips if checkpoint exists)
python victim.py

# 3. Run full 4-phase evaluation pipeline
python evaluate.py
#    Phase 1: Checks victim checkpoint exists
#    Phase 2: Runs all attacks WITHOUT defence → results_no_defense.csv
#    Phase 3: Runs all attacks WITH defences   → results_with_defense.csv
#    Phase 4: Generates 5 charts + prints final conclusion

# 4. Results saved to:
#    experiments/results/results_no_defense.csv
#    experiments/results/results_with_defense.csv
#    experiments/charts/fidelity_comparison.png
#    experiments/charts/protection_scores.png
#    experiments/charts/security_utility_tradeoff.png
#    experiments/charts/fidelity_vs_query_budget.png
#    experiments/charts/attack_defense_heatmap.png
```

---

*Document generated for MSc Dissertation*
