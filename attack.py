"""
attack.py
=========
Model extraction attack implementations.

Three attack strategies:
  1. **Random Query Attack** — queries the victim with random images.
  2. **Knockoff Nets Attack** — queries with in-domain images and
     trains via knowledge distillation.
  3. **Active Learning Attack** — uses uncertainty sampling to
     select the most informative queries.

Two substitute architectures:
  • SmallCNN  (3 conv layers — lightweight baseline)
  • MobileNetV3-Small (modern efficient architecture)

All training uses automatic mixed precision (AMP) for RTX 4070.

Author  : MSc Advanced Computer Science (Data Analytics) Dissertation
Python  : 3.9.25 (strict – no 3.10+ features)
GPU     : RTX 4070 / CUDA 12.1
"""

import logging
import time
from functools import partial
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, TensorDataset
import torchvision.models as models
from tqdm import tqdm

from data_loader import get_cifar10_datasets, get_subset_loader
from victim import VictimModel, DEVICE, SEED, NUM_CLASSES
from utils import AMP_DEVICE, AMP_ENABLED, seed_everything
from threat_model import (
    AttackerKnowledge,
    QueryBudget,
    QueryExecutor,
    QueryGenerator,
    ThreatConfig,
)

# ── logging ──────────────────────────────────────────────────
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)

# ── reproducibility ──────────────────────────────────────────
seed_everything(SEED)


# ═════════════════════════════════════════════════════════════
#  SUBSTITUTE MODEL ARCHITECTURES
# ═════════════════════════════════════════════════════════════

class SmallCNN(nn.Module):
    """Lightweight 3-conv-layer CNN for CIFAR-10.

    Architecture
    ------------
    Conv(3→32)→BN→ReLU→Pool →
    Conv(32→64)→BN→ReLU→Pool →
    Conv(64→128)→BN→ReLU→AdaptivePool →
    FC(128→num_classes)
    """ 

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def build_substitute_model(
    architecture: str = "small_cnn",
    num_classes: int = NUM_CLASSES,
) -> nn.Module:
    """Instantiate a substitute model.

    Parameters
    ----------
    architecture : str
        ``"small_cnn"`` or ``"mobilenetv3_small"``.
    num_classes : int
        Number of output classes.

    Returns
    -------
    nn.Module
        Model on *DEVICE*.
    """
    arch = architecture.lower().strip()
    logger.info("Building substitute model: %s", arch)

    if arch == "small_cnn":
        model = SmallCNN(num_classes=num_classes)
    elif arch == "mobilenetv3_small":
        model = models.mobilenet_v3_small(weights=None)
        # Replace classifier head
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unknown substitute architecture: {arch}")

    model = model.to(DEVICE)
    total = sum(p.numel() for p in model.parameters())
    logger.info("Substitute params: %s", f"{total:,}")
    return model


# ═════════════════════════════════════════════════════════════
#  KNOWLEDGE DISTILLATION TRAINING
# ═════════════════════════════════════════════════════════════

def soft_cross_entropy(
    logits: torch.Tensor,
    soft_targets: torch.Tensor,
    temperature: float = 3.0,
) -> torch.Tensor:
    """KL-divergence-based soft cross-entropy for knowledge distillation.

    Parameters
    ----------
    logits : Tensor
        Raw model outputs ``(B, C)``.
    soft_targets : Tensor
        Teacher probability distribution ``(B, C)``.
    temperature : float
        Distillation temperature (higher = softer).

    Returns
    -------
    Tensor
        Scalar loss.
    """
    log_probs = F.log_softmax(logits / temperature, dim=1)
    soft_targets_temp = F.softmax(soft_targets / temperature, dim=1)
    loss = F.kl_div(log_probs, soft_targets_temp, reduction="batchmean")
    return loss * (temperature ** 2)


def train_substitute(
    model: nn.Module,
    images_list: List[torch.Tensor],
    labels_list: List[torch.Tensor],
    epochs: int = 30,
    lr: float = 1e-3,
    temperature: float = 3.0,
    batch_size: int = 64,
) -> nn.Module:
    """Train the substitute model on stolen (image, soft-label) pairs.

    Uses automatic mixed precision and cosine annealing.

    Parameters
    ----------
    model : nn.Module
        Substitute model.
    images_list : List[Tensor]
        List of image batches collected from victim queries.
    labels_list : List[Tensor]
        Corresponding soft-label batches.
    epochs : int
        Training epochs.
    lr : float
        Learning rate.
    temperature : float
        Distillation temperature.
    batch_size : int
        Mini-batch size for training.

    Returns
    -------
    nn.Module
        Trained substitute model.
    """
    # Concatenate all collected data
    all_images = torch.cat(images_list, dim=0)
    all_labels = torch.cat(labels_list, dim=0)
    logger.info(
        "Training substitute  |  samples=%d  epochs=%d  T=%.1f",
        all_images.size(0), epochs, temperature,
    )

    dataset = TensorDataset(all_images, all_labels)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler(enabled=AMP_ENABLED)

    model.train()
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        pbar = tqdm(
            loader,
            desc=f"Sub epoch {epoch}/{epochs}",
            leave=False,
            ncols=100,
        )
        for imgs, soft_labels in pbar:
            imgs = imgs.to(DEVICE, non_blocking=True)
            soft_labels = soft_labels.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()
            try:
                with autocast(device_type=AMP_DEVICE, enabled=AMP_ENABLED):
                    outputs = model(imgs)
                    loss = soft_cross_entropy(outputs, soft_labels, temperature)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.warning("OOM in substitute training — skipping batch")
                    torch.cuda.empty_cache()
                    continue
                raise

            running_loss += loss.item() * imgs.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        avg_loss = running_loss / len(dataset)
        if epoch % 5 == 0 or epoch == 1:
            logger.info("Sub epoch %d/%d  |  loss=%.4f", epoch, epochs, avg_loss)

    model.eval()
    return model


# ═════════════════════════════════════════════════════════════
#  ATTACKS 1 & 2: QUERY-AND-DISTILL (random / knockoff)
# ═════════════════════════════════════════════════════════════

def _query_distillation_attack(
    victim: VictimModel,
    config: ThreatConfig,
    defense_fn: Optional[object] = None,
    sub_epochs: int = 30,
    *,
    attack_name: str,
    knowledge: AttackerKnowledge,
) -> Tuple[nn.Module, Dict]:
    """Shared query-then-distil extraction routine.

    Both the Random Query attack (Level-1, synthetic inputs) and the
    Knockoff Nets attack (Level-2, in-domain inputs) follow the identical
    pipeline — query the victim under a budget, then train the substitute via
    knowledge distillation. They differ only in the attacker knowledge level,
    so they share one implementation parameterised by ``knowledge``.

    Parameters
    ----------
    victim : VictimModel
        Black-box victim interface.
    config : ThreatConfig
        Threat-model configuration (budget, substitute arch, etc.).
    defense_fn : callable, optional
        Applied to victim outputs before the attacker sees them.
    sub_epochs : int
        Substitute training epochs.
    attack_name : str
        Label recorded in the returned metadata.
    knowledge : AttackerKnowledge
        ``LEVEL_1`` (random queries) or ``LEVEL_2`` (in-domain queries).

    Returns
    -------
    Tuple[nn.Module, Dict]
        (trained_substitute, metadata_dict)
    """
    logger.info("=" * 60)
    logger.info("ATTACK: %s  (knowledge=%s)", attack_name, knowledge.value)
    logger.info("=" * 60)

    cfg = ThreatConfig(
        budget=config.budget,
        knowledge=knowledge,
        substitute_arch=config.substitute_arch,
        victim_arch=config.victim_arch,
        batch_size=config.batch_size,
        image_size=config.image_size,
    )

    generator = QueryGenerator(cfg)
    query_loader = generator.get_query_loader()
    executor = QueryExecutor(victim, cfg)

    t0 = time.time()
    images_list, labels_list = executor.execute(query_loader, defense_fn=defense_fn)
    query_time = time.time() - t0

    substitute = build_substitute_model(cfg.substitute_arch)
    t1 = time.time()
    substitute = train_substitute(substitute, images_list, labels_list, epochs=sub_epochs)
    train_time = time.time() - t1

    meta = {
        "attack": attack_name,
        "queries_used": executor.queries_used,
        "query_time_s": round(query_time, 2),
        "train_time_s": round(train_time, 2),
    }
    logger.info("%s attack complete  |  %s", attack_name, meta)
    return substitute, meta


# Thin, explicit wrappers preserve the public attack names used everywhere.
random_query_attack = partial(
    _query_distillation_attack,
    attack_name="random_query",
    knowledge=AttackerKnowledge.LEVEL_1,
)

knockoff_attack = partial(
    _query_distillation_attack,
    attack_name="knockoff_nets",
    knowledge=AttackerKnowledge.LEVEL_2,
)


# ═════════════════════════════════════════════════════════════
#  ATTACK 3: ACTIVE LEARNING ATTACK
# ═════════════════════════════════════════════════════════════

def _uncertainty_scores(probs: torch.Tensor) -> torch.Tensor:
    """Compute entropy-based uncertainty for each sample.

    Parameters
    ----------
    probs : Tensor
        Probability vectors ``(N, C)``.

    Returns
    -------
    Tensor
        Entropy scores ``(N,)`` — higher = more uncertain.
    """
    log_probs = torch.log(probs + 1e-10)
    entropy = -(probs * log_probs).sum(dim=1)
    return entropy


def active_learning_attack(
    victim: VictimModel,
    config: ThreatConfig,
    defense_fn: Optional[object] = None,
    sub_epochs: int = 30,
    initial_fraction: float = 0.3,
    rounds: int = 5,
    data_dir: str = "./data/cifar10",
) -> Tuple[nn.Module, Dict]:
    """Attack 3 — Active learning with uncertainty sampling.

    Strategy
    --------
    1. Start with a small seed set (``initial_fraction`` of budget).
    2. Train an intermediate substitute.
    3. Score remaining pool by entropy of substitute predictions.
    4. Query the victim with the most uncertain samples.
    5. Repeat for *rounds* iterations.

    Parameters
    ----------
    victim : VictimModel
        Black-box victim interface.
    config : ThreatConfig
        Threat-model configuration.
    defense_fn : callable, optional
        Defense applied to victim outputs.
    sub_epochs : int
        Substitute training epochs per round.
    initial_fraction : float
        Fraction of budget used in the first (seed) round.
    rounds : int
        Number of active-learning rounds.
    data_dir : str
        CIFAR-10 data directory.

    Returns
    -------
    Tuple[nn.Module, Dict]
        (trained_substitute, metadata_dict)
    """
    logger.info("=" * 60)
    logger.info("ATTACK 3: Active Learning Attack")
    logger.info("=" * 60)

    budget = config.budget
    initial_budget = int(budget * initial_fraction)
    remaining_budget = budget - initial_budget
    per_round_budget = remaining_budget // rounds if rounds > 0 else 0

    # Prepare the full candidate pool (CIFAR-10 test set)
    _, test_dataset = get_cifar10_datasets(
        data_dir=data_dir,
        image_size=config.image_size,
    )
    pool_indices = list(range(len(test_dataset)))
    np.random.shuffle(pool_indices)

    collected_images: List[torch.Tensor] = []
    collected_labels: List[torch.Tensor] = []
    queries_used = 0
    query_time = 0.0   # cumulative time spent querying the victim
    train_time = 0.0   # cumulative time spent training the substitute

    # ── Round 0: seed set ────────────────────────────────────
    seed_indices = pool_indices[:initial_budget]
    pool_indices = pool_indices[initial_budget:]

    seed_loader = get_subset_loader(
        test_dataset,
        indices=seed_indices,
        batch_size=config.batch_size,
        shuffle=False,
    )
    logger.info("Active learning  |  seed round  |  %d queries", len(seed_indices))

    seed_cfg = ThreatConfig(
        budget=initial_budget,
        knowledge=config.knowledge,
        substitute_arch=config.substitute_arch,
        victim_arch=config.victim_arch,
        batch_size=config.batch_size,
        image_size=config.image_size,
    )
    executor = QueryExecutor(victim, seed_cfg)
    t_q = time.time()
    imgs, lbls = executor.execute(seed_loader, defense_fn=defense_fn)
    query_time += time.time() - t_q
    collected_images.extend(imgs)
    collected_labels.extend(lbls)
    queries_used += executor.queries_used

    # ── Active rounds ────────────────────────────────────────
    substitute: Optional[nn.Module] = None

    for rnd in range(1, rounds + 1):
        if per_round_budget <= 0 or len(pool_indices) == 0:
            break

        logger.info("Active learning  |  round %d/%d", rnd, rounds)

        # Train intermediate substitute
        substitute = build_substitute_model(config.substitute_arch)
        t_t = time.time()
        substitute = train_substitute(
            substitute, collected_images, collected_labels,
            epochs=max(5, sub_epochs // rounds),
        )
        train_time += time.time() - t_t

        # Score remaining pool by uncertainty
        score_loader = get_subset_loader(
            test_dataset,
            indices=pool_indices[:min(len(pool_indices), 10000)],
            batch_size=config.batch_size,
            shuffle=False,
        )

        all_scores: List[torch.Tensor] = []
        substitute.eval()
        with torch.no_grad():
            for batch_imgs, _ in score_loader:
                batch_imgs = batch_imgs.to(DEVICE, non_blocking=True)
                with autocast(device_type=AMP_DEVICE, enabled=AMP_ENABLED):
                    logits = substitute(batch_imgs)
                probs = torch.softmax(logits, dim=1).cpu()
                scores = _uncertainty_scores(probs)
                all_scores.append(scores)

        all_scores_tensor = torch.cat(all_scores, dim=0).float()  # ensure float32 for topk
        n_scored = all_scores_tensor.size(0)

        # Select top-k most uncertain
        k = min(per_round_budget, n_scored, len(pool_indices))
        _, top_idx = torch.topk(all_scores_tensor[:n_scored], k)
        selected_pool_positions = top_idx.tolist()
        selected_indices = [pool_indices[i] for i in selected_pool_positions]

        # Remove selected from pool
        selected_set = set(selected_pool_positions)
        pool_indices = [
            idx for i, idx in enumerate(pool_indices)
            if i not in selected_set
        ]

        # Query victim with selected samples
        round_loader = get_subset_loader(
            test_dataset,
            indices=selected_indices,
            batch_size=config.batch_size,
            shuffle=False,
        )
        round_cfg = ThreatConfig(
            budget=k,
            knowledge=config.knowledge,
            substitute_arch=config.substitute_arch,
            victim_arch=config.victim_arch,
            batch_size=config.batch_size,
            image_size=config.image_size,
        )
        round_executor = QueryExecutor(victim, round_cfg)
        t_q = time.time()
        rnd_imgs, rnd_lbls = round_executor.execute(round_loader, defense_fn=defense_fn)
        query_time += time.time() - t_q
        collected_images.extend(rnd_imgs)
        collected_labels.extend(rnd_lbls)
        queries_used += round_executor.queries_used

        logger.info(
            "Round %d  |  selected=%d  total_queries=%d",
            rnd, k, queries_used,
        )

    # ── Final substitute training ────────────────────────────
    substitute = build_substitute_model(config.substitute_arch)
    t_t = time.time()
    substitute = train_substitute(
        substitute, collected_images, collected_labels, epochs=sub_epochs,
    )
    train_time += time.time() - t_t

    meta = {
        "attack": "active_learning",
        "queries_used": queries_used,
        "rounds": rounds,
        "query_time_s": round(query_time, 2),
        "train_time_s": round(train_time, 2),
    }
    logger.info("Active learning attack complete  |  %s", meta)
    return substitute, meta


# ═════════════════════════════════════════════════════════════
#  DISPATCHER
# ═════════════════════════════════════════════════════════════

ATTACK_REGISTRY: Dict[str, object] = {
    "random_query": random_query_attack,
    "knockoff_nets": knockoff_attack,
    "active_learning": active_learning_attack,
}


def run_attack(
    attack_name: str,
    victim: VictimModel,
    config: ThreatConfig,
    defense_fn: Optional[object] = None,
    sub_epochs: int = 30,
) -> Tuple[nn.Module, Dict]:
    """Run a named attack.

    Parameters
    ----------
    attack_name : str
        One of ``"random_query"``, ``"knockoff_nets"``,
        ``"active_learning"``.
    victim : VictimModel
        Black-box victim.
    config : ThreatConfig
        Threat-model configuration.
    defense_fn : callable, optional
        Defense to apply to victim outputs.
    sub_epochs : int
        Substitute training epochs.

    Returns
    -------
    Tuple[nn.Module, Dict]
        (trained_substitute, attack_metadata)
    """
    if attack_name not in ATTACK_REGISTRY:
        raise ValueError(
            f"Unknown attack '{attack_name}'. "
            f"Choose from: {list(ATTACK_REGISTRY.keys())}"
        )
    fn = ATTACK_REGISTRY[attack_name]
    return fn(
        victim=victim,
        config=config,
        defense_fn=defense_fn,
        sub_epochs=sub_epochs,
    )


# ─────────────────────────────────────────────────────────────
#  Stand-alone demo
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from victim import load_victim

    logger.info("Attack module — stand-alone demo")

    # Load victim (untrained for demo — replace with checkpoint)
    victim = load_victim(architecture="resnet50", checkpoint_path="models/victim_resnet50.pth")

    cfg = ThreatConfig(
        budget=QueryBudget.LOW.value,
        knowledge=AttackerKnowledge.LEVEL_2,
        substitute_arch="small_cnn",
        victim_arch="resnet50",
    )

    for attack_name in ATTACK_REGISTRY:
        sub, meta = run_attack(attack_name, victim, cfg, sub_epochs=5)
        logger.info("Demo result  |  attack=%s  meta=%s", attack_name, meta)
