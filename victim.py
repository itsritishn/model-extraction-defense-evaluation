"""
victim.py
=========
Victim model training and black-box query interface.

Uses **ResNet-50** as the victim, fine-tuned on CIFAR-10 (10 classes).
(The model factory also supports EfficientNet-B0, retained for reference but
not used in the single-victim evaluation.)  The attacker interacts with
the victim *only* through :func:`query_victim`, which returns
probability vectors — no weights or gradients are ever exposed.

Author  : MSc Advanced Computer Science (Data Analytics) Dissertation
Python  : 3.9.25 (strict – no 3.10+ features)
GPU     : RTX 4070 / CUDA 12.1
"""

import logging
import os
import time
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
import torchvision.models as models
from tqdm import tqdm

from data_loader import get_data_loaders, CIFAR10_CLASSES
from utils import AMP_DEVICE, AMP_ENABLED, SEED, get_device, seed_everything

# ── logging ──────────────────────────────────────────────────
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)

# ── reproducibility ──────────────────────────────────────────
seed_everything(SEED)

# ── device ───────────────────────────────────────────────────
DEVICE: torch.device = get_device()
logger.info("Device: %s", DEVICE)

NUM_CLASSES: int = 10


# ─────────────────────────────────────────────────────────────
#  Model factory
# ─────────────────────────────────────────────────────────────
def build_victim_model(
    architecture: str = "resnet50",
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
) -> nn.Module:
    """Instantiate and prepare a victim model for CIFAR-10 fine-tuning.

    Parameters
    ----------
    architecture : str
        ``"resnet50"`` or ``"efficientnet_b0"``.
    num_classes : int
        Number of output classes.
    pretrained : bool
        Use ImageNet-pretrained weights.

    Returns
    -------
    nn.Module
        Model moved to *DEVICE* and ready for training.
    """
    architecture = architecture.lower().strip()
    logger.info("Building victim model: %s  (pretrained=%s)", architecture, pretrained)

    if architecture == "resnet50":
        # Use the older `pretrained` kwarg for torchvision 0.16 compat
        model = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None,
        )
        # Freeze all layers except the final FC
        for param in model.parameters():
            param.requires_grad = False
        # Replace final FC for CIFAR-10
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes),
        )

    elif architecture == "efficientnet_b0":
        if pretrained:
            # Load without hash check to avoid corrupted-cache errors
            model = models.efficientnet_b0(weights=None)
            state_dict_url = models.EfficientNet_B0_Weights.IMAGENET1K_V1.url
            state_dict = torch.hub.load_state_dict_from_url(
                state_dict_url, progress=True, check_hash=False,
            )
            model.load_state_dict(state_dict)
        else:
            model = models.efficientnet_b0(weights=None)
        for param in model.parameters():
            param.requires_grad = False
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes),
        )
    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    model = model.to(DEVICE)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        "Model params  |  trainable=%s  total=%s  (%.1f%%)",
        f"{trainable:,}", f"{total:,}", 100 * trainable / total,
    )
    return model


# ─────────────────────────────────────────────────────────────
#  Training loop
# ─────────────────────────────────────────────────────────────
def train_victim(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int = 15,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    save_path: Optional[str] = None,
) -> nn.Module:
    """Fine-tune the victim model on CIFAR-10 using AMP.

    Parameters
    ----------
    model : nn.Module
        Victim model (final layer unfrozen).
    train_loader : DataLoader
        Training data.
    test_loader : DataLoader
        Test data for periodic evaluation.
    epochs : int
        Number of training epochs.
    lr : float
        Learning rate.
    weight_decay : float
        L2 regularisation.
    save_path : str, optional
        Path to save the trained model checkpoint.

    Returns
    -------
    nn.Module
        Trained victim model.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler(enabled=AMP_ENABLED)

    best_acc = 0.0
    logger.info("Starting victim training  |  epochs=%d  lr=%.4f", epochs, lr)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{epochs}",
            leave=True,
            ncols=100,
        )

        for images, labels in pbar:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()
            try:
                with autocast(device_type=AMP_DEVICE, enabled=AMP_ENABLED):
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.warning("OOM — clearing cache and skipping batch")
                    torch.cuda.empty_cache()
                    continue
                raise

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{100.0 * correct / total:.1f}%",
            )

        scheduler.step()

        train_loss = running_loss / total
        train_acc = 100.0 * correct / total

        # ── periodic evaluation ──────────────────────────────
        test_acc = evaluate_model(model, test_loader)

        logger.info(
            "Epoch %d/%d  |  train_loss=%.4f  train_acc=%.2f%%  test_acc=%.2f%%",
            epoch, epochs, train_loss, train_acc, test_acc,
        )

        # ── checkpoint best model ────────────────────────────
        if test_acc > best_acc:
            best_acc = test_acc
            if save_path is not None:
                save_dir = os.path.dirname(save_path)
                if save_dir:
                    os.makedirs(save_dir, exist_ok=True)
                torch.save(model.state_dict(), save_path)
                logger.info("Saved best model (acc=%.2f%%) → %s", best_acc, save_path)

    logger.info("Training complete  |  best_test_acc=%.2f%%", best_acc)
    return model


# ─────────────────────────────────────────────────────────────
#  Evaluation helper
# ─────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
) -> float:
    """Compute top-1 accuracy on *data_loader*.

    Returns
    -------
    float
        Accuracy in **percent** (0–100).
    """
    model.eval()
    correct = 0
    total = 0

    for images, labels in data_loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        with autocast(device_type=AMP_DEVICE, enabled=AMP_ENABLED):
            outputs = model(images)

        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    acc = 100.0 * correct / total
    return acc


# ─────────────────────────────────────────────────────────────
#  BLACK-BOX QUERY INTERFACE
# ─────────────────────────────────────────────────────────────
class VictimModel:
    """Wrapper that exposes the victim *only* through probability queries.

    The attacker **never** has access to:
      - model weights / architecture details
      - gradients
      - intermediate feature maps
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device = DEVICE,
    ) -> None:
        self.model = model
        self.model.eval()
        self.device = device
        self._query_count: int = 0
        logger.info("VictimModel interface initialised (black-box only)")

    @property
    def query_count(self) -> int:
        """Total number of queries served so far."""
        return self._query_count

    def reset_query_count(self) -> None:
        """Reset the query counter to zero."""
        self._query_count = 0

    @torch.no_grad()
    def query(self, images: torch.Tensor) -> torch.Tensor:
        """Query the victim model and return **soft-label probabilities**.

        Parameters
        ----------
        images : torch.Tensor
            Batch of input images ``(B, C, H, W)``.

        Returns
        -------
        torch.Tensor
            Probability vectors ``(B, num_classes)`` on CPU.
        """
        images = images.to(self.device, non_blocking=True)
        with autocast(device_type=AMP_DEVICE, enabled=AMP_ENABLED):
            logits = self.model(images)
            probs = torch.softmax(logits, dim=1)
        self._query_count += images.size(0)
        return probs.cpu()


# ─────────────────────────────────────────────────────────────
#  Convenience loaders
# ─────────────────────────────────────────────────────────────
def load_victim(
    architecture: str = "resnet50",
    checkpoint_path: Optional[str] = None,
) -> VictimModel:
    """Build (and optionally load) a victim model wrapped in VictimModel.

    Parameters
    ----------
    architecture : str
        ``"resnet50"`` or ``"efficientnet_b0"``.
    checkpoint_path : str, optional
        Path to a saved ``.pth`` state dict.

    Returns
    -------
    VictimModel
        Black-box victim interface.
    """
    have_ckpt = checkpoint_path is not None and os.path.isfile(checkpoint_path)
    # Skip the ImageNet download when a checkpoint will supply all weights.
    model = build_victim_model(architecture=architecture, pretrained=not have_ckpt)
    if have_ckpt:
        state = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state)
        logger.info("Loaded checkpoint: %s", checkpoint_path)
    return VictimModel(model)


# ─────────────────────────────────────────────────────────────
#  Stand-alone entry point
# ─────────────────────────────────────────────────────────────
def main() -> None:
    """Train the victim model and persist its checkpoint (skips if already saved)."""
    loaders = get_data_loaders(batch_size=64, data_dir="./data/cifar10")

    for arch, save_name in [
        ("resnet50", "models/victim_resnet50.pth"),
    ]:
        logger.info("=" * 60)

        if os.path.exists(save_name):
            logger.info("Checkpoint found → %s  (skipping training)", save_name)
            model = build_victim_model(architecture=arch, pretrained=False)
            model.load_state_dict(torch.load(save_name, map_location="cpu", weights_only=True))
            model.to(DEVICE).eval()
        else:
            logger.info("Training victim: %s", arch)
            model = build_victim_model(architecture=arch, pretrained=True)
            train_victim(
                model,
                train_loader=loaders["train"],
                test_loader=loaders["test"],
                epochs=15,
                save_path=save_name,
            )

        logger.info("=" * 60)

        # Quick sanity check via black-box interface
        victim = VictimModel(model)
        sample_images, _ = next(iter(loaders["test"]))
        probs = victim.query(sample_images)
        logger.info(
            "Black-box sanity check  |  probs shape=%s  sum≈%.4f",
            probs.shape, probs[0].sum().item(),
        )


if __name__ == "__main__":
    main()
