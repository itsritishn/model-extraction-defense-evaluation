"""
data_loader.py
==============
CIFAR-10 data loading and preprocessing pipeline.

Provides train / test DataLoaders with transforms suitable for
ImageNet-pretrained models (resize to 224x224, ImageNet normalisation).

Author  : MSc Advanced Computer Science (Data Analytics) Dissertation
Python  : 3.9.25 (strict – no 3.10+ features)
"""

import logging
import os
from typing import Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision
import torchvision.transforms as transforms

from utils import make_generator, seed_worker

# ── logging ──────────────────────────────────────────────────
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)

# ── CIFAR-10 class names ────────────────────────────────────
CIFAR10_CLASSES: Tuple[str, ...] = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)

# ── ImageNet normalisation constants ────────────────────────
IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)


# ─────────────────────────────────────────────────────────────
#  Transform builders
# ─────────────────────────────────────────────────────────────
def get_train_transforms(image_size: int = 224) -> transforms.Compose:
    """Training transforms with data augmentation.

    Parameters
    ----------
    image_size : int
        Target spatial size (default 224 for ResNet-50).

    Returns
    -------
    transforms.Compose
        Composed transform pipeline.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(image_size, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean=list(IMAGENET_MEAN), std=list(IMAGENET_STD)),
    ])


def get_test_transforms(image_size: int = 224) -> transforms.Compose:
    """Deterministic test / validation transforms (no augmentation).

    Parameters
    ----------
    image_size : int
        Target spatial size.

    Returns
    -------
    transforms.Compose
        Composed transform pipeline.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=list(IMAGENET_MEAN), std=list(IMAGENET_STD)),
    ])


# ─────────────────────────────────────────────────────────────
#  Dataset loaders
# ─────────────────────────────────────────────────────────────
def get_cifar10_datasets(
    data_dir: str = "./data/cifar10",
    image_size: int = 224,
) -> Tuple[torchvision.datasets.CIFAR10, torchvision.datasets.CIFAR10]:
    """Download (if needed) and return CIFAR-10 train and test datasets.

    Parameters
    ----------
    data_dir : str
        Root directory for CIFAR-10 storage.
    image_size : int
        Target image size.

    Returns
    -------
    Tuple[Dataset, Dataset]
        (train_dataset, test_dataset)
    """
    os.makedirs(data_dir, exist_ok=True)

    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=get_train_transforms(image_size),
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=get_test_transforms(image_size),
    )

    logger.info(
        "CIFAR-10 loaded  |  train=%d  test=%d  classes=%d",
        len(train_dataset), len(test_dataset), len(CIFAR10_CLASSES),
    )
    return train_dataset, test_dataset


def get_data_loaders(
    batch_size: int = 64,
    num_workers: int = 4,
    data_dir: str = "./data/cifar10",
    image_size: int = 224,
    pin_memory: bool = True,
) -> Dict[str, DataLoader]:
    """Build train and test DataLoaders for CIFAR-10.

    Parameters
    ----------
    batch_size : int
        Mini-batch size.
    num_workers : int
        Number of data-loading worker processes.
    data_dir : str
        Root directory for CIFAR-10 storage.
    image_size : int
        Target image size.
    pin_memory : bool
        Pin memory for faster GPU transfer.

    Returns
    -------
    Dict[str, DataLoader]
        ``{"train": ..., "test": ...}``
    """
    train_dataset, test_dataset = get_cifar10_datasets(data_dir, image_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=make_generator(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
    )

    logger.info(
        "DataLoaders ready  |  batch_size=%d  workers=%d  pin_memory=%s",
        batch_size, num_workers, pin_memory,
    )
    return {"train": train_loader, "test": test_loader}


def get_subset_loader(
    dataset: Dataset,
    indices: Optional[list] = None,
    num_samples: Optional[int] = None,
    batch_size: int = 64,
    shuffle: bool = False,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoader:
    """Return a DataLoader over a subset of *dataset*.

    Useful for limiting the attacker's query budget.

    Parameters
    ----------
    dataset : Dataset
        Source dataset.
    indices : list, optional
        Explicit index list.  If ``None``, the first *num_samples*
        indices are used.
    num_samples : int, optional
        Number of samples when *indices* is not given.
    batch_size : int
        Mini-batch size.
    shuffle : bool
        Whether to shuffle the subset.
    num_workers : int
        Data-loading workers.
    pin_memory : bool
        Pin memory for GPU transfer.

    Returns
    -------
    DataLoader
        Subset data loader.
    """
    if indices is None:
        if num_samples is None:
            raise ValueError("Provide either `indices` or `num_samples`.")
        indices = list(range(min(num_samples, len(dataset))))

    subset = Subset(dataset, indices)
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        generator=make_generator() if shuffle else None,
    )
    logger.info("Subset loader created  |  samples=%d", len(subset))
    return loader


# ─────────────────────────────────────────────────────────────
#  Stand-alone sanity check
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    loaders = get_data_loaders(batch_size=64, data_dir="./data/cifar10")
    images, labels = next(iter(loaders["train"]))
    logger.info(
        "Sample batch  |  images=%s  labels=%s  dtype=%s",
        images.shape, labels.shape, images.dtype,
    )
    logger.info("Classes: %s", CIFAR10_CLASSES)
