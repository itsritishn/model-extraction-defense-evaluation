"""
utils.py
========
Shared utilities: AMP configuration, reproducibility helpers.

Centralises the two cross-cutting concerns that were previously duplicated
(and inconsistent) across modules:

  1. Automatic Mixed Precision (AMP) device selection — device-agnostic so the
     same code runs on CUDA (RTX 4070) or CPU without edits.
  2. Global seeding — deterministic, dissertation-grade reproducibility applied
     to Python ``random``, NumPy, PyTorch, CUDA, and DataLoader workers.

Author  : MSc Advanced Computer Science (Data Analytics) Dissertation
Python  : 3.9.25 (strict - no 3.10+ features)
"""

import os
import random

import numpy as np
import torch

# ── canonical seed ───────────────────────────────────────────
SEED: int = 42

# ── AMP configuration (device-agnostic) ──────────────────────
# autocast requires a device_type; GradScaler is only meaningful on CUDA.
# We keep autocast on CUDA only and run full precision on MPS/CPU (MPS autocast
# support is limited), so AMP_DEVICE stays a valid "cuda"/"cpu" string.
AMP_DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
AMP_ENABLED: bool = torch.cuda.is_available()


def get_device() -> "torch.device":
    """Select the best available compute device.

    Order of preference: CUDA (NVIDIA GPU) → MPS (Apple Silicon GPU) → CPU.
    This lets the same code run on a CUDA server, an M-series MacBook, or a
    plain CPU machine without modification.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int = SEED, deterministic: bool = True) -> int:
    """Seed all RNGs for reproducible experiments.

    Parameters
    ----------
    seed : int
        Seed value applied to ``random``, ``numpy``, and ``torch``.
    deterministic : bool
        If True, force deterministic cuDNN behaviour (disables benchmarking).

    Returns
    -------
    int
        The seed that was applied.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    return seed


def seed_worker(worker_id: int) -> None:
    """DataLoader ``worker_init_fn`` for reproducible multi-process loading."""
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int = SEED) -> torch.Generator:
    """Return a seeded ``torch.Generator`` for DataLoader shuffling."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator
