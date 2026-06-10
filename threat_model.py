"""
threat_model.py
===============
Black-box threat model definition and query-budget management.

Defines attacker capabilities, knowledge levels, query budgets,
and the query-orchestration logic that enforces the black-box
constraint throughout every experiment.

Author  : MSc Advanced Computer Science (Data Analytics) Dissertation
Python  : 3.9.25 (strict – no 3.10+ features)
"""

import logging
import random
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_loader import get_cifar10_datasets, get_subset_loader
from utils import SEED, seed_everything, seed_worker

# ── logging ──────────────────────────────────────────────────
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)

# ── reproducibility ──────────────────────────────────────────
seed_everything(SEED)


# ─────────────────────────────────────────────────────────────
#  Enums & constants
# ─────────────────────────────────────────────────────────────
class QueryBudget(Enum):
    """Predefined query-budget tiers for experiments."""
    LOW = 1_000
    MEDIUM = 5_000
    HIGH = 10_000


class AttackerKnowledge(Enum):
    """Attacker knowledge levels about the victim's domain.

    LEVEL_1 : The attacker has *no* in-domain data; queries are
              generated from random noise or an unrelated dataset.
    LEVEL_2 : The attacker has *some* in-domain data (e.g. CIFAR-10
              test split) and crafts queries from that domain.
    """
    LEVEL_1 = "random_queries"
    LEVEL_2 = "domain_queries"


# ─────────────────────────────────────────────────────────────
#  Threat-model configuration
# ─────────────────────────────────────────────────────────────
@dataclass
class ThreatConfig:
    """Immutable configuration for one experiment run.

    Attributes
    ----------
    budget : int
        Maximum number of queries the attacker may issue.
    knowledge : AttackerKnowledge
        Attacker knowledge level.
    substitute_arch : str
        Architecture tag for the substitute model (e.g. ``"small_cnn"``
        or ``"mobilenetv3_small"``).
    victim_arch : str
        Architecture tag for the victim model.
    batch_size : int
        Batch size used when querying the victim.
    image_size : int
        Spatial dimension of query images.
    """
    budget: int = QueryBudget.MEDIUM.value
    knowledge: AttackerKnowledge = AttackerKnowledge.LEVEL_2
    substitute_arch: str = "small_cnn"
    victim_arch: str = "resnet50"
    batch_size: int = 64
    image_size: int = 224

    def describe(self) -> str:
        """Human-readable summary string."""
        return (
            f"ThreatConfig(budget={self.budget}, "
            f"knowledge={self.knowledge.value}, "
            f"sub={self.substitute_arch}, "
            f"victim={self.victim_arch})"
        )


# ─────────────────────────────────────────────────────────────
#  Query generation strategies
# ─────────────────────────────────────────────────────────────
class QueryGenerator:
    """Generate query images according to the attacker's knowledge level.

    Parameters
    ----------
    config : ThreatConfig
        Threat-model configuration.
    data_dir : str
        Directory containing CIFAR-10 data (used for Level-2 queries).
    """

    def __init__(
        self,
        config: ThreatConfig,
        data_dir: str = "./data/cifar10",
    ) -> None:
        self.config = config
        self.data_dir = data_dir
        logger.info("QueryGenerator initialised  |  %s", config.describe())

    # ── Level 1: random noise images ─────────────────────────
    def _generate_random_queries(self) -> DataLoader:
        """Produce a DataLoader of uniform-random images.

        Each image is ``(3, image_size, image_size)`` drawn from
        a standard normal distribution and clamped to [−2, 2] so
        that values roughly span the normalised pixel range.
        """
        n = self.config.budget
        sz = self.config.image_size
        logger.info("Generating %d random-noise query images (%dx%d)", n, sz, sz)

        images = torch.randn(n, 3, sz, sz).clamp(-2.0, 2.0)
        labels = torch.zeros(n, dtype=torch.long)  # dummy labels

        dataset = torch.utils.data.TensorDataset(images, labels)
        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            pin_memory=True,
            worker_init_fn=seed_worker,
        )
        return loader

    # ── Level 2: in-domain CIFAR-10 test images ──────────────
    def _generate_domain_queries(self) -> DataLoader:
        """Produce a DataLoader from the CIFAR-10 test set.

        If the budget is smaller than the test set, a random subset
        is sampled; if larger, indices are repeated with wrapping.
        """
        _, test_dataset = get_cifar10_datasets(
            data_dir=self.data_dir,
            image_size=self.config.image_size,
        )
        n = self.config.budget
        total = len(test_dataset)

        if n <= total:
            indices = random.sample(range(total), n)
        else:
            # wrap around
            reps = n // total
            remainder = n % total
            indices = list(range(total)) * reps + random.sample(range(total), remainder)

        logger.info(
            "Domain query set: %d images from CIFAR-10 test (%d unique)",
            len(indices), min(n, total),
        )
        return get_subset_loader(
            test_dataset,
            indices=indices,
            batch_size=self.config.batch_size,
            shuffle=False,
        )

    # ── public dispatch ──────────────────────────────────────
    def get_query_loader(self) -> DataLoader:
        """Return the appropriate query DataLoader for the
        configured knowledge level.

        Returns
        -------
        DataLoader
            Iterable of ``(images, dummy_labels)`` batches.
        """
        if self.config.knowledge == AttackerKnowledge.LEVEL_1:
            return self._generate_random_queries()
        elif self.config.knowledge == AttackerKnowledge.LEVEL_2:
            return self._generate_domain_queries()
        else:
            raise ValueError(f"Unknown knowledge level: {self.config.knowledge}")


# ─────────────────────────────────────────────────────────────
#  Query executor (enforces budget + black-box constraint)
# ─────────────────────────────────────────────────────────────
class QueryExecutor:
    """Issue queries to a :class:`VictimModel` and collect soft labels.

    Enforces the query budget: once the budget is exhausted the
    executor refuses additional queries.

    Parameters
    ----------
    victim : object
        Any object with a ``.query(images) -> probs`` method
        (see :class:`victim.VictimModel`).
    config : ThreatConfig
        Threat-model configuration.
    """

    def __init__(self, victim: object, config: ThreatConfig) -> None:
        self.victim = victim
        self.config = config
        self._queries_used: int = 0
        self._query_log: List[int] = []  # timestamps (batch sizes)

    @property
    def queries_used(self) -> int:
        return self._queries_used

    @property
    def budget_remaining(self) -> int:
        return max(0, self.config.budget - self._queries_used)
    
    def reset(self) -> None:
        """Reset query counters for reuse across experiments."""
        self._queries_used = 0
        self._query_log = []

    def execute(
        self,
        query_loader: DataLoader,
        defense_fn: Optional[object] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Run all queries and collect (images, soft_labels) pairs.

        Parameters
        ----------
        query_loader : DataLoader
            Iterable of ``(images, _)`` batches.
        defense_fn : callable, optional
            A function ``f(probs) -> probs`` that applies a defense
            to the victim's output before the attacker sees it.

        Returns
        -------
        Tuple[List[Tensor], List[Tensor]]
            ``(all_images, all_soft_labels)`` — lists of per-batch
            tensors on CPU.
        """
        all_images: List[torch.Tensor] = []
        all_labels: List[torch.Tensor] = []

        logger.info(
            "Executing queries  |  budget=%d  used=%d  remaining=%d",
            self.config.budget, self._queries_used, self.budget_remaining,
        )

        pbar = tqdm(query_loader, desc="Querying victim", ncols=100)
        for images, _ in pbar:
            batch_size = images.size(0)

            # ── enforce budget ───────────────────────────────
            if self._queries_used + batch_size > self.config.budget:
                remaining = self.budget_remaining
                if remaining <= 0:
                    logger.warning("Query budget exhausted — stopping.")
                    break
                images = images[:remaining]
                batch_size = remaining

            # ── black-box query ──────────────────────────────
            probs = self.victim.query(images)

            # ── apply defense (if any) ───────────────────────
            if defense_fn is not None:
                probs = defense_fn(probs)

            all_images.append(images.cpu())
            all_labels.append(probs.cpu())

            self._queries_used += batch_size
            self._query_log.append(batch_size)

            pbar.set_postfix(
                used=self._queries_used,
                remaining=self.budget_remaining,
            )

            if self.budget_remaining <= 0:
                break

        logger.info(
            "Query phase complete  |  total_queries=%d  batches=%d",
            self._queries_used, len(self._query_log),
        )
        return all_images, all_labels


# ─────────────────────────────────────────────────────────────
#  Stand-alone sanity check
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Print all budget tiers
    for b in QueryBudget:
        logger.info("Budget tier: %-10s = %d", b.name, b.value)
