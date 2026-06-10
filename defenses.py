"""
defenses.py
===========
Defense mechanisms applied to victim model outputs.

All defenses are implemented as callable objects that transform
probability vectors **before** the attacker sees them.  They are
designed to be drop-in replaceable via the ``defense_fn`` argument
in the attack pipeline.

Three effective defenses (weak ones removed after empirical evaluation):
  1. Throttling          — blocks attacker after a query budget is exceeded
  2. Prediction Poisoning — corrupts a fraction of returned probability vectors
  3. Adaptive Noise      — escalates noise once a query-count threshold is exceeded

Author  : MSc Advanced Computer Science (Data Analytics) Dissertation
Python  : 3.9.25 (strict – no 3.10+ features)
"""

import logging
from typing import List, Optional, Tuple

import torch

# ── logging ──────────────────────────────────────────────────
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)


# ─────────────────────────────────────────────────────────────
#  Utility: renormalise a probability tensor so rows sum to 1
# ─────────────────────────────────────────────────────────────
def _renormalise(probs: torch.Tensor) -> torch.Tensor:
    """Clamp negatives and renormalise each row to sum to 1."""
    probs = probs.clamp(min=0.0)
    row_sums = probs.sum(dim=1, keepdim=True)
    # Avoid division by zero
    row_sums = row_sums.clamp(min=1e-10)
    return probs / row_sums


# ═════════════════════════════════════════════════════════════
#  DEFENSE 1: THROTTLING (QUERY-RATE LIMIT)
# ═════════════════════════════════════════════════════════════

class ThrottlingDefense:
    """Block queries after a maximum count is reached.

    When the budget is exceeded, the defense returns a **uniform
    distribution** so that the attacker gains no useful information.

    Parameters
    ----------
    max_queries : int
        Maximum allowed queries (250, 500, 1000 recommended).
    """

    def __init__(self, max_queries: int = 1000) -> None:
        self.max_queries = max_queries
        self._count: int = 0
        logger.info("ThrottlingDefense initialised  |  max=%d", max_queries)

    def __call__(self, probs: torch.Tensor) -> torch.Tensor:
        batch_size = probs.size(0)
        num_classes = probs.size(1)

        # How many can still go through?
        remaining = max(0, self.max_queries - self._count)

        if remaining <= 0:
            # All blocked — return uniform
            logger.debug("Throttled: returning uniform for batch of %d", batch_size)
            return torch.ones_like(probs) / num_classes

        if remaining < batch_size:
            # Partial block: only `remaining` real predictions get through;
            # the rest are replaced with a uniform distribution.
            result = probs.clone()
            result[remaining:] = 1.0 / num_classes
            self._count += remaining          # count only the queries served
            return result

        self._count += batch_size
        return probs

    def reset(self) -> None:
        self._count = 0

    def __repr__(self) -> str:
        return f"Throttling(max={self.max_queries}, used={self._count})"


# ═════════════════════════════════════════════════════════════
#  DEFENSE 2: PREDICTION POISONING
# ═════════════════════════════════════════════════════════════

class PredictionPoisoningDefense:
    """Deliberately return wrong probability vectors.

    With probability ``poison_rate``, the output is replaced by a
    randomly shuffled version of the true probabilities.  This
    corrupts the attacker's training set without dramatically
    changing the average output distribution.

    Parameters
    ----------
    poison_rate : float
        Fraction of queries to poison (0–1).  E.g. 0.3 = 30%.
    """

    def __init__(self, poison_rate: float = 0.3) -> None:
        self.poison_rate = poison_rate
        logger.info(
            "PredictionPoisoningDefense initialised  |  rate=%.2f", poison_rate,
        )

    def __call__(self, probs: torch.Tensor) -> torch.Tensor:
        batch_size = probs.size(0)
        num_classes = probs.size(1)

        result = probs.clone()
        # Decide which samples to poison (fully vectorised — no Python loop)
        mask = torch.rand(batch_size, device=probs.device) < self.poison_rate
        if mask.any():
            rows = mask.nonzero(as_tuple=True)[0]
            # One independent random permutation per poisoned row, built by
            # argsort-ing a random key matrix, then gathered in a single op.
            perms = torch.rand(
                rows.numel(), num_classes, device=probs.device
            ).argsort(dim=1)
            result[rows] = torch.gather(probs[rows], 1, perms)

        return result
    
    def reset(self) -> None:
        pass  # stateless — no counters or timestamps to reset

    def __repr__(self) -> str:
        return f"PredictionPoisoning(rate={self.poison_rate})"


# ═════════════════════════════════════════════════════════════
#  DEFENSE 3: ADAPTIVE NOISE
# ═════════════════════════════════════════════════════════════

class AdaptiveNoiseDefense:
    """Increase noise dynamically when a high-volume query burst is detected.

    Escalation is driven by a **logical query-count window** rather than
    wall-clock time, so behaviour is deterministic and reproducible across
    machines (dissertation-grade repeatability). Once the cumulative number of
    queries served exceeds ``escalation_after`` queries, the noise ``sigma`` is
    multiplied by ``escalation_factor`` (capped at ``max_sigma``). This models
    an owner that ramps up output perturbation once a client's query volume
    crosses a suspicious threshold.

    Parameters
    ----------
    base_sigma : float
        Baseline Gaussian noise standard deviation.
    escalation_after : int
        Number of cumulative queries after which noise escalates.
    escalation_factor : float
        Multiplier applied to ``base_sigma`` when triggered.
    max_sigma : float
        Hard upper cap on noise sigma.
    """

    def __init__(
        self,
        base_sigma: float = 0.01,
        escalation_after: int = 1000,
        escalation_factor: float = 5.0,
        max_sigma: float = 0.5,
    ) -> None:
        self.base_sigma = base_sigma
        self.escalation_after = escalation_after
        self.escalation_factor = escalation_factor
        self.max_sigma = max_sigma

        self._count: int = 0
        self._current_sigma: float = base_sigma
        self._escalated: bool = False

        logger.info(
            "AdaptiveNoiseDefense initialised  |  base_sigma=%.3f  "
            "escalate_after=%d queries  escalation=%.1fx",
            base_sigma, escalation_after, escalation_factor,
        )

    def _update_state(self, batch_size: int) -> None:
        """Record served queries and recompute the current noise level."""
        self._count += batch_size

        if self._count > self.escalation_after:
            self._current_sigma = min(
                self.base_sigma * self.escalation_factor,
                self.max_sigma,
            )
            if not self._escalated:
                logger.warning(
                    "Adaptive defense ESCALATED  |  queries=%d  sigma=%.3f -> %.3f",
                    self._count, self.base_sigma, self._current_sigma,
                )
                self._escalated = True
        else:
            self._current_sigma = self.base_sigma
            self._escalated = False

    def __call__(self, probs: torch.Tensor) -> torch.Tensor:
        self._update_state(probs.size(0))
        noise = torch.randn_like(probs) * self._current_sigma
        noisy = probs + noise
        return _renormalise(noisy)

    def reset(self) -> None:
        self._count = 0
        self._current_sigma = self.base_sigma
        self._escalated = False

    def __repr__(self) -> str:
        return (
            f"AdaptiveNoise(sigma={self._current_sigma:.3f}, "
            f"escalated={self._escalated})"
        )


# ═════════════════════════════════════════════════════════════
#  DEFENSE REGISTRY
# ═════════════════════════════════════════════════════════════

def get_all_defenses() -> List[Tuple[str, object]]:
    """Return a list of ``(name, defense_fn)`` tuples covering
    every defence configuration used in the evaluation.

    Returns
    -------
    List[Tuple[str, callable]]
        Named defenses with their parameter variants.
    """
    # Only the defenses proven effective (attack result = LOW in evaluation).
    # Weak defenses (rounding, gaussian, laplace, topK, poison_0.10,
    # adaptive_base) have been removed — they offered ≤ 10 pp fidelity drop
    # and the attacker still achieved MEDIUM success against them.
    defenses: List[Tuple[str, object]] = [
        # Baseline — no defense
        ("none", None),

        # Defense 1 — Throttling  [STRONG: ~40–49 pp fidelity drop]
        ("throttle_250",  ThrottlingDefense(max_queries=250)),
        ("throttle_500",  ThrottlingDefense(max_queries=500)),
        ("throttle_1000", ThrottlingDefense(max_queries=1000)),

        # Defense 2 — Prediction Poisoning  [MODERATE-STRONG: ~16–26 pp drop]
        ("poison_0.30", PredictionPoisoningDefense(poison_rate=0.30)),
        ("poison_0.50", PredictionPoisoningDefense(poison_rate=0.50)),

        # Defense 3 — Adaptive Noise (aggressive only)  [MODERATE: ~7 pp drop]
        ("adaptive_aggressive", AdaptiveNoiseDefense(
            base_sigma=0.02, escalation_factor=10.0, escalation_after=500,
        )),
    ]
    return defenses


def get_defense_by_name(name: str) -> Optional[object]:
    """Look up a single defence by name.

    Parameters
    ----------
    name : str
        Defence name as listed in :func:`get_all_defenses`.

    Returns
    -------
    callable or None
        The defence callable, or ``None`` if name is ``"none"``.
    """
    registry = dict(get_all_defenses())
    if name not in registry:
        raise ValueError(
            f"Unknown defense '{name}'. "
            f"Available: {list(registry.keys())}"
        )
    return registry[name]


# ─────────────────────────────────────────────────────────────
#  Stand-alone sanity check
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Quick smoke test with random probabilities
    dummy_probs = torch.softmax(torch.randn(4, 10), dim=1)
    logger.info("Original probs (row 0): %s", dummy_probs[0].tolist())

    for name, defense_fn in get_all_defenses():
        if defense_fn is None:
            result = dummy_probs
        else:
            result = defense_fn(dummy_probs)
        logger.info(
            "%-25s  sum=%.4f  max=%.4f",
            name, result[0].sum().item(), result[0].max().item(),
        )
