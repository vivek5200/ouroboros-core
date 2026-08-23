"""Module 4, §3.3: Coupled-GRPO with antithetic variate sampling.

For each training instance a pair of timesteps is selected and two
*complementary* masks perfectly cover the target sequence:

    m(1) ∪ m(2) = 1        m(1) ∩ m(2) = ∅

Because the couple partitions the same sequence, its two rollouts share all
information; scoring each side with the Fuzzy Proxy reward and averaging over
the couple reduces gradient estimator variance relative to independent masks.

Masks are plain ``list[bool]`` (stdlib only) — ``True`` marks a token selected
for that side of the couple.
"""

import random

from src.reward import compute_reward

__all__ = [
    "CURRICULUM",
    "antithetic_pair",
    "coupled_reward",
    "sample_batch",
    "validate_coupled",
]


# ---------------------------------------------------------------------------
# Antithetic mask pairs (§3.3)
# ---------------------------------------------------------------------------


def antithetic_pair(n: int, rng: random.Random) -> tuple[list[bool], list[bool]]:
    """Draw one coupled pair of complementary length-``n`` masks.

    Each position gets one fair Bernoulli draw from ``rng``; side 1 takes the
    token iff the draw succeeds and side 2 takes exactly the complement. The
    complement property therefore holds by construction, deterministically for
    any given ``rng`` state, and exactly ``n`` draws are consumed per pair so
    successive pairs stay reproducible under a single seeded stream.

    Args:
        n: Length of both masks (each side covers ~n/2 tokens in expectation).
        rng: Source of randomness; advanced by exactly ``n`` draws.

    Returns:
        Tuple ``(m1, m2)`` of ``list[bool]`` with ``m1[i] != m2[i]`` for all i,
        i.e. ``m1 | m2 == all-True`` and ``m1 & m2 == all-False``.

    Raises:
        ValueError: If ``n`` is negative.
    """
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    m1 = [rng.random() < 0.5 for _ in range(n)]
    m2 = [not b for b in m1]
    return m1, m2


def validate_coupled(m1: list[bool], m2: list[bool]) -> bool:
    """Check the complement property: disjoint AND covering, same length.

    Args:
        m1: First mask of the couple.
        m2: Second mask of the couple.

    Returns:
        True iff ``len(m1) == len(m2)`` and ``m1[i] != m2[i]`` at every
        position (equivalently XOR holds everywhere). Vacuously True for two
        empty masks.
    """
    return len(m1) == len(m2) and all(a != b for a, b in zip(m1, m2))


def sample_batch(
    n_tokens: int, batch: int, seed: int
) -> list[tuple[list[bool], list[bool]]]:
    """Draw ``batch`` independent antithetic pairs from one seeded rng.

    A single ``random.Random(seed)`` stream feeds every pair, so the whole
    batch is bit-reproducible from ``seed`` while each pair consumes its own
    fresh draws (no aliasing or reuse between couples).

    Args:
        n_tokens: Length of every mask (tokens covered by the couple).
        batch: Number of pairs to draw.
        seed: Seed for the shared random stream.

    Returns:
        List of ``batch`` ``(m1, m2)`` tuples as produced by
        :func:`antithetic_pair`.

    Raises:
        ValueError: If ``n_tokens`` or ``batch`` is negative.
    """
    if n_tokens < 0:
        raise ValueError(f"n_tokens must be non-negative, got {n_tokens}")
    if batch < 0:
        raise ValueError(f"batch must be non-negative, got {batch}")
    rng = random.Random(seed)
    return [antithetic_pair(n_tokens, rng) for _ in range(batch)]


# ---------------------------------------------------------------------------
# Three-phase curriculum (§3.3): easy → hard subtree migration workloads
# ---------------------------------------------------------------------------

CURRICULUM: list[dict[str, str]] = [
    {
        "name": "syntactic_boundaries",
        "description": (
            "Phase 1 (SFT warm-up): spans anchored at syntactic boundaries "
            "(statement and block edges); the model learns local grammar by "
            "completing small well-delimited regions."
        ),
        "typical_mask_density": "sparse",
    },
    {
        "name": "trivial_subtrees",
        "description": (
            "Phase 2 (easy subtrees): mechanically invertible edits such as "
            "variable renaming; medium-sized AST regions must be reproduced "
            "from context on both sides of the couple."
        ),
        "typical_mask_density": "medium",
    },
    {
        "name": "macro_migrations",
        "description": (
            "Phase 3 (hard migrations): structure-moving refactors such as "
            "method extraction and class splitting; dense masks force "
            "long-range coherent regeneration across the target sequence."
        ),
        "typical_mask_density": "dense",
    },
]


# ---------------------------------------------------------------------------
# Coupled Fuzzy Proxy reward
# ---------------------------------------------------------------------------


def coupled_reward(
    parses1: bool,
    checks1: bool,
    tests1: bool,
    parses2: bool,
    checks2: bool,
    tests2: bool,
) -> tuple[float, float]:
    """Apply the Fuzzy Proxy reward to each side of an antithetic couple.

    Reuses :func:`src.reward.compute_reward`
    (``0.1·Parses + 0.3·TypeChecks + 0.6·PassesTests``) per mask's outcome
    triple; antithetic pairing correlates the two outcomes, which is what
    yields variance reduction when the couple is averaged downstream.

    Returns:
        Tuple ``(reward_side1, reward_side2)``, each in ``[0.0, 1.0]``.
    """
    return (
        compute_reward(parses1, checks1, tests1),
        compute_reward(parses2, checks2, tests2),
    )
