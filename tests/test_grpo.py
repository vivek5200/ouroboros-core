"""Tests for Coupled-GRPO antithetic variate sampling (Module 4, §3.3).

Paper contract: pairs of timesteps whose masks perfectly cover the target
sequence — ``m(1) ∪ m(2) = 1`` and ``m(1) ∩ m(2) = ∅`` — so the couple
shares one rollout's information and reward variance drops.
"""

import random

from src.grpo import (
    CURRICULUM,
    antithetic_pair,
    coupled_reward,
    sample_batch,
    validate_coupled,
)
from src.reward import compute_reward


# ---------------------------------------------------------------------------
# antithetic_pair: complement property across seeds and sizes
# ---------------------------------------------------------------------------


def test_complement_property_across_100_seeds_n_1_to_64():
    """m(1)[i] != m(2)[i] for every position, 100 seeds × n in 1..64."""
    for seed in range(100):
        rng = random.Random(seed)
        for n in range(1, 65):
            m1, m2 = antithetic_pair(n, rng)
            assert isinstance(m1, list) and isinstance(m2, list)
            assert len(m1) == n, f"seed={seed} n={n}: bad m1 length"
            assert len(m2) == n, f"seed={seed} n={n}: bad m2 length"
            for i, (a, b) in enumerate(zip(m1, m2)):
                assert type(a) is bool and type(b) is bool
                assert a != b, (
                    f"seed={seed} n={n} pos={i}: complement violated "
                    f"(m1={a}, m2={b})"
                )


def test_disjointness_explicit():
    """No position may be selected by both masks: m(1) ∩ m(2) = ∅."""
    m1, m2 = antithetic_pair(256, random.Random(7))
    assert not any(a and b for a, b in zip(m1, m2))


def test_coverage_explicit():
    """Every position is selected by at least one mask: m(1) ∪ m(2) = 1."""
    m1, m2 = antithetic_pair(256, random.Random(7))
    assert all(a or b for a, b in zip(m1, m2))


def test_both_sides_non_degenerate_over_stream():
    """Each side sees roughly half the tokens over a long stream (sanity)."""
    rng = random.Random(42)
    ones1 = sum(sum(m1) for m1, _ in (antithetic_pair(64, rng) for _ in range(50)))
    total = 50 * 64
    assert 0 < ones1 < total  # neither mask is empty in aggregate
    assert abs(ones1 / total - 0.5) < 0.1  # fair coin, ±10%


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def _stream(seed: int, n: int, pairs: int) -> list[tuple[bool, ...]]:
    rng = random.Random(seed)
    out = []
    for _ in range(pairs):
        m1, m2 = antithetic_pair(n, rng)
        out.append(tuple(m1))
        out.append(tuple(m2))
    return out


def test_same_seed_identical_masks():
    """Same seed ⇒ byte-identical mask stream (deterministic given rng state)."""
    assert _stream(1234, 33, 25) == _stream(1234, 33, 25)


def test_antithetic_pair_consumes_exactly_n_draws():
    """Contract 'deterministic given rng state' ⇒ exactly n draws per pair."""

    class CountingRng(random.Random):
        draws = 0

        def random(self):
            self.draws += 1
            return super().random()

    for n in (1, 5, 64):
        rng = CountingRng()
        antithetic_pair(n, rng)
        assert rng.draws == n, f"expected {n} draws, got {rng.draws}"


def _stream_from(rng: random.Random, n: int = 17, pairs: int = 40):
    out = []
    for _ in range(pairs):
        m1, m2 = antithetic_pair(n, rng)
        out.append(tuple(m1))
        out.append(tuple(m2))
    return out


def test_rng_state_determinism_continues_after_foreign_draws():
    """Same state ⇒ identical masks; once states diverge, streams separate."""
    rng_a = random.Random(99)
    rng_b = random.Random(99)
    assert antithetic_pair(32, rng_a) == antithetic_pair(32, rng_b)
    rng_a.getrandbits(777)  # disturb only rng_a's state
    # Streams over 40×34 bits coincide with probability ~2⁻¹³⁶⁰ if states differ.
    assert _stream_from(rng_a) != _stream_from(rng_b)


def test_different_seeds_differ_with_overwhelming_probability():
    """Two 40×17-bit streams colliding has probability ~2⁻⁶⁸⁰: demand difference."""
    assert _stream(1, 17, 40) != _stream(2, 17, 40)
    assert _stream(123, 17, 40) != _stream(456, 17, 40)


# ---------------------------------------------------------------------------
# sample_batch: shape + independence
# ---------------------------------------------------------------------------


def test_sample_batch_shape_and_types():
    pairs = sample_batch(n_tokens=48, batch=12, seed=2024)
    assert type(pairs) is list
    assert len(pairs) == 12
    for pair in pairs:
        assert type(pair) is tuple and len(pair) == 2
        m1, m2 = pair
        assert type(m1) is list and type(m2) is list
        assert len(m1) == 48 and len(m2) == 48
        assert all(type(b) is bool for b in m1 + m2)


def test_sample_batch_pairs_are_valid_couples():
    for m1, m2 in sample_batch(n_tokens=64, batch=16, seed=5):
        assert validate_coupled(m1, m2)


def test_sample_batch_pairs_independent_not_aliased():
    pairs = sample_batch(n_tokens=32, batch=10, seed=31337)
    ids = {id(m) for pair in pairs for m in pair}
    assert len(ids) == 20, "masks must be distinct objects, not aliased views"
    contents = {(tuple(m1), tuple(m2)) for m1, m2 in pairs}
    assert len(contents) == 10, "each pair draws fresh randomness"


def test_sample_batch_reproducible_from_seed():
    assert sample_batch(24, 6, seed=77) == sample_batch(24, 6, seed=77)
    assert sample_batch(24, 6, seed=77) != sample_batch(24, 6, seed=78)


def test_sample_batch_empty():
    assert sample_batch(n_tokens=16, batch=0, seed=1) == []


# ---------------------------------------------------------------------------
# validate_coupled
# ---------------------------------------------------------------------------


def test_validate_accepts_true_complement():
    assert validate_coupled([True, False, True], [False, True, False])
    assert validate_coupled([], [])
    assert validate_coupled([False], [True])


def test_validate_rejects_overlap_breaks_disjointness():
    # Both True at index 1: m(1) ∩ m(2) ≠ ∅.
    assert not validate_coupled([True, True, False], [False, True, False])


def test_validate_rejects_gap_breaks_coverage():
    # Both False at index 0: m(1) ∪ m(2) ≠ 1.
    assert not validate_coupled([False, True], [False, False])


def test_validate_rejects_length_mismatch():
    assert not validate_coupled([True, False], [False])
    assert not validate_coupled([], [True])


def test_antithetic_pair_output_passes_its_own_validator():
    for seed in range(20):
        m1, m2 = antithetic_pair((seed % 64) + 1, random.Random(seed))
        assert validate_coupled(m1, m2)


# ---------------------------------------------------------------------------
# CURRICULUM: exactly the 3 paper stages
# ---------------------------------------------------------------------------

_ALLOWED_DENSITIES = ("sparse", "medium", "dense")


def test_curriculum_exactly_three_stages_with_required_fields():
    assert isinstance(CURRICULUM, list)
    assert len(CURRICULUM) == 3
    for stage in CURRICULUM:
        assert isinstance(stage, dict)
        assert {"name", "description", "typical_mask_density"} <= set(stage)
        assert isinstance(stage["name"], str) and stage["name"]
        assert isinstance(stage["description"], str) and stage["description"]
        assert stage["typical_mask_density"] in _ALLOWED_DENSITIES


def test_curriculum_stage_order_matches_paper():
    names = [stage["name"] for stage in CURRICULUM]
    assert names == [
        "syntactic_boundaries",  # phase 1: SFT on syntactic boundaries
        "trivial_subtrees",      # phase 2: easy, e.g. variable renaming
        "macro_migrations",      # phase 3: hard, e.g. method extraction
    ]


def test_curriculum_descriptions_name_the_paper_workloads():
    blob = " ".join(stage["description"].lower() for stage in CURRICULUM)
    assert "variable renaming" in blob          # phase 2 exemplar
    assert "method extraction" in blob          # phase 3 exemplar
    assert "class splitting" in blob            # phase 3 exemplar
    assert "syntactic" in blob                  # phase 1 focus


def test_curriculum_density_is_monotonically_nondecreasing():
    ranks = [
        _ALLOWED_DENSITIES.index(stage["typical_mask_density"])
        for stage in CURRICULUM
    ]
    assert ranks == sorted(ranks), "harder stages mask more of the sequence"


# ---------------------------------------------------------------------------
# coupled_reward: Fuzzy Proxy applied per side of the couple
# ---------------------------------------------------------------------------


def test_coupled_reward_equals_compute_reward_per_side():
    r1, r2 = coupled_reward(True, True, False, False, True, True)
    assert r1 == compute_reward(True, True, False)
    assert r2 == compute_reward(False, True, True)
    assert abs(r1 - 0.4) < 1e-9
    assert abs(r2 - 0.9) < 1e-9


def test_coupled_reward_extremes():
    assert coupled_reward(True, True, True, False, False, False) == (1.0, 0.0)


def test_coupled_reward_returns_tuple_of_floats():
    out = coupled_reward(True, False, False, True, True, True)
    assert type(out) is tuple and len(out) == 2
    assert all(type(r) is float for r in out)


def test_coupled_reward_symmetric_under_side_swap():
    a = coupled_reward(True, False, True, False, True, False)
    b = coupled_reward(False, True, False, True, False, True)
    assert a == (b[1], b[0])
