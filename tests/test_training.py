"""Tests for the Coupled-GRPO training-step scaffold (Module 1 RL wave).

Paper contract under test: ``PhantomPolicy`` is a tiny policy over fixed
L_MAX-style buffers whose gradients flow end-to-end through an antithetic
coupled-GRPO step — Embedding → mean-pool over unmasked (logical) positions
→ Linear action head (KEEP / EXPAND / DELETE placeholders). Real EXPAND and
DELETE semantics land with the diffusion loop; this scaffold only has to make
the learning signal measurable: positive param movement, bounded advantages,
seed determinism, complement property preserved through the pipeline, and a
zero-advantage couple producing ~zero parameter change.
"""

import math

import pytest
import torch

torch.set_num_threads(1)  # tiny tensors: single-threaded ops are faster here

from src.grpo import sample_batch, validate_coupled
from src.training import (
    ACTION_DELETE,
    ACTION_EXPAND,
    ACTION_KEEP,
    PhantomPolicy,
    grpo_step,
)

SEED = 20240607
VOCAB = 64          # vocab_size <= 64 per scaffold budget
D_MODEL = 16
L_MAX_TEST = 32     # L_MAX-style buffer length <= 32 for training tests


def _fresh_policy() -> PhantomPolicy:
    torch.manual_seed(SEED)
    return PhantomPolicy(
        vocab_size=VOCAB, d_model=D_MODEL, n_actions=3, l_max=L_MAX_TEST
    )


def _default_run(lr: float = 1e-3):
    """One full grpo_step on a fresh policy with sample_batch masks."""
    policy = _fresh_policy()
    pairs = sample_batch(n_tokens=16, batch=8, seed=123)
    result = grpo_step(policy, pairs, lr=lr, seed=SEED)
    return result


# ---------------------------------------------------------------------------
# PhantomPolicy shape/structure contracts
# ---------------------------------------------------------------------------


def test_policy_logits_shape_and_action_constants():
    policy = _fresh_policy()
    ids = torch.arange(16) % (VOCAB - 1) + 1  # real ids start at 1 (no pad)
    mask = [True] * 16
    logits = policy(ids, mask)
    assert logits.shape == (1, 3)
    # Actions >= 3: KEEP=0, EXPAND-ish=1, DELETE-ish=2 are distinct slots.
    assert (ACTION_KEEP, ACTION_EXPAND, ACTION_DELETE) == (0, 1, 2)


def test_policy_rejects_fewer_than_three_actions():
    with pytest.raises(ValueError):
        PhantomPolicy(vocab_size=VOCAB, d_model=4, n_actions=2)


def test_forward_pools_only_unmasked_logical_positions():
    """Gradient reaches embeddings of unmasked positions only.

    Mean-pooling over unmasked (logical) positions means masked-out rows of
    the buffer contribute nothing: after backward, embedding rows for ids
    that only appear at masked-out positions must have exactly zero grad.
    """
    policy = _fresh_policy()
    L = 6
    ids = torch.tensor([1, 2, 3, 4, 5, 6])
    mask = [True, True, True, False, False, False]
    logits = policy(ids, mask)
    logits.sum().backward()
    grad = policy.embed.weight.grad
    assert grad is not None
    for tok in (1, 2, 3):
        assert grad[tok].abs().sum() > 0, f"unmasked id {tok} got no gradient"
    for tok in (4, 5, 6):
        assert grad[tok].abs().sum() == 0, (
            f"masked-out id {tok} leaked gradient through the pool"
        )


# ---------------------------------------------------------------------------
# grpo_step: learning signal, bounds, determinism
# ---------------------------------------------------------------------------


def test_param_delta_positive_after_step():
    """Learning happens: parameters move measurably in one Adam step."""
    result = _default_run()
    assert result["param_delta"] > 0.0


def test_loss_finite_and_advantage_within_reward_bounds():
    """Loss is finite; advantage stays in [-1, 1] because rewards are [0, 1]."""
    result = _default_run()
    assert math.isfinite(result["loss"])
    assert -1.0 <= result["advantage_mean"] <= 1.0


def test_same_seed_identical_loss_across_two_fresh_runs():
    r1 = _default_run()
    r2 = _default_run()
    assert r1["loss"] == pytest.approx(r2["loss"], abs=1e-12)
    assert r1["advantage_mean"] == pytest.approx(r2["advantage_mean"], abs=1e-12)
    assert r1["param_delta"] == pytest.approx(r2["param_delta"], abs=1e-12)


def test_lr_argument_controls_step_size():
    """First Adam step magnitude scales with lr: bigger lr ⇒ bigger delta."""
    small = _default_run(lr=1e-3)["param_delta"]
    big = _default_run(lr=1e-1)["param_delta"]
    assert big > small > 0.0


def test_result_keys_are_plain_floats():
    result = _default_run()
    assert set(result) == {"loss", "advantage_mean", "param_delta"}
    for value in result.values():
        assert type(value) is float


# ---------------------------------------------------------------------------
# Coupling with src.grpo: complement property survives the pipeline
# ---------------------------------------------------------------------------


def test_complement_property_holds_through_pipeline():
    """Masks from sample_batch stay perfect complements entering grpo_step."""
    pairs = sample_batch(n_tokens=L_MAX_TEST, batch=10, seed=999)
    for m1, m2 in pairs:
        assert len(m1) == L_MAX_TEST and len(m2) == L_MAX_TEST
        assert validate_coupled(m1, m2), "pipeline broke m(1) ∪ m(2)=1, m∩=∅"
    # And the step consumes exactly such masks without error.
    result = grpo_step(_fresh_policy(), pairs, seed=SEED)
    assert math.isfinite(result["loss"])


def test_custom_reward_fn_is_used_for_both_sides():
    """A reward_fn(m1, m2)-style callable drives both outcome triples."""
    def max_vs_none(m1, m2):
        return (True, True, True), (False, False, False)

    result = grpo_step(_fresh_policy(), sample_batch(16, 4, seed=7),
                       reward_fn=max_vs_none, seed=SEED)
    # Every couple: r1 = 1.0, r2 = 0.0 ⇒ advantage ≡ +1.
    assert result["advantage_mean"] == pytest.approx(1.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Zero-advantage couples ⇒ ~zero parameter change
# ---------------------------------------------------------------------------


def test_zero_advantage_couples_produce_zero_param_change():
    """Exactly-zero advantage ⇒ zero loss signal ⇒ Adam moves nothing."""
    def identical_outcomes(m1, m2):
        return (True, True, False), (True, True, False)  # r1 == r2 == 0.4

    policy = _fresh_policy()
    pairs = sample_batch(n_tokens=16, batch=6, seed=42)
    result = grpo_step(policy, pairs, reward_fn=identical_outcomes, seed=SEED)
    assert result["advantage_mean"] == 0.0
    assert result["loss"] == pytest.approx(0.0, abs=1e-12)
    assert result["param_delta"] == pytest.approx(0.0, abs=1e-9)
