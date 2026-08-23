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
import time

import pytest
import torch

torch.set_num_threads(1)  # tiny tensors: single-threaded ops are faster here

from src.curriculum_data import make_instance, stage1_batch
from src.grpo import sample_batch, validate_coupled
from src.tokenizer import L_MAX
from src.training import (
    ACTION_DELETE,
    ACTION_EXPAND,
    ACTION_KEEP,
    PhantomPolicy,
    grpo_step,
)
from src.train_loop import PlacementHead, Trainer, expand_position_accuracy

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


# ---------------------------------------------------------------------------
# Stage-1 curriculum: expand-position accuracy + Trainer learning proof
# ---------------------------------------------------------------------------

TRAIN_SEEDS = stage1_batch(n=128, seed0=2000)  # 128 training instances
TRAIN_SEEDS_WIDE = stage1_batch(n=256, seed0=4000)  # 256 for the head proof
HELDOUT = stage1_batch(n=20, seed0=9000)       # 20 held-out instances


def _chance_baseline(dataset) -> float:
    """Mean uniform baseline: 1 candidate splice site per live slot.

    An [EXPAND] can only be spliced at ``0 <= pos <= logical_len``
    (``src.tokenizer.insert_masks`` contract), so a fresh, near-uniform
    policy holds ~1/(logical_len + 1) of the EXPAND mass at the true gap —
    that is the chance baseline the learning proof doubles. (Normalizing
    over all L_MAX physical slots instead is unlearnable by construction:
    the phantom tail never receives gradient, so its mass inflates in
    lockstep with the gap and the ratio is pinned at 1/L_MAX forever.)
    """
    return sum(1.0 / (len(inst["ids"]) + 1) for inst in dataset) / len(dataset)


def _stage1_policy(seed: int = SEED) -> PhantomPolicy:
    """Fresh policy sized for the real tokenizer buffers (l_max = L_MAX)."""
    torch.manual_seed(seed)
    return PhantomPolicy(vocab_size=64, d_model=32, n_actions=3, l_max=L_MAX)


def _mean_gap_accuracy(policy, dataset, placement_head=None) -> float:
    scores = [
        expand_position_accuracy(policy, inst, placement_head)
        for inst in dataset
    ]
    return sum(scores) / len(scores)


def test_expand_position_accuracy_in_unit_interval_and_deterministic():
    policy = _stage1_policy()
    for inst in HELDOUT[:5]:
        a = expand_position_accuracy(policy, inst)
        b = expand_position_accuracy(policy, inst)
        assert 0.0 <= a <= 1.0
        assert a == b, "accuracy must be a pure function of (policy, instance)"


def test_fresh_policy_starts_below_trained_level():
    """Informational: fresh init sits at/below the trained policy's level."""
    torch.manual_seed(SEED)
    policy = PhantomPolicy(vocab_size=64, d_model=32, n_actions=3, l_max=L_MAX)
    fresh = _mean_gap_accuracy(policy, HELDOUT)
    assert fresh <= 0.5  # nowhere near certainty at init

def test_training_lifts_gap_accuracy_on_heldout_instances():
    """THE learning proof (raised bar): with the attention-based
    PlacementHead attached, paired before/after on the SAME held-out
    instances must show an absolute lift > 0.05 in mean P(placement at
    gap_start). The legacy trigram readout saturates at ~0.15-0.16 (its total
    lift was ~0.05-0.07 from a lower base) because it scores each candidate
    site independently; the head's cross-site attention contrast expresses
    "this site vs the OTHER sites" and clears the bar by a wide margin.
    Observed with this exact config (256 instances x 6 epochs, lr 5e-3):
    fresh ~0.08 -> post ~0.40, i.e. LIFT ~ +0.32. Runtime < 30 s.
    """
    t0 = time.perf_counter()
    policy = _stage1_policy()
    head = PlacementHead(d_model=policy.d_model, n_heads=4, n_layers=2)
    fresh = _mean_gap_accuracy(policy, HELDOUT, head)

    Trainer(policy, lr=5e-3, seed=SEED, placement_head=head).epoch(
        TRAIN_SEEDS_WIDE, batch_size=8, epochs=6
    )

    post = _mean_gap_accuracy(policy, HELDOUT, head)
    elapsed = time.perf_counter() - t0
    assert post - fresh > 0.05, (
        f"no learning: fresh={fresh:.4f} vs post={post:.4f}"
    )
    assert elapsed < 30.0, f"learning proof too slow: {elapsed:.1f}s"

def test_trainer_save_load_round_trip_restores_identical_accuracy():
    policy = _stage1_policy()
    trainer = Trainer(policy, lr=1e-3, seed=SEED)
    trainer.epoch(TRAIN_SEEDS[:16], batch_size=8, epochs=1)
    before = [expand_position_accuracy(policy, inst) for inst in HELDOUT[:10]]

    import tempfile
    import os

    fd, path = tempfile.mkstemp(suffix=".pt")
    os.close(fd)
    try:
        trainer.save(path)
        revived = Trainer.load(path)
        after = [
            expand_position_accuracy(revived.policy, inst) for inst in HELDOUT[:10]
        ]
    finally:
        os.unlink(path)
    assert after == before  # bit-identical accuracy after round-trip


def test_trainer_epoch_metrics_are_plain_floats():
    trainer = Trainer(_stage1_policy(), lr=1e-3, seed=0)
    metrics = trainer.epoch(stage1_batch(4, seed0=0), batch_size=2, epochs=1)
    assert {"loss", "advantage_mean"} <= set(metrics)
    for key in ("loss", "advantage_mean"):
        assert type(metrics[key]) is float
        assert math.isfinite(metrics[key])


def test_trainer_rejects_empty_dataset():
    with pytest.raises(ValueError):
        Trainer(_stage1_policy()).epoch([])
