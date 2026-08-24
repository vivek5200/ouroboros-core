"""Phase 2-A TDD suite: ``DiffusionPolicy`` — a real (tiny) bidirectional
diffusion transformer behind the stage-1 ``Trainer`` interface.

Paper contract under test: the PhantomPolicy mean-pool stub proved learning
but cannot express per-position structural decisions. ``DiffusionPolicy``
replaces the pool with a genuine transformer stack (token + learned
positional embeddings → pre-norm bidirectional self-attention → per-position
action head) while keeping every stage-1 contract: fixed-shape buffers,
sentinel-aware masking (``src.tokenizer.IGNORE_ID`` excluded from attention),
bit-reproducible evaluation, and the paired-lift learning proof through the
UNCHANGED ``Trainer``/``expand_position_accuracy`` machinery.
"""

import os
import tempfile
import time

import pytest
import torch

torch.set_num_threads(1)  # tiny tensors: single-threaded ops are faster here

from src.curriculum_data import stage1_batch
from src.tokenizer import IGNORE_ID, L_MAX, MASK_ID, PAD_ID
from src.training import (
    ACTION_EXPAND,
    DiffusionPolicy,
    PhantomPolicy,
)
from src.train_loop import (
    Trainer,
    _placement_probs,
    _position_logits,
    expand_position_accuracy,
)
from src.grpo import antithetic_pair  # noqa: F401  (interface smoke)

SEED = 20240607
VOCAB = 64


def _fresh(d_model: int = 64) -> DiffusionPolicy:
    torch.manual_seed(SEED)
    return DiffusionPolicy(
        vocab_size=VOCAB, d_model=d_model, n_actions=3, l_max=L_MAX
    )


# ---------------------------------------------------------------------------
# forward: shapes, sentinel handling, validation
# ---------------------------------------------------------------------------


def test_forward_shapes_per_position_logits_and_hidden_states():
    policy = _fresh()
    ids = [3, 1, 4, 1, 5, 9, 2, 6, MASK_ID, 5]  # MASK placeholder rides along
    logits, h = policy(ids)
    assert logits.shape == (len(ids), 3)
    assert h.shape == (len(ids), policy.d_model)
    assert torch.isfinite(logits).all()


def test_forward_accepts_full_l_max_front_packed_buffer():
    policy = _fresh()
    live = [7, 2, 9, 4]
    ids = torch.tensor(live + [PAD_ID] * (L_MAX - len(live)))
    logits, h = policy(ids)
    assert logits.shape == (L_MAX, 3)
    assert h.shape == (L_MAX, policy.d_model)


def test_forward_validates_length_vocab_and_sentinel_floor():
    policy = _fresh()
    with pytest.raises(ValueError):
        policy(torch.arange(VOCAB + 4))          # id outside vocabulary
    with pytest.raises(ValueError):
        policy([1, -3, 4])                       # below IGNORE_ID floor
    with pytest.raises(ValueError):
        policy(torch.zeros(policy.l_max + 1, dtype=torch.long))  # overlong
    with pytest.raises(ValueError):
        policy([])                               # empty buffer


def test_all_ignore_buffer_degrades_gracefully_without_nans():
    """Fully-masked buffers fall back to unmasked attention instead of NaN."""
    policy = _fresh()
    logits, _ = policy([IGNORE_ID] * 6)
    assert torch.isfinite(logits).all()


# ---------------------------------------------------------------------------
# key_padding_mask: IGNORE slots are invisible to attention
# ---------------------------------------------------------------------------


def test_ignore_tail_leaves_live_outputs_unchanged():
    """Appending [IGNORE] filler must not move live-row outputs.

    Masked keys receive exactly zero attention mass, so the live rows'
    softmax renormalizes to the identical distribution they had without the
    filler — outputs agree to float tolerance even though the tensor is
    longer (different kernel tiling forbids bitwise equality).
    """
    policy = _fresh()
    base = torch.tensor([3, 1, 4, 1, 5, 9, 2, 6])
    padded = torch.cat([base, torch.full((6,), IGNORE_ID, dtype=torch.long)])
    with torch.no_grad():
        short, _ = policy(base)
        long_, _ = policy(padded)
    assert torch.allclose(short, long_[: base.shape[0]], atol=1e-5, rtol=0.0)


def test_ignore_positions_excluded_via_gradient_flow():
    """Gradients reach NOTHING whose only exposure is IGNORE slots.

    Loss reads live rows only: IGNORE rows' values are masked out of every
    live attention (zero mass) and their own output rows are unread, so both
    their token-embedding row (shared PAD row via sentinel clamp) and their
    positional rows get exactly zero gradient while every live row moves.
    """
    policy = _fresh()
    live = [3, 1, 4, 1, 5, 9, 2, 6]
    ids = torch.tensor(live + [IGNORE_ID] * 4)
    logits, _ = policy(ids)
    logits[: len(live)].sum().backward()

    emb_grad = policy.embed.weight.grad
    assert emb_grad is not None
    assert float(emb_grad[0].abs().sum()) == 0.0, (
        "PAD/sentinel embedding row moved through IGNORE-only slots"
    )
    for tok in set(live):
        assert float(emb_grad[tok].abs().sum()) > 0.0, (
            f"live id {tok} received no gradient"
        )

    pos_grad = policy.pos_embed.weight.grad
    assert pos_grad is not None
    for pos in range(len(live), len(live) + 4):
        assert float(pos_grad[pos].abs().sum()) == 0.0, (
            f"positional row {pos} (IGNORE slot) received gradient"
        )
    for pos in range(len(live)):
        assert float(pos_grad[pos].abs().sum()) > 0.0


def test_gradients_flow_through_whole_stack():
    """End-to-end signal: encoder weights + head all participate."""
    policy = _fresh()
    logits, _ = policy([2, 8, 5, 1, 3])
    logits.sum().backward()
    last = policy.encoder.layers[-1]
    assert last.self_attn.in_proj_weight.grad is not None
    assert float(last.self_attn.in_proj_weight.grad.abs().sum()) > 0.0
    assert float(last.linear1.weight.grad.abs().sum()) > 0.0
    assert float(policy.action_head.weight.grad.abs().sum()) > 0.0


# ---------------------------------------------------------------------------
# Parameter budget: CPU epochs must stay in seconds
# ---------------------------------------------------------------------------


def test_parameter_count_under_budget_at_d_model_64():
    policy = _fresh()
    n_params = sum(p.numel() for p in policy.parameters())
    assert 0 < n_params <= 1_500_000, f"parameter budget blown: {n_params}"


def test_architecture_shape_is_two_pre_norm_encoder_layers_plus_final_norm():
    policy = _fresh()
    assert len(policy.encoder.layers) == 2
    layer = policy.encoder.layers[0]
    assert layer.norm_first is True
    assert layer.self_attn.num_heads == 4
    assert layer.linear1.out_features == 256
    assert policy.encoder.norm is not None  # final LayerNorm


# ---------------------------------------------------------------------------
# site_logits protocol: BOTH policies expose it; Phantom stays bit-identical
# ---------------------------------------------------------------------------


def test_site_logits_exists_on_both_policies():
    torch.manual_seed(SEED)
    phantom = PhantomPolicy(vocab_size=VOCAB, d_model=16, n_actions=3, l_max=32)
    diffusion = _fresh()
    ids_p = torch.arange(1, 13) % (VOCAB - 1) + 1
    for pol, ids in ((phantom, ids_p), (diffusion, ids_p.tolist())):
        assert callable(getattr(pol, "site_logits", None))
        out = pol.site_logits(ids)
        assert out.shape == (ids.shape[0] if hasattr(ids, "shape") else len(ids), 3)
        assert torch.isfinite(out).all()


def test_phantom_site_logits_matches_legacy_position_logits_bitwise():
    torch.manual_seed(SEED)
    policy = PhantomPolicy(vocab_size=VOCAB, d_model=16, n_actions=3, l_max=32)
    ids = torch.tensor([5, 2, 7, 1, 3, 8, 4, 6, 2, 9])
    for logical_len in (None, 1, 5, len(ids)):
        legacy = _position_logits(policy, ids, logical_len)
        mine = policy.site_logits(ids, logical_len)
        assert torch.equal(mine, legacy), f"divergence at logical_len={logical_len}"
        # Keyword form must be accepted too and agree bit-for-bit.
        assert torch.equal(
            policy.site_logits(ids, logical_len=logical_len), legacy
        )


def test_diffusion_site_logits_scopes_to_candidate_site_window():
    """With ``logical_len`` given, exactly rows 0..logical_len come back."""
    policy = _fresh()
    live = [4, 2, 8, 6, 3]
    ids = torch.tensor(live + [PAD_ID] * 5)
    out = policy.site_logits(ids, logical_len=len(live))
    assert out.shape == (len(live) + 1, 3)
    # Without logical_len the trailing PAD filler is trimmed (+1 safety row
    # for the right-edge splice site), never dragging 1000 phantom slots
    # through attention.
    trimmed = policy.site_logits(ids)
    assert trimmed.shape[0] <= len(live) + 2


def test_placement_probs_semantics_unchanged_for_phantom():
    """_placement_probs routes through site_logits yet stays bit-identical."""
    torch.manual_seed(SEED)
    policy = PhantomPolicy(vocab_size=64, d_model=32, n_actions=3, l_max=L_MAX)
    for inst in stage1_batch(3, seed0=800):
        buffer, logical_len = _front_pack(inst["ids"])
        ids = torch.tensor(buffer, dtype=torch.long)
        manual = torch.softmax(
            _position_logits(policy, ids, logical_len)[: logical_len + 1], dim=-1
        )[:, ACTION_EXPAND]
        routed = _placement_probs(policy, inst)
        assert torch.equal(routed, manual)


def _front_pack(ids):
    from src.tokenizer import front_pack

    return front_pack(ids)


# ---------------------------------------------------------------------------
# Trainer integration: transparent acceptance, metrics, accuracy metric
# ---------------------------------------------------------------------------


def test_trainer_accepts_diffusion_policy_and_produces_finite_metrics():
    policy = _fresh()
    trainer = Trainer(policy, lr=2e-3, seed=SEED)
    metrics = trainer.epoch(stage1_batch(4, seed0=300), batch_size=2, epochs=1)
    assert {"loss", "advantage_mean", "updates", "instances"} <= set(metrics)
    assert metrics["updates"] > 0
    import math

    for key in ("loss", "advantage_mean"):
        assert type(metrics[key]) is float
        assert math.isfinite(metrics[key])


def test_expand_position_accuracy_works_and_is_deterministic():
    policy = _fresh()
    instances = stage1_batch(5, seed0=400)
    for inst in instances:
        a = expand_position_accuracy(policy, inst)
        b = expand_position_accuracy(policy, inst)
        assert 0.0 <= a <= 1.0
        assert a == b


def test_trainer_with_placement_head_accepts_diffusion_policy_embeddings():
    """The opt-in PlacementHead composes with the transformer's embeddings."""
    from src.train_loop import PlacementHead

    policy = _fresh()
    head = PlacementHead(d_model=policy.d_model, n_heads=4, n_layers=1)
    trainer = Trainer(policy, lr=1e-3, seed=SEED, placement_head=head)
    metrics = trainer.epoch(stage1_batch(2, seed0=500), batch_size=2, epochs=1)
    import math

    assert math.isfinite(metrics["loss"])


def test_trainer_builds_two_speed_groups_for_diffusion_policy():
    """DiffusionPolicy's optimizer_param_groups reach the Trainer optimizer.

    The action head trains at the full lr while the backbone trails at
    ``backbone_lr_scale * lr`` (see DiffusionPolicy.optimizer_param_groups
    for why); an attached PlacementHead always rides at full lr.
    """
    policy = _fresh()
    trainer = Trainer(policy, lr=4e-3, seed=SEED)
    lrs = sorted(group["lr"] for group in trainer.optimizer.param_groups)
    assert lrs == [4e-3 * policy.backbone_lr_scale, 4e-3]
    head_group = [
        g for g in trainer.optimizer.param_groups if g["lr"] == 4e-3
    ][0]
    assert any("action_head" in n for n, _ in policy.named_parameters())

    from src.train_loop import PlacementHead

    head = PlacementHead(d_model=policy.d_model, n_heads=4, n_layers=1)
    with_head = Trainer(policy, lr=4e-3, seed=SEED, placement_head=head)
    full_lr_groups = [
        g for g in with_head.optimizer.param_groups if g["lr"] == 4e-3
    ]
    assert len(full_lr_groups) == 2  # action head + placement head


def test_trainer_keeps_single_flat_group_for_phantom_policy():
    """No optimizer_param_groups hook ⇒ historical flat group, unchanged."""
    torch.manual_seed(SEED)
    phantom = PhantomPolicy(vocab_size=VOCAB, d_model=16, n_actions=3, l_max=L_MAX)
    trainer = Trainer(phantom, lr=1e-3, seed=SEED)
    assert len(trainer.optimizer.param_groups) == 1
    assert trainer.optimizer.param_groups[0]["lr"] == 1e-3
    assert len(trainer.optimizer.param_groups[0]["params"]) == len(
        list(phantom.parameters())
    )


# ---------------------------------------------------------------------------
# THE learning proof (upgraded): paired lift through the unchanged Trainer
# ---------------------------------------------------------------------------

TRAIN_SEEDS_DP = stage1_batch(n=64, seed0=60000)   # 64 training instances
HELDOUT_DP = stage1_batch(n=16, seed0=70000)       # disjoint held-out set


def _mean_gap_accuracy(policy, dataset) -> float:
    scores = [expand_position_accuracy(policy, inst) for inst in dataset]
    return sum(scores) / len(scores)


def test_diffusion_policy_learning_proof_paired_lift_on_heldout():
    """Paired before/after lift > 0.01 absolute on held-out instances.

    Same criterion as the PhantomPolicy proof (which showed x2-3 relative),
    now driven through the transformer's per-position heads: 64 stage-1
    instances x 3 epochs at d_model=64, CPU, wall clock < 30 s.
    """
    t0 = time.perf_counter()
    torch.manual_seed(SEED)
    policy = DiffusionPolicy(
        vocab_size=VOCAB, d_model=64, n_actions=3, l_max=L_MAX
    )
    fresh = _mean_gap_accuracy(policy, HELDOUT_DP)

    Trainer(policy, lr=5e-3, seed=SEED).epoch(
        TRAIN_SEEDS_DP, batch_size=8, epochs=3
    )

    post = _mean_gap_accuracy(policy, HELDOUT_DP)
    elapsed = time.perf_counter() - t0
    assert post - fresh > 0.01, (
        f"no learning: fresh={fresh:.4f} vs post={post:.4f}"
    )
    assert elapsed < 30.0, f"learning proof too slow: {elapsed:.1f}s"


# ---------------------------------------------------------------------------
# Persistence: state_dict round-trip including the learned position table
# ---------------------------------------------------------------------------


def test_state_dict_round_trip_restores_identical_site_logits():
    policy = _fresh()
    trainer = Trainer(policy, lr=1e-3, seed=SEED)
    trainer.epoch(stage1_batch(4, seed0=600), batch_size=4, epochs=1)

    ids = torch.tensor([3, 1, 4, 1, 5, 9, 2, 6, PAD_ID, PAD_ID])
    reference = policy.site_logits(ids, logical_len=8)

    state = policy.state_dict()
    assert "pos_embed.weight" in state, "learned position table must persist"
    assert state["pos_embed.weight"].shape == (L_MAX, policy.d_model)

    fd, path = tempfile.mkstemp(suffix=".pt")
    os.close(fd)
    try:
        torch.save(state, path)
        torch.manual_seed(SEED + 1)  # different init on purpose
        revived = DiffusionPolicy(
            vocab_size=VOCAB, d_model=64, n_actions=3, l_max=L_MAX
        )
        revived.load_state_dict(torch.load(path), strict=True)
    finally:
        os.unlink(path)
    assert torch.equal(revived.site_logits(ids, logical_len=8), reference)
