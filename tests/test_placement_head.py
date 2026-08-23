"""Tests for the attention-based PlacementHead (cross-site contrast readout).

Contracts under test:

* the placement distribution lives over exactly ``logical_len + 1`` candidate
  splice sites and sums to 1 (the phantom tail never competes);
* gradients flow through every MultiheadAttention parameter — the cross-site
  contrast is trainable end-to-end;
* a FRESH head is not degenerate: no single site hoards mass and the argmax
  site varies across instances (i.e. it starts as a genuine competition, not
  a constant answer);
* Trainer integration: the strategy flag trains policy + head jointly,
  rejects mismatched widths, and save/load round-trips bit-identically.
"""

import math
import os
import tempfile
import time

import pytest
import torch

torch.set_num_threads(1)  # tiny tensors: single-threaded ops are faster here

from src.curriculum_data import stage1_batch
from src.tokenizer import L_MAX, front_pack
from src.training import PhantomPolicy
from src.train_loop import PlacementHead, Trainer, expand_position_accuracy

SEED = 20240607


def _policy_and_head(seed: int = SEED):
    torch.manual_seed(seed)
    policy = PhantomPolicy(vocab_size=64, d_model=32, n_actions=3, l_max=L_MAX)
    head = PlacementHead(d_model=32, n_heads=4, n_layers=2)
    return policy, head


def _site_probs(policy, head, instance) -> torch.Tensor:
    buffer, logical_len = front_pack(instance["ids"])
    ids = torch.tensor(buffer, dtype=torch.long)
    with torch.no_grad():
        return head(policy.embed(ids), logical_len)


# ---------------------------------------------------------------------------
# Distribution contract: one probability per candidate splice site
# ---------------------------------------------------------------------------


def test_placement_distribution_sums_to_one_over_candidate_sites():
    """Softmax over sites: right length, sums to 1, non-negative, finite."""
    policy, head = _policy_and_head()
    for inst in stage1_batch(8, seed0=300):
        probs = _site_probs(policy, head, inst)
        assert probs.shape == (len(inst["ids"]) + 1,)   # sites 0..logical_len
        assert bool(torch.isfinite(probs).all())
        assert float(probs.min()) >= 0.0
        assert float(probs.sum()) == pytest.approx(1.0, abs=1e-6)


def test_head_probs_match_expand_position_accuracy_readout():
    """expand_position_accuracy(head) reads the same distribution the head
    emits directly: accuracy == P(site=gap_start) under softmax(logits)."""
    policy, head = _policy_and_head()
    for inst in stage1_batch(5, seed0=310):
        probs = _site_probs(policy, head, inst)
        expected = float(probs[inst["gap_start"]] / probs.sum())
        assert expand_position_accuracy(policy, inst, head) == pytest.approx(
            expected, rel=1e-9
        )


def test_head_rejects_bad_width_or_negative_logical_len():
    head = PlacementHead(d_model=32)
    with pytest.raises(ValueError):
        head(torch.zeros(10, 16), 5)          # wrong embedding width
    with pytest.raises(ValueError):
        head(torch.zeros(10, 32), -1)         # negative logical length
    with pytest.raises(ValueError):
        PlacementHead(d_model=32, n_heads=3)  # indivisible
    with pytest.raises(ValueError):
        PlacementHead(d_model=32, n_layers=3)  # spec allows 1 or 2 layers


# ---------------------------------------------------------------------------
# Gradient flow through the MultiheadAttention parameters
# ---------------------------------------------------------------------------


def test_gradients_flow_through_all_attention_parameters():
    """NLL of the true gap sites must reach EVERY head parameter after
    backward — in particular in_proj/out_proj of each MHA layer — proving
    the cross-site contrast path is trainable, not a frozen bystander."""
    policy, head = _policy_and_head()
    instances = stage1_batch(3, seed0=320)

    nll = torch.zeros(())
    for inst in instances:
        logits = head.logits(
            policy.embed(torch.tensor(front_pack(inst["ids"])[0], dtype=torch.long)),
            front_pack(inst["ids"])[1],
        )
        nll = nll - torch.log_softmax(logits, dim=-1)[inst["gap_start"]]
    nll.backward()

    params = dict(head.named_parameters())
    # Both MHA blocks really contributed their attention matrices.
    for layer_idx in range(head.n_layers):
        for part in (
            f"attn.{layer_idx}.in_proj_weight",
            f"attn.{layer_idx}.in_proj_bias",
            f"attn.{layer_idx}.out_proj.weight",
            f"attn.{layer_idx}.out_proj.bias",
        ):
            assert part in params, f"missing expected MHA param {part}"
            assert float(params[part].grad.abs().sum()) > 0.0, (
                f"{part} got zero grad"
            )
    # Plus scorer (and norms) — every head parameter participates.
    for name, param in params.items():
        assert param.grad is not None, f"no gradient reached {name}"
        assert math.isfinite(float(param.grad.abs().sum()))
        assert float(param.grad.abs().sum()) > 0.0, f"{name} got zero grad"


def test_trainer_epoch_moves_both_policy_and_head_parameters():
    """The strategy flag wires head params into the optimizer: one epoch on
    seeded instances moves them measurably (deterministic given seeds)."""
    policy, head = _policy_and_head()
    trainer = Trainer(policy, lr=5e-3, seed=SEED, placement_head=head)
    embed_before = policy.embed.weight.detach().clone()
    head_before = [p.detach().clone() for p in head.parameters()]

    trainer.epoch(stage1_batch(16, seed0=500), batch_size=8, epochs=2)

    embed_delta = (policy.embed.weight.detach() - embed_before).abs().sum()
    assert float(embed_delta) > 0.0
    # ``final_norm.bias`` is the one deliberate exception: it shifts every
    # site logit by the same amount, and the coupled loss's coefficients sum
    # to exactly zero per instance (antithetic baseline), so softmax shift
    # invariance pins its gradient at 0 — a documented dead parameter.
    for before, (name, param) in zip(head_before, head.named_parameters()):
        if name == "final_norm.bias":
            continue
        assert not torch.equal(param.detach(), before), (
            f"head param {name} never moved"
        )


def test_optimizer_covers_union_of_policy_and_head_params():
    policy, head = _policy_and_head()
    trainer = Trainer(policy, placement_head=head)
    n_expected = len(list(policy.parameters())) + len(list(head.parameters()))
    n_grouped = sum(len(g["params"]) for g in trainer.optimizer.param_groups)
    assert n_grouped == n_expected


def test_trainer_rejects_mismatched_head_width():
    policy = PhantomPolicy(vocab_size=64, d_model=32, n_actions=3, l_max=L_MAX)
    narrow = PlacementHead(d_model=8)
    with pytest.raises(ValueError):
        Trainer(policy, placement_head=narrow)
    with pytest.raises(TypeError):
        Trainer(policy, placement_head="not a module")


# ---------------------------------------------------------------------------
# Fresh-head sanity: a real competition, not a constant answer
# ---------------------------------------------------------------------------


def test_fresh_head_is_not_degenerate():
    """Before training, no site hoards the mass and the winning site varies
    across instances — the softmax starts as a genuine competition."""
    policy, head = _policy_and_head()
    argmax_sites: set[int] = set()
    top_probs = []
    for inst in stage1_batch(24, seed0=330):
        probs = _site_probs(policy, head, inst)
        argmax_sites.add(int(probs.argmax()))
        top_probs.append(float(probs.max()))
    assert len(argmax_sites) >= 2, (
        f"degenerate fresh head: argmax always {argmax_sites}"
    )
    assert max(top_probs) < 0.9, "fresh head dumps all mass on one site"


# ---------------------------------------------------------------------------
# Persistence: round-trip with an attached head is behaviour-identical
# ---------------------------------------------------------------------------


def test_save_load_round_trip_with_head_restores_identical_accuracy():
    policy, head = _policy_and_head()
    trainer = Trainer(policy, lr=5e-3, seed=SEED, placement_head=head)
    trainer.epoch(stage1_batch(8, seed0=510), batch_size=8, epochs=1)
    heldout = stage1_batch(6, seed0=520)
    before = [expand_position_accuracy(policy, i, head) for i in heldout]

    fd, path = tempfile.mkstemp(suffix=".pt")
    os.close(fd)
    try:
        trainer.save(path)
        revived = Trainer.load(path)
        after = [
            expand_position_accuracy(revived.policy, i, revived.placement_head)
            for i in heldout
        ]
    finally:
        os.unlink(path)
    assert revived.placement_head is not None
    assert after == before  # bit-identical accuracy after round-trip
