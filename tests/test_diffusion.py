"""Tests for the Phantom Padding inference loop (Algorithm 1) — src/diffusion.py.

Paper contract under test: ``PhantomLoop`` runs the deterministic denoising
loop over a FIXED physical buffer (len == l_max forever — THE LAW):

    xbuf = FrontPack(x, L_max)            # tail = IGNORE filler
    while MASK in buffer:
        p = policy(buffer)                # action scores
        per masked position (ascending):
            [EXPAND]  -> splice a fresh MASK boundary (logical region grows)
            [DELETE]  -> logical removal (logical region shrinks)
            else KEEP -> remain masked this step (PhantomPolicy has no token head)

Documented engine decisions locked in by these tests:

* ``start`` matches ``tokenizer.front_pack`` semantics exactly: overlong
  sources are TRUNCATED (never ValueError), matching the real system.
* EXPAND degrades to KEEP (counted under ``"keep"``) when the per-step
  ``max_expand`` budget or the physical capacity ``l_max`` is exhausted.
* Convergence (no MASK left) is delete-driven: without a token head the only
  op that removes a MASK is [DELETE]; KEEP deliberately leaves masks in place.
* The policy sees a sanitized view of the buffer: sentinel ids (MASK_ID=-1,
  IGNORE_ID=-2) are presented as PAD_ID so a real ``PhantomPolicy`` (which
  rejects negative ids) plugs in directly; the true buffer is never mutated.
"""

import os
import tempfile
import time

import pytest
import torch

torch.set_num_threads(1)  # tiny tensors: single-threaded ops are faster here

from src.curriculum_data import stage1_batch
from src.diffusion import DiffusionStep, PhantomLoop
from src.tokenizer import (
    IGNORE_ID,
    L_MAX,
    MASK_ID,
    PAD_ID,
    front_pack,
    tokenize,
)
from src.training import ACTION_DELETE, ACTION_EXPAND, ACTION_KEEP, PhantomPolicy
from src.train_loop import PlacementHead, Trainer, expand_position_accuracy


# ---------------------------------------------------------------------------
# StubPolicy: deterministic scripted actions per position (tests only)
# ---------------------------------------------------------------------------


class StubPolicy:
    """Scripted stand-in for ``PhantomPolicy`` with per-position actions.

    ``script`` may be either:

    * a mapping ``{absolute_buffer_index: action_id}`` (unlisted indices get
      ``default``), or
    * a callable ``(absolute_index, call_index) -> action_id`` where
      ``call_index`` counts ``forward`` invocations from 0, enabling phased
      schedules (e.g. expand for the first two calls, then delete).

    ``forward`` returns one action-score row per buffer position (an
    ``(l_max, 3)`` nested list) with a one-hot peak of 10.0 on the scripted
    action — the per-position row layout ``PhantomLoop`` documents.
    """

    def __init__(self, script=None, default=ACTION_KEEP, l_max=L_MAX):
        self.script = script if script is not None else {}
        self.default = default
        self.l_max = l_max
        self.calls = 0

    def forward(self, token_ids, mask):
        assert len(token_ids) == self.l_max, "engine must pass a fixed buffer"
        assert len(mask) == self.l_max
        self.calls += 1
        call_index = self.calls - 1

        def action_at(i):
            if callable(self.script):
                return self.script(i, call_index)
            return self.script.get(i, self.default)

        return [
            [10.0 if a == j else 0.0 for j in range(3)]
            for a in (action_at(i) for i in range(self.l_max))
        ]


def _seeded_loop(script=None, default=ACTION_KEEP, source=None, **kwargs):
    """Loop over a small seeded source: ``[10, MASK, 20, MASK, 30]`` by default."""
    if source is None:
        source = [10, MASK_ID, 20, MASK_ID, 30]
    loop = PhantomLoop(StubPolicy(script, default), **kwargs)
    loop.start(source)
    return loop


# ---------------------------------------------------------------------------
# Construction / lifecycle contracts
# ---------------------------------------------------------------------------


def test_start_matches_front_pack_semantics_short_input():
    loop = PhantomLoop(StubPolicy())
    loop.start([7, MASK_ID, 9])
    expected_buffer, expected_len = front_pack([7, MASK_ID, 9])
    buffer, logical_len = loop.result()
    assert buffer == expected_buffer
    assert logical_len == expected_len == 3


def test_start_truncates_overlong_source_like_tokenizer():
    """Documented decision: MATCH tokenizer semantics — truncate, never raise."""
    loop = PhantomLoop(StubPolicy())
    loop.start(list(range(1, L_MAX + 500)))  # far beyond L_MAX
    buffer, logical_len = loop.result()
    assert logical_len == L_MAX
    assert len(buffer) == L_MAX
    assert buffer == list(range(1, L_MAX + 1))


def test_l_max_divergent_from_tokenizer_raises_valueerror():
    """Splice primitives hardcode tokenizer.L_MAX; a different l_max would
    silently corrupt the phantom tail, so the engine refuses it up front."""
    with pytest.raises(ValueError):
        PhantomLoop(StubPolicy(), l_max=32)


def test_operations_before_start_raise_runtimeerror():
    loop = PhantomLoop(StubPolicy())
    with pytest.raises(RuntimeError):
        loop.step()
    with pytest.raises(RuntimeError):
        loop.run(3)
    with pytest.raises(RuntimeError):
        loop.result()


def test_negative_max_expand_or_max_steps_raise_valueerror():
    loop = _seeded_loop(default=ACTION_EXPAND)
    with pytest.raises(ValueError):
        loop.step(max_expand=-1)
    with pytest.raises(ValueError):
        loop.run(-1)


# ---------------------------------------------------------------------------
# THE LAW: physical shape invariant at every iteration
# ---------------------------------------------------------------------------


def test_shape_stays_l_max_after_every_step_of_the_loop():
    """THE LAW: len(buffer) == l_max before AND after every single step."""
    loop = _seeded_loop(script={1: ACTION_EXPAND, 3: ACTION_DELETE})
    assert len(loop.buffer) == L_MAX
    for _ in range(3):
        outcome = loop.step()
        assert outcome is not None
        assert len(loop.buffer) == L_MAX, "physical buffer resized — LAW broken"
    final_buffer, _ = loop.result()
    assert len(final_buffer) == L_MAX


def test_diffusion_step_records_exact_counts_and_lengths():
    """Hand-checked mixed step: expand at 1, keep at 3 (snapshot indexing)."""
    loop = _seeded_loop(script={1: ACTION_EXPAND})  # default KEEP elsewhere
    outcome = loop.step()
    assert isinstance(outcome, DiffusionStep)
    assert outcome.logical_len_before == 5
    assert outcome.logical_len_after == 6
    assert outcome.action_counts == {"keep": 1, "expand": 1, "delete": 0}
    buffer, logical_len = loop.result()
    assert logical_len == 6
    assert buffer[:6] == [10, MASK_ID, MASK_ID, 20, MASK_ID, 30]


def test_action_counts_sum_equals_masked_positions_at_step_start():
    loop = _seeded_loop(script={1: ACTION_EXPAND, 3: ACTION_DELETE})
    outcome = loop.step()
    assert sum(outcome.action_counts.values()) == 2  # two MASKs were seeded
    assert set(outcome.action_counts) == {"keep", "expand", "delete"}


def test_argmax_ties_resolve_deterministically_to_keep():
    """Flat scores ⇒ argmax picks action slot 0 (KEEP): deterministic tie-break."""

    class FlatPolicy:
        def forward(self, token_ids, mask):
            return [[0.0, 0.0, 0.0] for _ in token_ids]

    loop = _seeded_loop()
    loop.policy = FlatPolicy()
    outcome = loop.step()
    assert outcome.action_counts == {"keep": 2, "expand": 0, "delete": 0}
    assert outcome.logical_len_before == outcome.logical_len_after == 5


# ---------------------------------------------------------------------------
# [EXPAND]: +1 logical length per op, degrade to KEEP at caps
# ---------------------------------------------------------------------------


def test_scripted_expand_inserts_fresh_mask_boundary_growing_len_by_one():
    loop = _seeded_loop(default=ACTION_EXPAND, source=[10, MASK_ID, 20])
    outcome = loop.step(max_expand=1)
    assert outcome.logical_len_before == 3
    assert outcome.logical_len_after == 4  # exactly +1 per executed EXPAND
    assert outcome.action_counts["expand"] == 1
    buffer, _ = loop.result()
    assert buffer[:4] == [10, MASK_ID, MASK_ID, 20]  # fresh MASK at pos 1


def test_expand_degrades_to_keep_when_per_step_budget_exhausted():
    loop = _seeded_loop(default=ACTION_EXPAND)  # both MASKs want EXPAND
    outcome = loop.step(max_expand=1)           # budget allows only one
    assert outcome.action_counts == {"keep": 1, "expand": 1, "delete": 0}
    assert outcome.logical_len_after == outcome.logical_len_before + 1


def test_expand_degrades_to_keep_at_physical_capacity():
    """At logical_len == l_max the splice cannot fit: EXPAND → KEEP, no growth,
    and the tokenizer's overflow guard is never tripped."""
    source = [1] * (L_MAX - 2) + [MASK_ID]  # logical_len == L_MAX - 1
    loop = _seeded_loop(default=ACTION_EXPAND, source=source)

    first = loop.step(max_expand=4)
    assert first.logical_len_after == L_MAX  # grew right up to capacity
    _, logical_len = loop.result()
    assert logical_len == L_MAX

    second = loop.step(max_expand=4)  # capacity exhausted: degrade
    # The capacity-filling EXPAND left TWO adjacent fresh boundaries (pos
    # stays MASK while the old MASK slid right): both degrade to keep.
    assert second.action_counts["expand"] == 0
    assert second.action_counts["keep"] == 2
    assert second.logical_len_before == second.logical_len_after == L_MAX


# ---------------------------------------------------------------------------
# [DELETE]: logical removal shrinks the region
# ---------------------------------------------------------------------------


def test_scripted_delete_removes_positions_and_shrinks_logical_len():
    loop = _seeded_loop(script={1: ACTION_DELETE, 3: ACTION_DELETE})
    outcome = loop.step()
    assert outcome.logical_len_before == 5
    assert outcome.logical_len_after == 3  # 5 - 2 deletes
    assert outcome.action_counts == {"keep": 0, "expand": 1 * 0 + 0, "delete": 2}
    buffer, logical_len = loop.result()
    assert logical_len == 3
    assert buffer[:3] == [10, 20, 30]  # masks gone, real tokens shifted left
    assert all(t != MASK_ID for t in buffer)


# ---------------------------------------------------------------------------
# Convergence, KEEP persistence, max_steps bound
# ---------------------------------------------------------------------------


def test_full_resolution_run_reaches_no_mask_state_via_phased_stub():
    """Phased schedule: EXPAND while calls < 2, then DELETE everything.

    Exercises the documented resolution path: expansion grows the region,
    deletion clears every MASK — run stops early on the no-MASK condition.
    """
    loop = _seeded_loop(
        script=lambda i, c: ACTION_EXPAND if c < 2 else ACTION_DELETE,
        source=[10, MASK_ID, 20],
    )
    steps = loop.run(max_steps=1000)
    assert 2 < len(steps) < 1000  # converged well before the bound
    buffer, logical_len = loop.result()
    assert all(t != MASK_ID for t in buffer), "no-MASK state not reached"
    assert buffer[:logical_len] == [10, 20]
    assert logical_len == 2
    # Every step obeyed THE LAW and reported coherent lengths.
    for s in steps:
        assert s.logical_len_before <= L_MAX and s.logical_len_after <= L_MAX


def test_step_returns_none_once_converged():
    loop = _seeded_loop(default=ACTION_DELETE, source=[5, MASK_ID])
    steps = loop.run(max_steps=10)
    assert len(steps) == 1
    assert loop.step() is None          # converged: nothing left to denoise
    assert loop.run(max_steps=5) == []  # run on a converged loop is a no-op


def test_clean_source_converges_immediately_without_any_step():
    loop = PhantomLoop(StubPolicy())
    loop.start([1, 2, 3])
    assert loop.step() is None
    assert loop.run(max_steps=4) == []


def test_keep_leaves_masks_and_respects_max_steps_bound():
    """KEEP means 'remain masked' (no token head): masks persist, so run()
    must stop at exactly max_steps rather than spin forever."""
    loop = _seeded_loop(default=ACTION_KEEP)
    steps = loop.run(max_steps=5)
    assert len(steps) == 5  # bound respected exactly
    for s in steps:
        assert s.action_counts == {"keep": 2, "expand": 0, "delete": 0}
        assert s.logical_len_before == s.logical_len_after == 5
    buffer, logical_len = loop.result()
    assert logical_len == 5
    assert buffer.count(MASK_ID) >= 2  # still unresolved, as documented


# ---------------------------------------------------------------------------
# Integration: real tokenizer output + real PhantomPolicy feed the loop
# ---------------------------------------------------------------------------


class _IdGuard:
    """Asserts the engine never leaks negative sentinel ids into the policy."""

    def __init__(self, inner):
        self.inner = inner

    def forward(self, token_ids, mask):
        assert min(token_ids) >= PAD_ID, "sentinel id leaked into policy view"
        return self.inner(torch.tensor([token_ids]), torch.tensor([mask]))


def test_integration_tokenizer_source_with_real_phantom_policy():
    ids = tokenize("def f(x):\n    return x + 1\n")
    assert 0 < len(ids) < L_MAX
    policy = PhantomPolicy(vocab_size=64, d_model=8, n_actions=3, l_max=L_MAX)
    loop = PhantomLoop(_IdGuard(policy))
    seeded = ids + [MASK_ID] * 3  # seed placeholders for the loop to act on
    expected_buffer, expected_len = front_pack(seeded)
    loop.start(seeded)
    _, logical_len = loop.result()
    assert logical_len == expected_len == len(ids) + 3

    steps = loop.run(max_steps=4)
    assert 1 <= len(steps) <= 4
    buffer, logical_len = loop.result()
    assert isinstance(buffer, list) and isinstance(logical_len, int)
    assert len(buffer) == L_MAX  # THE LAW holds through the real-policy path
    assert 0 < logical_len <= L_MAX
    for s in steps:
        assert set(s.action_counts) == {"keep", "expand", "delete"}
        assert sum(s.action_counts.values()) >= 1
        # Recorded counts must explain the actual logical-length movement.
        delta = s.logical_len_after - s.logical_len_before
        assert delta == s.action_counts["expand"] - s.action_counts["delete"]
    # No sentinel may survive into the sanitized view the policy consumed;
    # _IdGuard already asserted min(id) >= PAD_ID on every forward call.


def test_result_returns_defensive_copy():
    loop = _seeded_loop()
    buffer, _ = loop.result()
    buffer[0] = -12345
    assert loop.buffer[0] != -12345


# ---------------------------------------------------------------------------
# Placement-head integration: expansion sites come from the learned head
# ---------------------------------------------------------------------------


class ScriptedPlacementHead(PlacementHead):
    """Deterministic head stand-in: fixed logits peaking on a scripted site.

    ``site`` is either an absolute candidate-site index or a callable
    ``(logical_len) -> site`` (enables the right-edge site
    ``site == logical_len``). ``forward`` overrides the learned pipeline with
    a one-hot softmax over the ``logical_len + 1`` candidate sites, and
    ``calls`` counts consultations so tests can assert the loop really
    consults the head.
    """

    def __init__(self, site):
        super().__init__(d_model=8)  # width irrelevant: forward is overridden
        self.site = site
        self.calls = 0

    def forward(self, h, logical_len):
        self.calls += 1
        chosen = self.site(logical_len) if callable(self.site) else self.site
        assert 0 <= chosen <= logical_len, "scripted site outside candidates"
        logits = torch.full((logical_len + 1,), -10.0)
        logits[chosen] = 10.0  # strictly dominant mass on the scripted site
        return torch.softmax(logits, dim=-1)


class _EmbedStubPolicy(StubPolicy):
    """:class:`StubPolicy` plus a throwaway ``embed``: the scripted head
    overrides ``forward`` and never reads ``h``, but the engine still (by
    contract) requires the policy to expose ``embed(ids)``."""

    def embed(self, ids):
        return torch.zeros(len(ids), 8)


def test_placement_head_drives_expand_site_selection():
    """With a placement_head attached, an EXPAND decision splices at the
    head's argmax site — NOT at the masked position that chose EXPAND.
    Scripted site 2 sits between the two seeded masks (1 and 3), so both
    fresh boundaries must pile up AT site 2."""
    policy = PhantomPolicy(vocab_size=64, d_model=8, n_actions=3, l_max=L_MAX)
    head = ScriptedPlacementHead(site=2)
    loop = PhantomLoop(
        _EmbedStubPolicy(default=ACTION_EXPAND), placement_head=head
    )
    loop.start([10, MASK_ID, 20, MASK_ID, 30])
    outcome = loop.step(max_expand=4)
    assert outcome.action_counts == {"keep": 0, "expand": 2, "delete": 0}
    assert outcome.logical_len_before == 5
    assert outcome.logical_len_after == 7
    assert head.calls == 2, "loop must consult the head per executed EXPAND"
    buffer, _ = loop.result()
    # Fresh MASKs spliced at candidate site 2 (never at the masked slots).
    assert buffer[:7] == [10, MASK_ID, MASK_ID, MASK_ID, 20, MASK_ID, 30]


def test_placement_head_may_choose_the_right_edge_site():
    """Candidate sites are 0..logical_len INCLUSIVE: a scripted right-edge
    choice must splice past the last live token (insert_masks contract)."""
    policy = PhantomPolicy(vocab_size=64, d_model=8, n_actions=3, l_max=L_MAX)
    head = ScriptedPlacementHead(site=lambda logical_len: logical_len)
    loop = PhantomLoop(
        _EmbedStubPolicy(script={1: ACTION_EXPAND}), placement_head=head
    )
    loop.start([10, MASK_ID, 20])
    outcome = loop.step(max_expand=4)
    assert outcome.action_counts["expand"] == 1
    buffer, logical_len = loop.result()
    assert logical_len == 4
    assert buffer[:4] == [10, MASK_ID, 20, MASK_ID]  # appended at right edge


def test_placement_head_keeps_keep_delete_on_action_logits():
    """Only SITE selection moves to the head: DELETE decisions still come
    from the policy's action rows and still act on the masked position."""
    policy = PhantomPolicy(vocab_size=64, d_model=8, n_actions=3, l_max=L_MAX)
    head = ScriptedPlacementHead(site=0)  # would hijack any expand to site 0
    loop = PhantomLoop(
        _EmbedStubPolicy(script={3: ACTION_DELETE}), placement_head=head
    )
    loop.start([10, MASK_ID, 20, MASK_ID, 30])
    outcome = loop.step(max_expand=4)
    assert outcome.action_counts == {"keep": 1, "expand": 0, "delete": 1}
    assert head.calls == 0, "head is never consulted without an EXPAND"
    buffer, logical_len = loop.result()
    assert logical_len == 4
    assert buffer[:4] == [10, MASK_ID, 20, 30]


def test_placement_head_without_policy_embed_raises_typeerror():
    """The head scores sites from embeddings: a policy without ``.embed``
    cannot drive it, so wiring fails fast at construction."""

    class NoEmbed:
        def forward(self, token_ids, mask):
            return [[0.0, 9.0, 0.0]] * len(token_ids)

    with pytest.raises(TypeError):
        PhantomLoop(NoEmbed(), placement_head=ScriptedPlacementHead(0))


def test_trainer_save_all_load_all_round_trip_restores_identical_accuracy():
    """Checkpoint completeness: save_all/load_all round-trip BOTH
    policy.state_dict AND placement_head.state_dict — bit-identical weights
    and therefore bit-identical placement accuracy after reload."""
    torch.manual_seed(20240607)
    policy = PhantomPolicy(vocab_size=64, d_model=32, n_actions=3, l_max=L_MAX)
    head = PlacementHead(d_model=32, n_heads=4, n_layers=2)
    trainer = Trainer(policy, lr=5e-3, seed=20240607, placement_head=head)
    trainer.epoch(stage1_batch(32, seed0=6200), batch_size=8, epochs=1)
    heldout = stage1_batch(6, seed0=6300)
    before = [expand_position_accuracy(policy, i, head) for i in heldout]

    fd, path = tempfile.mkstemp(suffix=".pt")
    os.close(fd)
    try:
        trainer.save_all(path)
        revived = Trainer.load_all(path)
        after = [
            expand_position_accuracy(revived.policy, i, revived.placement_head)
            for i in heldout
        ]
    finally:
        os.unlink(path)

    assert revived.placement_head is not None
    assert after == before  # bit-identical accuracy after round-trip
    # BOTH state dicts rode along, tensor-for-tensor.
    for key, tensor in policy.state_dict().items():
        assert torch.equal(revived.policy.state_dict()[key], tensor)
    for key, tensor in head.state_dict().items():
        assert torch.equal(revived.placement_head.state_dict()[key], tensor)


# ---------------------------------------------------------------------------
# Learned smoke: brief training steers the loop's first EXPAND to the gap
# ---------------------------------------------------------------------------


class _ExpandAllAdapter:
    """Forces the EXPAND action from every row while delegating embeddings.

    Isolates what the integration adds — head-driven SITE selection — from
    the (separately trained) action logits: every masked position decides
    EXPAND, so each executed splice lands exactly where the head points.
    """

    def __init__(self, inner):
        self.inner = inner

    def embed(self, ids):
        return self.inner.embed(ids)

    def forward(self, token_ids, mask):
        return [[0.0, 9.0, 0.0]] * len(token_ids)


def test_learned_head_steers_first_expand_toward_gap_start():
    """~5 s of stage-1 training (24 instances x 100 epochs, observed ~4 s)
    on the site-competition loss, then 8 held-out instances run through the
    real loop (EXPAND forced, site chosen by the trained head): the first
    executed EXPAND must land exactly on ``gap_start`` at least 2x the
    uniform-site share (soft assertion). Pinned seeds make it
    bit-reproducible; observed hit rate 4/8 vs threshold ~0.18."""
    t0 = time.perf_counter()
    torch.manual_seed(77)
    policy = PhantomPolicy(vocab_size=64, d_model=32, n_actions=3, l_max=L_MAX)
    head = PlacementHead(d_model=32, n_heads=4, n_layers=2)
    Trainer(policy, lr=5e-3, seed=77, placement_head=head).epoch(
        stage1_batch(24, seed0=6000), batch_size=8, epochs=100
    )
    train_seconds = time.perf_counter() - t0

    instances = stage1_batch(8, seed0=6100)
    hits = 0
    uniform_share_sum = 0.0
    for inst in instances:
        loop = PhantomLoop(_ExpandAllAdapter(policy), vocab_size=64,
                           placement_head=head)
        seeded = list(inst["ids"]) + [MASK_ID]  # one placeholder to act on
        loop.start(seeded)
        pre, logical_len = loop.result()
        assert logical_len == len(seeded)
        outcome = loop.step(max_expand=1)
        assert outcome is not None
        assert outcome.action_counts["expand"] == 1
        post, _ = loop.result()
        site = next(
            i for i in range(len(pre))
            if post[i] == MASK_ID and pre[i] != MASK_ID
        )
        hits += int(site == inst["gap_start"])
        uniform_share_sum += 1.0 / (len(inst["ids"]) + 1)

    hit_rate = hits / len(instances)
    chance = uniform_share_sum / len(instances)
    assert hit_rate >= 2.0 * chance > 0.0, (
        f"no learning signal: hit rate {hit_rate:.3f} vs uniform {chance:.3f}"
    )
    assert train_seconds < 30.0, f"smoke training too slow: {train_seconds:.1f}s"
