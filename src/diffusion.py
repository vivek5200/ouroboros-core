"""Module 1 inference — Algorithm 1: the Phantom Padding denoising loop.

A deterministic, testable engine for the paper's phantom-padding inference
loop. The physical buffer is created ONCE by ``front_pack`` and NEVER resized
(THE LAW: ``len(buffer) == l_max`` after every step, asserted in
:meth:`PhantomLoop.step`); all structure edits are purely logical:

    xbuf = FrontPack(x, L_max)              # tail filled [IGNORE]
    while MASK in buffer:
        p = policy(view(buffer))            # action scores, fixed shape in/out
        for each masked position (ascending):
            [EXPAND] -> insert_masks(pos, k=1): pos becomes a fresh MASK
                        boundary; the old MASK slides right; logical_len += 1
            [DELETE] -> logical_delete(pos): token logically removed;
                        logical_len -= 1
            else KEEP -> remain masked this step

Documented semantic decisions (locked by tests/test_diffusion.py):

* Truncation: ``start`` uses :func:`src.tokenizer.front_pack` semantics — an
  overlong source is TRUNCATED to l_max, never rejected (matches the real
  system; front_pack already truncates).
* EXPAND degradation: when the per-step ``max_expand`` budget is spent or the
  physical capacity is exhausted (``logical_len == l_max``), an EXPAND
  decision degrades to KEEP and is counted under ``"keep"``.
* KEEP means "remain masked": ``PhantomPolicy`` exposes only the three action
  logits and no vocabulary/token head, so the engine has no way to write a
  predicted token. Consequently convergence (no MASK left) is delete-driven —
  a policy that never emits [DELETE] never converges, and :meth:`PhantomLoop.run`
  is bounded by ``max_steps`` for exactly that reason.
* Sentinel sanitization: the policy sees a VIEW of the buffer in which
  MASK_ID/IGNORE_ID (negative sentinels) are presented as PAD_ID, because
  ``PhantomPolicy.forward`` rejects negative ids. The true buffer is never
  mutated by this view. The mask passed alongside is ``derive_mask`` of the
  real logical length.
* Policy output protocol: ``policy.forward(token_ids, mask)`` may return
  torch tensors or plain nested lists, shaped ``(A,)`` or ``(1, A)`` — one
  action-score row shared by every masked position this step — or
  ``(l_max, A)`` with one row per buffer position (row ``i`` scores position
  ``i``, evaluated on the step-start buffer). Any other leading size raises
  ``ValueError``.
* Snapshot indexing: masked positions are snapshotted at step start and
  processed ascending; a shift counter tracks insertions (+1) and deletions
  (-1) so each snapshotted MASK is acted on exactly once at its CURRENT
  location. Masks spliced in by an EXPAND during the step are fresh
  boundaries and are deliberately NOT revisited until the next step.
* Tie-breaking: ``argmax`` takes the lowest action index on exact ties, so
  equal scores resolve deterministically to KEEP (slot 0).

The engine itself imports no torch — policies carry the tensors; actions come
from :data:`src.training.ACTION_*` so the action space has one source of truth.
"""

from dataclasses import dataclass, field

from src.tokenizer import (
    L_MAX,
    MASK_ID,
    PAD_ID,
    derive_mask,
    front_pack,
    insert_masks,
    logical_delete,
)
from src.training import ACTION_DELETE, ACTION_EXPAND

__all__ = ["DiffusionStep", "PhantomLoop"]

_COUNT_KEYS = ("keep", "expand", "delete")


# ---------------------------------------------------------------------------
# One recorded iteration
# ---------------------------------------------------------------------------


@dataclass
class DiffusionStep:
    """Record of one denoising iteration (Algorithm 1 inner loop pass).

    Attributes:
        action_counts: How many masked positions resolved to each action this
            step; always contains all three keys ``"keep"``, ``"expand"``,
            ``"delete"`` (zeros included). Degraded EXPANDs count as keep.
        logical_len_before: Logical region length at step start.
        logical_len_after: Logical region length after all ops in this step.
    """

    action_counts: dict[str, int] = field(default_factory=dict)
    logical_len_before: int = 0
    logical_len_after: int = 0


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------


class PhantomLoop:
    """Algorithm 1 driver over a fixed ``front_pack`` buffer.

    Args:
        policy: Object exposing ``forward(token_ids, mask)`` returning action
            scores (see the module docstring for the accepted shapes). A real
            :class:`src.training.PhantomPolicy` plugs in directly.
        vocab_size: Vocabulary cardinality carried for callers/policies; the
            engine itself only routes ids, never embeds them.
        l_max: Fixed physical buffer length. Must equal
            :data:`src.tokenizer.L_MAX`: the splice primitives
            (``insert_masks`` / ``logical_delete`` / ``front_pack``) hardcode
            L_MAX internally, so any other value would silently corrupt the
            phantom tail. Divergent values raise ``ValueError``.
    """

    def __init__(self, policy, vocab_size: int = 64, l_max: int = L_MAX) -> None:
        if l_max != L_MAX:
            raise ValueError(
                f"l_max must equal tokenizer.L_MAX ({L_MAX}); the splice"
                f" primitives hardcode it, got {l_max}"
            )
        self.policy = policy
        self.vocab_size = vocab_size
        self.l_max = l_max
        self.buffer: list[int] | None = None
        self.logical_len: int = 0

    # -- lifecycle ----------------------------------------------------------

    def start(self, source_tokens: list[int]) -> None:
        """Front-pack ``source_tokens`` into the fixed buffer (lines 1-2).

        Matches tokenizer semantics exactly: sources longer than ``l_max``
        are truncated (never raised); the tail is pad-filled and reported as
        phantom via :meth:`result`. Callers seed denoising targets by placing
        ``MASK_ID`` placeholders directly in ``source_tokens``.
        """
        self.buffer, self.logical_len = front_pack(list(source_tokens))

    def result(self) -> tuple[list[int], int]:
        """``(buffer, logical_len)`` — the buffer as a defensive copy."""
        self._require_started()
        return list(self.buffer), self.logical_len

    # -- one iteration ------------------------------------------------------

    def step(self, max_expand: int = 4) -> DiffusionStep | None:
        """One denoising iteration, or ``None`` when no MASK remains.

        Args:
            max_expand: Maximum [EXPAND] operations permitted within THIS
                step; further EXPAND decisions degrade to KEEP.

        Returns:
            :class:`DiffusionStep` recording action counts and logical
            lengths, or ``None`` if the buffer holds no ``MASK_ID``.

        Raises:
            RuntimeError: If :meth:`start` was not called yet.
            ValueError: If ``max_expand`` is negative or the policy returns
                an unsupported score shape.
        """
        self._require_started()
        if max_expand < 0:
            raise ValueError(f"max_expand must be non-negative, got {max_expand}")
        assert len(self.buffer) == self.l_max  # THE LAW, checked on entry too

        snapshot = [
            i for i, t in enumerate(self.buffer[: self.logical_len]) if t == MASK_ID
        ]
        if not snapshot:
            return None

        before = self.logical_len
        rows = self._action_rows()
        counts = dict.fromkeys(_COUNT_KEYS, 0)
        shift = 0  # net logical displacement caused by ops earlier this step
        expands = 0

        for snapped in snapshot:
            pos = snapped + shift
            if not 0 <= pos < self.logical_len:  # unreachable by construction
                raise RuntimeError(
                    f"bookkeeping failure: position {pos} outside logical"
                    f" region [0, {self.logical_len})"
                )
            action = _argmax(rows[snapped])

            if action == ACTION_EXPAND:
                if expands < max_expand and self.logical_len < self.l_max:
                    # Fresh MASK boundary at pos; old MASK slides right.
                    self.logical_len = insert_masks(
                        self.buffer, self.logical_len, pos, 1
                    )
                    expands += 1
                    shift += 1
                    counts["expand"] += 1
                else:
                    counts["keep"] += 1  # documented degradation
            elif action == ACTION_DELETE:
                self.logical_len = logical_delete(self.buffer, self.logical_len, pos)
                shift -= 1
                counts["delete"] += 1
            else:
                counts["keep"] += 1  # KEEP: remain masked (no token head)

        assert len(self.buffer) == self.l_max  # THE LAW: shape never changes
        return DiffusionStep(counts, before, self.logical_len)

    def run(self, max_steps: int) -> list[DiffusionStep]:
        """Iterate :meth:`step` until no MASK remains or ``max_steps`` hit.

        Enforces THE LAW after every step: ``len(buffer) == l_max``.
        """
        if max_steps < 0:
            raise ValueError(f"max_steps must be non-negative, got {max_steps}")
        steps: list[DiffusionStep] = []
        for _ in range(max_steps):
            outcome = self.step()
            if outcome is None:
                break
            assert len(self.buffer) == self.l_max
            steps.append(outcome)
        return steps

    # -- internals -----------------------------------------------------------

    def _require_started(self) -> None:
        if self.buffer is None:
            raise RuntimeError("PhantomLoop.start() must be called first")

    def _action_rows(self) -> list[list[float]]:
        """Action-score rows from the policy, normalized to per-position rows.

        The policy receives a sanitized id view (negative sentinels mapped to
        PAD_ID) plus ``derive_mask`` of the true logical length. Output shapes
        ``(A,)``/``(1, A)`` broadcast one shared row to every masked position;
        ``(l_max, A)`` supplies one row per step-start buffer position.
        """
        view = [t if t >= PAD_ID else PAD_ID for t in self.buffer]
        out = self.policy.forward(view, derive_mask(self.buffer, self.logical_len))
        if hasattr(out, "detach"):  # torch.Tensor / eager conversion, no import
            out = out.detach().tolist()
        flat = [list(r) for r in out]
        if not flat:
            raise ValueError("policy returned no action scores")
        width = len(flat[0])
        if any(len(r) != width for r in flat):
            raise ValueError("ragged action-score rows from policy")
        if len(flat) == 1:
            return flat * self.l_max  # sequence-level scores: shared row
        if len(flat) == self.l_max:
            return flat  # per-position rows, step-start indexing
        raise ValueError(
            f"policy returned {len(flat)} score rows; expected 1 or {self.l_max}"
        )


def _argmax(scores: list[float]) -> int:
    """Deterministic argmax: first (lowest) index wins ties."""
    best = 0
    for i in range(1, len(scores)):
        if scores[i] > scores[best]:
            best = i
    return best
