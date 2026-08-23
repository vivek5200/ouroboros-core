"""Module 1 RL — training-step scaffold for Coupled-GRPO in torch.

Wires together the pieces that already exist on the stdlib side:

* ``src.grpo.sample_batch`` supplies antithetic mask couples
  (``m(1) ∪ m(2) = 1``, ``m(1) ∩ m(2) = ∅``),
* ``src.grpo.coupled_reward`` scores each side with the Fuzzy Proxy reward,
* this module turns ``advantage = r1 - r2`` per couple into one REINFORCE-style
  Adam step on a tiny policy.

``PhantomPolicy`` is a scaffold, not the paper's editor: actions 1 ([EXPAND])
and 2 ([DELETE]) are *placeholders* whose real semantics land with the
diffusion loop. The scaffold exists to exercise gradients end-to-end —
Embedding → mean-pool over unmasked (logical) positions → Linear action head —
under the fixed L_MAX-buffer discipline of ``src.tokenizer`` (buffers never
resize; only the logical region participates in pooling).

Everything is CPU-first and toy-scaled: default ``vocab_size=64``,
``l_max=32`` (a miniature of ``tokenizer.L_MAX=1024``) keeps the test battery
well under two seconds.
"""

import torch

from src.grpo import coupled_reward

__all__ = [
    "ACTION_DELETE",
    "ACTION_EXPAND",
    "ACTION_KEEP",
    "PhantomPolicy",
    "default_outcomes",
    "grpo_step",
]


# ---------------------------------------------------------------------------
# Action space: KEEP is real; EXPAND/DELETE are placeholders for now
# ---------------------------------------------------------------------------

ACTION_KEEP = 0    # keep the token as-is
ACTION_EXPAND = 1  # [EXPAND]-ish placeholder (real splice semantics: diffusion loop)
ACTION_DELETE = 2  # [DELETE]-ish placeholder (real delete semantics: diffusion loop)

_MIN_ACTIONS = 3  # KEEP + EXPAND + DELETE must all exist as slots


# ---------------------------------------------------------------------------
# PhantomPolicy: tiny policy over fixed L_MAX buffers
# ---------------------------------------------------------------------------


class PhantomPolicy(torch.nn.Module):
    """Embedding → mean-pool over unmasked (logical) positions → Linear head.

    The buffer shape is treated as fixed at ``l_max`` (the toy analogue of
    ``tokenizer.L_MAX``): ``forward`` accepts buffers no longer than it and
    derives everything from an explicit logical mask, mirroring
    ``derive_mask``/``front_pack`` — phantom tail slots never influence the
    output.

    Args:
        vocab_size: Embedding cardinality. Id 0 is padding per
            ``src.tokenizer.PAD_ID``; real ids start at 1.
        d_model: Embedding width of the single-layer policy.
        n_actions: Number of action logits; must be >= 3 so that the
            KEEP/EXPAND/DELETE slots (0/1/2) all exist.
        l_max: Fixed maximum buffer length accepted by ``forward``.
    """

    def __init__(
        self,
        vocab_size: int = 64,
        d_model: int = 16,
        n_actions: int = 3,
        l_max: int = 32,
    ) -> None:
        super().__init__()
        if n_actions < _MIN_ACTIONS:
            raise ValueError(
                f"n_actions must be >= {_MIN_ACTIONS} "
                f"(KEEP/EXPAND/DELETE), got {n_actions}"
            )
        if vocab_size < 1:
            raise ValueError(f"vocab_size must be >= 1, got {vocab_size}")
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_actions = n_actions
        self.l_max = l_max
        self.embed = torch.nn.Embedding(vocab_size, d_model)
        self.head = torch.nn.Linear(d_model, n_actions)

    def forward(self, token_ids, mask) -> torch.Tensor:
        """Logits ``(B, n_actions)`` for batched or single fixed buffers.

        Args:
            token_ids: Long values shaped ``(L,)`` or ``(B, L)``, each in
                ``[0, vocab_size)``.
            mask: Logical-region mask, same shape as ``token_ids``; ``True``
                marks an unmasked (live) position included in the pool.

        Returns:
            Float tensor of action logits shaped ``(B, n_actions)``. A row
            with an empty logical region pools to zeros (count clamped to 1)
            rather than producing NaNs.

        Raises:
            ValueError: If shapes disagree, the buffer exceeds ``l_max``,
                or ids fall outside the vocabulary.
        """
        ids = torch.as_tensor(token_ids, dtype=torch.long)
        m = torch.as_tensor(mask)
        if ids.shape != m.shape:
            raise ValueError(
                f"token_ids shape {tuple(ids.shape)} != mask shape {tuple(m.shape)}"
            )
        if ids.dim() == 1:
            ids, m = ids.unsqueeze(0), m.unsqueeze(0)
        L = ids.shape[-1]
        if L > self.l_max:
            raise ValueError(
                f"buffer length {L} exceeds fixed l_max {self.l_max}"
            )
        if bool((ids >= self.vocab_size).any()) or bool((ids < 0).any()):
            raise ValueError(
                f"token id out of range [0, {self.vocab_size})"
            )
        emb = self.embed(ids)                                  # (B, L, d)
        keep = m.to(emb.dtype).unsqueeze(-1)                   # (B, L, 1)
        counts = keep.sum(dim=1).clamp(min=1.0)                # (B, 1), no /0
        pooled = (emb * keep).sum(dim=1) / counts              # mean over live
        return self.head(pooled)                               # (B, A)


# ---------------------------------------------------------------------------
# Default deterministic outcome stub
# ---------------------------------------------------------------------------


def default_outcomes(m1: list[bool], m2: list[bool]):
    """Deterministic synthetic outcome triples derived from the masks alone.

    Each side's triple ``(parses, checks, tests)`` thresholds its share of
    selected tokens, so rewards are reproducible from the couple without any
    rollout machinery. Replace via ``grpo_step(..., reward_fn=...)`` once real
    rollouts exist.

    Returns:
        ``((parses1, checks1, tests1), (parses2, checks2, tests2))``.
    """

    def triple(mask: list[bool]) -> tuple[bool, bool, bool]:
        ones = sum(1 for b in mask if b)
        frac = ones / max(len(mask), 1)
        return (frac > 0.0, frac >= 0.5, frac >= 0.75)

    return triple(m1), triple(m2)


# ---------------------------------------------------------------------------
# One coupled-GRPO step
# ---------------------------------------------------------------------------


def _default_logprobs(policy: PhantomPolicy, token_ids, mask) -> torch.Tensor:
    """Default ``logprobs_fn``: log-softmax of the policy's action logits."""
    return torch.log_softmax(policy(token_ids, mask), dim=-1)


def grpo_step(
    policy: PhantomPolicy,
    batch_pairs,
    logprobs_fn=None,
    *,
    reward_fn=None,
    lr: float = 1e-3,
    seed: int = 0,
    token_ids=None,
) -> dict:
    """One antithetic coupled-GRPO update (REINFORCE loss + single Adam step).

    For each ``(m1, m2)`` couple from :func:`src.grpo.sample_batch`:

    1. Draw two synthetic outcome triples via ``reward_fn(m1, m2)`` (or the
       deterministic :func:`default_outcomes` stub) and score both sides with
       :func:`src.grpo.coupled_reward`.
    2. ``advantage = r1 - r2`` — the antithetic variance reduction: shared
       couple information cancels in the difference.
    3. Feed the same token buffer twice through the policy (mask ``m1`` then
       mask ``m2``), sample one action per side from the resulting
       distribution, and accumulate
       ``loss -= advantage * mean(logp_chosen_side1, logp_chosen_side2)``.

    After the batch: one Adam step at ``lr`` on a fresh optimizer (this
    scaffold is stateless; a persistent optimizer arrives with the trainer).

    Args:
        policy: A :class:`PhantomPolicy` (any module emitting action logits
            works if ``logprobs_fn`` matches).
        batch_pairs: List of ``(m1, m2)`` complementary ``list[bool]`` masks.
        logprobs_fn: Optional ``f(policy, token_ids, mask) -> (B, n_actions)`
            log-probability tensor overriding the default log-softmax head.
        reward_fn: Optional ``f(m1, m2) -> (triple1, triple2)`` of boolean
            outcome triples; default :func:`default_outcomes`.
        lr: Adam learning rate for the single step (default 1e-3).
        seed: Seed for the CPU generator driving action sampling, making the
            whole step bit-reproducible.
        token_ids: Optional shared buffer of ``len(m1)`` ids in
            ``[0, vocab_size)``; defaults to a deterministic cycle over the
            non-pad vocabulary (id 0 is reserved for padding).

    Returns:
        ``{"loss": float, "advantage_mean": float, "param_delta": float}``
        where ``param_delta`` is the L2 distance between the flattened
        parameters before vs after the step.

    Raises:
        ValueError: If ``batch_pairs`` is empty, mask lengths are ragged, or
            custom ``token_ids`` fall outside the vocabulary.
    """
    if not batch_pairs:
        raise ValueError("batch_pairs must contain at least one (m1, m2) couple")
    lengths = {len(m1) for m1, _ in batch_pairs} | {len(m2) for _, m2 in batch_pairs}
    if len(lengths) != 1:
        raise ValueError(f"ragged batch: couple mask lengths {sorted(lengths)}")
    L = lengths.pop()

    vocab = getattr(policy, "vocab_size", None)
    if token_ids is None:
        span = max(vocab - 1, 1)
        ids_list = [1 + (i % span) for i in range(L)]  # never pad id 0
    else:
        ids_list = list(token_ids)
        if len(ids_list) != L:
            raise ValueError(f"token_ids length {len(ids_list)} != mask length {L}")
        if vocab is not None and any(i < 0 or i >= vocab for i in ids_list):
            raise ValueError(f"token id out of range [0, {vocab})")
    ids = torch.tensor(ids_list, dtype=torch.long)

    gen = torch.Generator(device="cpu").manual_seed(seed)
    logprobs_of = logprobs_fn or _default_logprobs

    # Snapshot θ_before for the param_delta report.
    before = torch.cat([p.detach().reshape(-1) for p in policy.parameters()])

    total_loss = torch.zeros((), dtype=torch.float32)
    advantages: list[float] = []
    for m1, m2 in batch_pairs:
        t1, t2 = reward_fn(m1, m2) if reward_fn is not None else default_outcomes(m1, m2)
        r1, r2 = coupled_reward(*t1, *t2)
        adv = r1 - r2
        advantages.append(adv)

        pair_masks = torch.tensor([m1, m2])                    # (2, L) bool
        pair_ids = ids.unsqueeze(0).expand(pair_masks.shape[0], -1)  # (2, L)
        logp = logprobs_of(policy, pair_ids, pair_masks)       # (2, A)
        with torch.no_grad():
            chosen = torch.multinomial(
                logp.detach().exp(), num_samples=1, generator=gen
            )                                                  # (2, 1)
        chosen_logp = logp.gather(1, chosen).mean()            # mean of the couple

        total_loss = total_loss - adv * chosen_logp

    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    after = torch.cat([p.detach().reshape(-1) for p in policy.parameters()])
    return {
        "loss": float(total_loss.detach()),
        "advantage_mean": float(sum(advantages) / len(advantages)),
        "param_delta": float(torch.linalg.vector_norm(after - before)),
    }
