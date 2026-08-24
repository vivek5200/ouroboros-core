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

Phase 2-A adds :class:`DiffusionPolicy`: a real (tiny) bidirectional
diffusion transformer — token + learned positional embeddings, pre-norm
self-attention, final LayerNorm, and a per-position action head — behind the
same stage-1 interface. Both policies expose ``site_logits(ids)`` so the
trainer reads per-position action scores without knowing the architecture;
``PhantomPolicy.site_logits`` reproduces the legacy trigram readout
(:func:`trigram_site_logits`, shared code ⇒ bit-identical behaviour).
"""

import torch

from src.grpo import coupled_reward
from src.tokenizer import IGNORE_ID, L_MAX, PAD_ID

__all__ = [
    "ACTION_DELETE",
    "ACTION_EXPAND",
    "ACTION_KEEP",
    "DiffusionPolicy",
    "PhantomPolicy",
    "default_outcomes",
    "grpo_step",
    "trigram_site_logits",
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

    def site_logits(
        self, ids: torch.Tensor, logical_len: int | None = None
    ) -> torch.Tensor:
        """Per-position action logits ``(L, n_actions)`` — the trainer API.

        Wraps :func:`trigram_site_logits` (the legacy readout formerly known
        as ``train_loop._position_logits``). Both share one code path, so
        ``site_logits`` is bit-identical to the legacy function by
        construction; passing ``logical_len`` reproduces its live-mean
        subtraction exactly, omitting it reproduces the raw readout.
        """
        return trigram_site_logits(self, ids, logical_len)


# ---------------------------------------------------------------------------
# Legacy trigram readout: single source of truth for per-position scores
# ---------------------------------------------------------------------------


def trigram_site_logits(
    policy: PhantomPolicy, ids: torch.Tensor, logical_len: int | None = None
) -> torch.Tensor:
    """Per-position action logits ``(L, n_actions)`` from scaffold parameters.

    Boundary-safe neighbor context: position 0 lacks a left neighbor and the
    last slot lacks a right one; missing neighbors contribute zeros.

    Two deliberate departures from naively reusing ``policy.forward``'s head,
    both required by stage-1 learning dynamics:

    * **No bias term.** ``policy.head``'s bias is shared by every slot, so a
      gap-only positive signal would inflate the EXPAND logit uniformly and
      saturate the whole buffer, erasing all contrast. The positional
      readout therefore uses ``head.weight`` only.
    * **Live-mean subtraction.** When ``logical_len`` is given, the mean
      context over the logical region is removed from every live slot,
      which cancels exactly the shared/global component of any update and
      keeps gap-site learning from leaking into a uniform shift.

    Slots beyond ``logical_len`` (phantom tail) get raw context; callers
    must exclude them from placement competitions anyway. Any policy with
    ``.embed`` and ``.head.weight`` shaped like a :class:`PhantomPolicy`
    works; :class:`DiffusionPolicy` intentionally does NOT route through
    here — its ``site_logits`` reads its own transformer heads.
    """
    h = policy.embed(ids)          # (L, d)
    ctx = h.clone()
    ctx[1:] += h[:-1]              # left neighbor
    ctx[:-1] += h[1:]              # right neighbor
    if logical_len is not None:
        mu = ctx[:logical_len].mean(dim=0, keepdim=True)
        ctx = torch.cat([ctx[:logical_len] - mu, ctx[logical_len:]])
    return torch.nn.functional.linear(ctx, policy.head.weight)  # (L, A)


# ---------------------------------------------------------------------------
# DiffusionPolicy (Phase 2-A): tiny bidirectional diffusion transformer
# ---------------------------------------------------------------------------


class DiffusionPolicy(torch.nn.Module):
    """A real (tiny) bidirectional diffusion transformer with per-position heads.

    Where :class:`PhantomPolicy` mean-pools the logical region into ONE
    action distribution per sequence, this policy keeps position: every
    slot gets its own action logits, so structural decisions ("which token
    is anomalous?") become expressible. Architecture at defaults
    (``d_model=64``):

        token Embedding(64, d) + learned pos Embedding(L_MAX=1024, d)
          → 2 x TransformerEncoderLayer(d, nhead=4, ffn=256,
                                        batch_first, norm_first)
          → final LayerNorm
          → action head Linear(d -> n_actions) applied PER POSITION

    ≈ 0.17 M parameters — far under the ~1.5 M budget, so CPU epochs stay
    in seconds.

    Sentinel handling (``src.tokenizer`` contracts, used as-is):
    ``MASK_ID=-1`` / ``IGNORE_ID=-2`` are negative list sentinels an
    ``nn.Embedding`` cannot index, so ids are clamped to ``min=0`` for the
    lookup — both sentinels share the PAD row's embedding vector — while a
    ``key_padding_mask = (ids == IGNORE_ID)`` removes [IGNORE] filler slots
    from attention entirely: they contribute exactly zero attention mass to
    every other position and receive zero gradient through masked paths.
    [MASK] placeholders sit INSIDE the logical region and participate like
    real tokens.

    Buffer discipline: ``forward`` runs the encoder over whatever buffer it
    is given (up to ``l_max`` rows; callers expose only the logical region).
    A degenerate all-[IGNORE] buffer falls back to unmasked attention rather
    than softmax over zero keys. Determinism: dropout defaults to ``0.0``
    and nothing samples, so evaluation is bit-reproducible — matching the
    ``Trainer`` contract.

    Args:
        vocab_size: Embedding cardinality; id 0 is padding, real ids >= 1.
        d_model: Model width (default 64).
        n_actions: Action logits per position; must be >= 3 so the
            KEEP/EXPAND/DELETE slots exist.
        l_max: Fixed maximum buffer length AND size of the learned
            positional table; defaults to ``tokenizer.L_MAX`` so the policy
            slots straight into stage-1 ``Trainer`` buffers.
        n_layers: Encoder blocks (default 2).
        n_heads: Attention heads per block; ``d_model % n_heads == 0``.
        dim_feedforward: FFN width per block (default 256).
        dropout: Attention/FFN dropout (default 0.0 for determinism).
        backbone_lr_scale: Learning-rate multiplier for everything except
            the action head when a ``Trainer`` builds its optimizer via
            :meth:`optimizer_param_groups` (default 0.05 — see there).
    """

    def __init__(
        self,
        vocab_size: int = 64,
        d_model: int = 64,
        n_actions: int = 3,
        l_max: int = L_MAX,
        n_layers: int = 2,
        n_heads: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.0,
        backbone_lr_scale: float = 0.05,
    ) -> None:
        super().__init__()
        if n_actions < _MIN_ACTIONS:
            raise ValueError(
                f"n_actions must be >= {_MIN_ACTIONS} "
                f"(KEEP/EXPAND/DELETE), got {n_actions}"
            )
        if vocab_size < 1:
            raise ValueError(f"vocab_size must be >= 1, got {vocab_size}")
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model {d_model} not divisible by n_heads {n_heads}"
            )
        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")
        if l_max < 1:
            raise ValueError(f"l_max must be >= 1, got {l_max}")
        if backbone_lr_scale < 0:
            raise ValueError(
                f"backbone_lr_scale must be >= 0, got {backbone_lr_scale}"
            )
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_actions = n_actions
        self.l_max = l_max
        self.embed = torch.nn.Embedding(vocab_size, d_model)
        # LEARNED positional table: l_max rows ride in state_dict so
        # save/load round-trips restore positions along with content.
        self.pos_embed = torch.nn.Embedding(l_max, d_model)
        layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        # enable_nested_tensor=False: keep the plain (deterministic) slow
        # path instead of mask-driven nested-tensor conversion.
        self.encoder = torch.nn.TransformerEncoder(
            layer,
            num_layers=n_layers,
            norm=torch.nn.LayerNorm(d_model),   # final LayerNorm
            enable_nested_tensor=False,
        )
        # Two-speed optimization (see optimizer_param_groups): the masked
        # coupled-GRPO advantage is a sparse +/-1 REINFORCE signal, and
        # Adam's per-parameter normalization marches every entry of the big
        # embedding tables ~lr per step no matter how thin the signal is.
        # Measured on this curriculum, that churn scrambles representations
        # faster than the action head can exploit them (head-only training
        # on the SAME frozen backbone lifts held-out placement by ~+0.06,
        # while joint single-lr training sinks below chance). The action
        # head therefore learns at full lr while the backbone trails at
        # ``backbone_lr_scale * lr`` — slower adaptation, never frozen.
        self.backbone_lr_scale = backbone_lr_scale
        # bias=False (legacy-readout parity): a shared bias is the cheapest
        # descent direction under the masked couple loss — it inflates
        # EXPAND uniformly across positions, and the placement metric's
        # live-mean subtraction cancels exactly that component. Without a
        # bias, every update must flow through content features.
        self.action_head = torch.nn.Linear(d_model, n_actions, bias=False)

    def forward(self, token_ids):
        """Encode one fixed-shape buffer into per-position predictions.

        Args:
            token_ids: Long values shaped ``(L,)``, each in
                ``[IGNORE_ID, vocab_size)`` (negative sentinel ids allowed).

        Returns:
            ``(action_logits, h)`` where ``action_logits`` is ``(L,
            n_actions)`` — one KEEP/EXPAND/DELETE distribution per slot —
            and ``h`` is the final-layer hidden state ``(L, d_model)``.

        Raises:
            ValueError: If the buffer is empty, exceeds ``l_max``, is not
                1-D, or carries ids outside the sentinel floor / vocabulary.
        """
        ids = torch.as_tensor(token_ids, dtype=torch.long)
        if ids.dim() != 1:
            raise ValueError(
                f"token_ids must be a 1-D buffer, got shape {tuple(ids.shape)}"
            )
        L = int(ids.shape[0])
        if L == 0:
            raise ValueError("buffer must contain at least one slot")
        if L > self.l_max:
            raise ValueError(
                f"buffer length {L} exceeds fixed l_max {self.l_max}"
            )
        lo, hi = int(ids.min()), int(ids.max())
        if hi >= self.vocab_size or lo < IGNORE_ID:
            raise ValueError(
                f"token ids must lie in [{IGNORE_ID}, {self.vocab_size}); "
                f"got range ({lo}, {hi})"
            )
        key_padding_mask = ids == IGNORE_ID              # (L,) True ⇒ invisible
        safe_ids = ids.clamp(min=0)                      # sentinels share PAD row
        h = self.embed(safe_ids) + self.pos_embed(torch.arange(L))  # (L, d)
        # Guard: a fully-[IGNORE] buffer would softmax over zero keys (NaN);
        # fall back to unmasked attention since there is nothing live left.
        mask = (
            None
            if bool(key_padding_mask.all())
            else key_padding_mask.unsqueeze(0)           # (1, L) for batch_first
        )
        h = self.encoder(h.unsqueeze(0), src_key_padding_mask=mask)  # (1,L,d)
        h = h.squeeze(0)
        return self.action_head(h), h                    # ((L, A), (L, d))

    def site_logits(
        self, token_ids, logical_len: int | None = None
    ) -> torch.Tensor:
        """Per-position action logits ``(L', n_actions)`` — the trainer API.

        Runs :meth:`forward` over the **candidate-site window** and returns
        its action logits:

        * with ``logical_len``: exactly rows ``0..logical_len`` — every
          splice site ``insert_masks`` accepts, phantom tail excluded — with
          the **live-mean subtracted** (mean action logits over the logical
          region removed from those rows, tail rows left raw). This mirrors
          the legacy readout's documented anti-drift device: the placement
          metric is a softmax ratio over sites, so any exactly-shared
          component (notably action-head bias drift under coupled-GRPO)
          would cancel from the ratio and freeze it; subtracting the region
          mean structurally restores the cross-site contrast.
        * without: trailing ``PAD_ID``/``IGNORE_ID`` filler trimmed to one
          safety row past the last live token (the right-edge splice site
          lives on that first filler slot), logits returned RAW — the
          couple-loss readout never mean-subtracts, matching the legacy
          split bit-for-bit in spirit. Front-packed stage-1 buffers never
          drag their ~1000-slot pad tail through quadratic attention.

        Because unmasked PAD filler would otherwise attend into live
        context, window-scoped outputs here are the canonical trainer-facing
        scores; full-buffer :meth:`forward` remains available for callers
        that want every physical slot.
        """
        ids = torch.as_tensor(token_ids, dtype=torch.long)
        L = int(ids.shape[0])
        if logical_len is not None:
            if not 0 <= logical_len <= L:
                raise ValueError(
                    f"logical_len {logical_len} outside buffer length {L}"
                )
            end = logical_len + 1                        # sites 0..logical_len
        else:
            filler = (ids == PAD_ID) | (ids == IGNORE_ID)
            live_idx = torch.nonzero(~filler, as_tuple=False)
            last_live = int(live_idx[-1]) if live_idx.numel() else -1
            end = max(min(L, last_live + 2), 1)          # +1 right-edge row
        logits, _ = self.forward(ids[:end])
        if logical_len is not None:
            mu = logits[:logical_len].mean(dim=0, keepdim=True)
            logits = torch.cat([logits[:logical_len] - mu, logits[logical_len:]])
        return logits

    def optimizer_param_groups(self, lr: float) -> list[dict]:
        """Two-speed Adam groups for :class:`src.train_loop.Trainer`.

        Returns ``[{"params": head, "lr": lr},
        {"params": backbone, "lr": lr * backbone_lr_scale}]`` — every
        parameter stays trainable, but the embedding tables and encoder
        weights adapt ``backbone_lr_scale`` times slower than the action
        head. Rationale: the stage-1 couple loss carries a sparse ±1
        advantage per instance; Adam's per-parameter normalization turns
        that into ~uniform-size steps, so at a single shared lr the big
        token/position tables random-walk fast enough to scramble the very
        content features the head is trying to read (head-only training on
        the frozen backbone lifts held-out placement by ~+0.06, single-lr
        joint training lands below chance). Slowing the backbone keeps all
        of it learnable while the head exploits features first.
        """
        head_params: list[torch.nn.Parameter] = []
        backbone_params: list[torch.nn.Parameter] = []
        for name, param in self.named_parameters():
            (head_params if "action_head" in name else backbone_params).append(
                param
            )
        return [
            {"params": head_params, "lr": lr},
            {"params": backbone_params, "lr": lr * self.backbone_lr_scale},
        ]


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
