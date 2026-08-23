"""Stage-1 training loop: teach PhantomPolicy WHERE statements were deleted.

Wires the stage-1 curriculum (:mod:`src.curriculum_data`) into Coupled-GRPO
(:mod:`src.grpo`) on top of the :class:`src.training.PhantomPolicy` scaffold:

* buffers obey the fixed-shape discipline of ``src.tokenizer`` — every
  corrupted stream is ``front_pack``-ed into exactly ``L_MAX`` slots and the
  logical region comes from ``derive_mask``;
* every instance contributes one antithetic couple
  (``src.grpo.antithetic_pair`` over the logical length) whose sides are
  scored with the Fuzzy Proxy reward (``compute_reward(True, True, True)
  == 1.0`` iff that side's sampled action at ``gap_start`` was ``EXPAND``,
  else ``compute_reward(False, False, False) == 0.0``);
* the loss is a **masked-loss variant implemented here**, not inside
  ``grpo_step`` (documented choice): log-probability contributions are
  gathered ONLY at ``gap_start``, so the advantage can only ever push
  P(EXPAND) up or down at gap sites and never wastes gradient budget on
  irrelevant positions.

Why per-position logits live in this module
-------------------------------------------
``PhantomPolicy.forward`` mean-pools over the whole logical region, which
erases position: one distribution per sequence cannot point at a gap. Rather
than change the scaffold (its contracts are pinned by tests), this module
derives *positional* action scores from the very same parameters::

    h_t            = policy.embed(id_t)
    context_t      = h_{t-1} + h_t + h_{t+1}     (boundary-safe, no new params)
    logits_t       = policy.head(context_t)

The ±1-token context is what makes the task learnable at all: after a
deletion the corrupted stream shows ``<number> <next-name> =`` exactly at
``gap_start``, whereas intact line starts show ``= <name>``, and the linear
head can separate those neighbor patterns through the shared embeddings.
Because no extra parameters are introduced, ``save``/``load`` round-trips
through ``policy.state_dict()`` alone stay complete.

The attention-based placement head (cross-site contrast)
--------------------------------------------------------
Experiment showed the trigram readout saturates at ~0.15–0.16 held-out
accuracy: in these snippets *every* candidate splice site shares a
near-identical local context shape, and ``_position_logits`` scores each site
independently — it cannot express "this site **vs the other sites**", which is
exactly what gap localization needs. :class:`PlacementHead` fixes that with a
learned comparison across candidate sites:

    site_i         = h_{i-1} + h_i                 (i = 0..logical_len, edge-clamped)
    x_i            = site_i + sinusoidal(site_i)   (site ORDER enters here)
    x              = MultiheadAttention(x)  x2     (the cross-site contrast)
    logit_i        = Linear(x_i -> 1)

and the placement distribution is the softmax over the ``logical_len + 1``
site logits. Because that final softmax competes every site against every
other, any shared/global component of the scores cancels — the head is the
live-mean-subtraction of the legacy readout done structurally. It is opt-in:
``Trainer(policy, ..., placement_head=head)`` and
``expand_position_accuracy(policy, inst, placement_head=head)``; passing
``None`` (default) keeps the legacy behaviour bit-for-bit.

Accuracy metric (and its chance baseline)
-----------------------------------------
``expand_position_accuracy`` normalizes the EXPAND probability mass at
``gap_start`` over the **candidate splice sites**: the logical region plus
its right edge — exactly the positions ``src.tokenizer.insert_masks``
accepts. A freshly initialized policy is near-uniform per slot, so the true
gap holds ~1/(logical_len + 1) of the mass; that uniform share is the chance
baseline training must double. (The phantom tail is excluded on purpose: a
real [EXPAND] can never land there, and since the masked loss never touches
tail slots, including them would let their mass inflate in lockstep with the
gap site through shared parameters — the ratio would be pinned at 1/L_MAX
forever and no amount of learning could move it.) Training concentrates mass
on true gap sites, so the ratio climbs far above baseline while staying in
[0, 1].

Determinism: all stochasticity (antithetic couples, action sampling) flows
from arithmetically derived seeds off ``Trainer.seed``; no global RNG state
is read, so epochs are bit-reproducible given (policy, dataset, seed).
"""

import math
import random

import torch

from src.grpo import antithetic_pair, coupled_reward
from src.tokenizer import L_MAX, derive_mask, front_pack
from src.training import ACTION_EXPAND, PhantomPolicy

__all__ = [
    "PlacementHead",
    "Trainer",
    "expand_position_accuracy",
]


def _position_logits(
    policy: PhantomPolicy, ids: torch.Tensor, logical_len: int | None = None
) -> torch.Tensor:
    """Per-position action logits ``(L, n_actions)`` from scaffold parameters.

    Boundary-safe neighbor context: position 0 lacks a left neighbor and the
    last slot lacks a right one; missing neighbors contribute zeros.

    Two deliberate departures from naively reusing ``policy.forward``'s head,
    both required by stage-1 learning dynamics (see module docstring):

    * **No bias term.** ``policy.head``'s bias is shared by every slot, so a
      gap-only positive signal would inflate the EXPAND logit uniformly and
      saturate the whole buffer, erasing all contrast. The positional
      readout therefore uses ``head.weight`` only.
    * **Live-mean subtraction.** When ``logical_len`` is given, the mean
      context over the logical region is removed from every live slot,
      which cancels exactly the shared/global component of any update and
      keeps gap-site learning from leaking into a uniform shift.

    Slots beyond ``logical_len`` (phantom tail) get raw context; callers
    must exclude them from placement competitions anyway.
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
# Attention-based placement head: cross-site contrast readout
# ---------------------------------------------------------------------------


class PlacementHead(torch.nn.Module):
    """Scores candidate splice sites AGAINST EACH OTHER with self-attention.

    The legacy readout (``_position_logits``) maps each site's trigram context
    through a shared linear head — every site is scored in isolation, so when
    all sites look locally alike (which they do in this curriculum) the best it
    can do is rank the shared shape. This head instead builds one representation
    per candidate site, lets the sites attend to one another, and emits a single
    logit per site; the placement distribution is the softmax over sites, i.e.
    an explicit "this site vs the OTHER sites" competition.

    Pipeline (``d = d_model``, ``S = logical_len + 1`` candidate sites):

    1. **Site representations.** ``site_i = h_{i-1} + h_i`` for
       ``i = 0..logical_len`` with edge clamps (site 0 reads ``h_0`` twice,
       the last site reads ``h_logical_len``), where ``h = policy.embed(ids)``
       is the per-position embedding of the front-packed buffer. Only rows
       ``0..logical_len`` are ever touched — the phantom tail is excluded by
       construction.
    2. **Sinusoidal site-index encoding.** Standard sin/cos positional codes
       are added so attention can use site ORDER (where the anomaly sits in
       the statement list), not just content. The table is a non-persistent
       buffer sized for the tokenizer's full ``L_MAX + 1`` sites and sliced
       per call — nothing position-dependent is ever a learned parameter.
    3. **Cross-site contrast.** ``n_layers`` ``torch.nn.MultiheadAttention``
       blocks (pre-norm residual: ``x = x + MHA(LN(x), LN(x), LN(x))``) over
       the site sequence. This is the step the legacy readout cannot express:
       attention weights are normalized ACROSS sites, so each site's output
       depends on how it compares to every other site in the instance.
    4. **Scorer.** ``Linear(d_model -> 1)`` gives one logit per site;
       :meth:`forward` softmaxes them into the placement distribution.

    Args:
        d_model: Embedding width; must equal the policy's ``d_model``.
        n_heads: Attention heads per block (``d_model`` must be divisible);
            default 4.
        n_layers: Attention blocks, 1 or 2 (default 2).
        max_sites: Longest site sequence the sinusoidal table precomputes;
            defaults to ``tokenizer.L_MAX + 1``.

    Determinism: pure functions of the input — no dropout, no sampling — so
    evaluation is bit-reproducible, matching the ``Trainer`` contract.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        n_layers: int = 2,
        max_sites: int = L_MAX + 1,
    ) -> None:
        super().__init__()
        if d_model < 1:
            raise ValueError(f"d_model must be >= 1, got {d_model}")
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model {d_model} not divisible by n_heads {n_heads}"
            )
        if n_layers not in (1, 2):
            raise ValueError(f"n_layers must be 1 or 2, got {n_layers}")
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.attn = torch.nn.ModuleList(
            torch.nn.MultiheadAttention(d_model, n_heads, batch_first=True)
            for _ in range(n_layers)
        )
        self.norms = torch.nn.ModuleList(
            torch.nn.LayerNorm(d_model) for _ in range(n_layers)
        )
        self.final_norm = torch.nn.LayerNorm(d_model)
        # bias=False on purpose: a scorer bias adds the SAME constant to every
        # site logit, which softmax cancels (shift invariance) — its gradient
        # is identically zero, so it would be dead weight.
        self.scorer = torch.nn.Linear(d_model, 1, bias=False)
        # Sin/cos table for site indices 0..max_sites-1 (non-persistent: it is
        # a pure function of shape, so state dicts stay lean and portable).
        position = torch.arange(max_sites, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_sites, d_model)
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer("site_pe", pe, persistent=False)

    # ------------------------------------------------------------------

    def logits(self, h: torch.Tensor, logical_len: int) -> torch.Tensor:
        """Raw placement logits over candidate sites, shape ``(S,)``.

        Args:
            h: Per-position embeddings ``(L, d_model)`` — e.g.
                ``policy.embed(front_packed_ids)``.
            logical_len: Live prefix length of the buffer; sites
                ``0..logical_len`` are scored (``S = logical_len + 1``).

        Returns:
            Float tensor of ``logical_len + 1`` site logits (softmax over it
            is the placement distribution).
        """
        if h.dim() != 2 or h.shape[-1] != self.d_model:
            raise ValueError(
                f"h must be (L, {self.d_model}), got {tuple(h.shape)}"
            )
        if logical_len < 0:
            raise ValueError(f"logical_len must be >= 0, got {logical_len}")
        # Site i needs rows i-1 and i; slicing to logical_len + 1 rows keeps
        # the phantom tail out of the competition entirely (Python slicing
        # clamps, so a degenerate full buffer degrades gracefully).
        hs = h[: logical_len + 1]                       # (S, d)
        prev = torch.cat([hs[:1], hs[:-1]], dim=0)      # edge-clamped h_{i-1}
        x = hs + prev                                   # (S, d) site contexts
        x = x + self.site_pe[: x.shape[0]].to(x.dtype)  # site order enters
        x = x.unsqueeze(0)                              # (1, S, d), batch_first
        for norm, attn in zip(self.norms, self.attn):
            attended, _ = attn(norm(x), norm(x), norm(x), need_weights=False)
            x = x + attended                            # pre-norm residual
        scores = self.scorer(self.final_norm(x))        # (1, S, 1)
        return scores.squeeze(0).squeeze(-1)            # (S,)

    def forward(self, h: torch.Tensor, logical_len: int) -> torch.Tensor:
        """Placement distribution over candidate sites, ``(S,)`` summing to 1."""
        return torch.softmax(self.logits(h, logical_len), dim=-1)


def _placement_probs(
    policy: PhantomPolicy,
    instance: dict,
    placement_head: "PlacementHead | None" = None,
) -> torch.Tensor:
    """EXPAND probability over the ``logical_len + 1`` candidate splice sites.

    Candidate sites are exactly the insertion points ``src.tokenizer.insert_masks``
    accepts (``0 <= pos <= logical_len``): the phantom tail is not a place a
    real [EXPAND] can go, so it never competes for mass.

    With ``placement_head`` given, the distribution comes from the attention
    head's site competition instead of the legacy per-position readout.
    """
    buffer, logical_len = front_pack(instance["ids"])
    ids = torch.tensor(buffer, dtype=torch.long)
    if int(ids.max()) >= policy.vocab_size:
        raise ValueError(
            f"instance id {int(ids.max())} outside policy vocab "
            f"[0, {policy.vocab_size})"
        )
    if placement_head is None:
        probs = torch.softmax(
            _position_logits(policy, ids, logical_len)[: logical_len + 1], dim=-1
        )
        return probs[:, ACTION_EXPAND]                # (logical_len + 1,)
    return placement_head(policy.embed(ids), logical_len)  # (logical_len + 1,)


def expand_position_accuracy(
    policy: PhantomPolicy,
    instance: dict,
    placement_head: "PlacementHead | None" = None,
) -> float:
    """P(the policy's EXPAND placement lands exactly on ``gap_start``).

    Runs the policy forward on the front-packed corrupted ids, converts the
    per-position action logits to probabilities, and reports the gap slot's
    share of total EXPAND mass over the candidate splice sites (the logical
    region plus its right edge — precisely the positions
    ``src.tokenizer.insert_masks`` accepts). I.e. the probability that a
    single stochastically-placed [EXPAND] hits the true deletion site.
    In [0, 1]; ≈ ``1 / (logical_len + 1)`` for a fresh policy, because an
    untrained policy spreads mass uniformly over the candidate sites.

    With ``placement_head`` given, the site distribution comes from the
    attention head's cross-site competition instead of the legacy readout;
    semantics are otherwise identical.
    """
    gap_start = instance["gap_start"]
    # Insertion-point semantics (cf. insert_masks): 0 <= pos <= logical_len.
    if not 0 <= gap_start <= len(instance["ids"]):
        raise ValueError(f"gap_start {gap_start} outside corrupted stream")
    with torch.no_grad():
        expand_mass = _placement_probs(policy, instance, placement_head)
    return float(expand_mass[gap_start] / expand_mass.sum())


class Trainer:
    """Coupled-GRPO trainer for stage-1 gap localization (masked loss)."""

    def __init__(
        self,
        policy: PhantomPolicy,
        lr: float = 1e-3,
        seed: int = 0,
        placement_head: "PlacementHead | None" = None,
    ):
        if not isinstance(policy, torch.nn.Module):
            raise TypeError("policy must be a torch.nn.Module (PhantomPolicy)")
        if getattr(policy, "l_max", 0) < L_MAX:
            raise ValueError(
                f"policy l_max {getattr(policy, 'l_max', None)} < tokenizer "
                f"L_MAX {L_MAX}: stage-1 buffers never resize"
            )
        if placement_head is not None:
            if not isinstance(placement_head, torch.nn.Module):
                raise TypeError(
                    "placement_head must be a torch.nn.Module (PlacementHead)"
                )
            if getattr(placement_head, "d_model", None) != policy.d_model:
                raise ValueError(
                    f"placement_head d_model "
                    f"{getattr(placement_head, 'd_model', None)} != policy "
                    f"d_model {policy.d_model}"
                )
        self.policy = policy
        self.lr = lr
        self.seed = seed
        self.placement_head = placement_head
        params = list(policy.parameters())
        if placement_head is not None:
            params += list(placement_head.parameters())
        self.optimizer = torch.optim.Adam(params, lr=lr)

    # ------------------------------------------------------------------
    # One epoch (= full passes over the dataset in batches)
    # ------------------------------------------------------------------

    def epoch(self, dataset, batch_size: int = 8, epochs: int = 1) -> dict:
        """Train on stage-1 instances; returns aggregate metrics.

        Per instance: build the antithetic couple over the logical length,
        sample each side's action AT ``gap_start`` from the shared per-position
        distribution, score the sides with the Fuzzy Proxy reward
        (EXPAND-at-gap ⇒ ``(True, True, True)`` ⇒ 1.0, else 0.0), form
        ``advantage = r1 - r2``, and add the masked REINFORCE term
        ``-adv * mean(logp_side1(chosen), logp_side2(chosen))`` where both
        log-probs are gathered at ``gap_start`` ONLY. One Adam step per batch.

        Args:
            dataset: List of :func:`src.curriculum_data.make_instance` dicts.
            batch_size: Instances accumulated per optimizer step.
            epochs: Number of passes over ``dataset``.

        Returns:
            ``{"loss": mean couple loss, "advantage_mean": mean advantage,
            "updates": optimizer steps, "instances": samples seen}``.
        """
        if not dataset:
            raise ValueError("dataset must contain at least one instance")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")

        loss_sum = 0.0
        adv_sum = 0.0
        couples = 0
        updates = 0
        seen = 0

        for ep in range(epochs):
            for start in range(0, len(dataset), batch_size):
                chunk = dataset[start : start + batch_size]
                self.optimizer.zero_grad()
                batch_loss = torch.zeros(())
                for offset, inst in enumerate(chunk):
                    idx = start + offset
                    couple_seed = self.seed + 7919 * ep + 104_729 * idx
                    act_seed = self.seed * 1_000_003 + 10_007 * ep + 97 * idx
                    loss_i, adv_i = self._couple_loss(inst, couple_seed, act_seed)
                    batch_loss = batch_loss + loss_i
                    loss_sum += float(loss_i.detach())
                    adv_sum += adv_i
                    couples += 1
                    seen += 1
                batch_loss.backward()
                self.optimizer.step()
                updates += 1

        return {
            "loss": loss_sum / couples,
            "advantage_mean": adv_sum / couples,
            "updates": updates,
            "instances": seen,
        }

    # ------------------------------------------------------------------
    # One antithetic couple on one instance (masked to the gap site)
    # ------------------------------------------------------------------

    def _couple_loss(self, instance: dict, couple_seed: int, act_seed: int):
        """Masked coupled-GRPO loss for a single instance.

        Returns ``(loss_tensor, advantage)``. Only the ``gap_start`` slice of
        the readout ever enters the graph.

        Strategy flag (``self.placement_head``):

        * **None — legacy action competition.** The categorical is over
          ACTIONS *at* the fixed gap site: a side wins iff its sampled action
          there was ``EXPAND``, and descent raises P(EXPAND) / lowers
          P(non-EXPAND) at the gap.
        * **PlacementHead — site competition.** The categorical is over
          CANDIDATE SITES ``0..logical_len``: a side wins iff its sampled
          splice site *is* the gap, and descent raises P(gap site) while
          pushing down the specific wrong site that stole mass — the
          contrastive signal the legacy per-position scores cannot express.
        """
        gap = instance["gap_start"]
        buffer, logical_len = front_pack(instance["ids"])
        derive_mask(buffer, logical_len)  # contract check: logical region well-formed

        ids = torch.tensor(buffer, dtype=torch.long)
        if self.placement_head is None:
            # Legacy: per-position action logits; only the gap row competes.
            cand_logits = _position_logits(self.policy, ids)[gap]     # (A,)
            target = ACTION_EXPAND            # winning candidate = EXPAND action
        else:
            # Head: cross-site logits; the whole site set competes.
            h = self.policy.embed(ids)
            cand_logits = self.placement_head.logits(h, logical_len)  # (S,)
            target = gap                      # winning candidate = true site

        logp = torch.log_softmax(cand_logits, dim=-1)

        rng = random.Random(couple_seed)
        m1, m2 = antithetic_pair(logical_len, rng)   # couple over live sites

        gen = torch.Generator(device="cpu").manual_seed(act_seed)
        # Exploration floor: sample from policy mixed with uniform. Pure-policy
        # sampling collapses (both sides draw the same losing candidate →
        # advantage 0 → zero gradient → frozen policy). The 20% floor keeps
        # every candidate in play so the gap signal never starves. Scaffold-level
        # off-policy bias is accepted and documented.
        with torch.no_grad():
            base = torch.softmax(cand_logits.detach(), dim=-1)
            mix = 0.8 * base + 0.2 / base.numel()
        a1 = int(torch.multinomial(mix, num_samples=1, generator=gen))
        a2 = int(torch.multinomial(mix, num_samples=1, generator=gen))

        # Fuzzy Proxy outcomes: a side scores (T,T,T) iff ITS sampled
        # candidate hit the target (EXPAND action at the gap / the gap site);
        # compute_reward maps that to 1.0 vs 0.0.
        def triple(candidate: int) -> tuple[bool, bool, bool]:
            hit = candidate == target
            return (hit, hit, hit)

        r1, r2 = coupled_reward(*triple(a1), *triple(a2))
        advantage = r1 - r2                          # antithetic difference

        # Masked coupled-GRPO loss. Because both sides read the SAME
        # distribution (the scaffold has no per-side rollouts), the classic
        # ``mean(chosen_logp)`` accumulation would push whatever either side
        # sampled *up* together and cancel. Instead each side's own chosen
        # candidate log-prob is weighted by its reward centred on the couple
        # mean (the antithetic baseline):
        #     winner (r=1) gets +w · logp[winner_candidate]
        #     loser  (r=0) gets -w · logp[loser_candidate]
        # so descent raises the winning candidate and lowers the losing one,
        # and contributes literally nothing anywhere else.
        baseline = 0.5 * (r1 + r2)
        loss = -(
            (r1 - baseline) * logp[a1]
            + (r2 - baseline) * logp[a2]
        )
        return loss, advantage

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Persist policy state + constructor config via ``torch.save``.

        When a placement head is attached, its config and weights are saved
        alongside so :meth:`load` rebuilds a behaviour-identical trainer.
        """
        blob = {
            "policy_state": self.policy.state_dict(),
            "config": {
                "vocab_size": self.policy.vocab_size,
                "d_model": self.policy.d_model,
                "n_actions": self.policy.n_actions,
                "l_max": self.policy.l_max,
                "lr": self.lr,
                "seed": self.seed,
                "placement_head": None
                if self.placement_head is None
                else {
                    "d_model": self.placement_head.d_model,
                    "n_heads": self.placement_head.n_heads,
                    "n_layers": self.placement_head.n_layers,
                },
            },
        }
        if self.placement_head is not None:
            blob["placement_head_state"] = self.placement_head.state_dict()
        torch.save(blob, path)

    @classmethod
    def load(cls, path: str) -> "Trainer":
        """Rebuild a Trainer (fresh Adam state) with identical policy weights.

        Optimizer moments are training state, not part of the accuracy
        contract, so they are intentionally not persisted: a loaded trainer
        reproduces the saved policy's behaviour bit-for-bit.
        """
        blob = torch.load(path)
        cfg = blob["config"]
        model_cfg = {
            key: cfg[key]
            for key in ("vocab_size", "d_model", "n_actions", "l_max")
        }
        policy = PhantomPolicy(**model_cfg)
        policy.load_state_dict(blob["policy_state"])
        head_cfg = cfg.get("placement_head")
        if head_cfg is None:
            return cls(policy, lr=cfg["lr"], seed=cfg["seed"])
        head = PlacementHead(**head_cfg)
        head.load_state_dict(blob["placement_head_state"])
        return cls(policy, lr=cfg["lr"], seed=cfg["seed"], placement_head=head)

    # ------------------------------------------------------------------
    # Checkpoint-completeness aliases: BOTH state dicts ride together
    # ------------------------------------------------------------------

    def save_all(self, path: str) -> None:
        """Alias of :meth:`save` under the explicit completeness name.

        ``save``/``load`` already round-trip ``policy.state_dict()`` AND
        ``placement_head.state_dict()`` (plus constructor config) in one
        blob; ``save_all``/``load_all`` expose that guarantee under names
        that say so — checkpoint helpers always restore identical accuracy.
        """
        self.save(path)

    @classmethod
    def load_all(cls, path: str) -> "Trainer":
        """Alias of :meth:`load`: rebuild Trainer + policy + placement head."""
        return cls.load(path)
