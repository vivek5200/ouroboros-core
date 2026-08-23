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

import random

import torch

from src.grpo import antithetic_pair, coupled_reward
from src.tokenizer import L_MAX, derive_mask, front_pack
from src.training import ACTION_EXPAND, PhantomPolicy

__all__ = [
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


def _placement_probs(policy: PhantomPolicy, instance: dict) -> torch.Tensor:
    """EXPAND probability over the ``logical_len + 1`` candidate splice sites.

    Candidate sites are exactly the insertion points ``src.tokenizer.insert_masks``
    accepts (``0 <= pos <= logical_len``): the phantom tail is not a place a
    real [EXPAND] can go, so it never competes for mass.
    """
    buffer, logical_len = front_pack(instance["ids"])
    ids = torch.tensor(buffer, dtype=torch.long)
    if int(ids.max()) >= policy.vocab_size:
        raise ValueError(
            f"instance id {int(ids.max())} outside policy vocab "
            f"[0, {policy.vocab_size})"
        )
    probs = torch.softmax(
        _position_logits(policy, ids, logical_len)[: logical_len + 1], dim=-1
    )
    return probs[:, ACTION_EXPAND]                # (logical_len + 1,)


def expand_position_accuracy(policy: PhantomPolicy, instance: dict) -> float:
    """P(the policy's EXPAND placement lands exactly on ``gap_start``).

    Runs the policy forward on the front-packed corrupted ids, converts the
    per-position action logits to probabilities, and reports the gap slot's
    share of total EXPAND mass over the candidate splice sites (the logical
    region plus its right edge — precisely the positions
    ``src.tokenizer.insert_masks`` accepts). I.e. the probability that a
    single stochastically-placed [EXPAND] hits the true deletion site.
    In [0, 1]; ≈ ``1 / (logical_len + 1)`` for a fresh policy, because an
    untrained policy spreads mass uniformly over the candidate sites.
    """
    gap_start = instance["gap_start"]
    # Insertion-point semantics (cf. insert_masks): 0 <= pos <= logical_len.
    if not 0 <= gap_start <= len(instance["ids"]):
        raise ValueError(f"gap_start {gap_start} outside corrupted stream")
    with torch.no_grad():
        expand_mass = _placement_probs(policy, instance)
    return float(expand_mass[gap_start] / expand_mass.sum())


class Trainer:
    """Coupled-GRPO trainer for stage-1 gap localization (masked loss)."""

    def __init__(self, policy: PhantomPolicy, lr: float = 1e-3, seed: int = 0):
        if not isinstance(policy, torch.nn.Module):
            raise TypeError("policy must be a torch.nn.Module (PhantomPolicy)")
        if getattr(policy, "l_max", 0) < L_MAX:
            raise ValueError(
                f"policy l_max {getattr(policy, 'l_max', None)} < tokenizer "
                f"L_MAX {L_MAX}: stage-1 buffers never resize"
            )
        self.policy = policy
        self.lr = lr
        self.seed = seed
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

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
        the per-position log-probabilities ever enters the graph.
        """
        gap = instance["gap_start"]
        buffer, logical_len = front_pack(instance["ids"])
        derive_mask(buffer, logical_len)  # contract check: logical region well-formed

        ids = torch.tensor(buffer, dtype=torch.long)
        logp = torch.log_softmax(_position_logits(self.policy, ids), dim=-1)

        rng = random.Random(couple_seed)
        m1, m2 = antithetic_pair(logical_len, rng)   # couple over live sites

        gen = torch.Generator(device="cpu").manual_seed(act_seed)
        # Exploration floor: sample from policy mixed with uniform. Pure-policy
        # sampling collapses (both sides draw the same non-EXPAND action →
        # advantage 0 → zero gradient → frozen policy). The 20% floor keeps
        # every action in play so the gap signal never starves. Scaffold-level
        # off-policy bias is accepted and documented.
        with torch.no_grad():
            base = torch.softmax(_position_logits(self.policy, ids)[gap], dim=-1)
            mix = 0.8 * base + 0.2 / base.numel()
        a1 = int(torch.multinomial(mix, num_samples=1, generator=gen))
        a2 = int(torch.multinomial(mix, num_samples=1, generator=gen))

        # Fuzzy Proxy outcomes: a side scores (T,T,T) iff ITS sampled action
        # at the gap is EXPAND; compute_reward maps that to 1.0 vs 0.0.
        def triple(action: int) -> tuple[bool, bool, bool]:
            hit = action == ACTION_EXPAND
            return (hit, hit, hit)

        r1, r2 = coupled_reward(*triple(a1), *triple(a2))
        advantage = r1 - r2                          # antithetic difference

        # Masked coupled-GRPO loss. Because both sides read the SAME
        # per-position distribution (the scaffold has no per-side rollouts),
        # the classic ``mean(chosen_logp)`` accumulation would push whatever
        # either side sampled *up* together and cancel. Instead each side's
        # own chosen-action log-prob is weighted by its reward centred on
        # the couple mean (the antithetic baseline):
        #     winner (r=1) gets +w · logp[winner_action]
        #     loser  (r=0) gets -w · logp[loser_action]
        # so descent raises P(EXPAND) and lowers P(non-EXPAND) AT THE GAP,
        # and contributes literally nothing anywhere else.
        baseline = 0.5 * (r1 + r2)
        loss = -(
            (r1 - baseline) * logp[gap, a1]
            + (r2 - baseline) * logp[gap, a2]
        )
        return loss, advantage

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Persist policy state + constructor config via ``torch.save``."""
        torch.save(
            {
                "policy_state": self.policy.state_dict(),
                "config": {
                    "vocab_size": self.policy.vocab_size,
                    "d_model": self.policy.d_model,
                    "n_actions": self.policy.n_actions,
                    "l_max": self.policy.l_max,
                    "lr": self.lr,
                    "seed": self.seed,
                },
            },
            path,
        )

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
        return cls(policy, lr=cfg["lr"], seed=cfg["seed"])
