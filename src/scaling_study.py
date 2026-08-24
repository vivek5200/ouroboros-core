"""Scaling-study harness: the SAME pipeline tiny-on-CPU and big-on-T4.

One code path answers "how does stage-1 gap localization scale with model
width / data / epochs?": a :class:`StudyConfig` fully describes one run,
:func:`run_config` executes it (fresh baseline -> train -> post measurement
on held-out instances), :func:`scaling_study` runs a list of configs, and
:func:`render_table` prints the results as an aligned ASCII table. Nothing
in the pipeline is CPU- or GPU-specific — swap the device, not the code.

What one config measures
------------------------
``run_config`` builds a fresh ``PhantomPolicy`` (vocab 64, ``l_max=1024``)
plus an attention ``PlacementHead``, evaluates mean held-out placement
accuracy BEFORE any training (``fresh`` — the ~1/(logical_len+1) chance
baseline), trains with ``Trainer(lr=cfg.lr, seed=cfg.seed)`` on
``cfg.instances`` stage-1 instances from the window starting at
``train_seed0``, re-evaluates on the SAME held-out instances (``post``),
and reports absolute ``lift = post - fresh``.

Data hygiene (documented): training and evaluation windows are DISJOINT by
construction — train seeds run ``[train_seed0, train_seed0 + instances)``,
held-out seeds run ``[held_seed0, held_seed0 + heldout)``; with the default
windows (2000+ vs 999999+) no instance is ever scored on data it trained on.
Held-out accuracy is therefore a genuine generalization measure, matching
the protocol of ``tests/test_training.py``.

Determinism: ``run_config`` owns ALL its randomness. A single
``torch.manual_seed(cfg.seed)`` fixes policy AND placement-head init;
every training-time stochastic choice (antithetic couples, action
sampling) already derives arithmetically from ``Trainer.seed``
(``src.train_loop`` contract), and evaluation is a pure function of
parameters. So identical config + seed reproduce bit-identical lifts
regardless of what ran before — no global-RNG leakage between runs.

CPU budget: :data:`QUICK_CONFIGS` is the smoke-scale preset — d in {16, 32},
epochs <= 3, instances <= 64 — sized to finish well under ~30 s total on
CPU while still showing real learning (lift >= 0.01). The same fields go
big (wider d_model, more instances/epochs) when a T4 shows up.
"""

from dataclasses import dataclass

import torch

from src.curriculum_data import stage1_batch
from src.tokenizer import L_MAX
from src.training import PhantomPolicy
from src.train_loop import PlacementHead, Trainer, expand_position_accuracy

__all__ = [
    "StudyConfig",
    "QUICK_CONFIGS",
    "run_config",
    "scaling_study",
    "render_table",
]

_VOCAB_SIZE = 64  # stage-1 curriculum ids all live below this scaffold budget


@dataclass
class StudyConfig:
    """One scaling-study cell: model width x data x schedule (+ seeding).

    Attributes:
        d_model: Policy/placement-head embedding width.
        epochs: Passes over the training window.
        instances: Training-window size (seeds ``[train_seed0, +instances)``).
        batch_size: Instances accumulated per Adam step.
        lr: Adam learning rate for the whole run.
        seed: Seeds BOTH torch init and the Trainer's derived stochasticity;
            equal (config, seed) pairs reproduce identical lifts.
        heldout: Number of held-out evaluation instances.
        train_seed0: First seed of the (disjoint) training window.
        held_seed0: First seed of the (disjoint) held-out window.
    """

    d_model: int
    epochs: int
    instances: int
    batch_size: int = 8
    lr: float = 5e-3
    seed: int = 0
    heldout: int = 20
    train_seed0: int = 2000
    held_seed0: int = 999999


# Smoke-scale preset: full sweep < ~30 s on CPU, yet each cell learns
# (lift >= 0.01). Widen/extend these fields for T4-scale studies — the
# pipeline is identical.
QUICK_CONFIGS: list[StudyConfig] = [
    StudyConfig(d_model=16, epochs=3, instances=64),
    StudyConfig(d_model=32, epochs=3, instances=64),
]


def _mean_placement_accuracy(policy, head, dataset) -> float:
    """Mean P(placement lands exactly on ``gap_start``) over a dataset."""
    scores = [expand_position_accuracy(policy, inst, head) for inst in dataset]
    return sum(scores) / len(scores)


def run_config(cfg: StudyConfig) -> dict:
    """Execute ONE study cell end-to-end and report fresh/post/lift.

    Builds a fresh policy + attention placement head, measures held-out
    placement accuracy before training, trains via ``Trainer`` on the
    disjoint stage-1 training window, re-measures on the same held-out
    instances, and returns ``{"d_model", "epochs", "instances", "fresh",
    "post", "lift"}`` with ``lift = post - fresh``. Bit-reproducible for a
    given (config, seed).
    """
    torch.manual_seed(cfg.seed)  # own ALL init randomness: reproducible cells

    policy = PhantomPolicy(
        vocab_size=_VOCAB_SIZE, d_model=cfg.d_model, n_actions=3, l_max=L_MAX
    )
    head = PlacementHead(d_model=cfg.d_model)

    train_set = stage1_batch(cfg.instances, cfg.train_seed0)
    heldout_set = stage1_batch(cfg.heldout, cfg.held_seed0)

    fresh = _mean_placement_accuracy(policy, head, heldout_set)

    trainer = Trainer(policy, lr=cfg.lr, seed=cfg.seed, placement_head=head)
    trainer.epoch(train_set, batch_size=cfg.batch_size, epochs=cfg.epochs)

    post = _mean_placement_accuracy(policy, head, heldout_set)

    return {
        "d_model": cfg.d_model,
        "epochs": cfg.epochs,
        "instances": cfg.instances,
        "fresh": fresh,
        "post": post,
        "lift": post - fresh,
    }


def scaling_study(configs: list[StudyConfig]) -> list[dict]:
    """Run every config in order; one result row per config."""
    return [run_config(cfg) for cfg in configs]


def render_table(rows: list[dict]) -> str:
    """Aligned ASCII table over the study rows (header + rule + data).

    Every printed line is padded to the same width so columns line up in
    any monospace view; floats show four decimals. Diagnostic keys such as
    ``_seconds`` are ignored — only the six contract metrics print.
    """
    headers = ("d_model", "epochs", "instances", "fresh", "post", "lift")

    def fmt(key: str, row) -> str:
        value = row[key] if isinstance(row, dict) else getattr(row, key)
        return f"{value:.4f}" if key in ("fresh", "post", "lift") else str(value)

    def cells_for(row):
        return [fmt(key, row) for key in headers]

    body = [cells_for(row) for row in rows]
    widths = [
        max(len(header), *(len(cells[i]) for cells in body)) if body else len(header)
        for i, header in enumerate(headers)
    ]
    pad = "  "  # column gutter
    total = sum(widths) + len(pad) * (len(headers) - 1)

    lines = [
        pad.join(header.rjust(width) for header, width in zip(headers, widths)),
        "-" * total,
    ]
    lines += [
        pad.join(cell.rjust(width) for cell, width in zip(cells, widths))
        for cells in body
    ]
    return "\n".join(lines)
