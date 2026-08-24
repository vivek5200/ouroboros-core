"""Tests for the scaling-study harness (``src/scaling_study.py``).

Contract under test: ONE code path measures stage-1 gap-localization
learning at any scale — the same ``StudyConfig -> run_config`` pipeline
must run tiny-on-CPU (``QUICK_CONFIGS``) and big-on-T4 alike:

* ``run_config`` builds a fresh policy + attention ``PlacementHead``,
  records held-out placement accuracy BEFORE training, trains with
  ``Trainer``, records AFTER, and reports ``{"d_model", "epochs",
  "instances", "fresh", "post", "lift"}`` with ``lift = post - fresh``;
* the quick CPU preset must show REAL learning — absolute held-out lift
  >= 0.01, i.e. clearly beyond evaluation noise at toy scale;
* identical config + seed ⇒ bit-identical lift across two runs
  (the harness owns its seeding; no leakage from global RNG state);
* ``render_table`` emits an aligned ASCII table listing every studied
  ``d_model``.
"""

import math

import pytest
import torch

torch.set_num_threads(1)  # tiny tensors: single-threaded ops are faster here

from src.scaling_study import (
    QUICK_CONFIGS,
    StudyConfig,
    render_table,
    run_config,
    scaling_study,
)


# ---------------------------------------------------------------------------
# run_config: result contract
# ---------------------------------------------------------------------------


def test_run_config_returns_exact_result_contract():
    row = run_config(StudyConfig(d_model=16, epochs=1, instances=4))
    assert set(row) == {
        "d_model",
        "epochs",
        "instances",
        "fresh",
        "post",
        "lift",
    }
    assert row["d_model"] == 16
    assert row["epochs"] == 1
    assert row["instances"] == 4
    for key in ("fresh", "post", "lift"):
        assert type(row[key]) is float
        assert math.isfinite(row[key])
    assert 0.0 <= row["fresh"] <= 1.0
    assert 0.0 <= row["post"] <= 1.0
    assert row["lift"] == pytest.approx(row["post"] - row["fresh"], abs=1e-12)


def test_fresh_accuracy_is_at_chance_level():
    """Sanity on semantics: 'fresh' must be the UNTRAINED baseline, i.e. near
    the uniform share ~1/(logical_len + 1) (~0.05 for these snippets) —
    definitely not already-trained-level accuracy."""
    cfg = StudyConfig(d_model=16, epochs=1, instances=4)
    row = run_config(cfg)
    assert row["fresh"] < 0.20


# ---------------------------------------------------------------------------
# THE learning proof at toy scale: quick preset really learns
# ---------------------------------------------------------------------------


def test_quick_preset_first_config_learns_on_heldout():
    """run_config(QUICK_CONFIGS[0]) shows finite fresh/post and lift >= 0.01.

    Chance baseline for these snippets is ~1/(logical_len+1), so a fresh
    policy sits around ~0.05; an absolute lift of +0.01 can only come from
    gradient descent concentrating EXPAND mass on true gap sites.
    """
    row = run_config(QUICK_CONFIGS[0])
    assert math.isfinite(row["fresh"])
    assert math.isfinite(row["post"])
    assert row["lift"] >= 0.01, (
        f"no real learning at toy scale: fresh={row['fresh']:.4f} "
        f"post={row['post']:.4f} lift={row['lift']:.4f}"
    )


def test_quick_preset_stays_in_cpu_budget():
    """The whole QUICK_CONFIGS preset finishes fast on CPU (< 30 s)."""
    import time

    t0 = time.perf_counter()
    rows = scaling_study(QUICK_CONFIGS)
    elapsed = time.perf_counter() - t0
    assert len(rows) == len(QUICK_CONFIGS)
    assert elapsed < 30.0, f"quick preset too slow: {elapsed:.1f}s"


# ---------------------------------------------------------------------------
# scaling_study: one row per config, order preserved
# ---------------------------------------------------------------------------


def test_scaling_study_runs_each_config_in_order():
    configs = [
        StudyConfig(d_model=16, epochs=1, instances=8),
        StudyConfig(d_model=32, epochs=1, instances=8),
    ]
    rows = scaling_study(configs)
    assert [r["d_model"] for r in rows] == [16, 32]
    assert [(r["epochs"], r["instances"]) for r in rows] == [(1, 8), (1, 8)]


# ---------------------------------------------------------------------------
# Determinism: same config + seed => identical lift across two runs
# ---------------------------------------------------------------------------


def test_same_config_and_seed_reproduce_identical_lift():
    cfg = StudyConfig(d_model=16, epochs=2, instances=24, seed=7)
    row_a = run_config(cfg)
    row_b = run_config(cfg)
    assert row_a["lift"] == row_b["lift"]
    assert row_a["fresh"] == row_b["fresh"]
    assert row_a["post"] == row_b["post"]


# ---------------------------------------------------------------------------
# render_table: aligned ASCII report of every studied d_model
# ---------------------------------------------------------------------------


def test_render_table_lists_every_d_model_with_aligned_columns():
    rows = [
        {"d_model": 16, "epochs": 3, "instances": 48,
         "fresh": 0.05, "post": 0.21, "lift": 0.16},
        {"d_model": 32, "epochs": 2, "instances": 64,
         "fresh": 0.0482, "post": 0.2513, "lift": 0.2031},
    ]
    table = render_table(rows)
    lines = table.splitlines()

    # Header names the metric columns and every d_model value appears.
    assert "d_model" in table
    assert {str(r["d_model"]) for r in rows} <= set(table.split())
    # The FIRST field of each data line is exactly that row's d_model, so the
    # value is a column entry, not a substring of some float.
    data_fields = [line.split() for line in lines[2:]]
    assert [fields[0] for fields in data_fields] == ["16", "32"]

    # Alignment: every printed line has identical length and every row splits
    # into the same number of whitespace-separated columns (6 metrics).
    assert len({len(line) for line in lines}) == 1, "columns are not aligned"
    assert all(len(fields) == 6 for fields in data_fields)


def test_render_table_handles_empty_rows():
    table = render_table([])
    assert "d_model" in table
