"""Tests for the Fuzzy Proxy Reward module."""

from src.reward import compute_reward


def test_all_pass():
    assert compute_reward(True, True, True) == 1.0


def test_all_fail():
    assert compute_reward(False, False, False) == 0.0


def test_only_parses():
    assert abs(compute_reward(True, False, False) - 0.1) < 1e-9


def test_parses_and_typechecks():
    assert abs(compute_reward(True, True, False) - 0.4) < 1e-9
