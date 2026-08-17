"""Tests for the Tokenizer module."""

from src.tokenizer import phantom_pad, L_MAX


def test_phantom_pad_short_sequence():
    tokens = [1, 2, 3]
    padded = phantom_pad(tokens)
    assert len(padded) == L_MAX
    assert padded[:3] == [1, 2, 3]
    assert all(t == 0 for t in padded[3:])


def test_phantom_pad_exact_length():
    tokens = list(range(L_MAX))
    padded = phantom_pad(tokens)
    assert len(padded) == L_MAX


def test_phantom_pad_truncates_long():
    tokens = list(range(L_MAX + 100))
    padded = phantom_pad(tokens)
    assert len(padded) == L_MAX
