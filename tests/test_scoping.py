"""Tests for Module 4: Block-Sparse Lexical Scoping (pure python, no torch).

Paper Table 1 — hierarchical attention mask rules over GLOBAL/LOCAL tokens:
  M[i,j] = 1 if i is LOCAL and j is GLOBAL          (bodies see signatures)
  M[i,j] = 1 if i == j                              (self)
  M[i,j] = 1 if i,j both LOCAL in the SAME scope    (coherent generation)
  M[i,j] = 0 if i is GLOBAL and j is LOCAL          (global KV insulated)
  M[i,j] = 0 if i,j LOCAL siblings in DIFFERENT     (no sibling bleed)
"""

import copy

import pytest

from src.scoping import (
    Scope,
    block_sparse_mask,
    build_scope_tags,
    frozen_global_columns,
    global_rows_frozen,
    same_local_scope,
)

G = Scope.GLOBAL  # "global"
L = Scope.LOCAL   # "local"

# Tiny canonical fixture: 1 GLOBAL span + 2 LOCAL sibling spans (8 tokens).
#   tokens:  0 1 | 2 3 4 | 5 6 7
#   scopes:  G G | L L L | L L L      (A = [2,5), B = [5,8))
SPANS_8 = [(0, 2, G), (2, 5, L), (5, 8, L)]

# Hand-written literal expected mask for SPANS_8 (Table 1 applied by hand).
EXPECTED_8 = [
    # j:    0      1      2      3      4      5      6      7
    [True,  True,  False, False, False, False, False, False],  # i=0 (G)
    [True,  True,  False, False, False, False, False, False],  # i=1 (G)
    [True,  True,  True,  True,  True,  False, False, False],  # i=2 (A)
    [True,  True,  True,  True,  True,  False, False, False],  # i=3 (A)
    [True,  True,  True,  True,  True,  False, False, False],  # i=4 (A)
    [True,  True,  False, False, False, True,  True,  True ],  # i=5 (B)
    [True,  True,  False, False, False, True,  True,  True ],  # i=6 (B)
    [True,  True,  False, False, False, True,  True,  True ],  # i=7 (B)
]


# ---------------------------------------------------------------------------
# Scope constants
# ---------------------------------------------------------------------------

def test_scope_constants():
    assert Scope.GLOBAL == "global"
    assert Scope.LOCAL == "local"


# ---------------------------------------------------------------------------
# build_scope_tags
# ---------------------------------------------------------------------------

def test_build_scope_tags_basic():
    assert build_scope_tags(SPANS_8) == [
        G, G, L, L, L, L, L, L,
    ]


def test_build_scope_tags_accepts_unordered_exact_partition():
    # Order of spans should not matter as long as [0, L) is partitioned exactly.
    assert build_scope_tags([(2, 5, L), (5, 8, L), (0, 2, G)]) == [
        G, G, L, L, L, L, L, L,
    ]


@pytest.mark.parametrize(
    "bad",
    [
        [(0, 2, G), (3, 5, L)],           # gap between regions
        [(0, 3, G), (2, 5, L)],           # overlap between regions
        [(1, 4, L)],                      # does not start at 0
        [(0, 0, G)],                      # zero-length region
        [(-1, 2, G)],                     # negative start
        [(0, 4, "friend")],               # unknown scope name
    ],
)
def test_build_scope_tags_partition_violation_raises_valueerror(bad):
    with pytest.raises(ValueError):
        build_scope_tags(bad)


def test_build_scope_tags_empty():
    assert build_scope_tags([]) == []


# ---------------------------------------------------------------------------
# same_local_scope
# ---------------------------------------------------------------------------

def test_same_local_scope_pairs():
    # Within local span A.
    assert same_local_scope(2, 4, SPANS_8) is True
    # Degenerate pair i == j inside a local span counts as same scope.
    assert same_local_scope(3, 3, SPANS_8) is True
    # Across sibling local spans A/B: False.
    assert same_local_scope(2, 6, SPANS_8) is False
    assert same_local_scope(4, 5, SPANS_8) is False
    # Globals never live in a local scope.
    assert same_local_scope(0, 1, SPANS_8) is False
    assert same_local_scope(0, 0, SPANS_8) is False
    # Mixed global/local: False.
    assert same_local_scope(0, 2, SPANS_8) is False


def test_same_local_scope_out_of_range_is_false():
    assert same_local_scope(-1, 2, SPANS_8) is False
    assert same_local_scope(2, 8, SPANS_8) is False
    assert same_local_scope(0, 0, []) is False


# ---------------------------------------------------------------------------
# block_sparse_mask — exact 8x8 literal
# ---------------------------------------------------------------------------

def test_block_sparse_mask_8token_exact_literal():
    assert block_sparse_mask(SPANS_8) == EXPECTED_8


def test_block_sparse_mask_entries_are_real_bools():
    mask = block_sparse_mask(SPANS_8)
    assert len(mask) == 8 and all(len(row) == 8 for row in mask)
    assert all(isinstance(v, bool) for row in mask for v in row)


def test_block_sparse_mask_l0_edge():
    assert block_sparse_mask([]) == []


# ---------------------------------------------------------------------------
# Table 1 structural properties
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "spans",
    [
        SPANS_8,
        [(0, 1, G), (1, 2, L), (2, 3, L), (3, 4, L)],       # singleton scopes
        [(0, 3, G), (3, 5, L), (5, 7, L), (7, 9, L)],       # three siblings
        [(0, 2, L), (2, 4, G), (4, 6, L)],                  # local before global
        [(0, 4, L)],                                        # single local span
    ],
)
def test_self_attention_always_true(spans):
    mask = block_sparse_mask(spans)
    n = len(mask)
    assert all(mask[i][i] is True for i in range(n))


def test_asymmetry_global_row_local_col_zero_vs_local_row_global_col_one():
    """Dedicated asymmetry proof: LOCAL row x GLOBAL col = 1 while the mirrored
    GLOBAL row x LOCAL col = 0. The mask is deliberately NOT globally symmetric
    (this is what insulates the global KV-cache from local mutations)."""
    mask = block_sparse_mask(SPANS_8)
    # i=2 is LOCAL, j=0 is GLOBAL -> body sees signature.
    assert mask[2][0] is True
    # Mirrored: i=0 is GLOBAL, j=2 is LOCAL -> insulated.
    assert mask[0][2] is False
    assert mask[2][0] != mask[0][2]
    # Same contrast on the other side of the layout.
    assert mask[7][1] is True and mask[1][7] is False


def test_symmetry_only_within_same_local_scope_and_global_pairs():
    """M[i,j] == M[j,i] iff (same local scope) or (both GLOBAL)."""
    mask = block_sparse_mask(SPANS_8)
    for i in range(8):
        for j in range(8):
            should_be_symmetric = (
                same_local_scope(i, j, SPANS_8)
                or (build_scope_tags(SPANS_8)[i] == G and build_scope_tags(SPANS_8)[j] == G)
            )
            if should_be_symmetric:
                assert mask[i][j] == mask[j][i], (i, j)
            else:
                assert mask[i][j] != mask[j][i] or (
                    not mask[i][j] and not mask[j][i]
                ), (i, j)


@pytest.mark.parametrize(
    "spans",
    [
        SPANS_8,
        [(0, 1, G), (1, 2, L), (2, 3, L), (3, 4, L)],
        [(0, 3, G), (3, 5, L), (5, 7, L), (7, 9, L)],
        [(0, 2, L), (2, 4, G), (4, 6, L)],
    ],
)
def test_table1_rules_hold_for_every_cell(spans):
    """Exhaustive per-cell check of the five Table 1 rules (+ global x global)."""
    tags = build_scope_tags(spans)
    owners = []
    for s, e, sc in sorted(spans):
        owners.extend([(s, e)] * (e - s))
    n = len(tags)
    mask = block_sparse_mask(spans)
    assert len(mask) == n and all(len(r) == n for r in mask)
    for i in range(n):
        for j in range(n):
            if i == j:
                expected = True                                   # self
            elif tags[i] == L and tags[j] == G:
                expected = True                                   # body sees sig
            elif tags[i] == L and tags[j] == L:
                expected = owners[i] == owners[j]                 # same scope only
            else:  # i is GLOBAL
                expected = tags[j] == G                           # insulated from L
            assert mask[i][j] == expected, (i, j)


# ---------------------------------------------------------------------------
# frozen_global_columns — cache-coherency under [EXPAND]
# ---------------------------------------------------------------------------

def test_frozen_global_columns_after_expand():
    # Layout: 2 global, then two local siblings; expand middle local span
    # [2,6) -> [2,8) (splice 2 tokens). Everything after shifts right by 2.
    spans = [(0, 2, G), (2, 6, L), (6, 9, L)]
    mask_before = block_sparse_mask(spans)
    assert len(mask_before) == 9

    after = frozen_global_columns(mask_before, spans, (2, 8))

    # Shape grew by delta=2.
    assert len(after) == 11
    assert all(len(row) == 11 for row in after)

    # GLOBAL rows are bit-for-bit unchanged over the old extent...
    for gi in (0, 1):
        assert after[gi][:9] == mask_before[gi]
        # ...and see NOTHING toward newly appended LOCAL columns.
        assert after[gi][9] is False
        assert after[gi][10] is False

    # GLOBAL rows never gain True entries toward ANY LOCAL token.
    tags_after = build_scope_tags([(0, 2, G), (2, 8, L), (8, 11, L)])
    for gi in (0, 1):
        for j in range(11):
            if tags_after[j] == L:
                assert after[gi][j] is False

    # GLOBAL <-> GLOBAL visibility survived the freeze-copy.
    assert after[0][1] is True and after[1][0] is True

    # LOCAL structure rebuilt correctly around the mutation:
    assert after[8][8] is True        # self in new span B'
    assert after[8][0] is True        # local row sees global col
    assert after[8][5] is False       # sibling bleed blocked (B' vs A')
    assert after[10][2] is False      # sibling bleed blocked
    assert after[4][5] is True        # same scope A' coherent


def test_frozen_global_columns_identity_when_no_growth():
    spans = [(0, 2, G), (2, 5, L)]
    before = block_sparse_mask(spans)
    after = frozen_global_columns(before, spans, (2, 5))
    assert after == before


def test_frozen_global_columns_rejects_bad_mutation_target():
    spans = [(0, 2, G), (2, 5, L)]
    before = block_sparse_mask(spans)
    with pytest.raises(ValueError):
        frozen_global_columns(before, spans, (0, 4))   # target is the GLOBAL span
    with pytest.raises(ValueError):
        frozen_global_columns(before, spans, (3, 9))   # no span starts at 3


def test_global_rows_frozen_predicate_true_and_false():
    spans = [(0, 2, G), (2, 6, L), (6, 9, L)]
    before = block_sparse_mask(spans)
    after = frozen_global_columns(before, spans, (2, 8))
    assert global_rows_frozen(before, after, [(0, 2, G), (2, 8, L), (8, 11, L)]) is True

    # Corrupt one global-row entry toward a LOCAL token -> predicate flips False.
    bad = copy.deepcopy(after)
    bad[0][4] = True
    assert global_rows_frozen(before, bad, [(0, 2, G), (2, 8, L), (8, 11, L)]) is False

    # Truncating/duplicating rows also reads as not-frozen, never raises.
    assert global_rows_frozen(before, after[1:], [(0, 2, G)]) is False
