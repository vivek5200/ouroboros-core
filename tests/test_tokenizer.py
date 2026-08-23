"""Tests for the Tokenizer module."""

import pytest

from src.tokenizer import (
    IGNORE_ID,
    L_MAX,
    MASK_ID,
    Vocab,
    ast_spd_matrix,
    derive_mask,
    front_pack,
    insert_masks,
    lexical_tokens,
    logical_delete,
    phantom_pad,
    tokenize,
    tokenize_with_nodes,
)

GOLDEN = "def f(x):\n    return x + 1\n"


# ---------------------------------------------------------------------------
# tokenize — deterministic stable vocabulary, ids from 1 (0 = pad)
# ---------------------------------------------------------------------------


def test_tokenize_deterministic_across_calls():
    ids1 = tokenize(GOLDEN)
    ids2 = tokenize(GOLDEN)
    assert ids1 == ids2


def test_tokenize_ids_consecutive_from_one_no_pad():
    ids = tokenize(GOLDEN)
    assert ids and ids[0] == 1                    # first distinct string -> id 1
    assert 0 not in ids                           # 0 is reserved for padding
    assert sorted(set(ids)) == list(range(1, len(set(ids)) + 1))


def test_tokenize_matches_tokenize_with_nodes_ids():
    _, ids, _ = tokenize_with_nodes(GOLDEN)
    assert tokenize(GOLDEN) == ids                # plain tokenize = ids only


def test_tokenize_empty_source():
    assert tokenize("") == []


# ---------------------------------------------------------------------------
# Vocab — reusable / persistable mapping
# ---------------------------------------------------------------------------


def test_vocab_assigns_consecutive_ids_from_one():
    vocab = Vocab()
    assert vocab.encode(["def", "f", "x"]) == [1, 2, 3]


def test_vocab_existing_strings_keep_their_ids():
    vocab = Vocab()
    assert vocab.encode(["def", "f", "x"]) == [1, 2, 3]
    assert vocab.id_of("def") == 1
    assert vocab.encode(["x", "def"]) == [3, 1]   # no reassignment on reuse
    assert vocab.id_of("newcomer") == 4           # next id continues the run


def test_vocab_decode_is_inverse_of_encode():
    vocab = Vocab()
    toks = ["a", "=", "b"]
    ids = vocab.encode(toks)
    assert vocab.decode(ids) == toks
    assert vocab.decode([0]) == ["<pad>"]         # 0 = pad decodes to sentinel


# ---------------------------------------------------------------------------
# tokenize_with_nodes — AST-aware node kinds aligned with ids
# ---------------------------------------------------------------------------


def test_node_kinds_align_with_tokens():
    toks = lexical_tokens(GOLDEN)
    _, ids, kinds = tokenize_with_nodes(GOLDEN)
    assert len(kinds) == len(ids) == len(toks)
    assert kinds[toks.index("def")] == "FunctionDef"
    assert kinds[toks.index("return")] == "Return"
    assert kinds[toks.index("1")] == "Constant"
    assert kinds[toks.index("+")] == "BinOp"


def test_same_string_maps_to_same_id_within_call():
    vocab, ids, _ = tokenize_with_nodes("x = x + x\n")
    assert vocab.id_of("x") == ids[0]
    assert ids[0] == ids[2] == ids[4]


def test_round_trip_via_inverse_vocab():
    toks = lexical_tokens(GOLDEN)
    vocab, ids, _ = tokenize_with_nodes(GOLDEN)
    assert vocab.decode(ids) == toks


def test_malformed_source_falls_back_to_lexical_module_kinds():
    vocab, ids, kinds = tokenize_with_nodes("def f(:\n")
    assert ids == [1, 2, 3, 4]
    assert vocab.decode(ids) == ["def", "f", "(", ":"]
    assert kinds == ["Module"] * 4


def test_empty_source_with_nodes():
    vocab, ids, kinds = tokenize_with_nodes("")
    assert isinstance(vocab, Vocab)
    assert ids == [] and kinds == []


# ---------------------------------------------------------------------------
# ast_spd_matrix — Module 3 additive-bias input phi(i, j)
# ---------------------------------------------------------------------------


def test_spd_matrix_shape_symmetry_and_zero_diagonal():
    src = "def f(x):\n    return x\n"
    toks = lexical_tokens(src)
    m = ast_spd_matrix(src)
    t = len(toks)
    assert len(m) == t and all(len(row) == t for row in m)
    assert all(m[i][i] == 0 for i in range(t))
    assert all(m[i][j] == m[j][i] for i in range(t) for j in range(t))


def test_spd_functiondef_adjacent_to_return():
    src = "def f(x):\n    return x\n"
    toks = lexical_tokens(src)
    m = ast_spd_matrix(src)
    assert m[toks.index("def")][toks.index("return")] == 1


def test_spd_distances_through_the_tree():
    src = "def f(x):\n    return x\n"
    toks = lexical_tokens(src)
    m = ast_spd_matrix(src)
    i_param_x = toks.index("x")          # parameter name -> FunctionDef node
    i_body_x = toks.index("x", i_param_x + 1)  # body name -> Name node
    i_ret = toks.index("return")
    assert m[i_ret][i_body_x] == 1       # Name is a direct child of Return
    assert m[toks.index("def")][i_body_x] == 2  # FunctionDef->Return->Name
    assert m[i_param_x][i_body_x] == 2


def test_spd_identical_node_tokens_are_zero():
    src = "if x:\n    pass\n"            # 'if' and ':' both sit on the If node
    toks = lexical_tokens(src)
    m = ast_spd_matrix(src)
    i_if, i_colon = toks.index("if"), toks.index(":")
    assert m[i_if][i_colon] == 0         # same attributed node -> 0
    i_test_x, i_pass = toks.index("x"), toks.index("pass")
    assert m[i_colon][i_pass] == 1       # Pass is a direct child of If
    assert m[i_test_x][i_pass] == 2      # Name(test)->If->Pass


def test_ast_spd_matrix_empty_source():
    assert ast_spd_matrix("") == []


def test_ast_spd_matrix_unparseable_raises_value_error():
    with pytest.raises(ValueError):
        ast_spd_matrix("def f(:\n")


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


# ---------------------------------------------------------------------------
# front_pack — Algorithm 1 lines 1-2: fixed buffer + logical length
# ---------------------------------------------------------------------------


def test_front_pack_short_sequence():
    buffer, logical_len = front_pack([1, 2, 3])
    assert len(buffer) == L_MAX          # physical shape is always fixed
    assert logical_len == 3
    assert buffer[:3] == [1, 2, 3]
    assert all(t == 0 for t in buffer[3:])


def test_front_pack_exact_length():
    tokens = list(range(L_MAX))
    buffer, logical_len = front_pack(tokens)
    assert len(buffer) == L_MAX
    assert logical_len == L_MAX
    assert buffer == tokens


def test_front_pack_truncates_long():
    tokens = list(range(L_MAX + 100))
    buffer, logical_len = front_pack(tokens)
    assert len(buffer) == L_MAX
    assert logical_len == L_MAX
    assert buffer[:L_MAX] == tokens[:L_MAX]


def test_front_pack_custom_pad_id():
    buffer, logical_len = front_pack([5, 6], pad_id=999)
    assert logical_len == 2
    assert buffer[0] == 5 and buffer[1] == 6
    assert all(t == 999 for t in buffer[2:])


def test_front_pack_does_not_mutate_input():
    tokens = [7, 8, 9]
    front_pack(tokens, pad_id=1)
    assert tokens == [7, 8, 9]


# ---------------------------------------------------------------------------
# Sentinel ids — reserved, never colliding with real vocab ids (which are >= 0)
# ---------------------------------------------------------------------------


def test_sentinel_ids():
    assert MASK_ID == -1
    assert IGNORE_ID == -2


# ---------------------------------------------------------------------------
# insert_masks — Algorithm 1 [EXPAND]: splice k MASKs into the logical region
# ---------------------------------------------------------------------------


def test_insert_masks_middle():
    buffer, logical_len = front_pack([1, 2, 3])
    new_len = insert_masks(buffer, logical_len, 1, 2)
    assert new_len == 5
    assert buffer[:5] == [1, MASK_ID, MASK_ID, 2, 3]
    assert len(buffer) == L_MAX                      # physical shape invariant


def test_insert_masks_at_start_and_at_end():
    buffer, logical_len = front_pack([1, 2, 3])
    new_len = insert_masks(buffer, logical_len, 0, 2)
    assert new_len == 5
    assert buffer[:5] == [MASK_ID, MASK_ID, 1, 2, 3]
    new_len = insert_masks(buffer, new_len, 5, 2)    # pos == logical_len appends
    assert new_len == 7
    assert buffer[:7] == [MASK_ID, MASK_ID, 1, 2, 3, MASK_ID, MASK_ID]
    assert len(buffer) == L_MAX


def test_insert_masks_mutates_in_place_never_resizes():
    buffer, logical_len = front_pack([10, 20])
    alias = buffer                                   # same object reference
    snapshot = list(buffer)
    new_len = insert_masks(buffer, logical_len, 1, 3)
    assert alias is buffer                           # identity preserved
    assert len(alias) == L_MAX                       # never resized
    assert alias != snapshot                         # mutated through original ref
    assert alias[1:4] == [MASK_ID] * 3


def test_insert_masks_exact_fill_boundary_succeeds():
    tokens = list(range(L_MAX - 2))
    buffer, logical_len = front_pack(tokens)
    new_len = insert_masks(buffer, logical_len, 0, 2)   # logical_len + k == L_MAX
    assert new_len == L_MAX
    assert len(buffer) == L_MAX
    assert buffer[2:] == tokens


def test_insert_masks_overflow_raises():
    buffer, logical_len = front_pack([1, 2, 3])
    with pytest.raises(RuntimeError):
        insert_masks(buffer, logical_len, 0, L_MAX - 2)  # 3 + (L_MAX - 2) > L_MAX


def test_insert_masks_full_buffer_any_insert_raises():
    buffer, logical_len = front_pack(list(range(L_MAX)))
    with pytest.raises(RuntimeError):
        insert_masks(buffer, logical_len, 0, 1)


def test_insert_masks_bad_pos_raises():
    buffer, logical_len = front_pack([1, 2, 3])
    with pytest.raises(RuntimeError):
        insert_masks(buffer, logical_len, -1, 1)         # negative pos
    with pytest.raises(RuntimeError):
        insert_masks(buffer, logical_len, 4, 1)          # beyond logical region


def test_insert_masks_negative_k_raises():
    buffer, logical_len = front_pack([1, 2, 3])
    with pytest.raises(RuntimeError):
        insert_masks(buffer, logical_len, 1, -1)


# ---------------------------------------------------------------------------
# logical_delete — Algorithm 1 [DELETE]: logically remove one token
# ---------------------------------------------------------------------------


def test_logical_delete_removes_token():
    buffer, logical_len = front_pack([1, 2, 3])
    new_len = logical_delete(buffer, logical_len, 1)
    assert new_len == 2
    assert buffer[:2] == [1, 3]
    assert len(buffer) == L_MAX                          # shape still fixed


def test_logical_delete_last_token_then_empty():
    buffer, logical_len = front_pack([7])
    new_len = logical_delete(buffer, logical_len, 0)
    assert new_len == 0
    assert len(buffer) == L_MAX
    with pytest.raises(IndexError):
        logical_delete(buffer, new_len, 0)               # empty logical region


def test_logical_delete_rejects_pos_beyond_logical_region():
    buffer, logical_len = front_pack([1, 2])
    with pytest.raises(IndexError):
        logical_delete(buffer, logical_len, 2)           # index 2 is ignore tail
    with pytest.raises(IndexError):
        logical_delete(buffer, logical_len, -1)


def test_logical_delete_preserves_len_l_max():
    buffer, logical_len = front_pack(list(range(L_MAX)))
    for _ in range(5):
        logical_len = logical_delete(buffer, logical_len, 0)
        assert len(buffer) == L_MAX                      # invariant after every op
    assert logical_len == L_MAX - 5
    assert buffer[logical_len:] == [IGNORE_ID] * 5       # tail refilled with IGNORE


# ---------------------------------------------------------------------------
# derive_mask — attention mask from the logical length alone
# ---------------------------------------------------------------------------


def test_derive_mask_basic():
    buffer, logical_len = front_pack([1, 2, 3])
    mask = derive_mask(buffer, logical_len)
    assert mask == [True] * 3 + [False] * (L_MAX - 3)
    assert len(mask) == L_MAX


def test_derive_mask_empty_and_full():
    buffer, _ = front_pack([])
    assert derive_mask(buffer, 0) == [False] * L_MAX
    full_buffer, full_len = front_pack(list(range(L_MAX)))
    assert derive_mask(full_buffer, full_len) == [True] * L_MAX


def test_derive_mask_ignores_buffer_contents():
    # The mask depends only on logical_len, not on what sits in the buffer.
    buffer, logical_len = front_pack([1, 2], pad_id=IGNORE_ID)
    n = insert_masks(buffer, logical_len, 0, 1)
    n = logical_delete(buffer, n, 2)
    assert derive_mask(buffer, n) == [True, True] + [False] * (L_MAX - 2)


# ---------------------------------------------------------------------------
# Round trip: [EXPAND] then [DELETE] keeps the physical buffer shape forever
# ---------------------------------------------------------------------------


def test_expand_delete_roundtrip_shape_invariant():
    buffer, logical_len = front_pack([1, 2, 3], pad_id=IGNORE_ID)
    n = insert_masks(buffer, logical_len, 2, 100)
    assert len(buffer) == L_MAX
    n = logical_delete(buffer, n, 0)
    assert len(buffer) == L_MAX
    assert n == 102
    mask = derive_mask(buffer, n)
    assert mask[:104] == [True] * 102 + [False] * 2
    assert buffer[n:] == [IGNORE_ID] * (L_MAX - n)
