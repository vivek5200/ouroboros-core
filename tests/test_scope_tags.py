"""Tests for automatic GLOBAL/LOCAL scope tagging from real Python source.

Module 4 (Paper Table 1): tokens are partitioned GLOBAL vs LOCAL by
tree-sitter-style boundaries — module-level statements are GLOBAL; function/
class bodies are LOCAL. ``auto_scope_spans`` derives that partition straight
from source (stdlib ``ast`` + ``tokenize``) instead of hand-written spans:

  * module-level statements whose entire span is module-level  -> GLOBAL
  * FunctionDef/AsyncFunctionDef/ClassDef bodies               -> LOCAL
    (decorators + def-line/signature up to and incl. ':' stay GLOBAL;
     nested defs: innermost enclosing function wins -> LOCAL at any depth)
"""

import pytest

from src.scoping import (
    GLOBAL,
    LOCAL,
    Scope,
    auto_scope_spans,
    block_sparse_mask,
    block_sparse_mask_for_source,
    build_scope_tags,
)
from src.tokenizer import lexical_tokens

G = Scope.GLOBAL  # "global"
L = Scope.LOCAL   # "local"


def _tags(source):
    """Per-token scope tags (length == number of lexical tokens)."""
    return build_scope_tags(auto_scope_spans(source))


# ---------------------------------------------------------------------------
# 1. Single module-level assignment: everything GLOBAL.
# ---------------------------------------------------------------------------

def test_module_assign_all_global():
    src = "x = 1\n"
    assert lexical_tokens(src) == ["x", "=", "1"]
    assert auto_scope_spans(src) == [(0, 3, G)]
    assert _tags(src) == [G, G, G]


# ---------------------------------------------------------------------------
# 2. One function: def-line tokens GLOBAL, body tokens LOCAL.
# ---------------------------------------------------------------------------

def test_def_line_global_body_local():
    src = "def f():\n    return 1\n"
    # tokens: def f ( ) : | return 1
    assert lexical_tokens(src) == ["def", "f", "(", ")", ":", "return", "1"]
    spans = auto_scope_spans(src)
    assert spans == [(0, 5, G), (5, 7, L)]
    # The colon itself ends the GLOBAL signature; body sees the signature.
    mask = block_sparse_mask(spans)
    assert mask[5][0] is True   # body row sees def-line column
    assert mask[0][5] is False  # def-line row insulated from body column


# ---------------------------------------------------------------------------
# 3. Two sibling functions: separate LOCAL scopes, no bleed either direction.
# ---------------------------------------------------------------------------

SIBLINGS = (
    "def f():\n"
    "    return 1\n"
    "\n"
    "def g():\n"
    "    return 2\n"
)


def test_sibling_bodies_are_distinct_local_scopes():
    # tokens: def f ( ) : return 1 | def g ( ) : return 2
    assert lexical_tokens(SIBLINGS) == [
        "def", "f", "(", ")", ":", "return", "1",
        "def", "g", "(", ")", ":", "return", "2",
    ]
    spans = auto_scope_spans(SIBLINGS)
    assert spans == [(0, 5, G), (5, 7, L), (7, 12, G), (12, 14, L)]
    mask = block_sparse_mask(spans)
    # Sibling bodies ignore each other in BOTH directions (Table 1 row 5).
    assert mask[5][12] is False
    assert mask[12][5] is False
    # Each still sees every GLOBAL token.
    for i in (5, 6, 12, 13):
        for j in (0, 1, 2, 3, 4, 7, 8, 9, 10, 11):
            assert mask[i][j] is True


# ---------------------------------------------------------------------------
# 4. Nested def inside def: innermost enclosing function wins -> all inner
#    body tokens LOCAL (including the inner def line).
# ---------------------------------------------------------------------------

NESTED = (
    "def outer():\n"
    "    x = 1\n"
    "\n"
    "    def inner():\n"
    "        return x\n"
    "\n"
    "    return inner\n"
)


def test_nested_def_body_all_local():
    tokens = lexical_tokens(NESTED)
    assert tokens == [
        "def", "outer", "(", ")", ":",
        "x", "=", "1",
        "def", "inner", "(", ")", ":",
        "return", "x",
        "return", "inner",
    ]  # 17 tokens
    spans = auto_scope_spans(NESTED)
    assert spans == [(0, 5, G), (5, 17, L)]


# ---------------------------------------------------------------------------
# 5. Class at module level: class-line GLOBAL, methods/body LOCAL.
# ---------------------------------------------------------------------------

KLASS = (
    "class C:\n"
    "    def m(self):\n"
    "        return 1\n"
    "\n"
    "    x = 2\n"
)


def test_class_header_global_methods_local():
    # tokens: class C : | def m ( self ) : return 1 x = 2
    assert lexical_tokens(KLASS) == [
        "class", "C", ":",
        "def", "m", "(", "self", ")", ":", "return", "1",
        "x", "=", "2",
    ]  # 14 tokens
    spans = auto_scope_spans(KLASS)
    assert spans == [(0, 3, G), (3, 14, L)]
    tags = _tags(KLASS)
    assert tags[:3] == [G, G, G]          # class header line
    assert tags[3:] == [L] * 11           # method line + bodies + attrs


# ---------------------------------------------------------------------------
# 6. Full-pipeline smoke on a 10-token snippet vs hand-derived expectations.
# ---------------------------------------------------------------------------

SNIPPET_10 = "def f():\n    return x\ny = 1\n"


def test_full_pipeline_smoke_10_tokens():
    toks = lexical_tokens(SNIPPET_10)
    assert len(toks) == 10
    assert toks == ["def", "f", "(", ")", ":", "return", "x", "y", "=", "1"]
    # Hand-derived scopes:  G G G G G L L G G G
    expected_tags = [G, G, G, G, G, L, L, G, G, G]
    mask = block_sparse_mask_for_source(SNIPPET_10)
    assert len(mask) == 10 and all(len(row) == 10 for row in mask)
    assert all(isinstance(v, bool) for row in mask for v in row)
    for i in range(10):
        for j in range(10):
            if i == j:
                assert mask[i][j] is True
            elif expected_tags[i] == L:
                assert mask[i][j] is (expected_tags[j] == G or True)
            else:
                assert mask[i][j] is (expected_tags[j] == G)
    # Spot-check the interesting cells explicitly:
    assert mask[0][5] is False    # global row never sees local col
    assert mask[5][0] is True     # local row sees global cols
    assert mask[6][8] is True     # local row, trailing global col


# ---------------------------------------------------------------------------
# Attribution-rule edges pinned by the documented rule.
# ---------------------------------------------------------------------------

def test_decorated_function_decorator_and_signature_global():
    src = "@dec\ndef f():\n    return 1\n"
    # tokens: @ dec def f ( ) : | return 1
    assert lexical_tokens(src) == ["@", "dec", "def", "f", "(", ")", ":", "return", "1"]
    assert auto_scope_spans(src) == [(0, 7, G), (7, 9, L)]


def test_multiline_signature_stays_global():
    src = "def f(\n    a,\n    b,\n):  # sig\n    return a\n"
    # tokens: def f ( a , b , ) : #sig | return a   (comment rides GLOBAL:
    # it belongs to no body statement, mirroring tokenizer attribution)
    assert lexical_tokens(src) == [
        "def", "f", "(", "a", ",", "b", ",", ")", ":", "# sig", "return", "a",
    ]
    assert auto_scope_spans(src) == [(0, 10, G), (10, 12, L)]


def test_oneliner_def_body_after_colon_is_local():
    src = "def f(): x = 1\ny = 2\n"
    assert lexical_tokens(src) == ["def", "f", "(", ")", ":", "x", "=", "1", "y", "=", "2"]
    assert auto_scope_spans(src) == [(0, 5, G), (5, 8, L), (8, 11, G)]


def test_async_def_and_trailing_module_stmts():
    src = "async def f():\n    return 1\nz = 0\n"
    assert lexical_tokens(src) == [
        "async", "def", "f", "(", ")", ":", "return", "1", "z", "=", "0",
    ]
    assert auto_scope_spans(src) == [(0, 6, G), (6, 8, L), (8, 11, G)]


def test_compound_nondef_statements_stay_global():
    src = "if x:\n    y = 1\n"
    assert auto_scope_spans(src) == [(0, len(lexical_tokens(src)), G)]


# ---------------------------------------------------------------------------
# Contract: exact partition over the lexical stream; errors; docstring.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "src",
    [
        "x = 1\n",
        SIBLINGS,
        NESTED,
        KLASS,
        SNIPPET_10,
        "import os\nfrom math import sqrt\nA: int = 3\n",
        "def f():\n    d = {'k': 1}\n    return d['k']\n",
    ],
)
def test_spans_exactly_partition_token_stream(src):
    spans = auto_scope_spans(src)
    n = len(lexical_tokens(src))
    covered = 0
    for start, end, scope in sorted(spans):
        assert start == covered          # contiguous, starts at 0, no gaps
        assert end > start
        assert scope in (G, L)
        covered = end
    assert covered == n                  # covers ALL tokens exactly once
    # Downstream consumers accept the partition unchanged.
    assert len(build_scope_tags(spans)) == n
    assert len(block_sparse_mask(spans)) == n


def test_empty_source_gives_no_spans():
    assert auto_scope_spans("") == []
    assert auto_scope_spans("   \n\n") == []
    assert block_sparse_mask_for_source("") == []


def test_unparseable_source_raises_value_error():
    with pytest.raises(ValueError):
        auto_scope_spans("def f(:\n")


def test_docstring_states_table1_rationale_and_attribution_rule():
    doc = auto_scope_spans.__doc__ or ""
    assert "Table 1" in doc
    assert "smallest" in doc      # exact attribution rule is stated


def test_convenience_matches_manual_pipeline():
    for src in (SIBLINGS, NESTED, KLASS):
        assert block_sparse_mask_for_source(src) == block_sparse_mask(
            auto_scope_spans(src)
        )
