"""Stage-1 curriculum data: WHERE-was-the-statement-deleted instances (§3.3).

Paper stage 1 ("syntactic_boundaries") in one sentence: synthesize a toy
statement list, delete exactly one NON-FIRST statement's token span, and ask
the policy to point at the deletion site. Every instance is fully described
by the corrupted stream plus ``gap_start``/``gap_len``.

Template synthesis
------------------
Snippets are 2-5 single-line assignment statements over an rng-chosen name
subset of :data:`NAME_POOL`, e.g.::

    x0 = 7
    y1 = x0 + 3      # chains: rhs may reference earlier variables
    z = y1 * 2

Right-hand sides are integer literals, earlier-variable references,
``var op int`` or ``var op var`` expressions with ``op ∈ {+, -, *}``. The
first statement must be a literal (nothing exists to chain to); later
statements may chain on any earlier variable, which is what makes the gap
content partially predictable from surrounding context.

Statement → token-id span attribution (documented choice)
---------------------------------------------------------
Because every generated statement is a *single-line* simple assignment, the
top-level AST nodes partition the source line-by-line. ``make_instance``
parses the full snippet with :mod:`ast`, asserts that ``tree.body`` has
exactly one statement per line and that each is a single-line ``ast.Assign``
(``lineno == end_lineno``), and then derives statement k's id-span as the
lexical tokens of line k encoded through the instance's shared vocab. A
consistency check ties the two views together: concatenating all per-line
spans must reproduce ``full_ids = Vocab().encode(lexical_tokens(source))``
exactly, so no token is ever attributed twice or dropped.

Vocabulary sharing (documented)
-------------------------------
ONE :class:`src.tokenizer.Vocab` is built over the **full** token stream per
instance and used to encode both streams. Corrupted ids therefore live in
exactly the same id space as ``full_ids`` (no re-mapping when tokens shift),
and the returned dict needs no vocab field: any caller can rebuild it via
``Vocab().encode(lexical_tokens(snippet_source(seed)))``.

Determinism
-----------
``random.Random(seed)`` drives every rng choice; torch is never touched.
``stage1_batch(n, seed0)`` is simply ``[make_instance(s) for s in
range(seed0, seed0 + n)]``, so batches are reproducible and seed windows are
disjoint by construction.
"""

import ast
import random

from src.tokenizer import L_MAX, Vocab, lexical_tokens

__all__ = [
    "NAME_POOL",
    "make_instance",
    "snippet_source",
    "stage1_batch",
]


# Short identifiers the rng samples statement names from (distinct per snippet).
NAME_POOL: tuple[str, ...] = ("x0", "y1", "z2", "w3", "u4", "v5", "p6", "q7")

_OPS: tuple[str, ...] = ("+", "-", "*")
_MIN_STATEMENTS = 2
_MAX_STATEMENTS = 5


def _synthesise_lines(rng: random.Random) -> list[str]:
    """Generate 2-5 chained assignment lines from the seeded rng."""
    n_stmts = rng.randint(_MIN_STATEMENTS, _MAX_STATEMENTS)
    names = rng.sample(NAME_POOL, n_stmts)
    lines: list[str] = []
    for i in range(n_stmts):
        if i == 0:
            rhs = str(rng.randint(0, 12))
        else:
            roll = rng.random()
            if roll < 0.35:
                rhs = str(rng.randint(0, 12))
            elif roll < 0.60:
                rhs = names[rng.randrange(i)]
            elif roll < 0.85 or i < 2:
                # ``var op int``; also the fallback while only one earlier
                # variable exists (two-var form needs at least two).
                rhs = f"{names[rng.randrange(i)]} {rng.choice(_OPS)} {rng.randint(0, 9)}"
            else:
                j = rng.randrange(i)
                other = rng.choice(
                    [n for idx, n in enumerate(names[:i]) if idx != j]
                )
                rhs = f"{names[j]} {rng.choice(_OPS)} {other}"
        lines.append(f"{names[i]} = {rhs}")
    return lines


def snippet_source(seed: int) -> str:
    """The deterministic toy snippet for ``seed`` (newline-joined lines)."""
    return "\n".join(_synthesise_lines(random.Random(seed)))


def _statement_id_spans(source: str, vocab: Vocab) -> tuple[list[list[int]], int]:
    """Per-statement id spans via ast + per-line lexical tokenization.

    Returns ``(spans, n_lines)`` where ``len(spans) == n_lines`` and
    ``sum(map(len, spans))`` equals the full-stream token count (asserted).
    """
    tree = ast.parse(source)
    lines = source.splitlines()
    stmts = list(tree.body)
    if len(stmts) != len(lines):
        raise RuntimeError("template invariant broken: multi-statement line")
    for stmt in stmts:
        if not isinstance(stmt, ast.Assign) or stmt.lineno != stmt.end_lineno:
            raise RuntimeError(
                "template invariant broken: non-assignment or multi-line statement"
            )
    spans = [vocab.encode(lexical_tokens(line)) for line in lines]
    return spans, len(lines)


def make_instance(seed: int) -> dict:
    """Build one stage-1 instance: a deleted non-first statement to localize.

    Args:
        seed: Fully determines the snippet, the vocabulary ids, and which
            statement is deleted (uniform among statements 1..n-1).

    Returns:
        ``{"ids": corrupted_id_list, "gap_start": idx, "gap_len": n_removed,
        "l_max": tokenizer.L_MAX, "full_ids": original_list}`` where
        ``ids == full_ids[:gap_start] + full_ids[gap_start+gap_len:]``.
    """
    source = snippet_source(seed)

    # One shared Vocab over the FULL stream (see module docstring): both the
    # intact and corrupted streams are encoded through this same mapping.
    vocab = Vocab()
    full_ids = vocab.encode(lexical_tokens(source))

    spans, n_stmts = _statement_id_spans(source, vocab)
    flat: list[int] = [i for span in spans for i in span]
    assert flat == full_ids, "span attribution lost or duplicated tokens"

    # Uniform choice among NON-FIRST statements (index 1 .. n_stmts-1):
    # deleting statement 0 would leave no left context for the task.
    victim = random.Random(seed ^ 0x5EED).randrange(1, n_stmts)
    gap_start = sum(len(spans[k]) for k in range(victim))
    gap_len = len(spans[victim])
    ids = full_ids[:gap_start] + full_ids[gap_start + gap_len :]

    return {
        "ids": ids,
        "gap_start": gap_start,
        "gap_len": gap_len,
        "l_max": L_MAX,
        "full_ids": full_ids,
    }


def stage1_batch(n: int, seed0: int) -> list[dict]:
    """``n`` instances from consecutive seeds ``seed0 .. seed0+n-1``."""
    return [make_instance(seed0 + offset) for offset in range(n)]
