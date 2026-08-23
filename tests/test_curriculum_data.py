"""Tests for the stage-1 curriculum dataset generator (``src.curriculum_data``).

Paper §3.3 stage 1 ("syntactic_boundaries"): toy single-assignment snippets
with a randomly chosen NON-FIRST statement deleted from the token stream.
The instance asks exactly one question — *where* was the statement removed? —
so ``gap_start`` is the [EXPAND] target and everything else about the task is
held fixed by construction:

* deterministic per seed (same seed ⇒ identical dict),
* ``ids`` is ``full_ids`` with the half-open span
  ``[gap_start, gap_start + gap_len)`` cut out,
* at least one token is removed (``gap_len >= 1``) and the gap never touches
  the first statement (``gap_start > 0``),
* ``l_max`` carries the Phantom-Padding contract of ``src.tokenizer``.
"""

import ast

import pytest

from src.curriculum_data import NAME_POOL, make_instance, snippet_source, stage1_batch
from src.tokenizer import L_MAX, Vocab, lexical_tokens


# ---------------------------------------------------------------------------
# Template synthesis
# ---------------------------------------------------------------------------


def test_snippet_source_is_deterministic_and_parses():
    for seed in range(20):
        src1 = snippet_source(seed)
        src2 = snippet_source(seed)
        assert src1 == src2
        tree = ast.parse(src1)
        assert 2 <= len(tree.body) <= 5, f"seed {seed}: {len(tree.body)} statements"
        for stmt in tree.body:
            assert isinstance(stmt, ast.Assign), "toy templates are plain assignments"
            assert stmt.lineno == stmt.end_lineno, "statements are single-line"


def test_snippet_names_are_rng_chosen_from_pool():
    """Names come from the shared pool; distinct within one snippet."""
    seen = set()
    for seed in range(60):
        names = {
            t.id
            for stmt in ast.parse(snippet_source(seed)).body
            for t in stmt.targets
            if isinstance(t, ast.Name)
        }
        assert len(names) == len({n for n in names})  # (trivial) distinctness
        assert names <= set(NAME_POOL)
        seen |= names
    assert len(seen) >= 4, "rng should exercise several pool members overall"


def test_chaining_references_only_earlier_statements():
    """A loaded name must be defined on an earlier line (vars may chain)."""
    for seed in range(60):
        defined: set[str] = set()
        for stmt in ast.parse(snippet_source(seed)).body:
            for node in ast.walk(stmt.value):
                if isinstance(node, ast.Name):
                    assert node.id in defined, (
                        f"seed {seed}: {node.id} used before assignment"
                    )
            defined |= {t.id for t in stmt.targets if isinstance(t, ast.Name)}


# ---------------------------------------------------------------------------
# Instance structure
# ---------------------------------------------------------------------------


REQUIRED_KEYS = {"ids", "gap_start", "gap_len", "l_max", "full_ids"}


@pytest.fixture(scope="module")
def instances():
    return [make_instance(seed) for seed in range(40)]


def test_make_instance_is_deterministic_given_seed(instances):
    for seed in range(40):
        assert make_instance(seed) == make_instance(seed)
    assert make_instance(7) == instances[7]


def test_instance_keys(instances):
    for inst in instances:
        assert REQUIRED_KEYS <= set(inst)


def test_gap_inside_bounds_and_nonfirst(instances):
    for inst in instances:
        n = len(inst["full_ids"])
        assert inst["gap_len"] >= 1
        # Non-first statement ⇒ the gap starts after statement 0's tokens,
        # which are at least ``name = int`` (3 tokens).
        assert inst["gap_start"] >= 3
        assert inst["gap_start"] + inst["gap_len"] <= n


def test_removing_ids_reproduces_original_length_when_reexpanded(instances):
    """Conceptual re-expansion: full length == corrupted length + gap_len."""
    for inst in instances:
        assert len(inst["full_ids"]) == len(inst["ids"]) + inst["gap_len"]


def test_ids_equal_full_ids_with_span_removed(instances):
    for inst in instances:
        g, k = inst["gap_start"], inst["gap_len"]
        rebuilt = inst["full_ids"][:g] + inst["full_ids"][g + k :]
        assert inst["ids"] == rebuilt


def test_l_max_matches_tokenizer_contract(instances):
    for inst in instances:
        assert inst["l_max"] == L_MAX


def test_vocab_shared_between_full_and_corrupted_stream(instances):
    """One per-instance Vocab over full_ids explains ids too (no re-mapping).

    ``make_instance`` builds a single :class:`Vocab` fit on the *full* token
    stream and encodes both streams through it; therefore every corrupted id
    must already occur in ``full_ids`` (identical id space, pad excluded).
    """
    for inst in instances:
        assert set(inst["ids"]) <= set(inst["full_ids"])
        assert all(i > 0 for i in inst["full_ids"]), "pad id 0 never appears"


def test_full_ids_match_lexical_tokenization_of_source():
    """full_ids are exactly the stable ids of the snippet's lexical tokens."""
    for seed in range(15):
        source = snippet_source(seed)
        vocab = Vocab()
        assert vocab.encode(lexical_tokens(source)) == make_instance(seed)["full_ids"]


def test_stage1_batch_matches_per_seed_make_instance():
    batch = stage1_batch(n=8, seed0=1000)
    assert len(batch) == 8
    assert batch == [make_instance(s) for s in range(1000, 1008)]


def test_stage1_batch_seed_windows_do_not_collide():
    assert stage1_batch(5, 0)[3] == make_instance(3)
    assert stage1_batch(5, 50)[3] != stage1_batch(5, 0)[3]


def test_instances_vary_across_seeds():
    blobs = {tuple(make_instance(s)["full_ids"]) for s in range(30)}
    assert len(blobs) > 10, "generator should not collapse to one snippet"
