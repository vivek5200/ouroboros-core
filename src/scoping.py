"""Module 4: Block-Sparse Lexical Scoping (pure python — no torch).

Implements Paper Table 1: hierarchical attention mask rules over tokens
grouped into contiguous regions tagged GLOBAL or LOCAL.

Table 1 (M[i,j] as query-i-sees-key-j):
    M[i,j] = 1  if i is LOCAL and j is GLOBAL       function bodies see signatures
    M[i,j] = 1  if i == j                           self
    M[i,j] = 1  if i, j both LOCAL, same scope      coherent generation in block
    M[i,j] = 0  if i is GLOBAL and j is LOCAL       global KV-cache insulated
    M[i,j] = 0  if i, j LOCAL siblings, diff scopes no sibling bleed

Spec ambiguities resolved here (documented deliberately):
  * GLOBAL x GLOBAL visibility is not listed in Table 1; it is set to 1 so
    that signatures see each other and so the mask is symmetric exactly on
    {same-local-scope} U {global-global} pairs (per the module symmetry spec).
    Hence the mask is intentionally NOT symmetric overall:
    LOCAL-row/GLOBAL-col = 1 but the mirrored GLOBAL-row/LOCAL-col = 0.
  * ``frozen_global_columns`` returns the post-mutation L'xL' mask (matching
    its ``-> list[list[bool]]`` signature); the assert-style True/False check
    "did global rows stay frozen?" lives in ``global_rows_frozen``, which
    never raises. The helper freezes GLOBAL *rows* (a global token's output
    depends only on other globals, so its cached state survives local [EXPAND]
    mutations); the historical name mentions columns.
  * Spans may be given unordered; they are accepted iff they form an exact
    partition of [0, L). Zero-length, negative, unknown-scope, gapped or
    overlapping regions raise ValueError. Empty spans list => L = 0.
"""

from bisect import bisect_right

GLOBAL = "global"
LOCAL = "local"


class Scope:
    """Enum-like scope constants (plain strings, per Module 4 spec)."""

    GLOBAL = "global"
    LOCAL = "local"


def _validated_sorted(spans) -> list[tuple[int, int, str]]:
    """Validate that ``spans`` exactly partitions [0, L); return them sorted.

    Raises ValueError on gaps, overlaps, non-zero start, zero-length or
    negative regions, and unknown scope names.
    """
    cleaned: list[tuple[int, int, str]] = []
    for span in spans:
        try:
            start, end, scope = span
        except (TypeError, ValueError) as exc:
            raise ValueError(f"malformed span (want (start, end, scope)): {span!r}") from exc
        if isinstance(start, bool) or isinstance(end, bool):
            raise ValueError(f"span bounds must be ints, got {span!r}")
        if not isinstance(start, int) or not isinstance(end, int):
            raise ValueError(f"span bounds must be ints, got {span!r}")
        if start < 0:
            raise ValueError(f"span start must be >= 0, got {start}")
        if end <= start:
            raise ValueError(f"span must be non-empty, got ({start}, {end})")
        if scope not in (Scope.GLOBAL, Scope.LOCAL):
            raise ValueError(f"unknown scope {scope!r}; want {Scope.GLOBAL!r}/{Scope.LOCAL!r}")
        cleaned.append((start, end, scope))
    cleaned.sort()
    if cleaned and cleaned[0][0] != 0:
        raise ValueError(f"spans must partition starting at 0, first span starts at {cleaned[0][0]}")
    for prev, cur in zip(cleaned, cleaned[1:]):
        if cur[0] != prev[1]:
            kind = "overlap" if cur[0] < prev[1] else "gap"
            raise ValueError(f"spans do not partition [0, L): {kind} between {prev} and {cur}")
    return cleaned


def build_scope_tags(spans) -> list[str]:
    """Per-token scope tags of length L.

    Args:
        spans: List of ``(start, end_exclusive, scope)`` regions which must
            exactly partition ``[0, L)`` (order-insensitive).

    Returns:
        List of length L with Scope.GLOBAL / Scope.LOCAL per token.
        Empty spans list yields ``[]`` (L = 0 edge).

    Raises:
        ValueError: If the regions do not exactly partition [0, L).
    """
    ordered = _validated_sorted(spans)
    tags: list[str] = []
    for start, end, scope in ordered:
        tags.extend([scope] * (end - start))
    return tags


def same_local_scope(i: int, j: int, spans) -> bool:
    """True iff tokens i and j fall inside the same single LOCAL span."""
    ordered = _validated_sorted(spans)
    starts = [s for s, _, _ in ordered]

    def owner(tok: int):
        idx = bisect_right(starts, tok) - 1
        if idx < 0:
            return None
        s, e, scope = ordered[idx]
        return (s, e) if (s <= tok < e and scope == Scope.LOCAL) else None

    oi, oj = owner(i), owner(j)
    return oi is not None and oj is not None and oi == oj


def block_sparse_mask(spans) -> list[list[bool]]:
    """Full LxL boolean mask implementing Table 1 exactly.

    Row semantics: a GLOBAL row sees only GLOBAL columns (its KV-cache state
    is computed purely from other globals, insulating it from local edits);
    a LOCAL row sees all GLOBAL columns plus every column inside its own
    single LOCAL span, plus itself.
    """
    ordered = _validated_sorted(spans)
    if not ordered:
        return []
    total = ordered[-1][1]
    tags: list[str] = []
    owners: list[tuple[int, int]] = []
    for start, end, scope in ordered:
        tags.extend([scope] * (end - start))
        owners.extend([(start, end)] * (end - start))

    mask: list[list[bool]] = []
    for i in range(total):
        row: list[bool] = []
        tag_i = tags[i]
        own_i = owners[i]
        for j in range(total):
            if i == j:
                row.append(True)
            elif tag_i == Scope.LOCAL:
                # Bodies see signatures + coherent same-scope siblings.
                row.append(tags[j] == Scope.GLOBAL or owners[j] == own_i)
            else:  # GLOBAL row: insulated from every LOCAL column.
                row.append(tags[j] == Scope.GLOBAL)
        mask.append(row)
    return mask


def frozen_global_columns(mask_before, spans, mutated_span) -> list[list[bool]]:
    """Rebuild the mask after an [EXPAND] of one LOCAL span, freezing GLOBAL rows.

    Demonstrates the cache-coherency guarantee: when a LOCAL span grows by
    splicing tokens (Paper Algorithm 1), every later span shifts right, yet
    each GLOBAL row of the new mask is bit-for-bit the old GLOBAL row extended
    with False — global cached states never gain attention toward mutated
    LOCAL content, so the global KV-cache stays valid without recompute.

    Args:
        mask_before: Square mask from ``block_sparse_mask`` over ``spans``.
        spans: Current exact partition of [0, L).
        mutated_span: ``(new_start, new_end_exclusive)`` — the expanded extent
            of an existing LOCAL span whose start is unchanged. All spans at
            or after ``new_start`` shift right by the growth delta.

    Returns:
        The post-mutation L'xL' mask with frozen GLOBAL rows. Use
        ``global_rows_frozen(mask_before, result, new_spans)`` for the
        assert-style boolean check (returns True/False, never raises).

    Raises:
        ValueError: On malformed input (non-square mask_before, unknown
            mutation target, target not LOCAL, or a GLOBAL span sitting at or
            after the mutation point — then global indices would move and
            freezing by position would be ill-defined).
    """
    ordered = _validated_sorted(spans)
    total_old = ordered[-1][1] if ordered else 0

    if len(mask_before) != total_old or any(len(r) != total_old for r in mask_before):
        raise ValueError(
            f"mask_before must be square {total_old}x{total_old} to match spans"
        )

    try:
        m_start, m_end = mutated_span
    except (TypeError, ValueError) as exc:
        raise ValueError(f"mutated_span must be (start, end): {mutated_span!r}") from exc

    matches = [t for t in ordered if t[0] == m_start]
    if len(matches) != 1:
        raise ValueError(f"no unique span starts at {m_start}: {matches}")
    old_start, old_end, old_scope = matches[0]
    if old_scope != Scope.LOCAL:
        raise ValueError(f"[EXPAND] targets LOCAL spans only, but span {matches[0]} is {old_scope!r}")
    if not isinstance(m_end, int) or isinstance(m_end, bool) or m_end <= m_start:
        raise ValueError(f"mutated_span end must be int > start, got {mutated_span!r}")

    delta = m_end - old_end
    # Freezing by index requires global token positions to stay put: every
    # GLOBAL span must lie entirely before the mutation point.
    for s, e, sc in ordered:
        if sc == Scope.GLOBAL and e > m_start:
            raise ValueError(
                f"cannot freeze GLOBAL rows: GLOBAL span ({s}, {e}) overlaps/moves past "
                f"mutation point {m_start}"
            )

    new_spans = [
        (m_start, m_end, Scope.LOCAL) if (s, e) == (old_start, old_end)
        else ((s + delta, e + delta, sc) if s > m_start else (s, e, sc))
        for s, e, sc in ordered
    ]
    mask_after = block_sparse_mask(new_spans)

    total_new = total_old + delta
    for i, (_, _, scope) in enumerate(ordered):
        if scope != Scope.GLOBAL:
            continue
        before_row = mask_before[i]
        grown_row = mask_after[i]
        for j in range(total_new):
            # Old extent keeps its exact prior values; appended LOCAL columns
            # are invisible to the frozen global row (Table 1: G row x L col = 0).
            grown_row[j] = before_row[j] if j < total_old else False
    return mask_after


def global_rows_frozen(mask_before, mask_after, spans_after) -> bool:
    """Assert-style check: are all GLOBAL rows unchanged vs ``mask_before``?

    A row tagged GLOBAL under ``spans_after`` must equal the corresponding
    ``mask_before`` row over the old extent and be False beyond it. Returns
    True/False instead of raising; shape mismatches simply read as False.
    """
    try:
        ordered = _validated_sorted(spans_after)
        total_new = ordered[-1][1] if ordered else 0
        total_old = len(mask_before)
        if total_new < total_old or len(mask_after) != total_new:
            return False
        if any(len(r) != total_new for r in mask_after):
            return False
        tags = build_scope_tags(ordered)
        for i, tag in enumerate(tags):
            if tag != Scope.GLOBAL:
                continue
            expected = list(mask_before[i]) + [False] * (total_new - total_old)
            if list(mask_after[i]) != expected:
                return False
        return True
    except (TypeError, ValueError, IndexError):
        return False
