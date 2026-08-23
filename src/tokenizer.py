"""Module 1: AST-aware Tokenizer with Phantom Padding.

Phantom Padding: L_max = 1024 tokens.
All sequences are padded to L_max before entering the attention mechanism.

AST-awareness (paper Module 1 + Module 3): every lexical token is attributed
to its nearest enclosing statement/expression AST node. The attribution feeds
``ast_spd_matrix``, which produces the shortest-path-distance matrix phi(i,j)
consumed by Module 3's additive graph bias b_phi = scale * log1p(phi).

LAW math-rope (confidence 0.99): lexical ORDER stays 1-D RoPE's job (Module
3); the AST enters ONLY through phi. Nothing here re-encodes position from
tree structure — ids are purely lexical, node info is carried separately.
"""

import ast
import io
import re
import tokenize as _pytokenize
from collections import deque
from token import DEDENT, ENCODING, ENDMARKER, INDENT, NEWLINE, NL

L_MAX = 1024  # Phantom Padding maximum sequence length

PAD_ID = 0  # vocabulary id 0 is reserved for padding; real ids start at 1

# Lexical noise: structural tokenizer events, never emitted as vocab entries.
_NOISE_TOKEN_TYPES = frozenset({ENCODING, NL, NEWLINE, INDENT, DEDENT, ENDMARKER})

# Last-resort scanner if generate_tokens dies before yielding any usable token.
_FALLBACK_LEXEME_RE = re.compile(
    r"[A-Za-z_]\w*"                      # NAME / KEYWORD
    r"|\d[\w.]*"                         # NUMBER
    r"|[rRbBuUfF]{0,2}'''(?:.|\n)*?'''"  # triple-quoted STRING
    r"|[rRbBuUfF]{0,2}\"\"\"(?:.|\n)*?\"\"\""
    r"|[rRbBuUfF]{0,2}'(?:\\.|[^'\\\n])*'"
    r"|[rRbBuUfF]{0,2}\"(?:\\.|[^\"\\\n])*\""
    r"|\S"                               # any remaining OP / ERRORTOKEN char
)


class Vocab:
    """Deterministic, stable string->id mapping (0 = pad, real ids start at 1).

    Ids are assigned consecutively in order of first appearance, so two calls
    on identical input always agree — and a caller may keep an instance around
    (or serialize ``token_to_id``) to reuse one mapping across calls/documents.
    """

    PAD_ID = 0

    def __init__(self) -> None:
        self.token_to_id: dict[str, int] = {}
        self._id_to_token: dict[int, str] = {}

    def id_of(self, token: str) -> int:
        """Id of ``token``, assigning the next consecutive id if unseen."""
        tid = self.token_to_id.get(token)
        if tid is None:
            tid = len(self.token_to_id) + 1  # 0 reserved for padding
            self.token_to_id[token] = tid
            self._id_to_token[tid] = token
        return tid

    def encode(self, tokens) -> list[int]:
        """Map lexical token strings to their stable ids."""
        return [self.id_of(t) for t in tokens]

    def decode(self, ids) -> list[str]:
        """Inverse of :meth:`encode` (pad id 0 decodes to ``"<pad>"``)."""
        out = []
        for i in ids:
            if i == self.PAD_ID:
                out.append("<pad>")
                continue
            tok = self._id_to_token.get(i)
            if tok is None:
                raise KeyError(f"id {i!r} not in vocabulary")
            out.append(tok)
        return out

    def __len__(self) -> int:
        return len(self.token_to_id)

    def __contains__(self, token: object) -> bool:
        return token in self.token_to_id


def _line_start_offsets(source_code: str) -> tuple[list[int], int]:
    """Absolute offset of each 1-indexed line's start, plus total length."""
    offsets: list[int] = []
    off = 0
    for line in source_code.splitlines(keepends=True):
        offsets.append(off)
        off += len(line)
    return offsets, off


def _absolute_offset(line_offsets: list[int], eof_offset: int, pos: tuple[int, int]) -> int:
    """Convert a tokenize (row 1-based, col 0-based) position to an offset."""
    row, col = pos
    idx = row - 1
    if 0 <= idx < len(line_offsets):
        return line_offsets[idx] + col
    return eof_offset


def _lexical_token_spans(source_code: str) -> list[tuple[str, int]]:
    """(token_string, start_offset) per kept token, in lexical order.

    Uses ``tokenize.generate_tokens`` and skips pure structural noise
    (ENCODING/NL/NEWLINE/INDENT/DEDENT/ENDMARKER). Malformed source never
    raises here: tokens yielded before the failure are kept, and if nothing
    at all was captured a conservative regex scan supplies the lexemes so
    callers always get a usable lexical sequence (fallback path).
    """
    line_offsets, eof = _line_start_offsets(source_code)
    spans: list[tuple[str, int]] = []
    gen = _pytokenize.generate_tokens(io.StringIO(source_code).readline)
    while True:
        try:
            tok = next(gen)
        except StopIteration:
            break
        except Exception:
            # TokenError / SyntaxError family: malformed source — keep prefix.
            break
        if tok.type in _NOISE_TOKEN_TYPES:
            continue
        spans.append(
            (tok.string, _absolute_offset(line_offsets, eof, tok.start))
        )
    if not spans and source_code.strip():
        spans = [
            (m.group(0), m.start()) for m in _FALLBACK_LEXEME_RE.finditer(source_code)
        ]
    return spans


def lexical_tokens(source_code: str) -> list[str]:
    """The kept lexical tokens of ``source_code`` in order (strings only)."""
    return [string for string, _ in _lexical_token_spans(source_code)]


class _AstIndex:
    """Preorder DFS index of an AST with parent/child adjacency by node id."""

    __slots__ = ("nodes", "index_of", "child_ids")

    def __init__(self, tree: ast.AST):
        self.nodes: list[ast.AST] = []
        self.index_of: dict[int, int] = {}
        self.child_ids: dict[int, list[int]] = {}
        stack: list[ast.AST] = [tree]
        while stack:
            node = stack.pop()
            self.index_of[id(node)] = len(self.nodes)
            self.nodes.append(node)
            kids = list(ast.iter_child_nodes(node))
            self.child_ids[id(node)] = [id(k) for k in kids]
            stack.extend(reversed(kids))  # deterministic preorder


def _candidate_spans(index: _AstIndex, source: str) -> list[tuple[int, int, ast.AST]]:
    """Absolute ``(start, end, node)`` spans of every stmt/expr node, preorder."""
    line_offsets, eof = _line_start_offsets(source)

    def to_abs(row: int | None, col: int) -> int:
        if row is None:
            return eof
        idx = row - 1
        if not 0 <= idx < len(line_offsets):
            return eof
        return line_offsets[idx] + col

    spans: list[tuple[int, int, ast.AST]] = []
    for node in index.nodes:
        if not isinstance(node, (ast.stmt, ast.expr)):
            continue
        start = to_abs(getattr(node, "lineno", None), node.col_offset)
        end = to_abs(getattr(node, "end_lineno", None), node.end_col_offset)
        if end > start:
            spans.append((start, end, node))
    return spans


def _attributed_node(spans: list[tuple[int, int, ast.AST]], offset: int, root: ast.AST) -> ast.AST:
    """Smallest stmt/expr span containing ``offset`` (else ``root``).

    Containment is start-inclusive / end-exclusive. Ties (identical span
    size) resolve to the first in preorder — the outermost node — which is
    deterministic.
    """
    best: tuple[int, ast.AST] | None = None  # (span_size, node)
    for start, end, node in spans:
        if start <= offset < end:
            size = end - start
            if best is None or size < best[0]:
                best = (size, node)
    return best[1] if best is not None else root


def tokenize_with_nodes(source_code: str) -> tuple[Vocab, list[int], list[str]]:
    """AST-aware tokenization: ``(vocab, ids, node_kinds)``.

    Each id corresponds to the lexical token at the same index of
    ``node_kinds``, the kind name of that token's nearest enclosing
    statement/expression AST node. Malformed source falls back to the pure
    lexical token stream with every kind reported as ``"Module"``.
    """
    spans = _lexical_token_spans(source_code)
    strings = [s for s, _ in spans]
    try:
        tree = ast.parse(source_code)
    except Exception:
        # Malformed: pure lexical fallback, all nodes "Module".
        vocab = Vocab()
        return vocab, vocab.encode(strings), ["Module"] * len(strings)
    index = _AstIndex(tree)
    candidates = _candidate_spans(index, source_code)
    kinds = [
        _attributed_node(candidates, off, tree).__class__.__name__
        for _, off in spans
    ]
    vocab = Vocab()
    return vocab, vocab.encode(strings), kinds


def tokenize(source_code: str) -> list[int]:
    """Tokenize Python source into stable lexical token ids (0 = pad).

    Deterministic: distinct token strings map to consecutive ids starting at
    1 in order of first appearance. Purely lexical — per math-rope, sequence
    order belongs to RoPE, never to these ids.
    """
    vocab = Vocab()
    return vocab.encode(lexical_tokens(source_code))


def ast_spd_matrix(source_code: str) -> list[list[int]]:
    """T x T shortest-path distances between the tokens' AST nodes.

    Builds an undirected parent<->child adjacency over all AST nodes,
    computes all-pairs shortest paths by BFS from each node, and reports the
    node-level SPD for every pair of lexical tokens. Tokens sharing one
    attributed node have distance 0. This matrix phi(i, j) is the ONLY channel
    through which syntax structure reaches attention (Module 3 bias
    b_phi = log1p(phi)); it carries no positional information.

    Raises:
        ValueError: If ``source_code`` cannot be parsed as a Python AST.
    """
    spans = _lexical_token_spans(source_code)
    try:
        tree = ast.parse(source_code)
    except Exception as exc:
        raise ValueError(f"source_code is not parseable Python: {exc}") from exc
    if not spans:
        return []
    index = _AstIndex(tree)
    candidates = _candidate_spans(index, source_code)
    n = len(index.nodes)
    adjacency: list[list[int]] = [[] for _ in range(n)]
    for p_idx, node in enumerate(index.nodes):
        for child_id in index.child_ids[id(node)]:
            c_idx = index.index_of[child_id]
            adjacency[p_idx].append(c_idx)
            adjacency[c_idx].append(p_idx)
    attributed = [
        index.index_of[id(_attributed_node(candidates, off, tree))]
        for _, off in spans
    ]
    dist_from: dict[int, list[int]] = {}

    def bfs(src: int) -> list[int]:
        dist = [-1] * n
        dist[src] = 0
        queue = deque([src])
        while queue:
            u = queue.popleft()
            for v in adjacency[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    queue.append(v)
        return dist

    for u in set(attributed):
        dist_from[u] = bfs(u)
    return [[dist_from[a][b] for b in attributed] for a in attributed]


def phantom_pad(token_ids: list[int], pad_id: int = 0) -> list[int]:
    """Pad token sequence to L_max using Phantom Padding.

    Args:
        token_ids: Input token IDs.
        pad_id: Padding token ID.

    Returns:
        Padded token sequence of length L_MAX.
    """
    if len(token_ids) > L_MAX:
        return token_ids[:L_MAX]
    return token_ids + [pad_id] * (L_MAX - len(token_ids))


def front_pack(token_ids: list[int], pad_id: int = 0) -> tuple[list[int], int]:
    """Front-pack tokens into a fixed L_MAX buffer (Algorithm 1, lines 1-2).

    Unlike ``phantom_pad`` this exposes the logical length so attention masks
    can be derived without ever resizing the physical buffer: the returned
    buffer shape is invariant at exactly L_MAX regardless of input length.

    Args:
        token_ids: Input token IDs.
        pad_id: Padding token ID for the unused tail.

    Returns:
        Tuple ``(buffer, logical_len)`` where ``buffer`` has exactly L_MAX
        entries and ``logical_len`` counts leading real tokens (equal to
        L_MAX when the input was truncated).
    """
    trimmed = token_ids[:L_MAX]
    buffer = trimmed + [pad_id] * (L_MAX - len(trimmed))
    return buffer, len(trimmed)


# ---------------------------------------------------------------------------
# Reserved sentinel ids (real vocabulary ids are always >= 0)
# ---------------------------------------------------------------------------

MASK_ID = -1    # [EXPAND]: placeholder token spliced in by ``insert_masks``
IGNORE_ID = -2  # [IGNORE]: filler for the unused tail of the fixed buffer


def insert_masks(buffer: list[int], logical_len: int, pos: int, k: int) -> int:
    """[EXPAND]: splice ``k`` MASK tokens into the logical region, in place.

    Algorithm 1 inserts placeholders without ever resizing the physical
    L_MAX buffer: the ``k`` rightmost ignore-filler slots are sacrificed to
    make room (they are guaranteed unused because the overflow guard below
    ensures ``logical_len + k <= L_MAX``).

    Args:
        buffer: Fixed-length token buffer of exactly L_MAX entries.
        logical_len: Number of leading live tokens in ``buffer``.
        pos: Insertion index within the logical region, ``0 <= pos <= logical_len``.
        k: Number of MASK_ID tokens to insert.

    Returns:
        The new logical length (``logical_len + k``).

    Raises:
        RuntimeError: If ``pos`` is outside ``[0, logical_len]``, if ``k`` is
            negative, or if the buffer cannot fit the insertion
            (``logical_len + k > L_MAX``, i.e. exhausted).
    """
    if not 0 <= pos <= logical_len:
        raise RuntimeError(
            f"insert pos {pos} outside logical region [0, {logical_len}]"
        )
    if k < 0:
        raise RuntimeError(f"k must be non-negative, got {k}")
    if logical_len + k > L_MAX:
        raise RuntimeError(
            f"buffer exhausted: logical_len {logical_len} + {k} masks"
            f" exceeds L_MAX {L_MAX}"
        )
    # In-place fixed-shape splice: len(buffer) is L_MAX before and after.
    buffer[:] = buffer[:pos] + [MASK_ID] * k + buffer[pos:L_MAX - k]
    return logical_len + k


def logical_delete(buffer: list[int], logical_len: int, pos: int) -> int:
    """[DELETE]: logically remove the token at ``pos``, keeping len(buffer) fixed.

    Uses ``list.pop(pos)`` to shift the logical region left, then refills the
    vacated tail slot with an IGNORE_ID filler ([IGNORE]) so the physical
    buffer shape stays invariant at exactly L_MAX.

    Args:
        buffer: Fixed-length token buffer of exactly L_MAX entries.
        logical_len: Number of leading live tokens in ``buffer``.
        pos: Index within the logical region, ``0 <= pos < logical_len``.

    Returns:
        The new logical length (``logical_len - 1``).

    Raises:
        IndexError: If ``pos`` is outside the logical region
            ``[0, logical_len)``.
    """
    if not 0 <= pos < logical_len:
        raise IndexError(
            f"delete pos {pos} outside logical region [0, {logical_len})"
        )
    buffer.pop(pos)
    buffer.append(IGNORE_ID)
    return logical_len - 1


def derive_mask(buffer: list[int], logical_len: int) -> list[bool]:
    """Derive the attention mask from the logical length alone.

    True for every index inside the logical region ``[0, logical_len)``,
    False for the phantom tail. Buffer contents are irrelevant: sentinels
    (MASK_ID / IGNORE_ID / pad ids) never leak into the mask.

    Args:
        buffer: Fixed-length token buffer (used only for documentation of
            the shape contract; the mask depends solely on ``logical_len``).
        logical_len: Number of leading live tokens.

    Returns:
        List of exactly L_MAX booleans.
    """
    return [i < logical_len for i in range(L_MAX)]
