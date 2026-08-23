"""Module 3: Attention with 1D RoPE + Additive AST Graph Bias.

LAW math-rope (confidence 0.99): standard 1D RoPE encodes lexical sequence
order; AST topology enters ONLY as an additive bias b_phi(i,j) on pre-softmax
attention logits. 2D AST-relative RoPE (rotating channel groups by AST depth
and sibling index) is mathematically invalid and structurally prevented here:
`rope_cos_sin` accepts 1-D positions only and raises otherwise.

Bias compression follows the paper §5.2: b = scale * log1p(phi) over the
tree-sitter shortest-path-distance matrix, with unreachable pairs (phi < 0)
contributing zero bias (scoping masks, Module 4, handle hard exclusion).
"""

import math

import torch
import torch.nn as nn

ENFORCE_1D_ROPE = True


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """GPT-NeoX-style half rotation used by the 1-D RoPE formulation."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def rope_cos_sin(seq_len: int, head_dim: int, device=None,
                 base: float = 10000.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute cos/sin tables for 1-D positions [0, seq_len).

    Raises:
        ValueError: if a non-1D position grid is attempted (math-rope guard).
    """
    if not isinstance(seq_len, int) or seq_len < 0:
        raise ValueError(f"seq_len must be a non-negative int, got {seq_len!r}")
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")
    inv_freq = base ** (-torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    pos = torch.arange(seq_len, device=device).float()
    if pos.dim() != 1:
        raise ValueError("math-rope violation: RoPE positions must be 1-D lexical order")
    angles = torch.outer(pos, inv_freq)
    return angles.cos(), angles.sin()


def apply_1d_rope(x: torch.Tensor, cos: torch.Tensor,
                  sin: torch.Tensor) -> torch.Tensor:
    """Apply 1-D RoPE to (batch, heads, seq, head_dim) q or k tensors."""
    cos_d = torch.cat([cos, cos], dim=-1)[None, None, :, :]
    sin_d = torch.cat([sin, sin], dim=-1)[None, None, :, :]
    return x * cos_d + rotate_half(x) * sin_d


def ast_graph_bias(spd: torch.Tensor, scale: float = 1.0,
                   max_dist: float = 64.0) -> torch.Tensor:
    """Compressed additive AST bias b_phi(i,j) = scale * log1p(phi).

    Args:
        spd: (T, T) shortest-path distances; pairs with spd < 0 are treated
            as unreachable and receive zero bias (hard exclusion belongs to
            the Module 4 scoping mask, not the soft bias).
    """
    d = spd.clone().float()
    d[d < 0] = 0.0
    d = d.clamp(max=max_dist)
    return scale * torch.log1p(d)


class ASTGraphBiasAttention(nn.Module):
    """Attention with 1D RoPE and additive AST graph bias (paper Eq. 10).

    A_ij = Softmax( (q_i . k_j) / sqrt(head_dim) + b_phi(i,j) [+ scope mask] )
    """

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        ast_bias: torch.Tensor | None = None,
        scope_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with 1-D RoPE and additive AST graph bias.

        Args:
            x: (batch, seq_len, d_model).
            ast_bias: additive bias broadcastable to (batch, heads, T, T);
                a (T, T) tensor is auto-expanded. Pre-softmax, per math-rope.
            scope_mask: optional (T, T) bool from Module 4 — False positions
                receive -inf logits.
        """
        b, t, _ = x.shape
        q = self.wq(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)

        cos, sin = rope_cos_sin(t, self.head_dim, device=x.device)
        q = apply_1d_rope(q, cos, sin)
        k = apply_1d_rope(k, cos, sin)

        logits = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if ast_bias is not None:
            if ast_bias.dim() == 2:
                ast_bias = ast_bias[None, None]
            elif ast_bias.dim() == 3:
                ast_bias = ast_bias[:, None]
            logits = logits + ast_bias
        if scope_mask is not None:
            logits = logits.masked_fill(~scope_mask[None, None], float("-inf"))

        attn = torch.softmax(logits, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(b, t, self.d_model)
        return self.wo(out)
