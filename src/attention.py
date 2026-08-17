"""Module 3: Attention mechanism with 1D RoPE + Additive AST Graph Bias.

CONSTRAINT: NEVER use 2D AST-Relative RoPE.
Use standard 1D RoPE for lexical sequence order plus Additive AST Graph Bias
(shortest-path scalar b_phi(i,j) injected into pre-softmax attention logits).
"""

import torch
import torch.nn as nn


class ASTGraphBiasAttention(nn.Module):
    """Attention with 1D RoPE and additive AST graph bias.

    TODO: Implement the full attention mechanism.
    """

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

    def forward(
        self,
        x: torch.Tensor,
        ast_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with optional AST graph bias.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).
            ast_bias: Additive bias of shape (batch, n_heads, seq_len, seq_len).

        Returns:
            Output tensor of shape (batch, seq_len, d_model).
        """
        raise NotImplementedError("Attention not yet implemented")
