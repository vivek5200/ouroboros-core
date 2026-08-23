"""Tests for Module 3: 1D RoPE + Additive AST Graph Bias attention.

These tests activate automatically once torch is installed on the host;
until then they skip (importorskip) so the suite stays green.
"""

import pytest

torch = pytest.importorskip("torch")

from src.attention import (  # noqa: E402
    ASTGraphBiasAttention,
    apply_1d_rope,
    ast_graph_bias,
    rope_cos_sin,
    rotate_half,
)


def test_rope_positions_must_be_1d():
    """LAW GUARD (math-rope): reject any 2-D position grid."""
    with pytest.raises(ValueError):
        rope_cos_sin((4, 4), 8)


def test_rope_head_dim_must_be_even():
    with pytest.raises(ValueError):
        rope_cos_sin(8, 7)


def test_rotate_half_pairs():
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    assert torch.equal(rotate_half(x), torch.tensor([[-3.0, -4.0, 1.0, 2.0]]))


def test_rope_score_depends_only_on_relative_offset():
    """Paper Eq. 3: <R_m q, R_k n> = g(m - n, q, k).

    Same q/k vectors placed at two position pairs with identical offset (+3)
    must yield identical inner products after rotation.
    """
    d = 8
    T = 16
    cos, sin = rope_cos_sin(T, d)
    qv = torch.randn(1, 1, d)
    kv = torch.randn(1, 1, d)
    q_full = torch.zeros(1, 1, T, d)
    k_full = torch.zeros(1, 1, T, d)
    q_full[0, 0, 0] = qv   # pair A: q @ 0
    k_full[0, 0, 3] = kv   #        k @ 3      -> offset +3
    q_full[0, 0, 5] = qv   # pair B: same q @ 5
    k_full[0, 0, 8] = kv   #        same k @ 8 -> offset +3
    qr = apply_1d_rope(q_full, cos, sin)
    kr = apply_1d_rope(k_full, cos, sin)
    s1 = (qr[0, 0, 0] * kr[0, 0, 3]).sum()
    s2 = (qr[0, 0, 5] * kr[0, 0, 8]).sum()
    assert torch.allclose(s1, s2, atol=1e-5)


def test_rope_preserves_norm():
    cos, sin = rope_cos_sin(32, 8)
    x = torch.randn(1, 2, 32, 8)
    xr = apply_1d_rope(x, cos, sin)
    assert torch.allclose(x.norm(dim=-1), xr.norm(dim=-1), atol=1e-5)


def test_bias_additive_pre_softmax():
    """Law math-rope: bias enters ADDITIVELY pre-softmax."""
    spd = torch.tensor([[0, 1, 2], [1, 0, 1], [2, 1, 0]], dtype=torch.float)
    bias = ast_graph_bias(spd)
    expected = torch.log1p(spd)
    assert torch.allclose(bias, expected, atol=1e-6)


def test_bias_unreachable_pairs_get_zero():
    spd = torch.tensor([[0, -1], [-1, 0]], dtype=torch.float)
    b = ast_graph_bias(spd)
    assert (b[0, 1] == 0) and (b[1, 0] == 0)


def test_forward_shapes_with_2d_bias_broadcast():
    torch.manual_seed(0)
    layer = ASTGraphBiasAttention(d_model=16, n_heads=2)
    x = torch.randn(2, 8, 16)
    out = layer(x, ast_bias=torch.zeros(8, 8))
    assert out.shape == (2, 8, 16)
    assert not torch.isnan(out).any()


def test_forward_scope_mask_blocks_attention():
    torch.manual_seed(1)
    layer = ASTGraphBiasAttention(d_model=16, n_heads=2)
    x = torch.randn(1, 6, 16)
    scope = torch.ones(6, 6, dtype=torch.bool)
    scope[:, :3] = False          # nothing may attend to tokens 0..2
    out_masked = layer(x, scope_mask=scope)
    out_full = layer(x)
    assert not torch.allclose(out_masked, out_full)


def test_module_is_nn_module_and_trainable():
    layer = ASTGraphBiasAttention(d_model=16, n_heads=2)
    assert sum(p.numel() for p in layer.parameters()) > 0
