"""Tests for GalerkinAttention layer."""
import pytest
import torch
from ..galerkin_attention import GalerkinAttention
from ..embeddings import RotaryEmbedding2D


def test_GalerkinAttention_basic():
    """Test GalerkinAttention basic forward pass."""
    head_n_channels = 32
    channels = 64
    n_heads = 4
    num_points = 128
    batch_size = 4

    layer = GalerkinAttention(
        in_channels=channels,
        out_channels=channels,
        n_heads=n_heads,
        head_n_channels=head_n_channels,
        n_modes=16,
    )

    x = torch.randn(batch_size, num_points, channels)
    out = layer(x)
    assert out.shape == (batch_size, num_points, channels)


def test_GalerkinAttention_with_pos_emb():
    """Test GalerkinAttention with 2D rotary positional embeddings."""
    head_n_channels = 32
    channels = 64
    n_heads = 4
    n_dim = 2
    num_points = 64
    batch_size = 4

    layer = GalerkinAttention(
        in_channels=channels,
        out_channels=channels,
        n_heads=n_heads,
        head_n_channels=head_n_channels,
        n_modes=16,
    )
    pos_emb = RotaryEmbedding2D(head_n_channels // n_dim)

    x = torch.randn(batch_size, num_points, channels)
    pos = torch.randn(batch_size, num_points, n_dim)

    out = layer(x, pos_src=pos, positional_embedding_module=pos_emb)
    assert out.shape == (batch_size, num_points, channels)


def test_GalerkinAttention_return_kernel():
    """Test that return_kernel=True returns a kernel tensor."""
    head_n_channels = 16
    channels = 32
    n_heads = 2
    n_modes = 8
    num_points = 64
    batch_size = 2

    layer = GalerkinAttention(
        in_channels=channels,
        out_channels=channels,
        n_heads=n_heads,
        head_n_channels=head_n_channels,
        n_modes=n_modes,
    )

    x = torch.randn(batch_size, num_points, channels)
    out, kernel = layer(x, return_kernel=True)

    assert out.shape == (batch_size, num_points, channels)
    assert kernel.shape[0] == batch_size
    assert kernel.shape[1] == n_heads
    assert kernel.shape[2] == n_modes


def test_GalerkinAttention_cross_attention():
    """Test cross-attention with same sequence length (different input tensors)."""
    head_n_channels = 16
    channels = 32
    n_heads = 2
    num_points = 64
    batch_size = 2

    layer = GalerkinAttention(
        in_channels=channels,
        out_channels=channels,
        n_heads=n_heads,
        head_n_channels=head_n_channels,
        n_modes=8,
    )

    src = torch.randn(batch_size, num_points, channels)
    qry = torch.randn(batch_size, num_points, channels)

    out = layer(u_src=src, u_qry=qry)
    assert out.shape == (batch_size, num_points, channels)


def test_GalerkinAttention_different_io_channels():
    """Test with different input and output channel dimensions."""
    head_n_channels = 16
    in_ch = 32
    out_ch = 64
    n_heads = 2
    num_points = 32
    batch_size = 2

    layer = GalerkinAttention(
        in_channels=in_ch,
        out_channels=out_ch,
        n_heads=n_heads,
        head_n_channels=head_n_channels,
        n_modes=8,
    )

    x = torch.randn(batch_size, num_points, in_ch)
    out = layer(x)
    assert out.shape == (batch_size, num_points, out_ch)


def test_GalerkinAttention_training_mode():
    """Test that alpha parameter is learnable and responds to training."""
    head_n_channels = 16
    channels = 32
    n_heads = 2
    n_modes = 8
    num_points = 32
    batch_size = 2

    layer = GalerkinAttention(
        in_channels=channels,
        out_channels=channels,
        n_heads=n_heads,
        head_n_channels=head_n_channels,
        n_modes=n_modes,
    )

    x = torch.randn(batch_size, num_points, channels)
    target = torch.randn_like(x)

    opt = torch.optim.Adam(layer.parameters(), lr=1e-3)
    alpha_before = layer.alpha.item()

    for _ in range(50):
        out = layer(x)
        loss = (out - target).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    alpha_after = layer.alpha.item()
    assert alpha_after != alpha_before
