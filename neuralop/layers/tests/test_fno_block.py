import pytest
import torch
from ..fno_block import FNOBlocks


def test_FNOBlock_resolution_scaling_factor():
    """Test FNOBlocks with upsampled or downsampled outputs"""
    max_n_modes = [8, 8, 8, 8]
    n_modes = [4, 4, 4, 4]

    size = [10] * 4
    channel_mlp_dropout = 0
    channel_mlp_expansion = 0.5
    channel_mlp_skip = "linear"
    for dim in [1, 2, 3, 4]:
        block = FNOBlocks(
            3,
            4,
            max_n_modes[:dim],
            max_n_modes=max_n_modes[:dim],
            n_layers=1,
            channel_mlp_skip=channel_mlp_skip,
        )

        assert block.convs[0].n_modes[:-1] == max_n_modes[: dim - 1]
        assert block.convs[0].n_modes[-1] == max_n_modes[dim - 1] // 2 + 1

        block.n_modes = n_modes[:dim]
        assert block.convs[0].n_modes[:-1] == n_modes[: dim - 1]
        assert block.convs[0].n_modes[-1] == n_modes[dim - 1] // 2 + 1

        block.n_modes = max_n_modes[:dim]
        assert block.convs[0].n_modes[:-1] == max_n_modes[: dim - 1]
        assert block.convs[0].n_modes[-1] == max_n_modes[dim - 1] // 2 + 1

        # Downsample outputs
        block = FNOBlocks(
            3,
            4,
            n_modes[:dim],
            n_layers=1,
            resolution_scaling_factor=0.5,
            use_channel_mlp=True,
            channel_mlp_dropout=channel_mlp_dropout,
            channel_mlp_expansion=channel_mlp_expansion,
            channel_mlp_skip=channel_mlp_skip,
        )

        x = torch.randn(2, 3, *size[:dim])
        res = block(x)
        assert list(res.shape[2:]) == [m // 2 for m in size[:dim]]

        # Upsample outputs
        block = FNOBlocks(
            3,
            4,
            n_modes[:dim],
            n_layers=1,
            resolution_scaling_factor=2,
            channel_mlp_dropout=channel_mlp_dropout,
            channel_mlp_expansion=channel_mlp_expansion,
            channel_mlp_skip=channel_mlp_skip,
        )

        x = torch.randn(2, 3, *size[:dim])
        res = block(x)
        assert res.shape[1] == 4  # Check out channels
        assert list(res.shape[2:]) == [m * 2 for m in size[:dim]]


@pytest.mark.parametrize("n_dim", [1, 2, 3, 4])
@pytest.mark.parametrize(
    "norm", ["instance_norm", "ada_in", "group_norm", "batch_norm"]
)
def test_FNOBlock_norm(norm, n_dim):
    """Test SpectralConv with upsampled or downsampled outputs"""
    modes = (8, 8, 8)
    size = [10] * 3
    channel_mlp_dropout = 0
    channel_mlp_expansion = 0.5
    channel_mlp_skip = "linear"
    ada_in_features = 4
    block = FNOBlocks(
        3,
        4,
        modes[:n_dim],
        n_layers=1,
        norm=norm,
        ada_in_features=ada_in_features,
        channel_mlp_dropout=channel_mlp_dropout,
        channel_mlp_expansion=channel_mlp_expansion,
        channel_mlp_skip=channel_mlp_skip,
    )

    if norm == "ada_in":
        embedding = torch.randn(ada_in_features)
        block.set_ada_in_embeddings(embedding)

    x = torch.randn(2, 3, *size[:n_dim])
    res = block(x)
    assert list(res.shape[2:]) == size[:n_dim]


@pytest.mark.parametrize("norm_groups", [1, 2, 4, 8])
def test_FNOBlock_group_norm(norm_groups):
    """Test FNOBlocks with group_norm and custom norm_groups"""
    modes = (8, 8, 8)
    hidden_channels = 16
    n_layers = 1
    
    block = FNOBlocks(
        in_channels=hidden_channels,
        out_channels=hidden_channels,
        n_modes=modes,
        n_layers=n_layers,
        norm="group_norm",
        norm_groups=norm_groups,
    )
    
    # Check that GroupNorm layers are correctly initialized
    assert block.norm is not None
    for norm_layer in block.norm:
        assert isinstance(norm_layer, torch.nn.GroupNorm)
        assert norm_layer.num_groups == norm_groups
        assert norm_layer.num_channels == hidden_channels


@pytest.mark.parametrize("n_dim", [1, 2, 3])
def test_FNOBlock_complex_data(n_dim):
    """Test FNO layers with complex input data"""
    modes = (8, 8, 8)
    size = [10] * 3
    channel_mlp_dropout = 0
    channel_mlp_expansion = 0.5
    channel_mlp_skip = "linear"
    # Instantiate a complex-valued FNO block
    block = FNOBlocks(
        3,
        4,
        modes[:n_dim],
        n_layers=1,
        channel_mlp_dropout=channel_mlp_dropout,
        channel_mlp_expansion=channel_mlp_expansion,
        channel_mlp_skip=channel_mlp_skip,
        complex_data=True,
    )

    x = torch.randn(2, 3, *size[:n_dim], dtype=torch.cfloat)
    res = block(x)

    assert list(res.shape[2:]) == size[:n_dim]


@pytest.mark.parametrize("fno_skip", ["linear", None])
@pytest.mark.parametrize("channel_mlp_skip", ["linear", None])
def test_FNOBlock_skip_connections(fno_skip, channel_mlp_skip):
    """Test FNOBlocks with different skip connection options including None"""
    modes = (8, 8, 8)
    size = [10, 10, 10]

    # Skip test cases that are incompatible
    # Soft-gating requires same input/output channels
    if fno_skip == "soft-gating" or channel_mlp_skip == "soft-gating":
        pytest.skip("Soft-gating requires same input/output channels")

    # Test with channel MLP enabled
    block = FNOBlocks(
        3,
        4,
        modes,
        n_layers=2,
        fno_skip=fno_skip,
        channel_mlp_skip=channel_mlp_skip,
        use_channel_mlp=True,
        channel_mlp_expansion=0.5,
        channel_mlp_dropout=0.0,
    )

    x = torch.randn(2, 3, *size)
    res = block(x)

    # Check output shape
    assert res.shape == (2, 4, *size)

    # Test with channel MLP disabled
    block_no_mlp = FNOBlocks(
        3,
        4,
        modes,
        n_layers=2,
        fno_skip=fno_skip,
        channel_mlp_skip=channel_mlp_skip,
        use_channel_mlp=False,
    )

    res_no_mlp = block_no_mlp(x)
    assert res_no_mlp.shape == (2, 4, *size)


@pytest.mark.parametrize("fno_skip", ["linear", None])
@pytest.mark.parametrize("channel_mlp_skip", ["linear", None])
def test_FNOBlock_skip_connections_preactivation(fno_skip, channel_mlp_skip):
    """Test FNOBlocks with preactivation and different skip connection options"""
    modes = (8, 8, 8)
    size = [10, 10, 10]

    # Test with preactivation enabled
    block = FNOBlocks(
        3,
        4,
        modes,
        n_layers=2,
        fno_skip=fno_skip,
        channel_mlp_skip=channel_mlp_skip,
        use_channel_mlp=True,
        channel_mlp_expansion=0.5,
        channel_mlp_dropout=0.0,
        preactivation=True,
    )

    x = torch.randn(2, 3, *size)
    res = block(x)

    # Check output shape
    assert res.shape == (2, 4, *size)


# ----------------------------------------------------------------------
# Optional conditioning pathway. The block threads a precomputed `mode_embedding` to its spectral conv and applies precomputed
# FiLM params (`mod_params`) around its norm sites. Both are computed upstream
# by the FNO. These tests exercise the block in isolation.


def _mode_mod(mod_type="real", k_embed_dim=8, type_k="power"):
    return {
        "enabled": True,
        "type": mod_type,
        "hidden_channels": 16,
        "k_embed_dim": k_embed_dim,
        "type_k": type_k,
    }


@pytest.mark.parametrize("norm", [None, "group_norm", "instance_norm"])
def test_block_default_ignores_mode_embedding(norm):
    """A block built without mode_modulation ignores mode_embedding."""
    torch.manual_seed(0)
    block = FNOBlocks(2, 2, (6, 6), n_layers=2, norm=norm)
    assert not block._mode_mod_enabled
    x = torch.randn(2, 2, 10, 10)
    y_plain = block(x, index=0)
    y_emb = block(x, index=0, mode_embedding=torch.randn(2, 8))
    torch.testing.assert_close(y_plain, y_emb)


@pytest.mark.parametrize("mod_type", ["real", "complex", "polar"])
def test_block_mode_modulation_forward(mod_type):
    torch.manual_seed(0)
    cond_embed_dim = 10
    block = FNOBlocks(
        2, 2, (6, 6), n_layers=2, norm="group_norm",
        mode_modulation=_mode_mod(mod_type), cond_embed_dim=cond_embed_dim,
    )
    x = torch.randn(2, 2, 10, 10)
    e = torch.randn(2, cond_embed_dim)
    y = block(x, index=0, mode_embedding=e)
    assert y.shape == (2, 2, 10, 10)
    assert torch.isfinite(y).all()


def test_block_mode_modulation_requires_embedding():
    block = FNOBlocks(
        2, 2, (6, 6), n_layers=2,
        mode_modulation=_mode_mod("real"), cond_embed_dim=8,
    )
    with pytest.raises(ValueError, match="mode_embedding"):
        block(torch.randn(2, 2, 10, 10), index=0)


def test_block_film_none_and_zero_are_identity():
    """No mod_params, empty dict, and zero scale/shift all leave output
    unchanged (x*(1+0)+0 == x, and no gate keys => no gating)."""
    torch.manual_seed(0)
    block = FNOBlocks(2, 2, (6, 6), n_layers=1, norm="group_norm")
    x = torch.randn(2, 2, 10, 10)
    y_none = block(x, index=0)
    y_empty = block(x, index=0, mod_params={})
    zero = {
        "scale1": torch.zeros(2, 2, 1, 1), "shift1": torch.zeros(2, 2, 1, 1),
        "scale2": torch.zeros(2, 2, 1, 1), "shift2": torch.zeros(2, 2, 1, 1),
    }
    y_zero = block(x, index=0, mod_params=zero)
    torch.testing.assert_close(y_none, y_empty)
    torch.testing.assert_close(y_none, y_zero)


def test_block_film_changes_output_and_backprops():
    """Non-trivial FiLM params change the output and are differentiable."""
    torch.manual_seed(0)
    block = FNOBlocks(2, 2, (6, 6), n_layers=1, norm="group_norm")
    x = torch.randn(2, 2, 10, 10)
    y_ref = block(x, index=0)

    mp = {
        "scale1": torch.randn(2, 2, 1, 1, requires_grad=True),
        "shift1": torch.randn(2, 2, 1, 1, requires_grad=True),
        "gate1": torch.randn(2, 1, 1, 1, requires_grad=True),
        "scale2": torch.randn(2, 2, 1, 1, requires_grad=True),
        "shift2": torch.randn(2, 2, 1, 1, requires_grad=True),
        "gate2": torch.randn(2, 1, 1, 1, requires_grad=True),
    }
    y = block(x, index=0, mod_params=mp)
    assert y.shape == (2, 2, 10, 10)
    assert not torch.allclose(y, y_ref)
    y.sum().backward()
    for name, p in mp.items():
        assert p.grad is not None, f"no grad for {name}"


@pytest.mark.parametrize("mod_type", ["real", "complex", "polar"])
def test_block_mode_and_film_backward(mod_type):
    """Both pathways together: grads reach x, conv params, and FiLM params."""
    torch.manual_seed(0)
    cond_embed_dim = 8
    block = FNOBlocks(
        2, 2, (6, 6), n_layers=1, norm="group_norm",
        mode_modulation=_mode_mod(mod_type), cond_embed_dim=cond_embed_dim,
    )
    x = torch.randn(2, 2, 10, 10, requires_grad=True)
    e = torch.randn(2, cond_embed_dim)
    mp = {
        "scale1": torch.randn(2, 2, 1, 1), "shift1": torch.randn(2, 2, 1, 1),
        "gate1": torch.randn(2, 1, 1, 1),
    }
    y = block(x, index=0, mode_embedding=e, mod_params=mp)
    y.sum().backward()
    assert x.grad is not None
    for name, param in block.named_parameters():
        assert param.grad is not None, f"no grad for {name}"
