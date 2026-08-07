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
# Optional time-conditioning pathway
# ----------------------------------------------------------------------


def _embed(dim=8):
    return {
        "type_t": "sinusoidal",
        "type_k": "power",
        "dim": dim,
        "alpha": -2.0,
        "r": 10000.0,
    }


def _mode_mod(mod_type="real"):
    return {"enabled": True, "type": mod_type, "hidden_channels": 16, "full_res": False}


def _norm_mod(modulate1=True, modulate1_gate=True, modulate2=True, modulate2_gate=True):
    return {
        "enabled": True,
        "hidden_channels": 16,
        "modulate1": modulate1,
        "modulate1_gate": modulate1_gate,
        "modulate2": modulate2,
        "modulate2_gate": modulate2_gate,
    }


@pytest.mark.parametrize("norm", [None, "group_norm", "instance_norm"])
def test_block_default_ignores_t(norm):
    """A FNOBlocks built without modulation dicts ignores `t`."""
    torch.manual_seed(0)
    block = FNOBlocks(2, 2, (6, 6), n_layers=2, norm=norm)
    assert block.norm_modulator is None
    assert not block._mode_mod_enabled
    assert not block._norm_mod_enabled
    x = torch.randn(2, 2, 10, 10)
    y_no_t = block(x, index=0)
    y_with_t = block(x, index=0, t=torch.zeros(2, 1))
    torch.testing.assert_close(y_no_t, y_with_t)


@pytest.mark.parametrize("mod_type", ["real", "complex", "polar"])
def test_block_mode_modulation_only_forward(mod_type):
    torch.manual_seed(0)
    block = FNOBlocks(
        2,
        2,
        (6, 6),
        n_layers=2,
        norm="group_norm",
        embed=_embed(),
        mode_modulation=_mode_mod(mod_type),
    )
    x = torch.randn(2, 2, 10, 10)
    t = torch.tensor([[0.5], [1.5]])
    y = block(x, index=0, t=t)
    assert y.shape == (2, 2, 10, 10)
    assert torch.isfinite(y).all()


def test_block_mode_modulation_requires_t():
    block = FNOBlocks(
        2,
        2,
        (6, 6),
        n_layers=2,
        embed=_embed(),
        mode_modulation=_mode_mod("real"),
    )
    with pytest.raises(ValueError, match="t"):
        block(torch.randn(2, 2, 10, 10), index=0)


@pytest.mark.parametrize("modulate1", [True, False])
@pytest.mark.parametrize("modulate1_gate", [True, False])
@pytest.mark.parametrize("modulate2", [True, False])
@pytest.mark.parametrize("modulate2_gate", [True, False])
def test_block_norm_modulation_flags(
    modulate1, modulate1_gate, modulate2, modulate2_gate
):
    if not (modulate1 or modulate1_gate or modulate2 or modulate2_gate):
        pytest.skip("All flags off → norm_modulator is None; covered elsewhere")

    torch.manual_seed(0)
    block = FNOBlocks(
        2,
        2,
        (6, 6),
        n_layers=1,
        norm="group_norm",
        embed=_embed(),
        norm_modulation=_norm_mod(
            modulate1=modulate1,
            modulate1_gate=modulate1_gate,
            modulate2=modulate2,
            modulate2_gate=modulate2_gate,
        ),
    )
    x = torch.randn(2, 2, 10, 10)
    t = torch.tensor([[0.5], [1.5]])
    y = block(x, index=0, t=t)
    assert y.shape == (2, 2, 10, 10)
    assert torch.isfinite(y).all()


def test_block_norm_modulation_all_flags_off_inert():
    """All four flags off should leave norm_modulator=None."""
    block = FNOBlocks(
        2,
        2,
        (6, 6),
        n_layers=1,
        norm="group_norm",
        embed=_embed(),
        norm_modulation=_norm_mod(
            modulate1=False,
            modulate1_gate=False,
            modulate2=False,
            modulate2_gate=False,
        ),
    )
    assert block.norm_modulator is None


def test_block_modulator_sized_by_enabled_specs():
    """A flag set to False omits its slot from the modulator MLP output."""
    block = FNOBlocks(
        2,
        2,
        (6, 6),
        n_layers=1,
        norm="group_norm",
        embed=_embed(),
        norm_modulation=_norm_mod(
            modulate1=True,
            modulate1_gate=False,
            modulate2=False,
            modulate2_gate=False,
        ),
    )
    # only scale1+shift1 should be in the output: out_channels * 2
    assert block.norm_modulator[0].out_channels == block.out_channels * 2


@pytest.mark.parametrize("mod_type", ["real", "complex", "polar"])
def test_block_both_modulations_forward(mod_type):
    torch.manual_seed(0)
    block = FNOBlocks(
        2,
        2,
        (6, 6),
        n_layers=2,
        norm="group_norm",
        embed=_embed(),
        mode_modulation=_mode_mod(mod_type),
        norm_modulation=_norm_mod(),
    )
    x = torch.randn(2, 2, 10, 10)
    t = torch.tensor([[0.5], [1.5]])
    y = block(x, index=0, t=t)
    assert y.shape == (2, 2, 10, 10)
    assert torch.isfinite(y).all()


@pytest.mark.parametrize("mod_type", ["real", "complex", "polar"])
def test_block_both_modulations_backward(mod_type):
    torch.manual_seed(0)
    block = FNOBlocks(
        2,
        2,
        (6, 6),
        n_layers=1,
        norm="group_norm",
        embed=_embed(),
        mode_modulation=_mode_mod(mod_type),
        norm_modulation=_norm_mod(),
    )
    x = torch.randn(2, 2, 10, 10, requires_grad=True)
    t = torch.tensor([[0.5], [1.5]])
    y = block(x, index=0, t=t)
    y.sum().backward()
    assert x.grad is not None
    for name, param in block.named_parameters():
        assert param.grad is not None, f"no grad for {name}"


@pytest.mark.parametrize(
    "t_factory",
    [
        lambda B: torch.tensor(0.5),
        lambda B: torch.full((B, 1), 0.5),
    ],
)
def test_block_t_broadcast_shapes(t_factory):
    torch.manual_seed(0)
    block = FNOBlocks(
        2,
        2,
        (6, 6),
        n_layers=1,
        norm="group_norm",
        embed=_embed(),
        mode_modulation=_mode_mod("real"),
        norm_modulation=_norm_mod(),
    )
    B = 2
    x = torch.randn(B, 2, 10, 10)
    t = t_factory(B)
    if t.ndim == 0:
        t = t.expand(B, 1)
    y = block(x, index=0, t=t)
    assert y.shape == (B, 2, 10, 10)


def test_block_norm_modulation_without_embed_raises():
    with pytest.raises(ValueError, match="embed"):
        FNOBlocks(
            2,
            2,
            (6, 6),
            n_layers=1,
            norm="group_norm",
            embed=None,
            norm_modulation=_norm_mod(),
        )


def test_block_preactivation_with_norm_modulation_raises():
    with pytest.raises(ValueError, match="preactivation"):
        FNOBlocks(
            2,
            2,
            (6, 6),
            n_layers=1,
            norm="group_norm",
            preactivation=True,
            embed=_embed(),
            norm_modulation=_norm_mod(),
        )
@pytest.mark.parametrize("n_dim", [1, 2, 3])
def test_FNOBlock_conv_bias_kernel(n_dim):
    """Test local convolutional bias kernels beside spectral convolution."""
    modes = (8, 8, 8)
    size = [10, 10, 10]
    conv_bias_kernel = 3

    block = FNOBlocks(
        3,
        4,
        modes[:n_dim],
        n_layers=2,
        conv_bias_kernel=conv_bias_kernel,
        fno_skip="linear",
        use_channel_mlp=False,
    )

    x = torch.randn(2, 3, *size[:n_dim])
    res = block(x)

    assert res.shape == (2, 4, *size[:n_dim])
    assert block.conv_bias_kernel == conv_bias_kernel
    assert block.fno_skips[0].kernel_size == (conv_bias_kernel,) * n_dim
