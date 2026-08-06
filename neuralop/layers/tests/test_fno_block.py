import pytest
import torch
import torch.nn.functional as F
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


def test_FNOBlock_no_double_nonlinearity():
    """Regression test for issue #728: non-linearity must not be applied twice when use_channel_mlp=False."""
    call_count = 0

    def counting_nonlin(x):
        nonlocal call_count
        call_count += 1
        return F.gelu(x)

    n_layers = 3
    block = FNOBlocks(
        3,
        3,
        (8, 8),
        n_layers=n_layers,
        use_channel_mlp=False,
        non_linearity=counting_nonlin,
    )
    x = torch.randn(2, 3, 16, 16)
    for i in range(n_layers):
        call_count = 0
        x = block(x, index=i)
        expected = 1 if i < n_layers - 1 else 0
        assert call_count == expected, (
            f"Layer {i}: expected {expected} non-linearity call(s), got {call_count}"
        )
