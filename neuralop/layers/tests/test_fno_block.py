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

# FNOBlocks conditioning tests
@pytest.mark.parametrize("mode_modulation", [False, True])
@pytest.mark.parametrize("dim", [1, 2, 3])
def test_fno_block_film_forward_shape(dim, mode_modulation):
    torch.manual_seed(0)
    cond_dim = 8
    n_modes = (6,) * dim
    spatial = (10,) * dim

    block = FNOBlocks(4, 4, n_modes, cond_embed_dim=cond_dim,
        mode_modulation=mode_modulation,  norm="group_norm",
    )
    x = torch.randn(2, 4, *spatial)
    e = torch.randn(2, cond_dim)
    y = block(x, cond_emb=e)
    assert y.shape == (2, 4, *spatial)
    assert torch.isfinite(y).all()


def test_fno_block_film_no_cond_emb_is_identity_path():
    """When cond_emb=None, FiLM projectors are unused; output is deterministic."""
    block = FNOBlocks(4, 4, (6, 6), cond_embed_dim=8)
    x = torch.randn(2, 4, 10, 10)
    y1 = block(x, cond_emb=None)
    y2 = block(x)
    torch.testing.assert_close(y1, y2)


def test_fno_block_film_all_params_get_grad():
    torch.manual_seed(0)
    cond_dim = 8
    block = FNOBlocks(3, 3, (6, 6), cond_embed_dim=cond_dim, mode_modulation=True)
    x = torch.randn(2, 3, 10, 10, requires_grad=True)
    e = torch.randn(2, cond_dim, requires_grad=True)
    block(x, cond_emb=e).sum().backward()
    assert x.grad is not None
    assert e.grad is not None
    for name, p in block.named_parameters():
        assert p.grad is not None, f"no grad for {name}"
