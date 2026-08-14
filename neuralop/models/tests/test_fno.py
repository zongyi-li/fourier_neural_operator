from math import prod

import pytest
import torch
import torch.nn.functional as F
from tensorly import tenalg

from neuralop.models import FNO, LatticeFNO

tenalg.set_backend("einsum")


def test_lattice_fno_helper_accepts_integer_output_shape():
    z = torch.tensor([1, 5])
    model = LatticeFNO(
        n=17,
        z=z,
        n_modes=(3, 3),
        in_channels=1,
        out_channels=1,
        hidden_channels=4,
        n_layers=2,
        lifting_channel_ratio=1,
        projection_channel_ratio=1,
    )
    x = torch.randn(2, 1, 17)

    out = model(x, output_shape=31)

    assert out.shape == (2, 1, 31)


@pytest.mark.parametrize("n_dim", [1, 2, 3, 4])
@pytest.mark.parametrize("fno_block_precision", ["full"])
@pytest.mark.parametrize("stabilizer", [None, "tanh"])
@pytest.mark.parametrize("lifting_channel_ratio", [1, 2])
@pytest.mark.parametrize("preactivation", [False, True])
@pytest.mark.parametrize("complex_data", [True, False])
@pytest.mark.parametrize("enforce_hermitian_symmetry", [True, False])
def test_fno(
    n_dim,
    fno_block_precision,
    stabilizer,
    lifting_channel_ratio,
    preactivation,
    complex_data,
    enforce_hermitian_symmetry,
):
    if torch.has_cuda:
        device = "cuda"
        s = 12
        modes = 8
        width = 12
        batch_size = 4
        n_layers = 4
    else:
        device = "cpu"
        fno_block_precision = "full"
        s = 12
        modes = 5
        width = 9
        batch_size = 3
        n_layers = 2

    dtype = torch.cfloat if complex_data else torch.float32
    rank = 0.2
    size = (s,) * n_dim
    n_modes = (modes,) * n_dim
    model = FNO(
        in_channels=3,
        out_channels=1,
        hidden_channels=width,
        n_modes=n_modes,
        rank=rank,
        fixed_rank_modes=False,
        n_layers=n_layers,
        stabilizer=stabilizer,
        lifting_channel_ratio=lifting_channel_ratio,
        preactivation=preactivation,
        complex_data=complex_data,
        fno_block_precision=fno_block_precision,
        enforce_hermitian_symmetry=enforce_hermitian_symmetry,
    ).to(device)

    in_data = torch.randn(batch_size, 3, *size, dtype=dtype).to(device)

    # Test forward pass
    out = model(in_data)

    # Check output size
    assert list(out.shape) == [batch_size, 1, *size]

    # Check backward pass
    loss = out.sum()
    # take the modulus if data is complex-valued to create grad
    if dtype == torch.cfloat:
        loss = (loss.real**2 + loss.imag**2) ** 0.5
    loss.backward()

    n_unused_params = 0
    for param in model.parameters():
        if param.grad is None:
            n_unused_params += 1
    assert n_unused_params == 0, f"{n_unused_params} parameters were unused!"


@pytest.mark.parametrize(
    "resolution_scaling_factor",
    [
        [2, 1, 1],
        [1, 2, 1],
        [1, 1, 2],
        [1, 2, 2],
        [1, 0.5, 1],
    ],
)
def test_fno_superresolution(resolution_scaling_factor):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    s = 12
    modes = 5
    hidden_channels = 9
    batch_size = 3
    n_layers = 3
    n_dim = 2
    rank = 0.2
    size = (s,) * n_dim
    n_modes = (modes,) * n_dim

    model = FNO(
        in_channels=3,
        out_channels=1,
        n_modes=n_modes,
        hidden_channels=hidden_channels,
        factorization="cp",
        implementation="reconstructed",
        rank=rank,
        resolution_scaling_factor=resolution_scaling_factor,
        n_layers=n_layers,
    ).to(device)

    print(f"{model.resolution_scaling_factor=}")

    in_data = torch.randn(batch_size, 3, *size).to(device)
    # Test forward pass
    out = model(in_data)

    # Check output size
    factor = prod(resolution_scaling_factor)

    assert list(out.shape) == [batch_size, 1] + [int(round(factor * s)) for s in size]


@pytest.mark.parametrize("norm", [None, "group_norm", "instance_norm"])
@pytest.mark.parametrize("use_channel_mlp", [True, False])
@pytest.mark.parametrize("channel_mlp_skip", ["linear", "identity", "soft-gating", None])
@pytest.mark.parametrize("fno_skip", ["linear", "identity", "soft-gating", None])
@pytest.mark.parametrize("complex_data", [True, False])
def test_fno_advanced_params(norm, use_channel_mlp, channel_mlp_skip, fno_skip, complex_data):
    """Test FNO with various advanced parameter combinations."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    s = 12
    modes = 5
    hidden_channels = 9
    batch_size = 3
    n_layers = 2
    n_dim = 2
    size = (s,) * n_dim
    n_modes = (modes,) * n_dim

    dtype = torch.cfloat if complex_data else torch.float32

    model = FNO(
        in_channels=3,
        out_channels=1,
        n_modes=n_modes,
        hidden_channels=hidden_channels,
        n_layers=n_layers,
        norm=norm,
        use_channel_mlp=use_channel_mlp,
        channel_mlp_skip=channel_mlp_skip,
        fno_skip=fno_skip,
        complex_data=complex_data,
    ).to(device)

    in_data = torch.randn(batch_size, 3, *size, dtype=dtype).to(device)

    # Test forward pass
    out = model(in_data)

    # Check output size
    assert list(out.shape) == [batch_size, 1, *size]

    # Test backward pass
    loss = out.sum()
    # take the modulus if data is complex-valued to create grad
    if dtype == torch.cfloat:
        loss = (loss.real**2 + loss.imag**2) ** 0.5
    loss.backward()


def test_fno_group_norm():
    """Test FNO with group_norm and custom norm_groups"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    modes = 5
    hidden_channels = 16
    batch_size = 2
    n_layers = 2
    n_dim = 2
    size = (12,) * n_dim
    n_modes = (modes,) * n_dim
    norm_groups = 4

    model = FNO(
        in_channels=3,
        out_channels=1,
        n_modes=n_modes,
        hidden_channels=hidden_channels,
        n_layers=n_layers,
        norm="group_norm",
        norm_groups=norm_groups,
    ).to(device)

    in_data = torch.randn(batch_size, 3, *size).to(device)

    # Test forward pass
    out = model(in_data)

    # Check output size
    assert list(out.shape) == [batch_size, 1, *size]

    # Verify norm_groups propagation
    for norm_layer in model.fno_blocks.norm:
        assert isinstance(norm_layer, torch.nn.GroupNorm)
        assert norm_layer.num_groups == norm_groups


@pytest.mark.parametrize("positional_embedding", ["grid", None])
@pytest.mark.parametrize("domain_padding", [None, 0.1, [0.1, 0.2]])
def test_fno_embedding_and_padding(positional_embedding, domain_padding):
    """Test FNO with different positional embeddings and domain padding."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    s = 12
    modes = 5
    hidden_channels = 9
    projection_channel_ratio = 2
    lifting_channel_ratio = 2
    batch_size = 3
    n_layers = 2
    n_dim = 2
    size = (s,) * n_dim
    n_modes = (modes,) * n_dim

    model = FNO(
        in_channels=3,
        out_channels=1,
        n_modes=n_modes,
        hidden_channels=hidden_channels,
        n_layers=n_layers,
        positional_embedding=positional_embedding,
        domain_padding=domain_padding,
        projection_channel_ratio=projection_channel_ratio,
        lifting_channel_ratio=lifting_channel_ratio,
    ).to(device)

    in_data = torch.randn(batch_size, 3, *size).to(device)

    # Test forward pass
    out = model(in_data)

    # Check output size
    assert list(out.shape) == [batch_size, 1, *size]


@pytest.mark.parametrize("channel_mlp_dropout", [0.0, 0.1, 0.5])
@pytest.mark.parametrize("channel_mlp_expansion", [0.25, 0.5, 1.0])
@pytest.mark.parametrize("non_linearity", [F.gelu, F.relu, F.tanh])
def test_fno_channel_mlp_params(channel_mlp_dropout, channel_mlp_expansion, non_linearity):
    """Test FNO with different channel MLP parameters."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    s = 12
    modes = 5
    hidden_channels = 9
    batch_size = 3
    n_layers = 2
    n_dim = 2
    size = (s,) * n_dim
    n_modes = (modes,) * n_dim

    model = FNO(
        in_channels=3,
        out_channels=1,
        n_modes=n_modes,
        hidden_channels=hidden_channels,
        n_layers=n_layers,
        channel_mlp_dropout=channel_mlp_dropout,
        channel_mlp_expansion=channel_mlp_expansion,
        non_linearity=non_linearity,
    ).to(device)

    in_data = torch.randn(batch_size, 3, *size).to(device)

    # Test forward pass
    out = model(in_data)

    # Check output size
    assert list(out.shape) == [batch_size, 1, *size]


def test_fno_conv_bias_kernel():
    """Test FNO with a local convolutional bias kernel."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    s = 12
    modes = 5
    hidden_channels = 9
    batch_size = 3
    n_layers = 2
    n_dim = 2
    size = (s,) * n_dim
    n_modes = (modes,) * n_dim
    conv_bias_kernel = 3

    model = FNO(
        in_channels=3,
        out_channels=1,
        n_modes=n_modes,
        hidden_channels=hidden_channels,
        n_layers=n_layers,
        conv_bias_kernel=conv_bias_kernel,
    ).to(device)

    in_data = torch.randn(batch_size, 3, *size).to(device)
    out = model(in_data)

    assert list(out.shape) == [batch_size, 1, *size]
    assert model.conv_bias_kernel == conv_bias_kernel
    assert model.fno_blocks.conv_bias_kernel == conv_bias_kernel

    out.sum().backward()
