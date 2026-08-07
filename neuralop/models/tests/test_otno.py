import torch
import pytest

from ..otno import OTNO
from ..readouts import ResolutionInvariantReadout

# Fixed variables
in_channels = 9
out_channels = 1
hidden_channels = 16
n_modes = [16, 16]
lifting_channel_ratio = 3
projection_channel_ratio = 3
n_layers = 4

# data parameters
n_s_sqrt = 18
n_target_points = 100


@pytest.mark.parametrize("positional_embedding", [None, "grid"])
@pytest.mark.parametrize("use_channel_mlp", [True, False])
@pytest.mark.parametrize("domain_padding", [None, 0.2])
def test_otno(positional_embedding, use_channel_mlp, domain_padding):
    """Test OTNO forward/backward with various constructor options."""
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu:0")

    # OTNO is designed for batch_size = 1
    batch_size = 1

    model = OTNO(
        n_modes=n_modes,
        hidden_channels=hidden_channels,
        in_channels=in_channels,
        out_channels=out_channels,
        lifting_channel_ratio=lifting_channel_ratio,
        projection_channel_ratio=projection_channel_ratio,
        use_channel_mlp=use_channel_mlp,
        channel_mlp_expansion=1.0 if use_channel_mlp else None,
        channel_mlp_dropout=0,
        norm="group_norm",
        n_layers=n_layers,
        positional_embedding=positional_embedding,
        domain_padding=domain_padding,
    ).to(device)

    # Create input data
    x_shape = [batch_size, in_channels, n_s_sqrt, n_s_sqrt]
    x = torch.randn(*x_shape, device=device)
    x.requires_grad_(True)

    # Create decoding indices
    n_latent_points = n_s_sqrt * n_s_sqrt
    ind_dec = torch.randint(0, n_latent_points, (n_target_points,), device=device)

    # Test forward pass
    out = model(x=x, ind_dec=ind_dec)

    # Check output size (batch_size fixed to 1)
    assert list(out.shape) == [batch_size, n_target_points]

    # Check backward pass
    assert out.isfinite().all()
    loss = out.sum()
    loss.backward()
    n_unused_params = 0
    for name, param in model.named_parameters():
        if param.grad is None:
            print(name)
            n_unused_params += 1
    assert n_unused_params == 0, f"{n_unused_params} parameters were unused!"


@pytest.mark.parametrize("n_s_sqrt", [8, 12, 18])
@pytest.mark.parametrize("n_target_points", [10, 50, 100])
def test_otno_output_shape(n_s_sqrt, n_target_points):
    """Test OTNO output shape for different grid sizes and target point counts."""
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu:0")

    batch_size = 1
    n_latent_points = n_s_sqrt * n_s_sqrt
    # ensure we don't request more target points than latent points
    n_target_points = min(n_target_points, n_latent_points)

    model = OTNO(
        n_modes=[min(8, n_s_sqrt)] * 2,
        hidden_channels=8,
        in_channels=4,
        out_channels=1,
        lifting_channel_ratio=2,
        projection_channel_ratio=2,
        n_layers=2,
        use_channel_mlp=False,
        norm="group_norm",
    ).to(device)

    x = torch.randn(batch_size, 4, n_s_sqrt, n_s_sqrt, device=device)
    ind_dec = torch.randint(0, n_latent_points, (n_target_points,), device=device)

    out = model(x=x, ind_dec=ind_dec)

    assert list(out.shape) == [batch_size, n_target_points]
    assert out.isfinite().all()


def test_otno_rejects_readout():
    """OTNO should raise ValueError when readout is passed.

    OTNO.forward removes the batch dimension and indexes into a target mesh
    before projection, so its output is incompatible with readout heads that
    expect (B, C, *spatial) input.  Silently accepting readout would let users
    think they are getting a scalar/vector QoI head when the head is never called.
    """
    with pytest.raises(ValueError, match="readout"):
        OTNO(
            n_modes=[8, 8],
            hidden_channels=8,
            in_channels=4,
            out_channels=1,
            n_layers=2,
            readout=ResolutionInvariantReadout(in_channels=1, out_dim=3),
        )


def test_otno_group_norm():
    """Test OTNO with group_norm and custom norm_groups"""
    norm_groups = 4
    model = OTNO(
        n_modes=n_modes,
        hidden_channels=hidden_channels,
        in_channels=in_channels,
        out_channels=out_channels,
        n_layers=n_layers,
        norm="group_norm",
        norm_groups=norm_groups,
    )

    for norm_layer in model.fno_blocks.norm:
        assert isinstance(norm_layer, torch.nn.GroupNorm)
        assert norm_layer.num_groups == norm_groups
