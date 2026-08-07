import torch
import pytest
from tensorly import tenalg

tenalg.set_backend("einsum")

# Parameterize use of torch_scatter if it is built
try:
    from torch_scatter import segment_csr

    use_torch_scatter = [True, False]
except:
    use_torch_scatter = [False]

from ..gino import GINO

# Fixed variables
in_channels = 3
out_channels = 2
projection_channels = 16
lifting_channels = 16
fno_n_modes = (8, 8, 8)

# data parameters
n_in = 100
n_out = 100
latent_density = 8
fno_ada_in_dim = 1
fno_ada_in_features = 4


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("gno_coord_dim", [2, 3])
@pytest.mark.parametrize("gno_pos_embed_type", [None, "transformer"])
@pytest.mark.parametrize(
    "gno_transform_type", ["linear", "nonlinear_kernelonly", "nonlinear"]
)
@pytest.mark.parametrize("latent_feature_dim", [None, 2])
@pytest.mark.parametrize("use_torch_scatter", use_torch_scatter)
@pytest.mark.parametrize("fno_domain_padding", [None, 0.125])
def test_gino(gno_transform_type, latent_feature_dim, gno_coord_dim, gno_pos_embed_type, batch_size, use_torch_scatter, fno_domain_padding):
    if torch.backends.cuda.is_built():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu:0")

    model = GINO(
        in_channels=in_channels,
        out_channels=out_channels,
        latent_feature_channels=latent_feature_dim,
        in_gno_radius=0.3,  # make this large to ensure neighborhoods overlap with queries on the domain
        out_gno_radius=0.3,
        gno_use_torch_scatter=use_torch_scatter,
        projection_channel_ratio=projection_channels,
        gno_coord_dim=gno_coord_dim,
        in_gno_pos_embed_type=gno_pos_embed_type,
        out_gno_pos_embed_type=gno_pos_embed_type,
        in_gno_channel_mlp_hidden_layers=[16, 16],
        out_gno_channel_mlp_hidden_layers=[16, 16],
        in_gno_transform_type=gno_transform_type,
        out_gno_transform_type=gno_transform_type,
        fno_n_modes=fno_n_modes[:gno_coord_dim],
        fno_ada_in_dim=fno_ada_in_dim,
        fno_ada_in_features=fno_ada_in_features,
        # keep the FNO model small for runtime
        fno_lifting_channel_ratio=lifting_channels,
        fno_domain_padding=fno_domain_padding,
    ).to(device)

    # create grid of latent queries on the unit cube
    latent_geom = torch.stack(
        torch.meshgrid(
            [torch.linspace(0, 1, latent_density)] * gno_coord_dim, indexing="xy"
        )
    )
    latent_geom = latent_geom.permute(*list(range(1, gno_coord_dim + 1)), 0).to(device)

    if latent_feature_dim is not None:
        latent_features_shape = [batch_size, *latent_geom.shape[:-1], latent_feature_dim]
        latent_features = torch.randn(*latent_features_shape, device=device)
    else:
        latent_features = None
    # create input geometry and output queries
    input_geom_shape = [n_in, gno_coord_dim]
    input_geom = torch.randn(*input_geom_shape, device=device)
    output_queries_shape = [n_out, gno_coord_dim]
    output_queries = torch.randn(*output_queries_shape, device=device)

    # create data and features
    x_shape = [batch_size, n_in, in_channels]
    x = torch.randn(*x_shape, device=device)
    # require and retain grad to check for backprop
    x.requires_grad_(True)

    # Test forward pass
    out = model(
        x=x,
        input_geom=input_geom,
        latent_queries=latent_geom,
        output_queries=output_queries,
        latent_features=latent_features,
        ada_in=None,
    )

    # Check output size
    assert list(out.shape) == [batch_size, n_out, out_channels]

    # Check backward pass
    assert out.isfinite().all()
    if batch_size > 1:
        loss = out[0].sum()
    else:
        loss = out.sum()
    loss.backward()
    n_unused_params = 0
    for name, param in model.named_parameters():
        if param.grad is None:
            print(name)
            n_unused_params += 1
    assert n_unused_params == 0, f"{n_unused_params} parameters were unused!"
    if batch_size > 1:
        # assert f[1:] accumulates no grad
        assert not x.grad[1:].nonzero().any()


def test_gino_group_norm():
    """Test GINO with group_norm and custom fno_norm_groups"""
    fno_norm_groups = 4
    model = GINO(
        in_channels=in_channels,
        out_channels=out_channels,
        gno_coord_dim=3,
        in_gno_radius=0.3,
        out_gno_radius=0.3,
        fno_n_modes=fno_n_modes,
        fno_hidden_channels=16,
        fno_norm="group_norm",
        fno_norm_groups=fno_norm_groups,
        gno_use_open3d=False,
    )

    for norm_layer in model.fno_blocks.norm:
        assert isinstance(norm_layer, torch.nn.GroupNorm)
        assert norm_layer.num_groups == fno_norm_groups
