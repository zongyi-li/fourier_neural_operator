import torch
from torch.autograd import grad
import pytest
from tensorly import tenalg

tenalg.set_backend("einsum")

from ..fnogno import FNOGNO

# Parameterize use of torch_scatter if it is built
try:
    from torch_scatter import segment_csr

    use_torch_scatter = [True, False]
except:
    use_torch_scatter = [False]


@pytest.mark.parametrize(
    "gno_transform_type", ["linear", "nonlinear_kernelonly", "nonlinear"]
)
@pytest.mark.parametrize("fno_n_modes", [(8,), (8, 8), (8, 8, 8)])
@pytest.mark.parametrize("gno_batched", [False, True])
@pytest.mark.parametrize("gno_pos_embed_type", [None, "transformer"])
@pytest.mark.parametrize("fno_norm", [None, "ada_in"])
@pytest.mark.parametrize("use_torch_scatter", use_torch_scatter)
@pytest.mark.parametrize("fno_domain_padding", [None, 0.125])
def test_fnogno(gno_transform_type, fno_n_modes, gno_batched, gno_pos_embed_type, fno_norm, use_torch_scatter, fno_domain_padding):
    if torch.has_cuda:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu:0")

    in_channels = 3
    out_channels = 2
    batch_size = 4
    n_dim = len(fno_n_modes)

    model = FNOGNO(
        in_channels=in_channels,
        out_channels=out_channels,
        gno_radius=0.2,
        gno_coord_dim=n_dim,
        gno_pos_embed_type=gno_pos_embed_type,
        gno_transform_type=gno_transform_type,
        gno_batched=gno_batched,
        fno_n_modes=fno_n_modes,
        fno_norm=fno_norm,
        fno_ada_in_features=4,
        gno_use_open3d=False,
        gno_use_torch_scatter=use_torch_scatter,
        fno_domain_padding=fno_domain_padding,
    ).to(device)

    in_p_shape = [
        32,
    ] * n_dim
    in_p_shape.append(n_dim)
    in_p = torch.randn(*in_p_shape, device=device)

    out_p = torch.randn(100, n_dim, device=device)

    f_shape = [
        32,
    ] * n_dim
    f_shape.append(in_channels)

    # repeat along batch dim if batched
    if gno_batched:
        f_shape = [batch_size] + f_shape

    f = torch.randn(*f_shape, device=device)
    # require and retain grad to check for backprop
    f.requires_grad_(True)
    # f.retain_grad()

    ada_in = torch.randn(1, device=device)

    # Test forward pass
    out = model(in_p, out_p, f, ada_in)

    # Check output size
    if gno_batched:
        assert list(out.shape) == [batch_size, 100, out_channels]
    else:
        assert list(out.shape) == [100, out_channels]

    # Check backward pass
    if gno_batched:
        loss = out[0].sum()
    else:
        loss = out.sum()
    loss.backward()
    n_unused_params = 0
    for param in model.parameters():
        if param.grad is None:
            n_unused_params += 1
    assert n_unused_params == 0, f"{n_unused_params} parameters were unused!"
    if gno_batched:
        # assert f[1:] accumulates no grad
        assert not f.grad[1:].nonzero().any()


def test_fnogno_group_norm():
    """Test FNOGNO with group_norm and custom fno_norm_groups"""
    fno_norm_groups = 4
    model = FNOGNO(
        in_channels=3,
        out_channels=2,
        gno_radius=0.2,
        gno_coord_dim=2,
        fno_n_modes=(8, 8),
        fno_hidden_channels=16,
        fno_norm="group_norm",
        fno_norm_groups=fno_norm_groups,
        gno_use_open3d=False,
    )

    for norm_layer in model.fno_blocks.norm:
        assert isinstance(norm_layer, torch.nn.GroupNorm)
        assert norm_layer.num_groups == fno_norm_groups
