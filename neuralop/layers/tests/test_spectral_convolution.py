import pytest
import torch
from tltorch import FactorizedTensor
from ..spectral_convolution import SpectralConv


@pytest.mark.parametrize("factorization", ["Dense", "CP", "Tucker", "TT"])
@pytest.mark.parametrize("implementation", ["factorized", "reconstructed"])
@pytest.mark.parametrize("separable", [False, True])
@pytest.mark.parametrize("dim", [1, 2, 3, 4])
@pytest.mark.parametrize("complex_data", [False, True])
def test_SpectralConv(factorization, implementation, separable, dim, complex_data):
    """Test for SpectralConv of any order

    Compares Factorized and Dense convolution output
    Verifies that a dense conv and factorized conv with the same weight produce the same output

    Checks the output size

    Verifies that dynamically changing the number of Fourier modes doesn't break the conv
    """
    modes = (10, 8, 6, 6)
    incremental_modes = (6, 6, 4, 4)
    dtype = torch.cfloat if complex_data else torch.float32

    # Test for Conv1D to Conv4D
    conv = SpectralConv(
        3,
        3,
        modes[:dim],
        bias=False,
        implementation=implementation,
        factorization=factorization,
        complex_data=complex_data,
        separable=separable,
    )

    conv_dense = SpectralConv(
        3,
        3,
        modes[:dim],
        bias=False,
        implementation="reconstructed",
        factorization=None,
        complex_data=complex_data,
    )

    x = torch.randn(2, 3, *(12,) * dim, dtype=dtype)

    assert torch.is_complex(conv.weight)
    assert torch.is_complex(conv_dense.weight)

    # this closeness test only works if the weights in full form have the same shape
    if not separable:
        conv_dense.weight = FactorizedTensor.from_tensor(
            conv.weight.to_tensor(), rank=None, factorization="ComplexDense"
        )

    res_dense = conv_dense(x)
    res = conv(x)
    res_shape = res.shape

    # this closeness test only works if the weights in full form have the same shape
    if not separable:
        torch.testing.assert_close(res_dense, res)

    # Dynamically reduce the number of modes in Fourier space
    conv.n_modes = incremental_modes[:dim]
    res = conv(x)
    assert res_shape == res.shape

    # Downsample outputs
    block = SpectralConv(3, 4, modes[:dim], resolution_scaling_factor=0.5)

    x = torch.randn(2, 3, *(12,) * dim)
    res = block(x)
    assert list(res.shape[2:]) == [12 // 2] * dim

    # Upsample outputs
    block = SpectralConv(3, 4, modes[:dim], resolution_scaling_factor=2)

    x = torch.randn(2, 3, *(12,) * dim)
    res = block(x)
    assert res.shape[1] == 4  # Check out channels
    assert list(res.shape[2:]) == [12 * 2] * dim


@pytest.mark.parametrize("enforce_hermitian_symmetry", [True, False])
@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize(
    "spatial_size", [8, 9]
)  # Even and odd: Nyquist handling differs
@pytest.mark.parametrize("resolution_scaling_factor", [None, 0.5, 2])
@pytest.mark.parametrize("modes", [(4, 4, 4), (4, 5, 7)])
def test_SpectralConv2(
    enforce_hermitian_symmetry, dim, spatial_size, modes, resolution_scaling_factor
):
    modes = modes[:dim]
    size = [spatial_size] * dim
    if resolution_scaling_factor is None:
        out_size = size
    else:
        out_size = [round(s * resolution_scaling_factor) for s in size]

    # Test with real-valued data
    conv = SpectralConv(
        3,
        4,
        modes,
        enforce_hermitian_symmetry=enforce_hermitian_symmetry,
        complex_data=False,
        resolution_scaling_factor=resolution_scaling_factor,
    )
    x = torch.randn(2, 3, *size, dtype=torch.float32)
    res = conv(x)

    assert res.shape == (2, 4, *out_size)
    assert res.dtype == torch.float32
    assert not torch.is_complex(res)


# ----------------------------------------------------------------------
# Optional (t, k)-modulation pathway
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


def test_modulator_default_disabled():
    """Default SpectralConv has no modulator and ignores `t`."""
    layer = SpectralConv(2, 3, (6, 6))
    assert layer.modulator is None
    x = torch.randn(2, 2, 10, 10)
    y_no_t = layer(x)
    y_with_t = layer(x, t=torch.zeros(2, 1))
    torch.testing.assert_close(y_no_t, y_with_t)


@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("mod_type", ["real", "complex", "polar"])
@pytest.mark.parametrize(
    "type_t,type_k",
    [
        ("sinusoidal", "power"),
        ("power", "sinusoidal"),
    ],
)
def test_modulated_forward_shape(dim, mod_type, type_t, type_k):
    torch.manual_seed(0)
    n_modes = (6,) * dim
    spatial = (10,) * dim

    embed = _embed(dim=8)
    embed["type_t"] = type_t
    embed["type_k"] = type_k

    layer = SpectralConv(
        2,
        3,
        n_modes,
        embed=embed,
        mode_modulation=_mode_mod(mod_type),
    )

    x = torch.randn(2, 2, *spatial)
    t = torch.tensor([[0.5], [1.5]])
    y = layer(x, t)
    assert y.shape == (2, 3, *spatial)
    assert torch.isfinite(y).all()


@pytest.mark.parametrize("mod_type", ["real", "complex", "polar"])
def test_modulated_backward_grads_all_params(mod_type):
    torch.manual_seed(0)
    layer = SpectralConv(
        3,
        3,
        (6, 6),
        embed=_embed(dim=8),
        mode_modulation=_mode_mod(mod_type),
    )
    x = torch.randn(2, 3, 10, 10, requires_grad=True)
    t = torch.tensor([[0.5], [1.5]])
    y = layer(x, t)
    y.sum().backward()
    assert x.grad is not None
    for name, param in layer.named_parameters():
        assert param.grad is not None, f"no grad for {name}"


@pytest.mark.parametrize(
    "t_factory",
    [
        lambda B: 0.5,
        lambda B: torch.tensor(0.5),
        lambda B: torch.full((B, 1), 0.5),
    ],
)
def test_t_broadcast_shapes(t_factory):
    torch.manual_seed(0)
    layer = SpectralConv(
        2,
        3,
        (6, 6),
        embed=_embed(dim=8),
        mode_modulation=_mode_mod("real"),
    )
    B = 2
    x = torch.randn(B, 2, 10, 10)
    t = t_factory(B)
    if not isinstance(t, torch.Tensor):
        t = torch.as_tensor(t, dtype=x.dtype)
    if t.ndim == 0:
        t = t.expand(B, 1)
    elif t.ndim == 1:
        t = t.unsqueeze(-1)
    y = layer(x, t)
    assert y.shape == (B, 3, 10, 10)


def test_enabled_flag_false_is_inert():
    """mode_modulation={'enabled': False, ...} behaves like no modulation."""
    layer = SpectralConv(
        2,
        3,
        (6, 6),
        embed=_embed(dim=8),
        mode_modulation={
            "enabled": False,
            "type": "real",
            "hidden_channels": 16,
            "full_res": False,
        },
    )
    assert layer.modulator is None
    y = layer(torch.randn(2, 2, 10, 10))
    assert y.shape == (2, 3, 10, 10)


def test_missing_t_when_enabled_raises():
    layer = SpectralConv(
        2,
        3,
        (6, 6),
        embed=_embed(dim=8),
        mode_modulation=_mode_mod("real"),
    )
    with pytest.raises(ValueError, match="t"):
        layer(torch.randn(2, 2, 10, 10))


def test_modulation_without_embed_raises():
    with pytest.raises(ValueError, match="embed"):
        SpectralConv(
            2,
            3,
            (6, 6),
            embed=None,
            mode_modulation=_mode_mod("real"),
        )


def test_unknown_modulation_type_raises():
    with pytest.raises(ValueError, match="mode_modulation"):
        SpectralConv(
            2,
            3,
            (6, 6),
            embed=_embed(dim=8),
            mode_modulation={
                "enabled": True,
                "type": "bogus",
                "hidden_channels": 16,
                "full_res": False,
            },
        )


def test_unknown_embed_type_raises():
    with pytest.raises(ValueError, match="type_t"):
        SpectralConv(
            2,
            3,
            (6, 6),
            embed={
                "type_t": "bogus",
                "type_k": "power",
                "dim": 8,
                "alpha": -2.0,
                "r": 10000.0,
            },
            mode_modulation=_mode_mod("real"),
        )
