import pytest
import torch
from tltorch import FactorizedTensor

try:
    import torch_harmonics
except ModuleNotFoundError:
    pytest.skip(
        "Skipping because torch_harmonics is not installed", allow_module_level=True
    )

from ..spherical_convolution import SphericalConv
from ..spherical_convolution import SHT


@pytest.mark.parametrize(
    "factorization", ["ComplexDense", "ComplexCP", "ComplexTucker", "ComplexTT"]
)
@pytest.mark.parametrize("implementation", ["factorized", "reconstructed"])
def test_SphericalConv(factorization, implementation):
    """Test for SphericalConv (2D only)

    Compares Factorized and Dense convolution output
    Verifies that a dense conv and factorized conv with the same weight produce the same output

    Checks the output size

    Verifies that dynamically changing the number of Fourier modes doesn't break the conv
    """
    n_modes = (6, 6)

    conv = SphericalConv(
        3,
        3,
        n_modes,
        bias=False,
        implementation=implementation,
        factorization=factorization,
    )

    conv_dense = SphericalConv(
        3, 3, n_modes, bias=False, implementation="reconstructed", factorization=None
    )

    conv_dense.weight = FactorizedTensor.from_tensor(
        conv.weight.to_tensor(), rank=None, factorization="ComplexDense"
    )
    x = torch.randn(2, 3, *(12, 12))

    res_dense = conv_dense(x)
    res = conv(x)

    torch.testing.assert_close(res_dense, res)

    # Downsample outputs
    block = SphericalConv(3, 4, n_modes, resolution_scaling_factor=0.5)

    x = torch.randn(2, 3, *(12, 12))
    res = block(x)
    assert list(res.shape[2:]) == [12 // 2, 12 // 2]

    # Upsample outputs
    block = SphericalConv(3, 4, n_modes, resolution_scaling_factor=2)

    x = torch.randn(2, 3, *(12, 12))
    res = block(x)
    assert res.shape[1] == 4  # Check out channels
    assert list(res.shape[2:]) == [12 * 2, 12 * 2]

    # Test change of grid
    block_0 = SphericalConv(4, 4, n_modes, sht_grids=["equiangular", "legendre-gauss"])

    block_1 = SphericalConv(4, 4, n_modes, sht_grids=["legendre-gauss", "equiangular"])

    x = torch.randn(2, 4, *(12, 12))
    res = block_0(x)
    res = block_1(res)
    assert res.shape[2:] == x.shape[2:]

    res = block_0.transform(x)
    res = block_1.transform(res)
    assert res.shape[2:] == x.shape[2:]


@pytest.mark.parametrize("grid", ["equiangular", "legendre-gauss"])
def test_sht(grid):
    nlat = 16
    nlon = 2 * nlat
    batch_size = 2
    if grid == "equiangular":
        mmax = nlat // 2
    else:
        mmax = nlat
    lmax = mmax
    norm = "ortho"
    dtype = torch.float32

    sht_handle = SHT(dtype=dtype)

    # Create input
    coeffs = torch.zeros(batch_size, lmax, mmax, dtype=torch.complex64)
    coeffs[:, :lmax, :mmax] = torch.randn(batch_size, lmax, mmax, dtype=torch.complex64)

    signal = sht_handle.isht(coeffs, s=(nlat, nlon), grid=grid, norm=norm).to(
        torch.float32
    )

    coeffs = sht_handle.sht(signal, s=(lmax, mmax), grid=grid, norm=norm)
    rec = sht_handle.isht(coeffs, s=(nlat, nlon), grid=grid, norm=norm)
    torch.testing.assert_close(signal, rec, rtol=1e-4, atol=1e-4)


# ----------------------------------------------------------------------
# Optional (t, l, m)-modulation pathway
# ----------------------------------------------------------------------


def _embed(dim=8):
    return {
        "type_t": "sinusoidal",
        "type_k": "power",
        "dim": dim,
        "alpha": -2.0,
        "r": 10000.0,
    }


def _mode_mod(mod_type="real", share_m=True, pre_modulate=True):
    return {
        "enabled": True,
        "type": mod_type,
        "hidden_channels": 16,
        "full_res": False,
        "share_m": share_m,
        "pre_modulate": pre_modulate,
    }


def test_sph_modulator_default_disabled():
    """Default SphericalConv has no modulator and ignores `t`."""
    layer = SphericalConv(2, 3, (6, 6), factorization="dense")
    assert layer.modulator is None
    x = torch.randn(2, 2, 12, 12)
    y_no_t = layer(x)
    y_with_t = layer(x, t=torch.zeros(2, 1))
    torch.testing.assert_close(y_no_t, y_with_t)


@pytest.mark.parametrize("mod_type", ["real", "complex", "polar"])
@pytest.mark.parametrize("share_m", [True, False])
@pytest.mark.parametrize("pre_modulate", [True, False])
def test_sph_modulated_forward_shape(mod_type, share_m, pre_modulate):
    torch.manual_seed(0)
    layer = SphericalConv(
        2,
        3,
        (6, 6),
        factorization="dense",
        embed=_embed(dim=8),
        mode_modulation=_mode_mod(mod_type, share_m=share_m, pre_modulate=pre_modulate),
    )
    x = torch.randn(2, 2, 12, 12)
    t = torch.tensor([[0.5], [1.5]])
    y = layer(x, t)
    assert y.shape == (2, 3, 12, 12)
    assert torch.isfinite(y).all()


@pytest.mark.parametrize(
    "type_t,type_k",
    [
        ("sinusoidal", "power"),
        ("power", "sinusoidal"),
        ("sinusoidal", "sinusoidal"),
        ("power", "power"),
    ],
)
def test_sph_modulated_embed_types(type_t, type_k):
    embed = _embed(dim=8)
    embed["type_t"] = type_t
    embed["type_k"] = type_k

    layer = SphericalConv(
        2,
        3,
        (6, 6),
        factorization="dense",
        embed=embed,
        mode_modulation=_mode_mod("real"),
    )
    x = torch.randn(2, 2, 12, 12)
    t = torch.tensor([[0.5], [1.5]])
    y = layer(x, t)
    assert y.shape == (2, 3, 12, 12)
    assert torch.isfinite(y).all()


@pytest.mark.parametrize("mod_type", ["real", "complex", "polar"])
def test_sph_modulated_backward_grads_all_params(mod_type):
    torch.manual_seed(0)
    layer = SphericalConv(
        2,
        3,
        (6, 6),
        factorization="dense",
        embed=_embed(dim=8),
        mode_modulation=_mode_mod(mod_type),
    )
    x = torch.randn(2, 2, 12, 12, requires_grad=True)
    t = torch.tensor([[0.5], [1.5]])
    y = layer(x, t)
    y.sum().backward()
    assert x.grad is not None
    for name, param in layer.named_parameters():
        assert param.grad is not None, f"no grad for {name}"


@pytest.mark.parametrize(
    "t_factory",
    [
        lambda B: torch.tensor(0.5),
        lambda B: torch.full((B, 1), 0.5),
    ],
)
def test_sph_t_broadcast_shapes(t_factory):
    torch.manual_seed(0)
    layer = SphericalConv(
        2,
        3,
        (6, 6),
        factorization="dense",
        embed=_embed(dim=8),
        mode_modulation=_mode_mod("real"),
    )
    B = 2
    x = torch.randn(B, 2, 12, 12)
    t = t_factory(B)
    if t.ndim == 0:
        t = t.expand(B, 1)
    y = layer(x, t)
    assert y.shape == (B, 3, 12, 12)


def test_sph_enabled_flag_false_is_inert():
    layer = SphericalConv(
        2,
        3,
        (6, 6),
        factorization="dense",
        embed=_embed(dim=8),
        mode_modulation={
            "enabled": False,
            "type": "real",
            "hidden_channels": 16,
            "full_res": False,
        },
    )
    assert layer.modulator is None
    y = layer(torch.randn(2, 2, 12, 12))
    assert y.shape == (2, 3, 12, 12)


def test_sph_missing_t_when_enabled_raises():
    layer = SphericalConv(
        2,
        3,
        (6, 6),
        factorization="dense",
        embed=_embed(dim=8),
        mode_modulation=_mode_mod("real"),
    )
    with pytest.raises(ValueError, match="t"):
        layer(torch.randn(2, 2, 12, 12))


def test_sph_modulation_without_embed_raises():
    with pytest.raises(ValueError, match="embed"):
        SphericalConv(
            2,
            3,
            (6, 6),
            factorization="dense",
            embed=None,
            mode_modulation=_mode_mod("real"),
        )


def test_sph_unknown_modulation_type_raises():
    with pytest.raises(ValueError, match="mode_modulation"):
        SphericalConv(
            2,
            3,
            (6, 6),
            factorization="dense",
            embed=_embed(dim=8),
            mode_modulation={
                "enabled": True,
                "type": "bogus",
                "hidden_channels": 16,
                "full_res": False,
            },
        )


def test_sph_share_m_changes_modulator_in_features():
    """share_m flips the modulator MLP's input width."""
    layer_share = SphericalConv(
        2,
        3,
        (6, 6),
        factorization="dense",
        embed=_embed(dim=8),
        mode_modulation=_mode_mod("real", share_m=True),
    )
    layer_full = SphericalConv(
        2,
        3,
        (6, 6),
        factorization="dense",
        embed=_embed(dim=8),
        mode_modulation=_mode_mod("real", share_m=False),
    )
    # share_m=True: 1 axis (l) + t → 2 * D
    # share_m=False: 2 axes (l, m) + t → 3 * D
    assert layer_share.modulator.in_channels == 8 * 2
    assert layer_full.modulator.in_channels == 8 * 3
