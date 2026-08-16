import pytest
import torch
from tltorch import FactorizedTensor
from ..spectral_convolution import SpectralConv, ConditionalSpectralConv


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
@pytest.mark.parametrize("spatial_size", [8, 9])  # Even and odd: Nyquist handling differs
@pytest.mark.parametrize("resolution_scaling_factor", [None, 0.5, 2])
@pytest.mark.parametrize("modes", [(4, 4, 4), (4, 5, 7)])
def test_SpectralConv2(enforce_hermitian_symmetry, dim, spatial_size, modes, resolution_scaling_factor):
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


# ConditionalSpectralConv tests

@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("modulation_type", ["real", "complex", "polar"])
@pytest.mark.parametrize("type_k", ["power", "sinusoidal"])
def test_conditional_spectral_conv_forward_shape(dim, modulation_type, type_k):
    torch.manual_seed(0)
    n_modes = (6,) * dim
    spatial = (10,) * dim
    cond_dim = 8

    layer = ConditionalSpectralConv(
        2, 3, n_modes,
        condition_embedding_channels=cond_dim,
        modulation_type=modulation_type,
        type_k=type_k,
    )
    x = torch.randn(2, 2, *spatial)
    e = torch.randn(2, cond_dim)
    y = layer(x, condition_embedding=e)
    assert y.shape == (2, 3, *spatial)
    assert torch.isfinite(y).all()

def test_conditional_spectral_conv_no_embedding_raises():
    layer = ConditionalSpectralConv(2, 3, (6, 6), condition_embedding_channels=8)
    with pytest.raises(ValueError, match="condition_embedding"):
        layer(torch.randn(2, 2, 10, 10), condition_embedding=None)

def test_conditional_spectral_conv_wrong_width_raises():
    layer = ConditionalSpectralConv(2, 3, (6, 6), condition_embedding_channels=8)
    with pytest.raises(ValueError, match="condition_embedding_channels"):
        layer(torch.randn(2, 2, 10, 10), condition_embedding=torch.randn(2, 5))

def test_conditional_spectral_conv_backward():
    """ConditionalSpectralConv is affine in x, so analytic grad == finite difference."""
    torch.manual_seed(0)
    cond_dim = 6
    layer = ConditionalSpectralConv(2, 2, (6, 6), condition_embedding_channels=cond_dim, modulation_type="polar")
    x = torch.randn(2, 2, 8, 8, requires_grad=True)
    e = torch.randn(2, cond_dim)
    w = torch.randn(2, 2, 8, 8)
    v = torch.randn(2, 2, 8, 8)
    loss = (layer(x, condition_embedding=e) * w).sum()
    (grad_x,) = torch.autograd.grad(loss, x)
    analytic = (grad_x * v).sum()
    eps = 1.0
    with torch.no_grad():
        lp = (layer(x + eps * v, condition_embedding=e) * w).sum()
        lm = (layer(x - eps * v, condition_embedding=e) * w).sum()
    fd = (lp - lm) / (2 * eps)
    torch.testing.assert_close(analytic, fd, rtol=1e-4, atol=1e-4)
