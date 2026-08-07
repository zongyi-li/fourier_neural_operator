import pytest
import torch
from ..spectral_convolution_wavelet import SpectralConvWavelet 

try:
    from pytorch_wavelets import DWT1D, IDWT1D
    from pytorch_wavelets import DWT, IDWT
    import pywt
    from ptwt.conv_transform_3 import wavedec3 as ptwt_wavedec3
    from ptwt.conv_transform_3 import waverec3 as ptwt_waverec3 
except ModuleNotFoundError:
    pytest.skip("Skipping because Wavelet transform libreries are not installed", allow_module_level=True)

# ----------------------- helpers -----------------------

def _size_for_dim(n_dim: int):
    if n_dim == 1:
        return 32
    if n_dim == 2:
        return (32, 32)
    if n_dim == 3:
        return (32, 32, 32)
    raise ValueError

# ----------------------- tests -----------------------

@pytest.mark.parametrize("n_dim", [1, 2, 3])
def test_waveconvnd_forward_and_shapes(n_dim, in_channels=4, out_channels=4, level=1):
    """Smoke test: forward pass completes and output shape matches input shape.

    Mirrors the reference style: multiple dims & channel counts, minimal assertions.
    """
    size = _size_for_dim(n_dim)
    # Modes: use 'symmetric' for 1/2D, 'periodic' for 3D (ptwt expects 'periodic')
    mode = "periodic" if n_dim == 3 else "symmetric"

    layer = SpectralConvWavelet(
        in_channels=in_channels,
        out_channels=out_channels,
        level=level,
        size=size,
        n_dim=n_dim,
        wavelet="db4",
        mode=mode,
    )

    x_shape = (2, in_channels) + (size if isinstance(size, tuple) else (size,))
    x = torch.randn(*x_shape)

    y = layer(x)

    assert y.shape[0] == x.shape[0]
    assert y.shape[1] == out_channels
    assert list(y.shape[2:]) == list(x.shape[2:])


@pytest.mark.parametrize("n_dim", [1, 2, 3])
def test_waveconvnd_zero_weights_outputs_zero(n_dim, level=1):
    """If all weights are zero, output should be (near) zero after IDWT/waverec.
    Useful for catching broadcasting/misalignment errors in coefficient mixing.
    """
    
    size = _size_for_dim(n_dim)
    mode = "periodic" if n_dim == 3 else "symmetric"

    layer = SpectralConvWavelet(
        in_channels=3,
        out_channels=3,
        level=level,
        size=size,
        n_dim=n_dim,
        wavelet="db6",
        mode=mode,
    )

    # Zero all learned weights
    for name, p in layer.named_parameters():
        if name.startswith("weights"):
            with torch.no_grad():
                p.zero_()

    x_shape = (2, 3) + (size if isinstance(size, tuple) else (size,))
    x = torch.randn(*x_shape)

    y = layer(x)
    assert torch.allclose(y, torch.zeros_like(y), atol=1e-6, rtol=0)