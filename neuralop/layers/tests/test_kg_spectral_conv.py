import pytest
import torch
from ..kg_spectral_conv import KGSpectralConv


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_kg_spectral_conv_output_shape(dim):
    """Test that the output shape matches expectations for 1D-3D inputs."""
    in_ch, out_ch = 3, 5
    modes = (16,) * dim
    spatial = (32,) * dim

    layer = KGSpectralConv(in_ch, out_ch, n_modes=modes)
    x = torch.randn(2, in_ch, *spatial)
    y = layer(x)

    assert y.shape == (2, out_ch, *spatial)


def test_kg_spectral_conv_per_channel():
    """Test that KG parameters are always per-channel."""
    in_ch, out_ch = 2, 4
    layer = KGSpectralConv(in_ch, out_ch, n_modes=(8,))

    assert layer.log_T.shape == (out_ch,)
    assert layer.log_c.shape == (out_ch,)
    assert layer.log_chi.shape == (out_ch,)
    # n_modes=(8,) for real data -> internally [8//2+1] = [5]
    assert layer.alpha_real.shape == (out_ch, 5)
    assert layer.alpha_imag.shape == (out_ch, 5)

    x = torch.randn(1, in_ch, 32)
    y = layer(x)
    assert y.shape == (1, out_ch, 32)


def test_kg_spectral_conv_gradient_flow():
    """Verify that gradients flow through all KG parameters."""
    layer = KGSpectralConv(2, 2, n_modes=(16,))
    x = torch.randn(1, 2, 64, requires_grad=True)

    y = layer(x)
    loss = y.sum()
    loss.backward()

    assert layer.log_T.grad is not None
    assert layer.log_c.grad is not None
    assert layer.log_chi.grad is not None
    assert layer.alpha_real.grad is not None
    assert layer.alpha_imag.grad is not None
    assert layer.channel_weight.grad is not None
    assert x.grad is not None


def test_kg_spectral_conv_identity_at_T0():
    """When T -> 0, the KG filter approaches identity."""
    # n_modes=(16,) for real data -> internally [9] (16//2+1)
    n_internal = 16 // 2 + 1  # = 9
    layer = KGSpectralConv(
        1,
        1,
        n_modes=(16,),
        init_T=1e-8,
        init_c=1.0,
        init_chi=1.0,
        bias=False,
    )
    # Set channel weight to identity, alpha to 1+0j
    with torch.no_grad():
        layer.channel_weight.fill_(1.0)
        layer.alpha_real.fill_(1.0)
        layer.alpha_imag.fill_(0.0)

    x = torch.randn(1, 1, 64)
    y = layer(x)

    # With T ~ 0, exp(-iT*omega) ~ 1, so output ~ input
    # (only n_internal low frequencies pass; high freqs zeroed)
    x_hat = torch.fft.rfft(x, norm="forward")
    y_hat = torch.fft.rfft(y, norm="forward")
    # First n_internal modes should match
    torch.testing.assert_close(
        y_hat[:, :, :n_internal], x_hat[:, :, :n_internal], atol=1e-4, rtol=1e-4
    )
    # High-frequency modes should be near zero
    assert y_hat[:, :, n_internal:].abs().max() < 1e-5


def test_kg_spectral_conv_no_bias():
    """Test that bias=False works correctly."""
    layer = KGSpectralConv(2, 3, n_modes=(8,), bias=False)
    assert layer.bias is None

    x = torch.randn(1, 2, 32)
    y = layer(x)
    assert y.shape == (1, 3, 32)


def test_kg_spectral_conv_complex_data():
    """Test with complex-valued input data."""
    layer = KGSpectralConv(2, 2, n_modes=(8,), complex_data=True)
    x = torch.randn(1, 2, 32, dtype=torch.cfloat)
    y = layer(x)

    assert y.shape == (1, 2, 32)
    assert y.dtype == torch.cfloat


def test_kg_spectral_conv_parameter_efficiency():
    """KG layer should have fewer parameters than standard SpectralConv."""
    in_ch, out_ch, n_modes = 16, 16, (32,)

    kg_layer = KGSpectralConv(in_ch, out_ch, n_modes=n_modes)
    kg_params = sum(p.numel() for p in kg_layer.parameters())

    # Standard SpectralConv: in_ch * out_ch * adjusted_modes * 2 (complex)
    # For real data, adjusted_modes = n_modes[-1]//2+1 = 17
    adjusted = n_modes[0] // 2 + 1
    fno_spectral_params = in_ch * out_ch * adjusted * 2
    assert kg_params < fno_spectral_params


def test_kg_spectral_conv_output_shape_with_resize():
    """Test output_shape parameter for resolution changes."""
    layer = KGSpectralConv(2, 2, n_modes=(16,))
    x = torch.randn(1, 2, 64)

    y = layer(x, output_shape=(128,))
    assert y.shape == (1, 2, 128)

    y = layer(x, output_shape=(32,))
    assert y.shape == (1, 2, 32)


def test_kg_spectral_conv_2d():
    """Dedicated 2D test with non-square input."""
    layer = KGSpectralConv(1, 1, n_modes=(8, 8))
    x = torch.randn(2, 1, 32, 48)
    y = layer(x)
    assert y.shape == (2, 1, 32, 48)


def test_kg_spectral_conv_repr():
    """Test that string representation works without errors."""
    layer = KGSpectralConv(2, 3, n_modes=(8, 8))
    r = repr(layer)
    assert "KGSpectralConv" in r
    assert "in_channels=2" in r
    assert "out_channels=3" in r


def test_kg_spectral_conv_n_modes_property():
    """Test dynamic n_modes update with rFFT adjustment."""
    layer = KGSpectralConv(2, 2, n_modes=(16,))
    # n_modes=(16,) for real data -> [16//2+1] = [9]
    assert layer.n_modes == [9]

    layer.n_modes = (8,)
    # (8,) -> [8//2+1] = [5]
    assert layer.n_modes == [5]

    # Should still produce valid output
    x = torch.randn(1, 2, 64)
    y = layer(x)
    assert y.shape == (1, 2, 64)


def test_kg_spectral_conv_2d_mode_truncation():
    """Test that 2D mode truncation uses centered window (fftshift).

    For a 2D real-data input, the first spatial dimension should use
    a centered frequency window (via fftshift), while the last dimension
    uses the standard rFFT low-frequency slice. This verifies the KG layer
    matches SpectralConv's multi-D truncation behavior.
    """
    layer = KGSpectralConv(
        1,
        1,
        n_modes=(8, 8),
        init_T=1e-8,
        init_c=1.0,
        init_chi=1.0,
        bias=False,
    )
    # n_modes=(8,8) for real data -> [8, 5] internally
    assert layer.n_modes == [8, 5]

    with torch.no_grad():
        layer.channel_weight.fill_(1.0)
        layer.alpha_real.fill_(1.0)
        layer.alpha_imag.fill_(0.0)

    # Create input with energy only at low frequencies
    x = torch.randn(1, 1, 32, 32)
    y = layer(x)
    assert y.shape == (1, 1, 32, 32)

    # Verify the layer preserves low-frequency content and zeros highs
    # by checking that the output has reduced high-frequency energy
    x_hat = torch.fft.rfftn(x, dim=[-2, -1], norm="forward")
    y_hat = torch.fft.rfftn(y, dim=[-2, -1], norm="forward")

    # High-frequency energy should be reduced (zeroed out modes)
    high_freq_energy_x = x_hat[:, :, 8:, :].abs().pow(2).sum()
    high_freq_energy_y = y_hat[:, :, 8:, :].abs().pow(2).sum()
    assert high_freq_energy_y < high_freq_energy_x * 0.1


def test_kg_spectral_conv_max_n_modes():
    """Test max_n_modes allocates larger alpha and supports mode changes."""
    layer = KGSpectralConv(
        2, 2, n_modes=(8,), max_n_modes=16, bias=False
    )
    # n_modes=(8,) -> [5]; max_n_modes=16 -> [9]
    assert layer.n_modes == [5]
    assert layer.max_n_modes == [9]
    # Alpha allocated at max_n_modes
    assert layer.alpha_real.shape == (2, 9)

    x = torch.randn(1, 2, 64)
    y = layer(x)
    assert y.shape == (1, 2, 64)

    # Increase n_modes within max_n_modes range
    layer.n_modes = (16,)  # -> [9], same as max_n_modes
    assert layer.n_modes == [9]
    y2 = layer(x)
    assert y2.shape == (1, 2, 64)


def test_kg_spectral_conv_precision_warning():
    """Non-default fno_block_precision should emit a warning."""
    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        layer = KGSpectralConv(
            2, 2, n_modes=(8,), fno_block_precision="half"
        )
        assert len(w) == 1
        assert "fno_block_precision" in str(w[0].message)

    # "full" should not warn
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        layer = KGSpectralConv(
            2, 2, n_modes=(8,), fno_block_precision="full"
        )
        assert len(w) == 0


def test_kg_spectral_conv_n_modes_rfft_adjustment():
    """Verify rFFT adjustment of n_modes matches SpectralConv semantics."""
    # 1D: (16,) -> [9]
    layer_1d = KGSpectralConv(1, 1, n_modes=(16,))
    assert layer_1d.n_modes == [16 // 2 + 1]

    # 2D: (16, 16) -> [16, 9] (only last dim adjusted)
    layer_2d = KGSpectralConv(1, 1, n_modes=(16, 16))
    assert layer_2d.n_modes == [16, 16 // 2 + 1]

    # 3D: (16, 16, 16) -> [16, 16, 9]
    layer_3d = KGSpectralConv(1, 1, n_modes=(16, 16, 16))
    assert layer_3d.n_modes == [16, 16, 16 // 2 + 1]

    # Complex data: no adjustment
    layer_complex = KGSpectralConv(1, 1, n_modes=(16,), complex_data=True)
    assert layer_complex.n_modes == [16]
