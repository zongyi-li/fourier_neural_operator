import pytest
import torch
import numpy as np
from ..spectral_convolution_laplace import SpectralConvLaplace


@pytest.mark.parametrize("dim", [1, 2, 3, 4])
@pytest.mark.parametrize("in_channels", [1, 4])
@pytest.mark.parametrize("out_channels", [1, 4])
@pytest.mark.parametrize("output_shape", [False, True])
def test_SpectralConvLaplace(dim, in_channels, out_channels, output_shape):
    """Test for SpectralConvLaplace of any order

    Verifies:
    - Forward pass completes successfully
    - Output shape is correct
    """
    modes = (8, 6, 5, 4)
    dtype = torch.float32
    batch_size = 2
    in_channels = in_channels
    out_channels = out_channels

    # Create the convolutional layer
    conv = SpectralConvLaplace(
        in_channels=in_channels,
        out_channels=out_channels,
        n_modes=modes[:dim],
        bias=False,
    )

    # Basic parameter checks - Complex weights
    assert torch.is_complex(conv.weight)
    # Expected weight last dimension
    expected_weight_last_dim = sum(modes[:dim]) + int(np.prod(modes[:dim]))
    assert tuple(conv.weight.shape) == (
        in_channels,
        out_channels,
        expected_weight_last_dim,
    )

    # Input tensor
    x = torch.randn(batch_size, in_channels, *(12,) * dim, dtype=dtype)
    expected_output_shape = (
        (batch_size, out_channels, *(6,) * dim) if output_shape else None
    )

    if output_shape:
        # Forward pass
        res = conv(x, output_shape=expected_output_shape[2:])

        # Check output shape
        assert res.shape[0] == batch_size
        assert res.shape[1] == out_channels
        assert list(res.shape[2:]) == list(expected_output_shape[2:])
        # Output should be real when input is real
        assert not torch.is_complex(res)

    else:
        # Forward pass
        res = conv(x)

        # Check output shape
        assert res.shape[0] == batch_size
        assert res.shape[1] == out_channels
        assert list(res.shape[2:]) == [12] * dim
        # Output should be real when input is real
        assert not torch.is_complex(res)

    # Check linspace with explicit parameters
    from ..spectral_convolution_laplace import _compute_dt_nd

    steps = tuple([10] * dim)
    starts = [-1.0] * dim
    ends = [2.0] * dim
    dt_list, grid_coords_list = _compute_dt_nd(
        shape=steps, start_points=starts, end_points=ends
    )
    for i in range(dim):
        expected_grid = torch.linspace(starts[i], ends[i], steps=steps[i])
        assert torch.allclose(grid_coords_list[i], expected_grid)
