from ..resample import resample
import torch
import numpy as np


def test_resample():
    a = torch.randn(10, 20, 40, 50)

    res_scale = [2, 3]
    axis = [-2, -1]

    b = resample(a, res_scale, axis)
    assert b.shape[-1] == 3 * a.shape[-1] and b.shape[-2] == 2 * a.shape[-2]

    a = torch.randn((10, 20, 40, 50, 60))

    res_scale = [0.5, 3, 4]
    axis = [-3, -2, -1]
    b = resample(a, res_scale, axis)

    assert (
        b.shape[-1] == 4 * a.shape[-1]
        and b.shape[-2] == 3 * a.shape[-2]
        and b.shape[-3] == int(0.5 * a.shape[-3])
    )


def test_resample_fourier_mode():
    test_cases = [
        ((8,), (12,), (2,)),
        ((5,), (7,), (2,)),
        ((6, 8), (10, 12), (1, -2)),
        ((5, 6, 8), (7, 9, 12), (2, 1, 2)),
        ((5, 6, 7, 8), (7, 9, 10, 12), (1, -1, 2, 2)),
    ]

    for spatial_shape, output_shape, frequencies in test_cases:
        grids = torch.meshgrid(
            *[torch.arange(n) for n in spatial_shape], indexing="ij"
        )
        phase = sum(
            f * grid / n for f, grid, n in zip(frequencies, grids, spatial_shape)
        )
        output_grids = torch.meshgrid(
            *[torch.arange(n) for n in output_shape], indexing="ij"
        )
        output_phase = sum(
            f * grid / n for f, grid, n in zip(frequencies, output_grids, output_shape)
        )

        x_complex = torch.exp(2j * torch.pi * phase).reshape(1, 1, *spatial_shape)
        expected_complex = torch.exp(2j * torch.pi * output_phase).reshape(
            1, 1, *output_shape
        )
        y_complex = resample(
            x_complex, 1.0, list(range(2, x_complex.ndim)), output_shape=output_shape
        )

        assert y_complex.shape == (1, 1, *output_shape)
        assert y_complex.dtype == x_complex.dtype
        assert torch.is_complex(y_complex)
        torch.testing.assert_close(
            y_complex, expected_complex, atol=1e-5, rtol=1e-5
        )

        if len(spatial_shape) > 2:
            x_real = torch.cos(2 * torch.pi * phase).reshape(1, 1, *spatial_shape)
            expected_real = torch.cos(2 * torch.pi * output_phase).reshape(
                1, 1, *output_shape
            )
            y_real = resample(
                x_real, 1.0, list(range(2, x_real.ndim)), output_shape=output_shape
            )

            assert y_real.shape == (1, 1, *output_shape)
            assert y_real.dtype == x_real.dtype
            assert not torch.is_complex(y_real)
            torch.testing.assert_close(y_real, expected_real, atol=1e-5, rtol=1e-5)


def test_resampling_odd_resolution():
    test_res = [15, 31, 63]
    resample_factor = 2

    for res in test_res:
        # Define discretizations
        res_increased = int(res * resample_factor)
        ticks = torch.tensor(np.linspace(0, 1, res_increased, endpoint=False), dtype=torch.float32)
        X_increased, _, _ = torch.meshgrid((ticks, ticks, ticks), indexing="ij")
        X = X_increased[::resample_factor, ::resample_factor, ::resample_factor]

        # Calculate test function for both original and increased resolution
        h = (res-1)//2
        f_eval = torch.cos(2 * torch.pi * (h * X))
        f_eval_increased = torch.cos(2 * torch.pi * (h * X_increased))

        # Resample test function and check that the error is close to zero.
        f_eval_resampled = resample(f_eval.unsqueeze(0).unsqueeze(0), res_scale=1.0, axis=list(range(2, 5)), output_shape=(res_increased, res_increased, res_increased)).squeeze(0).squeeze(0)
        relative_error = (torch.linalg.norm(f_eval_increased - f_eval_resampled) / torch.linalg.norm(f_eval_increased))
        assert(relative_error < 1e-5)
        
def test_resampling_even_resolution():
    test_res = [16, 32, 64]
    resample_factor = 2

    for res in test_res:
        # Define discretizations
        res_increased = int(res * resample_factor)
        ticks = torch.tensor(np.linspace(0, 1, res_increased, endpoint=False), dtype=torch.float32)
        X_increased, _, _ = torch.meshgrid((ticks, ticks, ticks), indexing="ij")
        X = X_increased[::resample_factor, ::resample_factor, ::resample_factor]

        # Calculate test function for both original and increased resolution
        h = res//2 - 1
        f_eval = torch.cos(2 * torch.pi * (h * X))
        f_eval_increased = torch.cos(2 * torch.pi * (h * X_increased))

        # Resample test function and check that the error is close to zero.
        f_eval_resampled = resample(f_eval.unsqueeze(0).unsqueeze(0), res_scale=1.0, axis=list(range(2, 5)), output_shape=(res_increased, res_increased, res_increased)).squeeze(0).squeeze(0)
        relative_error = (torch.linalg.norm(f_eval_increased - f_eval_resampled) / torch.linalg.norm(f_eval_increased))
        assert(relative_error < 1e-5)

def test_resampling_complex_odd_resolution():
    test_res = [15, 31, 63]
    resample_factor = 2

    for res in test_res:
        # Define discretizations
        res_increased = int(res * resample_factor)
        ticks = torch.tensor(np.linspace(0, 1, res_increased, endpoint=False), dtype=torch.float32)
        X_increased, _, _ = torch.meshgrid((ticks, ticks, ticks), indexing="ij")
        X = X_increased[::resample_factor, ::resample_factor, ::resample_factor]

        # Calculate test function for both original and increased resolution
        h = (res-1)//2
        f_eval = torch.cos(2 * torch.pi * (h * X)) + 1j * torch.sin(2 * torch.pi * (h * X))
        f_eval_increased = torch.cos(2 * torch.pi * (h * X_increased)) + 1j * torch.sin(2 * torch.pi * (h * X_increased))

        # Resample 'f_eval'
        f_eval_resampled = resample(f_eval.unsqueeze(0).unsqueeze(0), res_scale=1.0, axis=list(range(2, 5)), output_shape=(res_increased, res_increased, res_increased)).squeeze(0).squeeze(0)
        
        # The resampled function should be complex valued
        assert(torch.norm(f_eval_resampled.imag) > 1e-5)

        # Check that the error is close to zero.
        relative_error = (torch.linalg.norm(f_eval_increased - f_eval_resampled) / torch.linalg.norm(f_eval_increased))
        assert(relative_error < 1e-5)
    
    

def test_resampling_complex_even_resolution():
    test_res = [16, 32, 64]
    resample_factor = 2

    for res in test_res:
        # Define discretizations
        res_increased = int(res * resample_factor)
        ticks = torch.tensor(np.linspace(0, 1, res_increased, endpoint=False), dtype=torch.float32)
        X_increased, _, _ = torch.meshgrid((ticks, ticks, ticks), indexing="ij")
        X = X_increased[::resample_factor, ::resample_factor, ::resample_factor]

        # Calculate test function for both original and increased resolution
        h = res//2 - 1
        f_eval = torch.cos(2 * torch.pi * (h * X)) + 1j * torch.sin(2 * torch.pi * (h * X))
        f_eval_increased = torch.cos(2 * torch.pi * (h * X_increased)) + 1j * torch.sin(2 * torch.pi * (h * X_increased))

        # Resample 'f_eval'
        f_eval_resampled = resample(f_eval.unsqueeze(0).unsqueeze(0), res_scale=1.0, axis=list(range(2, 5)), output_shape=(res_increased, res_increased, res_increased)).squeeze(0).squeeze(0)
        
        # The resampled function should be complex valued
        assert(torch.norm(f_eval_resampled.imag) > 1e-5)

        # Check that the error is close to zero.
        relative_error = (torch.linalg.norm(f_eval_increased - f_eval_resampled) / torch.linalg.norm(f_eval_increased))
        assert(relative_error < 1e-5)
