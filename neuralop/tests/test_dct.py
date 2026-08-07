"""
Tests for DCT (Discrete Cosine Transform) functions.
"""

import pytest
import torch
import numpy as np
from scipy.fft import dct as scipy_dct, idct as scipy_idct
from torch.testing import assert_close

from ..dct import dct, idct, dct_2d, idct_2d, dct_3d, idct_3d


class TestDCTvsScipy:
    """Test DCT functions against scipy implementation."""

    def test_dct_1d(self):
        """Test 1D DCT against scipy."""
        x_np = np.random.randn(10)
        x_torch = torch.tensor(x_np, dtype=torch.float64)

        # Scipy DCT
        X_np = scipy_dct(x_np, type=2, norm="ortho")

        # Our DCT
        X_torch = dct(x_torch, norm="ortho")

        assert torch.allclose(X_torch, torch.tensor(X_np), atol=1e-10)

    def test_idct_1d(self):
        """Test 1D IDCT against scipy."""
        X_np = np.random.randn(10)
        X_torch = torch.tensor(X_np, dtype=torch.float64)

        # Scipy IDCT
        x_np = scipy_idct(X_np, type=2, norm="ortho")

        # Our IDCT
        x_torch = idct(X_torch, norm="ortho")

        assert torch.allclose(x_torch, torch.tensor(x_np), atol=1e-10)

    def test_dct_2d(self):
        """Test 2D DCT against scipy."""
        x_np = np.random.randn(5, 10)
        x_torch = torch.tensor(x_np, dtype=torch.float64)

        # Scipy 2D DCT
        X_np = scipy_dct(scipy_dct(x_np, type=2, norm="ortho", axis=1), type=2, norm="ortho", axis=0)

        # Our 2D DCT
        X_torch = dct_2d(x_torch, norm="ortho")

        assert torch.allclose(X_torch, torch.tensor(X_np), atol=1e-10)

    def test_idct_2d(self):
        """Test 2D IDCT against scipy."""
        X_np = np.random.randn(5, 10)
        X_torch = torch.tensor(X_np, dtype=torch.float64)

        # Scipy 2D IDCT
        x_np = scipy_idct(scipy_idct(X_np, type=2, norm="ortho", axis=1), type=2, norm="ortho", axis=0)

        # Our 2D IDCT
        x_torch = idct_2d(X_torch, norm="ortho")

        assert torch.allclose(x_torch, torch.tensor(x_np), atol=1e-10)

    def test_dct_3d(self):
        """Test 3D DCT against scipy."""
        x_np = np.random.randn(3, 5, 10)
        x_torch = torch.tensor(x_np, dtype=torch.float64)

        # Scipy 3D DCT
        X_np = scipy_dct(scipy_dct(scipy_dct(x_np, type=2, norm="ortho", axis=2), type=2, norm="ortho", axis=1), type=2, norm="ortho", axis=0)

        # Our 3D DCT
        X_torch = dct_3d(x_torch, norm="ortho")

        assert torch.allclose(X_torch, torch.tensor(X_np), atol=1e-10)

    def test_idct_3d(self):
        """Test 3D IDCT against scipy."""
        X_np = np.random.randn(3, 5, 10)
        X_torch = torch.tensor(X_np, dtype=torch.float64)

        # Scipy 3D IDCT
        x_np = scipy_idct(scipy_idct(scipy_idct(X_np, type=2, norm="ortho", axis=2), type=2, norm="ortho", axis=1), type=2, norm="ortho", axis=0)

        # Our 3D IDCT
        x_torch = idct_3d(X_torch, norm="ortho")

        assert torch.allclose(x_torch, torch.tensor(x_np), atol=1e-10)


class TestDCTRoundtrip:
    """Test roundtrip DCT/IDCT operations."""

    def test_dct_idct_roundtrip_1d(self):
        """Test that idct(dct(x)) == x."""
        x = torch.randn(10)
        X = dct(x, norm="ortho")
        y = idct(X, norm="ortho")
        assert_close(x, y)

    def test_dct_idct_roundtrip_2d(self):
        """Test that idct_2d(dct_2d(x)) == x."""
        x = torch.randn(5, 10)
        X = dct_2d(x, norm="ortho")
        y = idct_2d(X, norm="ortho")
        assert_close(x, y)

    def test_dct_idct_roundtrip_3d(self):
        """Test that idct_3d(dct_3d(x)) == x."""
        x = torch.randn(3, 5, 10)
        X = dct_3d(x, norm="ortho")
        y = idct_3d(X, norm="ortho")
        assert_close(x, y)


if __name__ == "__main__":
    pytest.main([__file__])
