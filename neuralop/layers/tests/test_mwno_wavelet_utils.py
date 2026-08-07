import numpy as np
import pytest

from ..mwno_block import WaveletUtils


@pytest.mark.parametrize("k", [2, 3, 4])
def test_chebyshev_wavelets_nonzero(k):
    phi, psi1, psi2 = WaveletUtils.get_phi_psi(k, "chebyshev")
    x_left = np.linspace(0.02, 0.48, 24)
    x_right = np.linspace(0.52, 0.98, 24)
    for i in range(k):
        assert np.max(np.abs(psi1[i](x_left))) > 1e-10, f"psi1[{i}] vanishes on (0,0.5)"
        assert np.max(np.abs(psi2[i](x_right))) > 1e-10, f"psi2[{i}] vanishes on (0.5,1)"
        assert np.max(np.abs(phi[i](np.linspace(0.05, 0.95, 30)))) > 1e-10


@pytest.mark.parametrize("k", [3, 4])
def test_chebyshev_filter_bank_finite(k):
    h0, h1, g0, g1, phi0, phi1 = WaveletUtils.get_filter("chebyshev", k)
    for m in (h0, h1, g0, g1, phi0, phi1):
        assert np.isfinite(m).all()
    assert np.max(np.abs(g0)) > 1e-12
    assert np.max(np.abs(g1)) > 1e-12


def test_legendre_wavelets_unchanged_nonzero():
    _, psi1, _ = WaveletUtils.get_phi_psi(4, "legendre")
    x = np.linspace(0.02, 0.48, 24)
    assert np.max(np.abs(psi1[0](x))) > 1e-10
