import pytest
import torch

from ..mwno import MWNO


def _run_mwno_forward_gradient_test(
    n_dim: int,
    n_modes: int,
    k: int,
    c: int,
    n_layers: int,
    L: int,
    base: str,
) -> None:
    """Forward + backward smoke test; all parameters must receive gradients."""
    if torch.cuda.is_available():
        device = "cuda"
        batch_size = 4
    else:
        device = "cpu"
        batch_size = 3

    s = 16
    s = 2 ** int(torch.log2(torch.tensor(s, dtype=torch.float)).item())

    model = MWNO(
        n_modes=n_modes,
        n_dim=n_dim,
        in_channels=3,
        out_channels=1,
        k=k,
        c=c,
        n_layers=n_layers,
        L=L,
        base=base,
    ).to(device)

    if n_dim == 1:
        in_data = torch.randn(batch_size, s, 3).to(device)
        expected_shape = [batch_size, s]
    elif n_dim == 2:
        in_data = torch.randn(batch_size, s, s, 3).to(device)
        expected_shape = [batch_size, s, s]
    else:
        t = 8
        in_data = torch.randn(batch_size, s, s, t, 3).to(device)
        expected_shape = [batch_size, s, s, t]

    out = model(in_data)
    assert list(out.shape) == expected_shape, (
        f"Expected shape {expected_shape}, got {list(out.shape)}"
    )

    loss = out.sum()
    loss.backward()

    unused = [name for name, p in model.named_parameters() if p.grad is None]
    assert not unused, f"{len(unused)} parameters had no gradient: {unused[:5]}"


@pytest.mark.parametrize("n_dim, n_modes", [(1, 16), (2, 12), (3, 8)])
@pytest.mark.parametrize("k", [4])
@pytest.mark.parametrize("c", [4, 16])
@pytest.mark.parametrize("n_layers", [1, 3])
@pytest.mark.parametrize("L", [0, 1, 2])
def test_mwno_legendre(n_dim, n_modes, k, c, n_layers, L):
    """Legendre multiwavelet base across dims, depths, and coarse-scale skips."""
    _run_mwno_forward_gradient_test(
        n_dim, n_modes, k, c, n_layers, L, "legendre",
    )


@pytest.mark.parametrize("n_dim, n_modes", [(1, 16), (2, 12), (3, 8)])
@pytest.mark.parametrize("k", [4])
@pytest.mark.parametrize("c", [4, 16])
@pytest.mark.parametrize("n_layers", [1, 3])
@pytest.mark.parametrize("L", [0, 1, 2])
def test_mwno_chebyshev(n_dim, n_modes, k, c, n_layers, L):
    """Chebyshev multiwavelet base (same coverage as legendre tests)."""
    _run_mwno_forward_gradient_test(
        n_dim, n_modes, k, c, n_layers, L, "chebyshev",
    )


def test_mwno_rejects_non_power_of_two_spatial():
    """Spatial sizes on wavelet axes must be powers of 2."""
    model = MWNO(
        n_modes=8,
        n_dim=2,
        in_channels=1,
        out_channels=1,
        k=4,
        c=4,
        n_layers=1,
        L=0,
    )
    x = torch.randn(2, 63, 63, 1)
    with pytest.raises(ValueError, match="powers of 2"):
        model(x)


def test_mwno_L_must_be_less_than_num_scales():
    """MWNO requires L < floor(log2(grid)) on the wavelet axis."""
    # 16 -> num_scales=4; L=4 is invalid
    model = MWNO(
        n_modes=8,
        n_dim=2,
        in_channels=1,
        out_channels=1,
        k=4,
        c=4,
        n_layers=1,
        L=4,
    )
    x = torch.randn(1, 16, 16, 1)
    with pytest.raises(ValueError, match="L .* must be less than num_scales"):
        model(x)


def test_mwno_spatial_check_reruns_after_reset():
    model = MWNO(
        n_modes=8,
        n_dim=2,
        in_channels=1,
        out_channels=1,
        k=4,
        c=4,
        n_layers=1,
        L=0,
    )
    model(torch.randn(1, 64, 64, 1))
    model.reset_spatial_resolution_check()
    with pytest.raises(ValueError, match="powers of 2"):
        model(torch.randn(1, 63, 63, 1))
