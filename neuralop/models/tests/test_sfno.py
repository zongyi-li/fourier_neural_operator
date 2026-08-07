"""Quick tests for SFNO (Spherical FNO). SFNO is FNO with SphericalConv; 2D only."""

import pytest
import torch

try:
    from neuralop.models import SFNO, t_emb_SFNO
except ImportError:
    SFNO = None
    t_emb_SFNO = None


def _embed(dim=8):
    return {
        "type_t": "sinusoidal",
        "type_k": "power",
        "dim": dim,
        "alpha": -2.0,
        "r": 10000.0,
    }


def _mode_mod(mod_type="real", pre_modulate=True, share_m=True):
    return {
        "enabled": True,
        "type": mod_type,
        "hidden_channels": 16,
        "full_res": False,
        "share_m": share_m,
        "pre_modulate": pre_modulate,
    }


def _norm_mod():
    return {
        "enabled": True,
        "hidden_channels": 16,
        "modulate1": True,
        "modulate1_gate": True,
        "modulate2": True,
        "modulate2_gate": True,
    }


@pytest.mark.skipif(
    SFNO is None, reason="SFNO not available (torch_harmonics required)"
)
@pytest.mark.parametrize("n_modes", [(4, 4), (5, 5), (4, 6), (5, 6)])
@pytest.mark.parametrize(
    "implementation,factorization,resolution_scaling_factor,rank,fixed_rank_modes",
    [
        ("factorized", "dense", None, 0.3, False),
        ("reconstructed", "cp", 1.0, 0.5, False),
        ("factorized", "tucker", None, 0.5, True),
        ("reconstructed", "tt", 1.0, 0.2, True),
    ],
)
def test_sfno_forward(
    n_modes,
    implementation,
    factorization,
    resolution_scaling_factor,
    rank,
    fixed_rank_modes,
):
    """SFNO forward over n_modes, implementation, factorization, resolution scaling, rank, fixed_rank_modes."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 2
    size = (12, 12)

    model = SFNO(
        in_channels=3,
        out_channels=1,
        hidden_channels=8,
        n_modes=n_modes,
        n_layers=2,
        implementation=implementation,
        factorization=factorization,
        resolution_scaling_factor=resolution_scaling_factor,
        rank=rank,
        fixed_rank_modes=fixed_rank_modes,
    ).to(device)
    x = torch.randn(batch_size, 3, *size, dtype=torch.float32).to(device)
    out = model(x)

    assert out.shape == (batch_size, 1, *size)


@pytest.mark.skipif(
    SFNO is None, reason="SFNO not available (torch_harmonics required)"
)
@pytest.mark.parametrize("n_modes", [(4, 4), (5, 5), (4, 6), (5, 6)])
@pytest.mark.parametrize(
    "resolution_scaling_factor,expected_scale", [(0.5, 0.5), (2.0, 2.0)]
)
def test_sfno_forward_superresolution(
    n_modes, resolution_scaling_factor, expected_scale
):
    """SFNO with single layer and resolution scaling; output spatial size scales accordingly."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 2
    size = (12, 12)

    model = SFNO(
        in_channels=3,
        out_channels=1,
        hidden_channels=8,
        n_modes=n_modes,
        n_layers=1,
        resolution_scaling_factor=resolution_scaling_factor,
    ).to(device)
    x = torch.randn(batch_size, 3, *size, dtype=torch.float32).to(device)
    out = model(x)

    expected_h = round(size[0] * expected_scale)
    expected_w = round(size[1] * expected_scale)
    assert out.shape == (batch_size, 1, expected_h, expected_w)


@pytest.mark.skipif(
    SFNO is None, reason="SFNO not available (torch_harmonics required)"
)
@pytest.mark.parametrize("n_modes", [(4, 4), (5, 5), (4, 6), (5, 6)])
@pytest.mark.parametrize("n_layers", [1, 2])
def test_sfno_backward(n_modes, n_layers):
    """SFNO backward pass runs and all parameters receive gradients."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 2
    size = (12, 12)

    model = SFNO(
        in_channels=2,
        out_channels=1,
        hidden_channels=4,
        n_modes=n_modes,
        n_layers=n_layers,
    ).to(device)

    x = torch.randn(batch_size, 2, *size, dtype=torch.float32).to(device)
    out = model(x)
    loss = out.sum()
    loss.backward()

    for param in model.parameters():
        assert param.grad is not None, "All parameters should receive gradients"


# ----------------------------------------------------------------------
# Time-conditioned SFNO
# ----------------------------------------------------------------------


@pytest.mark.skipif(
    t_emb_SFNO is None, reason="t_emb_SFNO not available (torch_harmonics required)"
)
@pytest.mark.parametrize("n_modes", [(4, 4), (5, 6)])
@pytest.mark.parametrize("mod_type", ["real", "complex", "polar"])
@pytest.mark.parametrize("pre_modulate", [True, False])
def test_t_emb_sfno_forward(n_modes, mod_type, pre_modulate):
    """t_emb_SFNO with mode + norm modulation runs and returns the right shape.

    Parametrized over pre_modulate so both branches (multiplier acting on
    in_channels before contraction, or out_channels after) are exercised.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 2
    size = (12, 12)

    model = t_emb_SFNO(
        in_channels=3,
        out_channels=1,
        hidden_channels=8,
        n_modes=n_modes,
        n_layers=2,
        embed=_embed(),
        mode_modulation=_mode_mod(mod_type, pre_modulate=pre_modulate),
        norm_modulation=_norm_mod(),
    ).to(device)

    assert model._time_conditioned is True

    x = torch.randn(batch_size, 3, *size, dtype=torch.float32).to(device)
    t = torch.tensor([[0.5], [1.5]], device=device)
    out = model(x, t=t)
    assert out.shape == (batch_size, 1, *size)


@pytest.mark.skipif(
    t_emb_SFNO is None, reason="t_emb_SFNO not available (torch_harmonics required)"
)
@pytest.mark.parametrize("n_layers", [1, 2])
def test_t_emb_sfno_backward(n_layers):
    """Backward pass; all parameters receive gradients."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 2
    size = (12, 12)

    model = t_emb_SFNO(
        in_channels=2,
        out_channels=1,
        hidden_channels=4,
        n_modes=(4, 4),
        n_layers=n_layers,
        embed=_embed(),
        mode_modulation=_mode_mod("real"),
        norm_modulation=_norm_mod(),
    ).to(device)

    x = torch.randn(batch_size, 2, *size, dtype=torch.float32).to(device)
    t = torch.tensor([[0.5], [1.5]], device=device)
    out = model(x, t=t)
    loss = out.sum()
    loss.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"no grad for {name}"


@pytest.mark.skipif(
    t_emb_SFNO is None, reason="t_emb_SFNO not available (torch_harmonics required)"
)
@pytest.mark.parametrize(
    "t_factory",
    [
        lambda B: 0.5,
        lambda B: torch.tensor(0.5),
        lambda B: torch.full((B, 1), 0.5),
    ],
)
def test_t_emb_sfno_t_broadcast(t_factory):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = t_emb_SFNO(
        in_channels=2,
        out_channels=1,
        hidden_channels=4,
        n_modes=(4, 4),
        n_layers=1,
        embed=_embed(),
        mode_modulation=_mode_mod("real"),
    ).to(device)

    B = 2
    x = torch.randn(B, 2, 12, 12, dtype=torch.float32).to(device)
    t = t_factory(B)
    if isinstance(t, torch.Tensor):
        t = t.to(device)
    out = model(x, t=t)
    assert out.shape == (B, 1, 12, 12)


@pytest.mark.skipif(
    t_emb_SFNO is None, reason="t_emb_SFNO not available (torch_harmonics required)"
)
def test_t_emb_sfno_default_t_runs():
    """t_emb_SFNO with t omitted defaults to t=1 and runs."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = t_emb_SFNO(
        in_channels=2,
        out_channels=1,
        hidden_channels=4,
        n_modes=(4, 4),
        n_layers=1,
        embed=_embed(),
        mode_modulation=_mode_mod("real"),
    ).to(device)
    x = torch.randn(2, 2, 12, 12, dtype=torch.float32).to(device)
    out = model(x)
    assert out.shape == (2, 1, 12, 12)
