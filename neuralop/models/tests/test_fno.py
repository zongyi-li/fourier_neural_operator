from math import prod

import pytest
import torch
import torch.nn.functional as F
from tensorly import tenalg

from neuralop.models import FNO, TFNO, t_emb_FNO, t_emb_TFNO

tenalg.set_backend("einsum")


@pytest.mark.parametrize("n_dim", [1, 2, 3, 4])
@pytest.mark.parametrize("fno_block_precision", ["full"])
@pytest.mark.parametrize("stabilizer", [None, "tanh"])
@pytest.mark.parametrize("lifting_channel_ratio", [1, 2])
@pytest.mark.parametrize("preactivation", [False, True])
@pytest.mark.parametrize("complex_data", [True, False])
@pytest.mark.parametrize("enforce_hermitian_symmetry", [True, False])
def test_fno(
    n_dim,
    fno_block_precision,
    stabilizer,
    lifting_channel_ratio,
    preactivation,
    complex_data,
    enforce_hermitian_symmetry,
):
    if torch.has_cuda:
        device = "cuda"
        s = 12
        modes = 8
        width = 12
        batch_size = 4
        n_layers = 4
    else:
        device = "cpu"
        fno_block_precision = "full"
        s = 12
        modes = 5
        width = 9
        batch_size = 3
        n_layers = 2

    dtype = torch.cfloat if complex_data else torch.float32
    rank = 0.2
    size = (s,) * n_dim
    n_modes = (modes,) * n_dim
    model = FNO(
        in_channels=3,
        out_channels=1,
        hidden_channels=width,
        n_modes=n_modes,
        rank=rank,
        fixed_rank_modes=False,
        n_layers=n_layers,
        stabilizer=stabilizer,
        lifting_channel_ratio=lifting_channel_ratio,
        preactivation=preactivation,
        complex_data=complex_data,
        fno_block_precision=fno_block_precision,
        enforce_hermitian_symmetry=enforce_hermitian_symmetry,
    ).to(device)

    in_data = torch.randn(batch_size, 3, *size, dtype=dtype).to(device)

    # Test forward pass
    out = model(in_data)

    # Check output size
    assert list(out.shape) == [batch_size, 1, *size]

    # Check backward pass
    loss = out.sum()
    # take the modulus if data is complex-valued to create grad
    if dtype == torch.cfloat:
        loss = (loss.real**2 + loss.imag**2) ** 0.5
    loss.backward()

    n_unused_params = 0
    for param in model.parameters():
        if param.grad is None:
            n_unused_params += 1
    assert n_unused_params == 0, f"{n_unused_params} parameters were unused!"


@pytest.mark.parametrize(
    "resolution_scaling_factor",
    [
        [2, 1, 1],
        [1, 2, 1],
        [1, 1, 2],
        [1, 2, 2],
        [1, 0.5, 1],
    ],
)
def test_fno_superresolution(resolution_scaling_factor):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    s = 12
    modes = 5
    hidden_channels = 9
    batch_size = 3
    n_layers = 3
    n_dim = 2
    rank = 0.2
    size = (s,) * n_dim
    n_modes = (modes,) * n_dim

    model = FNO(
        in_channels=3,
        out_channels=1,
        n_modes=n_modes,
        hidden_channels=hidden_channels,
        factorization="cp",
        implementation="reconstructed",
        rank=rank,
        resolution_scaling_factor=resolution_scaling_factor,
        n_layers=n_layers,
    ).to(device)

    print(f"{model.resolution_scaling_factor=}")

    in_data = torch.randn(batch_size, 3, *size).to(device)
    # Test forward pass
    out = model(in_data)

    # Check output size
    factor = prod(resolution_scaling_factor)

    assert list(out.shape) == [batch_size, 1] + [int(round(factor * s)) for s in size]


@pytest.mark.parametrize("norm", [None, "group_norm", "instance_norm"])
@pytest.mark.parametrize("use_channel_mlp", [True, False])
@pytest.mark.parametrize(
    "channel_mlp_skip", ["linear", "identity", "soft-gating", None]
)
@pytest.mark.parametrize("fno_skip", ["linear", "identity", "soft-gating", None])
@pytest.mark.parametrize("complex_data", [True, False])
def test_fno_advanced_params(
    norm, use_channel_mlp, channel_mlp_skip, fno_skip, complex_data
):
    """Test FNO with various advanced parameter combinations."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    s = 12
    modes = 5
    hidden_channels = 9
    batch_size = 3
    n_layers = 2
    n_dim = 2
    size = (s,) * n_dim
    n_modes = (modes,) * n_dim

    dtype = torch.cfloat if complex_data else torch.float32

    model = FNO(
        in_channels=3,
        out_channels=1,
        n_modes=n_modes,
        hidden_channels=hidden_channels,
        n_layers=n_layers,
        norm=norm,
        use_channel_mlp=use_channel_mlp,
        channel_mlp_skip=channel_mlp_skip,
        fno_skip=fno_skip,
        complex_data=complex_data,
    ).to(device)

    in_data = torch.randn(batch_size, 3, *size, dtype=dtype).to(device)

    # Test forward pass
    out = model(in_data)

    # Check output size
    assert list(out.shape) == [batch_size, 1, *size]

    # Test backward pass
    loss = out.sum()
    # take the modulus if data is complex-valued to create grad
    if dtype == torch.cfloat:
        loss = (loss.real**2 + loss.imag**2) ** 0.5
    loss.backward()


def test_fno_group_norm():
    """Test FNO with group_norm and custom norm_groups"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    modes = 5
    hidden_channels = 16
    batch_size = 2
    n_layers = 2
    n_dim = 2
    size = (12,) * n_dim
    n_modes = (modes,) * n_dim
    norm_groups = 4

    model = FNO(
        in_channels=3,
        out_channels=1,
        n_modes=n_modes,
        hidden_channels=hidden_channels,
        n_layers=n_layers,
        norm="group_norm",
        norm_groups=norm_groups,
    ).to(device)

    in_data = torch.randn(batch_size, 3, *size).to(device)

    # Test forward pass
    out = model(in_data)

    # Check output size
    assert list(out.shape) == [batch_size, 1, *size]

    # Verify norm_groups propagation
    for norm_layer in model.fno_blocks.norm:
        assert isinstance(norm_layer, torch.nn.GroupNorm)
        assert norm_layer.num_groups == norm_groups


@pytest.mark.parametrize("positional_embedding", ["grid", None])
@pytest.mark.parametrize("domain_padding", [None, 0.1, [0.1, 0.2]])
def test_fno_embedding_and_padding(positional_embedding, domain_padding):
    """Test FNO with different positional embeddings and domain padding."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    s = 12
    modes = 5
    hidden_channels = 9
    projection_channel_ratio = 2
    lifting_channel_ratio = 2
    batch_size = 3
    n_layers = 2
    n_dim = 2
    size = (s,) * n_dim
    n_modes = (modes,) * n_dim

    model = FNO(
        in_channels=3,
        out_channels=1,
        n_modes=n_modes,
        hidden_channels=hidden_channels,
        n_layers=n_layers,
        positional_embedding=positional_embedding,
        domain_padding=domain_padding,
        projection_channel_ratio=projection_channel_ratio,
        lifting_channel_ratio=lifting_channel_ratio,
    ).to(device)

    in_data = torch.randn(batch_size, 3, *size).to(device)

    # Test forward pass
    out = model(in_data)

    # Check output size
    assert list(out.shape) == [batch_size, 1, *size]


@pytest.mark.parametrize("channel_mlp_dropout", [0.0, 0.1, 0.5])
@pytest.mark.parametrize("channel_mlp_expansion", [0.25, 0.5, 1.0])
@pytest.mark.parametrize("non_linearity", [F.gelu, F.relu, F.tanh])
def test_fno_channel_mlp_params(
    channel_mlp_dropout, channel_mlp_expansion, non_linearity
):
    """Test FNO with different channel MLP parameters."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    s = 12
    modes = 5
    hidden_channels = 9
    batch_size = 3
    n_layers = 2
    n_dim = 2
    size = (s,) * n_dim
    n_modes = (modes,) * n_dim

    model = FNO(
        in_channels=3,
        out_channels=1,
        n_modes=n_modes,
        hidden_channels=hidden_channels,
        n_layers=n_layers,
        channel_mlp_dropout=channel_mlp_dropout,
        channel_mlp_expansion=channel_mlp_expansion,
        non_linearity=non_linearity,
    ).to(device)

    in_data = torch.randn(batch_size, 3, *size).to(device)

    # Test forward pass
    out = model(in_data)

    # Check output size
    assert list(out.shape) == [batch_size, 1, *size]


# ----------------------------------------------------------------------
# Time-conditioned FNO / TFNO
# ----------------------------------------------------------------------


def _embed(dim=8):
    return {
        "type_t": "sinusoidal",
        "type_k": "power",
        "dim": dim,
        "alpha": -2.0,
        "r": 10000.0,
    }


def _mode_mod(mod_type="real"):
    return {"enabled": True, "type": mod_type, "hidden_channels": 16, "full_res": False}


def _norm_mod():
    return {
        "enabled": True,
        "hidden_channels": 16,
        "modulate1": True,
        "modulate1_gate": True,
        "modulate2": True,
        "modulate2_gate": True,
    }


def test_fno_default_unchanged_no_modulated_modules():
    """Default FNO never imports or instantiates a modulated module."""
    model = FNO(
        in_channels=2,
        out_channels=2,
        n_modes=(6, 6),
        hidden_channels=8,
        n_layers=2,
    )
    for name, _ in model.named_modules():
        assert (
            "modulated" not in name.lower()
        ), f"unexpected modulated submodule in default FNO: {name}"
    assert model._time_conditioned is False


def test_fno_default_forward_bit_equal_across_seeds():
    """Two FNOs built with the same seed produce identical outputs."""

    def build_and_run():
        torch.manual_seed(0)
        model = FNO(
            in_channels=2,
            out_channels=2,
            n_modes=(6, 6),
            hidden_channels=8,
            n_layers=2,
        )
        x = torch.randn(2, 2, 12, 12)
        with torch.no_grad():
            return model(x)

    torch.testing.assert_close(build_and_run(), build_and_run())


@pytest.mark.parametrize("n_dim", [1, 2, 3])
@pytest.mark.parametrize("factorization", [None, "Tucker"])
@pytest.mark.parametrize("mod_type", ["real", "complex", "polar"])
def test_fno_time_conditioned_forward_backward(n_dim, factorization, mod_type):
    torch.manual_seed(0)
    n_modes = (6,) * n_dim
    spatial = (10,) * n_dim
    rank = 0.4 if factorization == "Tucker" else 1.0

    model = FNO(
        in_channels=2,
        out_channels=1,
        n_modes=n_modes,
        hidden_channels=8,
        n_layers=2,
        factorization=factorization,
        rank=rank,
        embed=_embed(),
        mode_modulation=_mode_mod(mod_type),
        norm_modulation=_norm_mod(),
    )
    assert model._time_conditioned is True

    x = torch.randn(2, 2, *spatial, requires_grad=True)
    t = torch.tensor([[0.5], [1.5]])
    out = model(x, t=t)
    assert list(out.shape) == [2, 1, *spatial]
    assert torch.isfinite(out).all()

    out.sum().backward()
    assert x.grad is not None
    for name, param in model.named_parameters():
        assert param.grad is not None, f"no grad for {name}"


@pytest.mark.parametrize(
    "t_factory",
    [
        lambda B: 0.5,
        lambda B: torch.tensor(0.5),
        lambda B: torch.full((B,), 0.5),
        lambda B: torch.full((B, 1), 0.5),
    ],
)
def test_fno_t_broadcast(t_factory):
    torch.manual_seed(0)
    model = FNO(
        in_channels=2,
        out_channels=1,
        n_modes=(6, 6),
        hidden_channels=8,
        n_layers=1,
        embed=_embed(),
        mode_modulation=_mode_mod("real"),
    )
    B = 2
    x = torch.randn(B, 2, 12, 12)
    out = model(x, t=t_factory(B))
    assert list(out.shape) == [B, 1, 12, 12]


@pytest.mark.parametrize("mod_type", ["real", "complex", "polar"])
def test_fno_vector_valued_conditioning(mod_type):
    """n_params > 1: t of shape (B, P) runs forward and backward."""
    torch.manual_seed(0)
    P = 5
    model = FNO(
        in_channels=2,
        out_channels=2,
        n_modes=(6, 6),
        hidden_channels=8,
        n_layers=2,
        embed=_embed(),
        mode_modulation=_mode_mod(mod_type),
        norm_modulation=_norm_mod(),
        n_params=P,
    )
    assert model._time_conditioned is True

    B = 4
    x = torch.randn(B, 2, 12, 12, requires_grad=True)
    t = torch.rand(B, P)
    out = model(x, t=t)
    assert list(out.shape) == [B, 2, 12, 12]
    assert torch.isfinite(out).all()

    out.sum().backward()
    assert x.grad is not None
    for name, param in model.named_parameters():
        assert param.grad is not None, f"no grad for {name}"


@pytest.mark.parametrize("modulation", ["mode", "norm"])
def test_fno_sinusoidal_conditioning_accepts_nonpositive_t(modulation):
    """Sinusoidal embedding must accept t <= 0 on both modulation paths.

    Regression test: sin/cos are well-defined for any real t, so the t > 0
    guard belongs only to the 'power' embedding. A stray unconditional guard
    (as once existed on the mode_modulation path) breaks signed conditioning
    parameters, which is the whole point of vector-valued conditioning.
    """
    torch.manual_seed(0)
    embed = _embed()
    embed["type_t"] = "sinusoidal"
    kwargs = dict(
        in_channels=2,
        out_channels=2,
        n_modes=(6, 6),
        hidden_channels=8,
        n_layers=1,
        embed=embed,
        n_params=3,
    )
    if modulation == "mode":
        kwargs["mode_modulation"] = _mode_mod("real")
    else:
        kwargs["norm_modulation"] = _norm_mod()
    model = FNO(**kwargs)

    x = torch.randn(2, 2, 12, 12)
    t = torch.tensor([[-1.0, 0.0, 2.0], [-0.5, -3.0, 1.0]])  # negatives and zero
    out = model(x, t=t)
    assert list(out.shape) == [2, 2, 12, 12]
    assert torch.isfinite(out).all()


def test_fno_power_conditioning_rejects_nonpositive_t():
    """The power embedding must still reject t <= 0 (per-parameter)."""
    torch.manual_seed(0)
    embed = _embed()
    embed["type_t"] = "power"
    model = FNO(
        in_channels=2,
        out_channels=1,
        n_modes=(6, 6),
        hidden_channels=8,
        n_layers=1,
        embed=embed,
        mode_modulation=_mode_mod("real"),
        n_params=2,
    )
    x = torch.randn(2, 2, 12, 12)
    with pytest.raises(ValueError, match=r"requires t > 0"):
        model(x, t=torch.tensor([[1.0, -1.0], [2.0, 3.0]]))


def test_fno_vector_conditioning_rejects_wrong_param_count():
    """A (B, P) t whose P != n_params raises a clear shape error."""
    torch.manual_seed(0)
    model = FNO(
        in_channels=2,
        out_channels=1,
        n_modes=(6, 6),
        hidden_channels=8,
        n_layers=1,
        embed=_embed(),
        mode_modulation=_mode_mod("real"),
        n_params=4,
    )
    x = torch.randn(2, 2, 12, 12)
    with pytest.raises(ValueError, match=r"shape \(2, 4\)"):
        model(x, t=torch.rand(2, 3))


def test_fno_time_conditioned_t_defaults_to_one():
    """A time-conditioned FNO with t omitted defaults to t=1 and runs."""
    torch.manual_seed(0)
    model = FNO(
        in_channels=2,
        out_channels=1,
        n_modes=(6, 6),
        hidden_channels=8,
        n_layers=1,
        embed=_embed(),
        mode_modulation=_mode_mod("real"),
    )
    x = torch.randn(2, 2, 12, 12)
    out = model(x)
    assert list(out.shape) == [2, 1, 12, 12]


def test_t_emb_fno_alias_sets_time_conditioned():
    model = t_emb_FNO(
        in_channels=2,
        out_channels=1,
        n_modes=(6, 6),
        hidden_channels=8,
        n_layers=1,
        embed=_embed(),
        mode_modulation=_mode_mod("real"),
    )
    assert isinstance(model, FNO)
    assert model._time_conditioned is True
    out = model(torch.randn(2, 2, 12, 12), t=0.5)
    assert list(out.shape) == [2, 1, 12, 12]


def test_t_emb_tfno_alias_sets_tucker_and_time_conditioned():
    model = t_emb_TFNO(
        in_channels=2,
        out_channels=1,
        n_modes=(6, 6),
        hidden_channels=8,
        n_layers=1,
        embed=_embed(),
        mode_modulation=_mode_mod("real"),
    )
    assert isinstance(model, t_emb_FNO)
    assert isinstance(model, FNO)
    assert model._time_conditioned is True
    assert model.factorization == "Tucker"
    out = model(torch.randn(2, 2, 12, 12), t=0.5)
    assert list(out.shape) == [2, 1, 12, 12]
def test_fno_conv_bias_kernel():
    """Test FNO with a local convolutional bias kernel."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    s = 12
    modes = 5
    hidden_channels = 9
    batch_size = 3
    n_layers = 2
    n_dim = 2
    size = (s,) * n_dim
    n_modes = (modes,) * n_dim
    conv_bias_kernel = 3

    model = FNO(
        in_channels=3,
        out_channels=1,
        n_modes=n_modes,
        hidden_channels=hidden_channels,
        n_layers=n_layers,
        conv_bias_kernel=conv_bias_kernel,
    ).to(device)

    in_data = torch.randn(batch_size, 3, *size).to(device)
    out = model(in_data)

    assert list(out.shape) == [batch_size, 1, *size]
    assert model.conv_bias_kernel == conv_bias_kernel
    assert model.fno_blocks.conv_bias_kernel == conv_bias_kernel

    out.sum().backward()
