import pytest
import torch

from neuralop.models import FNO, ResolutionInvariantReadout


@pytest.mark.parametrize("n_dim", [1, 2, 3])
@pytest.mark.parametrize("reduce", ["mean", "integral"])
def test_fno_resolution_invariant_readout_shapes(reduce, n_dim):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    out_dim = 7
    n_modes = (8,) * n_dim
    shape_1 = (12,) * n_dim
    shape_2 = (16,) * n_dim
    model = FNO(
        n_modes=n_modes,
        in_channels=2,
        out_channels=4,
        hidden_channels=12,
        n_layers=2,
        readout=ResolutionInvariantReadout(
            in_channels=4,
            out_dim=out_dim,
            reduce=reduce,
            measure_per_dim=[1.0] * n_dim,
        ),
    ).to(device)

    x_1 = torch.randn(3, 2, *shape_1, device=device)
    x_2 = torch.randn(3, 2, *shape_2, device=device)

    y_1 = model(x_1)
    y_2 = model(x_2)

    assert y_1.shape == (3, out_dim)
    assert y_2.shape == (3, out_dim)


@pytest.mark.parametrize("n_dim", [1, 2, 3])
def test_fno_resolution_invariant_readout_backward(n_dim):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    n_modes = (8,) * n_dim
    shape = (12,) * n_dim
    model = FNO(
        n_modes=n_modes,
        in_channels=2,
        out_channels=4,
        hidden_channels=12,
        n_layers=2,
        readout=ResolutionInvariantReadout(
            in_channels=4,
            out_dim=5,
            reduce="mean",
            head="mlp",
            mlp_hidden_dim=8,
        ),
    ).to(device)

    x = torch.randn(4, 2, *shape, device=device)
    target = torch.randn(4, 5, device=device)

    out = model(x)
    loss = (out - target).pow(2).mean()
    loss.backward()

    assert all(param.grad is not None for param in model.readout.head.parameters())


def test_resolution_invariant_readout_non_unit_measure():
    """integral reduce with non-unit measure_per_dim should scale pre-head pooling by domain volume.

    The measure scales the spatially-pooled tensor before the linear head, so the
    relationship is head(6 * pooled) not 6 * head(pooled) — these differ by the bias
    term. We zero the bias to isolate the scaling behaviour.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    readout_unit = ResolutionInvariantReadout(
        in_channels=4, out_dim=3, reduce="integral", measure_per_dim=[1.0, 1.0]
    ).to(device)
    readout_scaled = ResolutionInvariantReadout(
        in_channels=4, out_dim=3, reduce="integral", measure_per_dim=[2.0, 3.0]
    ).to(device)

    # Share weights and zero bias so head is purely linear: head(6x) == 6 * head(x)
    readout_scaled.head.load_state_dict(readout_unit.head.state_dict())
    with torch.no_grad():
        readout_unit.head.bias.zero_()
        readout_scaled.head.bias.zero_()

    x = torch.randn(2, 4, 8, 8, device=device)
    with torch.no_grad():
        out_unit = readout_unit(x)
        out_scaled = readout_scaled(x)

    # domain volume = 2.0 * 3.0 = 6.0, so scaled output should be exactly 6x unit
    torch.testing.assert_close(out_scaled, out_unit * 6.0)


def test_resolution_invariant_readout_measure_length_mismatch():
    """measure_per_dim with wrong length should raise ValueError on forward."""
    readout = ResolutionInvariantReadout(
        in_channels=4, out_dim=3, reduce="integral", measure_per_dim=[1.0, 1.0, 1.0]
    )
    x = torch.randn(2, 4, 8, 8)  # 2 spatial dims, but measure_per_dim has 3
    with pytest.raises(ValueError, match="measure_per_dim"):
        readout(x)


def test_resolution_invariant_readout_copies_measure_sequence():
    measure_per_dim = [1.0, 2.0]
    readout = ResolutionInvariantReadout(
        in_channels=4, out_dim=3, reduce="integral", measure_per_dim=measure_per_dim
    )

    measure_per_dim.append(3.0)
    x = torch.randn(2, 4, 8, 8)

    out = readout(x)
    assert out.shape == (2, 3)
    assert readout.measure_per_dim == [1.0, 2.0]


def test_resolution_invariant_readout_state_dict_keys_match():
    scalar = ResolutionInvariantReadout(
        in_channels=4, out_dim=3, reduce="integral", measure_per_dim=2.0
    )
    sequence = ResolutionInvariantReadout(
        in_channels=4, out_dim=3, reduce="integral", measure_per_dim=[2.0, 2.0]
    )

    assert set(scalar.state_dict().keys()) == set(sequence.state_dict().keys())


def test_resolution_invariant_readout_cross_load_scalar_sequence_raises():
    scalar = ResolutionInvariantReadout(
        in_channels=4, out_dim=3, reduce="integral", measure_per_dim=2.0
    )
    sequence = ResolutionInvariantReadout(
        in_channels=4, out_dim=3, reduce="integral", measure_per_dim=[2.0, 2.0]
    )
    with pytest.raises(RuntimeError, match="size mismatch"):
        sequence.load_state_dict(scalar.state_dict())


def test_resolution_invariant_readout_mean_vs_integral_differ():
    """mean and integral reduce should produce numerically different outputs."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    readout_mean = ResolutionInvariantReadout(
        in_channels=4, out_dim=3, reduce="mean", measure_per_dim=[2.0, 2.0]
    ).to(device)
    readout_integral = ResolutionInvariantReadout(
        in_channels=4, out_dim=3, reduce="integral", measure_per_dim=[2.0, 2.0]
    ).to(device)

    # Share weights so only the reduce mode differs
    readout_integral.head.load_state_dict(readout_mean.head.state_dict())

    x = torch.randn(2, 4, 8, 8, device=device)
    with torch.no_grad():
        out_mean = readout_mean(x)
        out_integral = readout_integral(x)

    assert not torch.allclose(
        out_mean, out_integral
    ), "mean and integral outputs should differ when measure_per_dim != 1"


@pytest.mark.parametrize("n_dim", [1, 2, 3])
@pytest.mark.parametrize("reduce", ["mean", "integral"])
def test_resolution_invariant_readout_constant_field(n_dim, reduce):
    """Readout of a constant field should be identical across resolutions.

    A spatially-constant input has the same pooled value regardless of grid
    size for both ``mean`` and ``integral`` reduction when the physical domain
    measure is held fixed. This validates the invariance semantics directly,
    not just output shapes.
    """
    torch.manual_seed(0)
    in_channels = 4
    out_dim = 3
    readout = ResolutionInvariantReadout(
        in_channels=in_channels,
        out_dim=out_dim,
        reduce=reduce,
        measure_per_dim=([0.5] * n_dim if reduce == "integral" else [1.0] * n_dim),
    )
    readout.eval()

    # Build a batch of constant fields: same value at every spatial location.
    value = torch.randn(2, in_channels)  # (B, C)

    outputs = []
    for size in (8, 16, 32):
        spatial = (size,) * n_dim
        # Expand value to (B, C, *spatial)
        x = value.reshape(2, in_channels, *([1] * n_dim)).expand(
            2, in_channels, *spatial
        )
        with torch.no_grad():
            outputs.append(readout(x))

    for out in outputs[1:]:
        torch.testing.assert_close(out, outputs[0])


def test_fno_complex_data_with_readout_raises():
    """FNO(complex_data=True, readout=...) should raise ValueError at construction."""
    with pytest.raises(ValueError, match="complex_data"):
        FNO(
            n_modes=(4, 4),
            in_channels=2,
            out_channels=4,
            hidden_channels=8,
            n_layers=1,
            complex_data=True,
            readout=ResolutionInvariantReadout(in_channels=4, out_dim=1),
        )


@pytest.mark.parametrize("n_dim", [1, 2, 3])
def test_fno_without_readout_preserves_field_output_shape(n_dim):
    model = FNO(
        n_modes=(8,) * n_dim,
        in_channels=2,
        out_channels=5,
        hidden_channels=12,
        n_layers=2,
        readout=None,
    )
    x = torch.randn(3, 2, *([16] * n_dim))
    y = model(x)
    assert y.shape == (3, 5, *([16] * n_dim))


def test_fno_readout_channel_mismatch_raises():
    """FNO should raise ValueError when readout.in_channels != out_channels."""
    with pytest.raises(ValueError, match="in_channels"):
        FNO(
            n_modes=(4, 4),
            in_channels=2,
            out_channels=4,
            hidden_channels=8,
            n_layers=1,
            readout=ResolutionInvariantReadout(in_channels=8, out_dim=1),
        )


def test_resolution_invariant_readout_invalid_reduce_raises():
    """Invalid reduce value should raise ValueError at construction."""
    with pytest.raises(ValueError, match="reduce"):
        ResolutionInvariantReadout(in_channels=4, out_dim=3, reduce="sum")


def test_resolution_invariant_readout_invalid_head_raises():
    """Invalid head value should raise ValueError at construction."""
    with pytest.raises(ValueError, match="head"):
        ResolutionInvariantReadout(in_channels=4, out_dim=3, head="transformer")


def test_resolution_invariant_readout_low_rank_input_raises():
    """Input with fewer than 3 dimensions should raise ValueError on forward."""
    readout = ResolutionInvariantReadout(in_channels=4, out_dim=3)
    x = torch.randn(2, 4)  # missing spatial dim
    with pytest.raises(ValueError, match=r"\(B, C, \*spatial\)"):
        readout(x)


def test_resolution_invariant_readout_complex_input_raises():
    """Complex input should raise a clear ValueError on forward."""
    readout = ResolutionInvariantReadout(in_channels=4, out_dim=3)
    x = torch.randn(2, 4, 8, 8, dtype=torch.cfloat)
    with pytest.raises(ValueError, match="complex"):
        readout(x)
