import pytest
import torch
from tltorch import FactorizedTensor
from ..index_sets import (
    ExplicitIndexSet,
    HyperRectangleIndexSet,
    HyperbolicCrossIndexSet,
)
from ..embeddings import LatticeEmbedding
from ..spectral_convolution import SpectralConv
from ..spectral_transforms import Rank1LatticeFFT
from ..lattice import (
    lattice_to_regular_grid,
    rank1_lattice_points,
    regular_grid_to_lattice,
)


@pytest.mark.parametrize("factorization", ["Dense", "CP", "Tucker", "TT"])
@pytest.mark.parametrize("implementation", ["factorized", "reconstructed"])
@pytest.mark.parametrize("separable", [False, True])
@pytest.mark.parametrize("dim", [1, 2, 3, 4])
@pytest.mark.parametrize("complex_data", [False, True])
def test_SpectralConv(factorization, implementation, separable, dim, complex_data):
    """Test for SpectralConv of any order

    Compares Factorized and Dense convolution output
    Verifies that a dense conv and factorized conv with the same weight produce the same output

    Checks the output size

    Verifies that dynamically changing the number of Fourier modes doesn't break the conv
    """
    modes = (10, 8, 6, 6)
    incremental_modes = (6, 6, 4, 4)
    dtype = torch.cfloat if complex_data else torch.float32

    # Test for Conv1D to Conv4D
    conv = SpectralConv(
        3,
        3,
        modes[:dim],
        bias=False,
        implementation=implementation,
        factorization=factorization,
        complex_data=complex_data,
        separable=separable,
    )

    conv_dense = SpectralConv(
        3,
        3,
        modes[:dim],
        bias=False,
        implementation="reconstructed",
        factorization=None,
        complex_data=complex_data,
    )

    x = torch.randn(2, 3, *(12,) * dim, dtype=dtype)

    assert torch.is_complex(conv.weight)
    assert torch.is_complex(conv_dense.weight)

    # this closeness test only works if the weights in full form have the same shape
    if not separable:
        conv_dense.weight = FactorizedTensor.from_tensor(
            conv.weight.to_tensor(), rank=None, factorization="ComplexDense"
        )

    res_dense = conv_dense(x)
    res = conv(x)
    res_shape = res.shape

    # this closeness test only works if the weights in full form have the same shape
    if not separable:
        torch.testing.assert_close(res_dense, res)

    # Dynamically reduce the number of modes in Fourier space
    conv.n_modes = incremental_modes[:dim]
    res = conv(x)
    assert res_shape == res.shape

    # Downsample outputs
    block = SpectralConv(3, 4, modes[:dim], resolution_scaling_factor=0.5)

    x = torch.randn(2, 3, *(12,) * dim)
    res = block(x)
    assert list(res.shape[2:]) == [12 // 2] * dim

    # Upsample outputs
    block = SpectralConv(3, 4, modes[:dim], resolution_scaling_factor=2)

    x = torch.randn(2, 3, *(12,) * dim)
    res = block(x)
    assert res.shape[1] == 4  # Check out channels
    assert list(res.shape[2:]) == [12 * 2] * dim


@pytest.mark.parametrize("enforce_hermitian_symmetry", [True, False])
@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("spatial_size", [8, 9])  # Even and odd: Nyquist handling differs
@pytest.mark.parametrize("resolution_scaling_factor", [None, 0.5, 2])
@pytest.mark.parametrize("modes", [(4, 4, 4), (4, 5, 7)])
def test_SpectralConv2(enforce_hermitian_symmetry, dim, spatial_size, modes, resolution_scaling_factor):
    modes = modes[:dim]
    size = [spatial_size] * dim
    if resolution_scaling_factor is None:
        out_size = size
    else:
        out_size = [round(s * resolution_scaling_factor) for s in size]

    # Test with real-valued data
    conv = SpectralConv(
        3,
        4,
        modes,
        enforce_hermitian_symmetry=enforce_hermitian_symmetry,
        complex_data=False,
        resolution_scaling_factor=resolution_scaling_factor,
    )
    x = torch.randn(2, 3, *size, dtype=torch.float32)
    res = conv(x)

    assert res.shape == (2, 4, *out_size)
    assert res.dtype == torch.float32
    assert not torch.is_complex(res)


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_SpectralConv_complex_output_shape_and_dtype(dim):
    conv = SpectralConv(
        2,
        4,
        (4,) * dim,
        complex_data=True,
        factorization=None,
    )
    x = torch.randn(3, 2, *((8,) * dim), dtype=torch.cfloat)

    y = conv(x)

    assert y.shape == (3, 4, *((8,) * dim))
    assert y.dtype == torch.cfloat
    assert torch.is_complex(y)


def test_SpectralConv_complex_centers_last_fft_dimension():
    size = 8
    conv = SpectralConv(
        1,
        1,
        (3, 3),
        complex_data=True,
        factorization=None,
        bias=False,
    )
    with torch.no_grad():
        conv.weight.tensor.zero_()
        conv.weight.tensor[0, 0, 1, 0] = 1 # For in_channel=0, out_channel=0, mode [0, -1]

    coordinates = torch.arange(size)
    # This function is constant in the first dimension and mode -1 in the second dimension
    signal = torch.exp(-2j * torch.pi * coordinates / size)
    x = signal.reshape(1, 1, 1, size).expand(1, 1, size, size).contiguous()

    y = conv(x)

    torch.testing.assert_close(y, x)


def test_SpectralConv_complex_odd_max_n_modes_difference():
    size = 8
    # We allocate weights for 5 modes per dimension but only activate 4 modes.
    # The difference max_n_modes - n_modes is odd, so the weight slices should
    # remove one mode from the positive side instead of accidentally removing no modes.
    conv = SpectralConv(
        1,
        1,
        (4, 4),
        max_n_modes=(5, 5),
        complex_data=True,
        factorization=None,
        bias=False,
    )
    with torch.no_grad():
        conv.weight.tensor.zero_()
        # For in_channel=0, out_channel=0, active mode [0, -1].
        # Since the weight tensor has max modes [-2, -1, 0, 1, 2],
        # mode [0, -1] is stored at weight index [2, 1].
        conv.weight.tensor[0, 0, 2, 1] = 1
        # This mode is outside the active 4-mode slice and should be ignored.
        conv.weight.tensor[0, 0, 2, 4] = 7

    coordinates = torch.arange(size)
    # This function is constant in the first dimension and mode -1 in the second dimension.
    signal = torch.exp(-2j * torch.pi * coordinates / size)
    x = signal.reshape(1, 1, 1, size).expand(1, 1, size, size).contiguous()

    y = conv(x)

    torch.testing.assert_close(y, x, atol=1e-6, rtol=1e-6)


def test_SpectralConv_explicit_index_set_passes_listed_mode():
    size = 8
    # We only allow Fourier mode [1, 2] to pass through
    conv = SpectralConv(
        1,
        1,
        (3, 5),
        complex_data=True,
        factorization=None,
        bias=False,
        index_set=ExplicitIndexSet([[1, 2]]),
    )
    with torch.no_grad():
        conv.weight.tensor.fill_(1) # For [1, 2]

    x0 = torch.arange(size).reshape(size, 1)
    x1 = torch.arange(size).reshape(1, size)
    # This function only has Fourier mode [1, 2]
    x = torch.exp(2j * torch.pi * (x0 + 2 * x1) / size).reshape(
        1, 1, size, size
    )

    torch.testing.assert_close(conv(x), x, atol=1e-6, rtol=1e-6)


def test_SpectralConv_explicit_index_set_filters_unlisted_mode():
    size = 8
    # We only allow Fourier mode [0, 0] to pass through
    conv = SpectralConv(
        1,
        1,
        (3, 3),
        complex_data=True,
        factorization=None,
        bias=False,
        index_set=ExplicitIndexSet([[0, 0]]),
    )
    with torch.no_grad():
        conv.weight.tensor.fill_(1) # For [0, 0]

    x0 = torch.arange(size).reshape(size, 1)
    x1 = torch.arange(size).reshape(1, size)
    # This function only has Fourier mode [1, 2]
    x = torch.exp(2j * torch.pi * (x0 + 2 * x1) / size).reshape(
        1, 1, size, size
    )

    torch.testing.assert_close(conv(x), torch.zeros_like(x), atol=1e-6, rtol=1e-6)


def test_SpectralConv_real_explicit_index_set_uses_rfft_representative():
    size = 8
    # We only allow Fourier modes [0, -1] and [0, 1] to pass through
    # but we also set this to be real-valued functions.
    # Real-valued functions have Fourier coefficients to have
    # complex conjugate symmetry.
    conv = SpectralConv(
        1,
        1,
        (3, 3),
        complex_data=False, # The function is real-valued
        factorization=None,
        bias=False,
        index_set=ExplicitIndexSet([[0, -1], [0, 1]]),
    )
    # We set multipliers 7 for mode [0, -1] @ index 0 and 1 for mode [0, 1] @ index 1.
    # However, for real FFTs, the last dimension is stored in rfft form: [0, 1] represents
    # both mode [0, 1] and mode [0, -1], and weights for negative last modes
    # are ignored. So the modes [0, 1] and [0, -1] are both multiplied by 1 (not by 7).
    with torch.no_grad():
        conv.weight.tensor.zero_()
        conv.weight.tensor[0, 0, 0] = 7 # For in_channel=0, out_channel=0, mode [0, -1]
        conv.weight.tensor[0, 0, 1] = 1 # For in_channel=0, out_channel=0, mode [0, 1]

    x1 = torch.arange(size).reshape(1, size)
    signal = torch.cos(2 * torch.pi * x1 / size) # This is using cosine (not exp)
    x = signal.reshape(1, 1, 1, size).expand(1, 1, size, size).contiguous()

    torch.testing.assert_close(conv(x), x, atol=1e-6, rtol=1e-6)


def test_SpectralConv_hyperbolic_cross_index_set():
    conv = SpectralConv(
        2,
        3,
        (5, 5),
        index_set=HyperbolicCrossIndexSet(radius=2, n_dim=2),
        factorization=None,
    )
    x = torch.randn(4, 2, 8, 8)

    y = conv(x)

    assert y.shape == (4, 3, 8, 8)
    assert conv.weight.shape == (2, 3, conv.index_set.n_modes)


def test_SpectralConv_hyperbolic_cross_uses_radius_from_n_modes():
    size = 8
    index_set = HyperbolicCrossIndexSet(radius=2, n_dim=2)
    conv = SpectralConv(
        1,
        1,
        (3, 3),
        max_n_modes=(5, 5),
        complex_data=True,
        index_set=index_set,
        factorization=None,
        bias=False,
    )
    mode_id = torch.nonzero(
        torch.all(index_set.modes() == torch.tensor([2, 0]), dim=1),
        as_tuple=False,
    ).item()
    with torch.no_grad():
        conv.weight.tensor.zero_()
        conv.weight.tensor[0, 0, mode_id] = 1

    coordinates = torch.arange(size)
    signal = torch.exp(2j * torch.pi * 2 * coordinates / size)
    x = signal.reshape(1, 1, size, 1).expand(1, 1, size, size).contiguous()

    # n_modes=(3, 3) gives active radius 1 for the hyperbolic cross, so mode
    # [2, 0] is allocated in the weight tensor but is not active yet.
    torch.testing.assert_close(conv(x), torch.zeros_like(x), atol=1e-6, rtol=1e-6)

    # Increasing n_modes grows the active radius to 2 without reallocating the
    # index set or the spectral weights, so mode [2, 0] now passes.
    conv.n_modes = (5, 5)
    torch.testing.assert_close(conv(x), x, atol=1e-6, rtol=1e-6)


def test_SpectralConv_keeps_true_modes_for_real_fields():
    conv = SpectralConv(
        1,
        1,
        (6, 8),
        max_n_modes=(10, 12),
        complex_data=False,
        factorization=None,
    )

    assert conv.true_n_modes == [6, 8]
    assert conv.n_modes == [6, 5]
    assert conv.max_n_modes == [10, 12]
    assert conv.practical_max_n_modes == [10, 7]
    assert conv.index_set.n_modes_per_dim == (10, 12)


def test_SpectralConv_index_set_radius_must_match_true_max_modes():
    with pytest.raises(ValueError):
        SpectralConv(
            1,
            1,
            (5, 5),
            complex_data=True,
            index_set=HyperbolicCrossIndexSet(radius=1, n_dim=2),
            factorization=None,
        )


def test_SpectralConv_rank1_lattice_transform():
    conv = SpectralConv(
        2,
        3,
        (3, 3),
        complex_data=True,
        index_set=HyperbolicCrossIndexSet(radius=1, n_dim=2),
        spectral_transform=Rank1LatticeFFT(
            n=17,
            z=torch.tensor([1, 5]),
            complex_data=True,
        ),
        factorization=None,
    )
    x = torch.randn(4, 2, 17, dtype=torch.cfloat)

    y = conv(x)

    assert y.shape == (4, 3, 17)
    assert y.dtype == torch.cfloat


def test_SpectralConv_rank1_lattice_hyperrectangle_index_set_filters_mode():
    n = 17
    z = torch.tensor([1, 5])
    # The hyperrectangle has modes [-1, 0, 1] x [-1, 0, 1].
    # On the rank-1 lattice, mode [1, 0] is stored at coefficient dot([1, 0], z) = 1.
    conv = SpectralConv(
        1,
        1,
        (3, 3),
        complex_data=True,
        index_set=HyperRectangleIndexSet.from_n_modes_per_dim((3, 3)),
        spectral_transform=Rank1LatticeFFT(n=n, z=z, complex_data=True),
        factorization=None,
        bias=False,
    )
    with torch.no_grad():
        conv.weight.tensor.zero_()
        # For in_channel=0, out_channel=0, mode [1, 0].
        # Hyperrectangle weights are stored as a grid, so [1, 0] is at [2, 1].
        conv.weight.tensor[0, 0, 2, 1] = 1

    coordinates = torch.arange(n)
    signal = torch.exp(2j * torch.pi * coordinates / n)
    x = signal.reshape(1, 1, n)

    torch.testing.assert_close(conv(x), x, atol=1e-6, rtol=1e-6)


def test_SpectralConv_rank1_lattice_transform_remaps_output_shape():
    n_in = 17
    n_out = 31
    z = torch.tensor([1, 5])
    conv = SpectralConv(
        1,
        1,
        (3, 3),
        complex_data=True,
        index_set=HyperRectangleIndexSet.from_n_modes_per_dim((3, 3)),
        spectral_transform=Rank1LatticeFFT(n=n_in, z=z, complex_data=True),
        factorization=None,
        bias=False,
    )

    coordinates_in = torch.arange(n_in)
    coordinates_out = torch.arange(n_out)
    x = torch.exp(2j * torch.pi * coordinates_in / n_in).reshape(1, 1, n_in)
    expected = torch.exp(2j * torch.pi * coordinates_out / n_out).reshape(
        1, 1, n_out
    )

    torch.testing.assert_close(
        conv.transform(x, output_shape=(n_out,)), expected, atol=1e-6, rtol=1e-6
    )


def test_SpectralConv_rank1_lattice_transform_remaps_real_output_shape():
    n_in = 17
    n_out = 31
    z = torch.tensor([1, 5])
    conv = SpectralConv(
        1,
        1,
        (3, 3),
        complex_data=False,
        index_set=HyperRectangleIndexSet.from_n_modes_per_dim((3, 3)),
        spectral_transform=Rank1LatticeFFT(n=n_in, z=z, complex_data=False),
        factorization=None,
        bias=False,
    )

    coordinates_in = torch.arange(n_in)
    coordinates_out = torch.arange(n_out)
    x = torch.cos(2 * torch.pi * coordinates_in / n_in).reshape(1, 1, n_in)
    expected = torch.cos(2 * torch.pi * coordinates_out / n_out).reshape(
        1, 1, n_out
    )

    torch.testing.assert_close(
        conv.transform(x, output_shape=(n_out,)), expected, atol=1e-6, rtol=1e-6
    )


def test_SpectralConv_rank1_lattice_forward_remaps_output_shape():
    n_in = 17
    n_out = 31
    z = torch.tensor([1, 5])
    conv = SpectralConv(
        1,
        1,
        (3, 3),
        complex_data=True,
        index_set=HyperRectangleIndexSet.from_n_modes_per_dim((3, 3)),
        spectral_transform=Rank1LatticeFFT(n=n_in, z=z, complex_data=True),
        factorization=None,
        bias=False,
    )
    with torch.no_grad():
        conv.weight.tensor.zero_()
        conv.weight.tensor[0, 0, 2, 1] = 1

    coordinates_in = torch.arange(n_in)
    coordinates_out = torch.arange(n_out)
    x = torch.exp(2j * torch.pi * coordinates_in / n_in).reshape(1, 1, n_in)
    expected = torch.exp(2j * torch.pi * coordinates_out / n_out).reshape(
        1, 1, n_out
    )

    torch.testing.assert_close(
        conv(x, output_shape=(n_out,)), expected, atol=1e-6, rtol=1e-6
    )


@pytest.mark.parametrize("complex_data", [False, True])
@pytest.mark.parametrize("n_in,n_out", [(2**9, 2**10), (2**10, 2**9)])
@pytest.mark.parametrize(
    "index_set,n_modes",
    [
        (HyperRectangleIndexSet.from_n_modes_per_dim((9, 9)), (9, 9)),
        (HyperbolicCrossIndexSet(radius=2, n_dim=2), (5, 5)),
    ],
)
def test_SpectralConv_rank1_lattice_transform_up_downsamples_lattice_modes(
    complex_data, n_in, n_out, index_set, n_modes
):
    z = torch.tensor([433461, 472323])
    conv = SpectralConv(
        1,
        1,
        n_modes,
        complex_data=complex_data,
        index_set=index_set,
        spectral_transform=Rank1LatticeFFT(n=n_in, z=z, complex_data=complex_data),
        factorization=None,
        bias=False,
    )

    lattice_in = rank1_lattice_points(z, n_in, dtype=torch.float32)
    lattice_out = rank1_lattice_points(z, n_out, dtype=torch.float32)
    x = torch.cos(2 * torch.pi * lattice_in[:, 0]).reshape(1, 1, n_in)
    expected = torch.cos(2 * torch.pi * lattice_out[:, 0]).reshape(1, 1, n_out)
    if complex_data:
        x = x.to(torch.cfloat)
        expected = expected.to(torch.cfloat)

    torch.testing.assert_close(
        conv.transform(x, output_shape=(n_out,)), expected, atol=1e-6, rtol=1e-6
    )


def test_rank1_lattice_coordinates():
    z = torch.tensor([1, 3])
    expected_coordinates = torch.tensor(
        [[0, 0], [1 / 5, 3 / 5], [2 / 5, 1 / 5], [3 / 5, 4 / 5], [4 / 5, 2 / 5]]
    )
    torch.testing.assert_close(
        rank1_lattice_points(z, 5), expected_coordinates
    )


def test_regular_grid_to_lattice_samples_rank1_lattice_points():
    z = torch.tensor([1, 3])
    grid_x = torch.arange(5).reshape(5, 1).expand(5, 5)
    grid = torch.cos(2 * torch.pi * grid_x / 5).reshape(1, 1, 5, 5)

    lattice_values = regular_grid_to_lattice(grid, z, n=5)

    expected = torch.cos(
        2 * torch.pi * rank1_lattice_points(z, 5)[:, 0]
    ).reshape(1, 1, 5)
    torch.testing.assert_close(lattice_values, expected)


def test_lattice_to_regular_grid_places_rank1_lattice_points():
    z = torch.tensor([5, 1])
    grid_x = torch.arange(5).reshape(5, 1).expand(5, 5)
    grid_y = torch.arange(5).reshape(1, 5).expand(5, 5)
    grid = (grid_x + 10 * grid_y).reshape(1, 1, 5, 5).to(torch.float32)
    lattice_indices = torch.floor(rank1_lattice_points(z, 25) * 5).to(torch.long)
    lattice_values = (
        lattice_indices[:, 0] + 10 * lattice_indices[:, 1]
    ).reshape(1, 1, 25).to(torch.float32)

    reconstructed = lattice_to_regular_grid(lattice_values, z, output_shape=(5, 5))

    torch.testing.assert_close(reconstructed, grid)


def test_lattice_embedding_appends_rank1_lattice_coordinates():
    z = torch.tensor([1, 3])
    embedding = LatticeEmbedding(in_channels=1, z=z)
    x = torch.zeros(2, 1, 5)

    y = embedding(x)

    expected_coordinates = torch.tensor(
        [[0, 1 / 5, 2 / 5, 3 / 5, 4 / 5], [0, 3 / 5, 1 / 5, 4 / 5, 2 / 5]]
    )
    assert y.shape == (2, 3, 5)
    torch.testing.assert_close(y[0, 1:], expected_coordinates)


def test_SpectralConv_rank1_lattice_hyperrectangle_rejects_factorization():
    with pytest.raises(ValueError, match="Rank1LatticeFFT and HyperRectangleIndexSet"):
        SpectralConv(
            1,
            1,
            (3, 3),
            complex_data=True,
            index_set=HyperRectangleIndexSet.from_n_modes_per_dim((3, 3)),
            spectral_transform=Rank1LatticeFFT(
                n=17,
                z=torch.tensor([1, 5]),
                complex_data=True,
            ),
            factorization="cp",
            bias=False,
        )


def test_SpectralConv_rank1_lattice_hyperbolic_cross_index_set_filters_mode():
    n = 17
    z = torch.tensor([1, 5])
    index_set = HyperbolicCrossIndexSet(radius=1, n_dim=2)
    # On the rank-1 lattice, mode [1, 1] is stored at coefficient dot([1, 1], z) = 6.
    conv = SpectralConv(
        1,
        1,
        (3, 3),
        complex_data=True,
        index_set=index_set,
        spectral_transform=Rank1LatticeFFT(n=n, z=z, complex_data=True),
        factorization=None,
        bias=False,
    )
    mode_id = torch.nonzero(
        torch.all(index_set.modes() == torch.tensor([1, 1]), dim=1),
        as_tuple=False,
    ).item()
    with torch.no_grad():
        conv.weight.tensor.zero_()
        # Hyperbolic-cross weights are stored as one flat slot per explicit mode.
        conv.weight.tensor[0, 0, mode_id] = 1

    coordinates = torch.arange(n)
    signal = torch.exp(2j * torch.pi * 6 * coordinates / n)
    x = signal.reshape(1, 1, n)

    torch.testing.assert_close(conv(x), x, atol=1e-6, rtol=1e-6)


def test_hyperrectangle_index_set_from_n_modes_derives_weights():
    index_set = HyperRectangleIndexSet.from_n_modes_per_dim((6, 3, 9))

    assert index_set.radius == 3
    assert index_set.weights == (1.0, 0.5, 1.5)
    assert index_set.n_modes_per_dim == (6, 3, 9)

    # The stored radius gives the maximum capacity. Smaller/larger active
    # radii can still be selected when computing nested mode sets.
    assert HyperRectangleIndexSet.n_modes_per_dim_for_radius(4, index_set.weights) == (
        8,
        4,
        12,
    )

    active_modes = index_set.modes(radius=4)
    assert active_modes.shape == (8 * 4 * 12, 3)
    assert index_set.n_modes_per_dim == (6, 3, 9)


def test_hyperrectangle_index_set_radius_uses_half_width():
    index_set = HyperRectangleIndexSet(radius=4, n_dim=2, weights=(1.0, 0.5))

    assert index_set.n_modes_per_dim == (8, 4)


def test_radial_index_sets_active_radius_uses_first_weight():
    hyperrectangle = HyperRectangleIndexSet(radius=3, n_dim=2, weights=(2.0, 1.0))
    hyperbolic_cross = HyperbolicCrossIndexSet(radius=3, n_dim=2, weights=(2.0, 1.0))

    # For weights[0] != 1, radius_from_n_modes has to invert the
    # constructor's first-axis mode extent.
    assert hyperrectangle.n_modes_per_dim[0] == 12
    assert hyperrectangle.radius_from_n_modes((12, 6)) == pytest.approx(
        hyperrectangle.radius
    )
    assert hyperbolic_cross.radius_from_n_modes((13, 7)) == pytest.approx(
        hyperbolic_cross.radius
    )


def test_hyperbolic_cross_index_set_from_n_modes_derives_weights():
    index_set = HyperbolicCrossIndexSet.from_n_modes_per_dim((7, 5, 9))

    assert index_set.radius == 3
    assert index_set.weights == (1.0, 2 / 3, 4 / 3)
    assert index_set.radius_from_n_modes((7, 5, 9)) == pytest.approx(index_set.radius)


def test_hyperbolic_cross_index_set_beta_two():
    radius = 9
    beta = 2
    weights = (1.0, 2.0)
    index_set = HyperbolicCrossIndexSet(
        radius=radius,
        n_dim=2,
        weights=weights,
        beta=beta,
    )
    modes = index_set.modes()

    expected_modes_1d = torch.arange(-5, 6)
    grids = torch.meshgrid(expected_modes_1d, expected_modes_1d, indexing="ij")
    expected_modes = torch.stack(grids, dim=-1).reshape(-1, 2)
    abs_modes = torch.abs(expected_modes).to(torch.float64)
    weights_tensor = torch.tensor(weights, dtype=torch.float64)
    expected_terms = abs_modes**beta / weights_tensor
    expected_terms = torch.where(abs_modes == 0, 1, expected_terms)
    expected_radii = torch.prod(expected_terms, dim=1)
    expected_modes = expected_modes[expected_radii <= radius]

    assert set(map(tuple, modes.tolist())) == set(map(tuple, expected_modes.tolist()))


def test_hyperbolic_cross_index_set_beta_zero_matches_hyperrectangle():
    index_set = HyperbolicCrossIndexSet(radius=3, n_dim=2, beta=0)

    assert index_set.modes().shape == (7 * 7, 2)
    assert index_set.radius_from_n_modes((7, 7)) == pytest.approx(3)


def test_hyperbolic_cross_index_set_tracks_integer_radius_starts():
    index_set = HyperbolicCrossIndexSet(radius=3, n_dim=2)
    modes = index_set.modes()
    mode_radii = torch.prod(torch.clamp(torch.abs(modes), min=1), dim=1)

    assert index_set.radius_starts[0] == 0
    assert index_set.radius_starts[1] == 0
    assert index_set.radius_starts[2] == torch.sum(mode_radii <= 1).item()
    assert index_set.radius_starts[3] == torch.sum(mode_radii <= 2).item()
    assert index_set.radius_starts[4] == modes.shape[0]

    for radius in range(1, 4):
        start = index_set.radius_starts[radius]
        end = index_set.radius_starts[radius + 1]
        assert torch.all(torch.ceil(mode_radii[start:end]) == radius)
