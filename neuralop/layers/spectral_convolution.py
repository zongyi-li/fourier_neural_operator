from typing import List, Optional, Tuple, Union

from ..utils import validate_scaling_factor

import torch
from torch import nn
from numpy import isclose

import tensorly as tl
from tensorly.plugins import use_opt_einsum
from tltorch.factorized_tensors.core import FactorizedTensor

from .einsum_utils import einsum_complexhalf
from .base_spectral_conv import BaseSpectralConv
from .index_sets import HyperRectangleIndexSet, RadialIndexSet
from .spectral_transforms import Rank1LatticeFFT, RegularGridFFT

tl.set_backend("pytorch")
use_opt_einsum("optimal")
einsum_symbols = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _contract_dense(x, weight, separable=False):
    order = tl.ndim(x)
    # batch-size, in_channels, x, y...
    x_syms = list(einsum_symbols[:order])

    # in_channels, out_channels, x, y...
    weight_syms = list(x_syms[1:])  # no batch-size

    # batch-size, out_channels, x, y...
    if separable:
        out_syms = [x_syms[0]] + list(weight_syms)
    else:
        weight_syms.insert(1, einsum_symbols[order])  # outputs
        out_syms = list(weight_syms)
        out_syms[0] = x_syms[0]

    eq = f'{"".join(x_syms)},{"".join(weight_syms)}->{"".join(out_syms)}'

    if not torch.is_tensor(weight):
        weight = weight.to_tensor()

    if x.dtype == torch.complex32:
        # if x is half precision, run a specialized einsum
        return einsum_complexhalf(eq, x, weight)
    else:
        return tl.einsum(eq, x, weight)


def _contract_dense_separable(x, weight, separable):
    if not torch.is_tensor(weight):
        weight = weight.to_tensor()
    return x * weight


def _contract_cp(x, cp_weight, separable=False):
    order = tl.ndim(x)

    x_syms = str(einsum_symbols[:order])
    rank_sym = einsum_symbols[order]
    out_sym = einsum_symbols[order + 1]
    out_syms = list(x_syms)
    if separable:
        factor_syms = [einsum_symbols[1] + rank_sym]  # in only
    else:
        out_syms[1] = out_sym
        factor_syms = [einsum_symbols[1] + rank_sym, out_sym + rank_sym]  # in, out
    factor_syms += [xs + rank_sym for xs in x_syms[2:]]  # x, y, ...
    eq = f'{x_syms},{rank_sym},{",".join(factor_syms)}->{"".join(out_syms)}'

    if x.dtype == torch.complex32:
        return einsum_complexhalf(eq, x, cp_weight.weights, *cp_weight.factors)
    else:
        return tl.einsum(eq, x, cp_weight.weights, *cp_weight.factors)


def _contract_tucker(x, tucker_weight, separable=False):
    order = tl.ndim(x)

    x_syms = str(einsum_symbols[:order])
    out_sym = einsum_symbols[order]
    out_syms = list(x_syms)
    if separable:
        core_syms = einsum_symbols[order + 1 : 2 * order]
        # factor_syms = [einsum_symbols[1]+core_syms[0]] #in only
        # x, y, ...
        factor_syms = [xs + rs for (xs, rs) in zip(x_syms[1:], core_syms)]

    else:
        core_syms = einsum_symbols[order + 1 : 2 * order + 1]
        out_syms[1] = out_sym
        factor_syms = [
            einsum_symbols[1] + core_syms[0],
            out_sym + core_syms[1],
        ]  # out, in
        # x, y, ...
        factor_syms += [xs + rs for (xs, rs) in zip(x_syms[2:], core_syms[2:])]

    eq = f'{x_syms},{core_syms},{",".join(factor_syms)}->{"".join(out_syms)}'

    if x.dtype == torch.complex32:
        return einsum_complexhalf(eq, x, tucker_weight.core, *tucker_weight.factors)
    else:
        return tl.einsum(eq, x, tucker_weight.core, *tucker_weight.factors)


def _contract_tt(x, tt_weight, separable=False):
    order = tl.ndim(x)

    x_syms = list(einsum_symbols[:order])
    weight_syms = list(x_syms[1:])  # no batch-size
    if not separable:
        weight_syms.insert(1, einsum_symbols[order])  # outputs
        out_syms = list(weight_syms)
        out_syms[0] = x_syms[0]
    else:
        out_syms = list(x_syms)
    rank_syms = list(einsum_symbols[order + 1 :])
    tt_syms = []
    for i, s in enumerate(weight_syms):
        tt_syms.append([rank_syms[i], s, rank_syms[i + 1]])
    eq = (
        "".join(x_syms)
        + ","
        + ",".join("".join(f) for f in tt_syms)
        + "->"
        + "".join(out_syms)
    )

    if x.dtype == torch.complex32:
        return einsum_complexhalf(eq, x, *tt_weight.factors)
    else:
        return tl.einsum(eq, x, *tt_weight.factors)


def get_contract_fun(weight, implementation="reconstructed", separable=False):
    """Generic ND implementation of Fourier Spectral Conv contraction

    Parameters
    ----------
    weight : tensorly-torch's FactorizedTensor
    implementation : {'reconstructed', 'factorized'}, default is 'reconstructed'
        whether to reconstruct the weight and do a forward pass (reconstructed)
        or contract directly the factors of the factorized weight with the input (factorized)
    separable: bool
        if True, performs contraction with individual tensor factors.
        if False,
    Returns
    -------
    function : (x, weight) -> x * weight in Fourier space
    """
    if implementation == "reconstructed":
        if separable:
            return _contract_dense_separable
        else:
            return _contract_dense
    elif implementation == "factorized":
        if torch.is_tensor(weight):
            return _contract_dense
        elif isinstance(weight, FactorizedTensor):
            if weight.name.lower().endswith("dense"):
                return _contract_dense
            elif weight.name.lower().endswith("tucker"):
                return _contract_tucker
            elif weight.name.lower().endswith("tt"):
                return _contract_tt
            elif weight.name.lower().endswith("cp"):
                return _contract_cp
            else:
                raise ValueError(f"Got unexpected factorized weight type {weight.name}")
        else:
            raise ValueError(
                f"Got unexpected weight type of class {weight.__class__.__name__}"
            )
    else:
        raise ValueError(
            f'Got implementation={implementation}, expected "reconstructed" or "factorized"'
        )


Number = Union[int, float]


class SpectralConv(BaseSpectralConv):
    """SpectralConv implements the Spectral Convolution component of a Fourier layer
    described. 

    It is implemented as described in [1]_ and [2]_.

    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    n_modes : int or int tuple
        Number of true Fourier modes to use for contraction in Fourier domain
        during training. This can be updated dynamically during training.
        These are the unreduced active mode extents along each dimension. The
        unreduced active extents are stored in ``self.true_n_modes`` and define
        the active radius used to select modes from the index set.

        .. warning::

            For real-valued data, pass the true Fourier extents here. The layer
            exposes ``self.n_modes`` in the legacy NeuralOp storage convention.
            Real-valued spectral transforms store only the nonnegative half of
            their FFT coefficient axis. For ``RegularGridFFT`` this is the last
            grid-FFT dimension; for ``Rank1LatticeFFT`` this is the
            one-dimensional lattice-FFT coefficient axis. ``self.true_n_modes``
            keeps the unreduced multi-dimensional Fourier extents.

        .. note::

            If ``index_set`` is provided, it defines the actual selected Fourier
            modes and its constructor radius defines the storage capacity. In
            that case, only the first entries of ``n_modes`` and
            ``max_n_modes`` matter: ``max_n_modes[0]`` is checked against
            ``index_set.radius`` when ``max_n_modes`` is given, otherwise
            ``n_modes[0]`` is checked; ``n_modes[0]`` defines the active radius
            selected during the forward pass.

            If ``index_set`` is not provided, the layer uses a
            ``HyperRectangleIndexSet``. Its storage capacity is defined by
            ``max_n_modes`` when given, otherwise by ``n_modes``. The active
            extents are defined by ``n_modes`` and the active radius is
            ``n_modes[0] / 2``.

    complex_data : bool, optional
        Whether data takes on complex values in the spatial domain, by default False.
        If True, uses different logic for FFT contraction and uses full FFT instead of real-valued.
    max_n_modes : int tuple or None, optional
        If not None, maximum number of modes to keep in Fourier Layer along each dim
        (n_modes cannot be increased beyond that). If None, all n_modes are used.
        Like ``n_modes``, this should be given in true, unreduced Fourier
        extents. If an explicit ``index_set`` with a ``radius`` is provided, its
        storage radius is checked only against ``max_n_modes[0]`` when
        ``max_n_modes`` is given, and against ``n_modes[0]`` otherwise.
        Hyperrectangles use radius ``M[0] / 2``; explicit radius-based sets such
        as hyperbolic crosses use ``(M[0] - 1) / 2``. Other entries of
        ``n_modes`` and ``max_n_modes`` do not affect the geometry of a provided
        ``index_set``.
        By default None.
    bias : bool, optional
        Whether to add a learnable bias to the output, by default True.
    separable : bool, optional
        Whether to use separable implementation of contraction.
        If True, contracts factors of factorized tensor weight individually.
        By default False.
    resolution_scaling_factor : float, list of float, or None, optional
        Scaling factor(s) for resolution scaling. If provided, the output resolution
        will be scaled by this factor along each spatial dimension.
        By default None.
    fno_block_precision : str, optional
        Precision mode for FNO block operations. Options: 'full', 'half', 'mixed'.
        By default 'full'.
    rank : float, optional
        Rank of the tensor factorization of the Fourier weights, by default 1.0.
        Ignored if ``factorization is None``.
    factorization : str or None, optional
        Tensor factorization type. Options: {'tucker', 'cp', 'tt'}.
        If None, a single dense weight is learned for the FNO.
        Otherwise, that weight, used for the contraction in the Fourier domain
        is learned in factorized form. In that case, `factorization` is the
        tensor factorization of the parameters weight used.
        By default None.
    implementation : {'factorized', 'reconstructed'}, optional
        If factorization is not None, forward mode to use:
        * `reconstructed` : the full weight tensor is reconstructed from the
          factorization and used for the forward pass
        * `factorized` : the input is directly contracted with the factors of
          the decomposition
        Ignored if ``factorization is None``.
        By default 'reconstructed'.
    enforce_hermitian_symmetry : bool, optional
        Whether to enforce Hermitian symmetry conditions when performing inverse FFT
        for real-valued data. When True, explicitly enforces that the 0th frequency
        and Nyquist frequency are real-valued before calling irfft. 
        When False, relies on cuFFT's irfftn to handle symmetry automatically, 
        which may fail on certain GPUs or input sizes, causing line artifacts. 
        Setting to True splits the inverse FFT into ifftn along (n-1) dimensions 
        followed by irfft on the last dimension, with a small computational overhead. 
        By default True.
    fixed_rank_modes : bool, optional
        Modes to not factorize, by default False.
        Ignored if ``factorization is None``.
    decomposition_kwargs : dict or None, optional
        Optional additional parameters to pass to the tensor decomposition.
        Ignored if ``factorization is None``.
        By default None.
    init_std : float or 'auto', optional
        Standard deviation to use for weight initialization, by default 'auto'.
        If 'auto', uses (2 / (in_channels + out_channels)) ** 0.5.
    fft_norm : str, optional
        FFT normalization parameter, by default 'forward'.
    index_set : IndexSet or None, optional
        Fourier modes to keep and learn weights for. If None, the layer uses a
        ``HyperRectangleIndexSet`` with storage capacity derived from
        ``max_n_modes`` when given, and otherwise from ``n_modes``.
        Non-rectangular index sets, such as a hyperbolic cross, are stored as
        one flat weight slot per selected Fourier mode, as in [3]_.
    spectral_transform : object or None, optional
        Spectral transform backend. If None, the layer uses the default
        regular-grid FFT backend, ``RegularGridFFT``. Custom transforms must
        provide ``forward_transform``, ``inverse_transform``, ``weight_shape``,
        and ``selection`` methods. Rank-1 lattice transforms use a one-dimensional
        FFT while mapping multi-dimensional Fourier modes into lattice
        coefficients [3]_.
    device : torch.device or None, optional
        Device to place the layer on, by default None.

    Notes
    -----
    Tensor decompositions are compatible with the default regular-grid FFT and
    with flat explicit index sets. They are not currently compatible with a
    rank-1 lattice transform together with a hyperrectangular grid-shaped weight:
    tensorly-torch factorized tensors use outer indexing for multiple tensor
    indices, while the lattice path needs paired coordinate indexing.

    References
    ----------
    .. [1] Li, Z. et al. "Fourier Neural Operator for Parametric Partial Differential
        Equations" (2021). ICLR 2021, https://arxiv.org/pdf/2010.08895.

    .. [2] Kossaifi, J., Kovachki, N., Azizzadenesheli, K., Anandkumar, A. "Multi-Grid
        Tensorized Fourier Neural Operator for High-Resolution PDEs" (2024).
        TMLR 2024, https://openreview.net/pdf?id=AWiDlO63bH.

    .. [3] Dilen, J., Keller, A., Kuo, F. Y., Nuyens, D. "Fourier Neural Operators
        with Rank-1 Lattice Points and Hyperbolic Cross" (2026).
        https://arxiv.org/abs/2606.08871.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        n_modes,
        complex_data=False,
        max_n_modes=None,
        bias=True,
        separable=False,
        resolution_scaling_factor: Optional[Union[Number, List[Number]]] = None,
        fno_block_precision="full",
        rank=1.0,
        factorization=None,
        implementation="reconstructed",
        enforce_hermitian_symmetry=True,
        fixed_rank_modes=False,
        decomposition_kwargs: Optional[dict] = None,
        init_std="auto",
        fft_norm="forward",
        index_set=None,
        spectral_transform=None,
        device=None,
    ):
        super().__init__(device=device)

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.complex_data = complex_data

        # n_modes is the active number of modes kept along each dimension.
        # For legacy NeuralOp compatibility, self.n_modes stores the FFT-storage
        # convention: for real-valued fields, the last dimension is reduced to
        # M // 2 + 1. self.true_n_modes keeps the unreduced active extents.
        # See the n_modes property below for the same convention during updates.
        self.n_modes = n_modes

        if max_n_modes is None:
            max_n_modes = list(self.true_n_modes)
        elif isinstance(max_n_modes, int):
            max_n_modes = [max_n_modes]
        else:
            max_n_modes = list(max_n_modes)
        self.max_n_modes = max_n_modes

        self.fft_norm = fft_norm
        self.index_set = (
            HyperRectangleIndexSet.from_n_modes_per_dim(self.max_n_modes)
            if index_set is None
            else index_set
        )
        if index_set is not None:
            self._validate_index_set_radius()
        self.spectral_transform = (
            RegularGridFFT(
                order=len(self.true_n_modes),
                complex_data=complex_data,
                fft_norm=fft_norm,
                enforce_hermitian_symmetry=enforce_hermitian_symmetry,
            )
            if spectral_transform is None
            else spectral_transform
        )
        self.order = getattr(self.spectral_transform, "order", len(self.true_n_modes))

        self.fno_block_precision = fno_block_precision
        self.rank = rank
        self.factorization = factorization
        self.implementation = implementation
        self.enforce_hermitian_symmetry = enforce_hermitian_symmetry

        self.resolution_scaling_factor: Union[
            None, List[List[float]]
        ] = validate_scaling_factor(resolution_scaling_factor, self.order)

        if init_std == "auto":
            init_std = (2 / (in_channels + out_channels)) ** 0.5

        if isinstance(fixed_rank_modes, bool):
            if fixed_rank_modes:
                # If bool, keep the number of layers fixed
                fixed_rank_modes = [0]
            else:
                fixed_rank_modes = None
        if (
            factorization is not None
            and isinstance(self.spectral_transform, Rank1LatticeFFT)
            and isinstance(self.index_set, HyperRectangleIndexSet)
        ):
            raise ValueError(
                "Tensor factorization is not currently compatible with "
                "Rank1LatticeFFT and HyperRectangleIndexSet. Use factorization=None "
                "or use a flat explicit index set."
            )
        if factorization is None:
            factorization = "Dense"  # No factorization

        weight_shape = self.spectral_transform.weight_shape(
            self.index_set, max_n_modes=self.max_n_modes
        )
        if separable:
            if in_channels != out_channels:
                raise ValueError(
                    "To use separable Fourier Conv, in_channels must be equal "
                    f"to out_channels, but got in_channels={in_channels} and "
                    f"out_channels={out_channels}",
                )
            weight_shape = (in_channels, *weight_shape)
        else:
            weight_shape = (in_channels, out_channels, *weight_shape)
        self.separable = separable

        tensor_kwargs = decomposition_kwargs if decomposition_kwargs is not None else {}

        # Create/init spectral weight tensor
        self.weight = FactorizedTensor.new(
            weight_shape,
            rank=self.rank,
            factorization=factorization,
            fixed_rank_modes=fixed_rank_modes,
            **tensor_kwargs,
            dtype=torch.cfloat,
        )
        self.weight.normal_(0, init_std)

        self._contract = get_contract_fun(
            self.weight, implementation=implementation, separable=separable
        )

        if bias:
            self.bias = nn.Parameter(
                init_std * torch.randn(*(tuple([self.out_channels]) + (1,) * self.order))
            )
        else:
            self.bias = None

    def transform(self, x, output_shape=None):
        """Upsample or downsample the skip link to the spectral path resolution."""
        in_shape = list(x.shape[2:])

        if self.resolution_scaling_factor is not None and output_shape is None:
            out_shape = tuple(
                [round(s * r) for (s, r) in zip(in_shape, self.resolution_scaling_factor)]
            )
        elif output_shape is not None:
            out_shape = self._as_shape_tuple(output_shape)
        else:
            out_shape = tuple(in_shape)

        if in_shape == out_shape:
            return x
        return self.spectral_transform.transform(
            x,
            output_shape=out_shape,
            index_set=self.index_set,
            true_n_modes=self.true_n_modes,
        )

    @property
    def n_modes(self):
        # Legacy NeuralOp convention: for real-valued fields this property is
        # stored in FFT-storage form, with the last dimension reduced to
        # M // 2 + 1. Use true_n_modes for the unreduced user-facing extents.
        return self._n_modes

    @n_modes.setter
    def n_modes(self, n_modes):
        # Legacy NeuralOp convention: n_modes is stored in FFT-storage form for
        # real-valued fields, so the last dimension is reduced to M // 2 + 1.
        # true_n_modes keeps the unreduced extents used by index sets and
        # spectral transforms that need the mathematical mode geometry.
        self.true_n_modes = self._as_mode_list(n_modes)
        n_modes = self._fft_storage_n_modes(self.true_n_modes)
        self._n_modes = n_modes

    def _as_mode_list(self, n_modes):
        if isinstance(n_modes, int):  # Should happen for 1D FNO only
            return [n_modes]
        else:
            return list(n_modes)

    def _as_shape_tuple(self, shape):
        if isinstance(shape, int):
            return (shape,)
        return tuple(shape)

    def _fft_storage_n_modes(self, n_modes):
        """This is only here for the legacy fields n_modes."""
        n_modes = self._as_mode_list(n_modes)
        if not self.complex_data:
            n_modes[-1] = n_modes[-1] // 2 + 1
        return n_modes

    def _validate_index_set_radius(self):
        if isinstance(self.index_set, RadialIndexSet):
            expected_radius = self.index_set.radius_from_n_modes(self.max_n_modes)
            if not isclose(self.index_set.radius, expected_radius):
                raise ValueError(
                    "Expected index_set.radius to agree with max_n_modes[0] "
                    f"(got radius={self.index_set.radius}, expected {expected_radius})."
                )

    def forward(self, x: torch.Tensor, output_shape: Optional[Tuple[int]] = None):
        """Generic forward pass for the Factorized Spectral Conv

        Parameters
        ----------
        x : torch.Tensor
            input activation of size (batch_size, channels, d1, ..., dN)

        Returns
        -------
        tensorized_spectral_conv(x)
        """
        batchsize = x.shape[0]

        if self.fno_block_precision == "half":
            x = x.half()

        x, transform_state = self.spectral_transform.forward_transform(x)

        if self.fno_block_precision == "mixed":
            # if 'mixed', the above fft runs in full precision, but the
            # following operations run at half precision
            x = x.chalf()

        if self.fno_block_precision in ["half", "mixed"]:
            out_dtype = torch.chalf
        else:
            out_dtype = torch.cfloat
        out_fft = torch.zeros(
            [batchsize, self.out_channels, *transform_state.fft_size],
            device=x.device,
            dtype=out_dtype,
        )

        selection = self.spectral_transform.selection(
            index_set=self.index_set,
            fft_size=transform_state.fft_size,
            max_n_modes=self.max_n_modes,
            true_n_modes=self.true_n_modes,
            separable=self.separable,
            device=x.device,
        )
        weight = self.weight[selection.weight_index]
        out_selected = self._contract(
            x[selection.x_index], weight, separable=self.separable
        )
        out_fft[selection.x_index] = out_selected

        mode_sizes = transform_state.mode_sizes
        if self.resolution_scaling_factor is not None and output_shape is None:
            mode_sizes = tuple([round(s * r) for (s, r) in zip(mode_sizes, self.resolution_scaling_factor)])

        if output_shape is not None:
            mode_sizes = self._as_shape_tuple(output_shape)

        if mode_sizes != transform_state.mode_sizes:
            out_fft, mode_sizes, transform_state = self.spectral_transform.resize_fft(
                out_fft,
                output_shape=mode_sizes,
                index_set=self.index_set,
                true_n_modes=self.true_n_modes,
                state=transform_state,
            )

        x = self.spectral_transform.inverse_transform(out_fft, mode_sizes, transform_state)

        if self.bias is not None:
            x = x + self.bias

        return x
