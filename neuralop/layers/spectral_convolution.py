from typing import List, Optional, Tuple, Union

from ..utils import validate_scaling_factor

import torch
from torch import nn

import tensorly as tl
from tensorly.plugins import use_opt_einsum
from tltorch.factorized_tensors.core import FactorizedTensor

from .channel_mlp import ChannelMLP
from .einsum_utils import einsum_complexhalf
from .base_spectral_conv import BaseSpectralConv
from .resample import resample

tl.set_backend("pytorch")
use_opt_einsum("optimal")
einsum_symbols = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


class _ContiguousBackward(torch.autograd.Function):
    """Identity in the forward pass; returns a contiguous gradient in backward.

    PyTorch's native FFT autograd dispatches the transform's backward (an
    ``_fft_r2c`` / ``_fft_c2r`` / ``_fft_c2c`` ATen op) on whatever gradient
    tensor it receives. On x86 CPU with Intel oneMKL, a gradient with
    non-standard strides makes DFTI raise "Inconsistent configuration
    parameters". The common offender is the zero-stride tensor that ``.sum()``
    / broadcast backward produces (e.g. ``loss.sum().backward()`` flowing into
    the inverse transform). Forcing the gradient contiguous before it reaches
    the FFT backward avoids the error.

    ``.contiguous()`` only changes memory layout, so gradient *values* are
    unchanged and stay exact. This is the key difference from re-deriving an
    FFT's adjoint by hand: ``torch.fft.irfftn`` is the *inverse* of ``rfftn``,
    not its *adjoint* (vector-Jacobian product), so using it as a custom
    backward silently corrupts gradients.
    """

    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.contiguous()


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
        core_syms = einsum_symbols[order + 1: 2 * order]
        # factor_syms = [einsum_symbols[1]+core_syms[0]] #in only
        # x, y, ...
        factor_syms = [xs + rs for (xs, rs) in zip(x_syms[1:], core_syms)]

    else:
        core_syms = einsum_symbols[order + 1: 2 * order + 1]
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
    rank_syms = list(einsum_symbols[order + 1:])
    tt_syms = []
    tt_syms.extend(
        [rank_syms[i], s, rank_syms[i + 1]] for i, s in enumerate(weight_syms)
    )
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
        return _contract_dense_separable if separable else _contract_dense
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
        Number of modes to use for contraction in Fourier domain during training.

        .. warning::

            We take care of the redundancy in the Fourier modes, therefore, for an input
            of size I_1, ..., I_N, please provide modes M_K that are I_1 < M_K <= I_N
            We will automatically keep the right amount of modes: specifically, for the
            last mode only, if you specify M_N modes we will use M_N // 2 + 1 modes
            as the real FFT is redundant along that last dimension. See the theory guide for mode truncation details.


        .. note::

            Provided modes should be even integers. odd numbers will be rounded to the closest even number.

        This can be updated dynamically during training.

    complex_data : bool, optional
        Whether data takes on complex values in the spatial domain, by default False.
        If True, uses different logic for FFT contraction and uses full FFT instead of real-valued.
    max_n_modes : int tuple or None, optional
        If not None, maximum number of modes to keep in Fourier Layer along each dim
        (n_modes cannot be increased beyond that). If None, all n_modes are used.
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
        Otherwise, that weight, used for the contraction in the Fourier domain,
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
    device : torch.device or None, optional
        Device to place the layer on, by default None.
    embed : dict or None, optional
        Configuration for the scalar-time and frequency-mode embeddings used
        by the optional ``(t, k)`` mode-modulation pathway. Required when
        ``mode_modulation`` is provided. Keys:

        - ``type_t`` : ``'sinusoidal'`` (default) or ``'power'``.
        - ``type_k`` : ``'power'`` (default) or ``'sinusoidal'``.
        - ``dim`` : int, default 32. Embedding dimension ``D``.
        - ``alpha`` : float, default ``-2.0``. Exponent range for power
          embedding: features are ``t**p`` with
          ``p in linspace(alpha, 0, D)``.
        - ``r`` : float, default ``10000.0``. Base for the sinusoidal
          embedding frequencies ``r ** (-2i/D)``.

        By default ``None``; no mode modulation is applied and ``forward``
        ignores ``t``.
    mode_modulation : dict or None, optional
        Configuration for the optional per-mode modulation MLP. When set,
        the layer applies a learned ``(t, k)``-dependent multiplier to the
        spectral coefficients before the convolution contraction. Keys:

        - ``enabled`` : bool, default ``True``. If ``False`` the layer
          ignores ``t`` and behaves like a vanilla :class:`SpectralConv`.
        - ``type`` : ``'real'``, ``'complex'``, or ``'polar'``.
        - ``hidden_channels`` : int, default 64.
        - ``full_res`` : bool, default ``False``. Reserved for future use.

        By default ``None``; no mode modulation is applied.

    References
    ----------
    .. [1] Li, Z. et al. "Fourier Neural Operator for Parametric Partial Differential
        Equations" (2021). ICLR 2021, https://arxiv.org/pdf/2010.08895.

    .. [2] Kossaifi, J., Kovachki, N., Azizzadenesheli, K., Anandkumar, A. "Multi-Grid
        Tensorized Fourier Neural Operator for High-Resolution PDEs" (2024).
        TMLR 2024, https://openreview.net/pdf?id=AWiDlO63bH.
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
        device=None,
        embed: Optional[dict] = None,
        mode_modulation: Optional[dict] = None,
        n_params: int = 1,
    ):
        super().__init__(device=device)

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.complex_data = complex_data

        # n_modes is the total number of modes kept along each dimension
        self.n_modes = n_modes
        self.order = len(self.n_modes)

        if max_n_modes is None:
            max_n_modes = self.n_modes
        elif isinstance(max_n_modes, int):
            max_n_modes = [max_n_modes]
        self.max_n_modes = max_n_modes

        self.fno_block_precision = fno_block_precision
        self.rank = rank
        self.factorization = factorization
        self.implementation = implementation
        self.enforce_hermitian_symmetry = enforce_hermitian_symmetry

        self.resolution_scaling_factor: Union[None, List[List[float]]] = (
            validate_scaling_factor(resolution_scaling_factor, self.order)
        )

        if init_std == "auto":
            init_std = (2 / (in_channels + out_channels)) ** 0.5

        if isinstance(fixed_rank_modes, bool):
            # If bool, keep the number of layers fixed
            fixed_rank_modes = [0] if fixed_rank_modes else None
        self.fft_norm = fft_norm

        if factorization is None:
            factorization = "Dense"  # No factorization

        if separable:
            if in_channels != out_channels:
                raise ValueError(
                    "To use separable Fourier Conv, in_channels must be equal "
                    f"to out_channels, but got in_channels={in_channels} and "
                    f"out_channels={out_channels}",
                )
            weight_shape = (in_channels, *max_n_modes)
        else:
            weight_shape = (in_channels, out_channels, *max_n_modes)
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
            self.bias = nn.Parameter((
                init_std
                * torch.randn(*((self.out_channels,) + (1,) * self.order))
            ))
        else:
            self.bias = None

        # Optional (t, k)-modulation pathway. When both dicts are None
        # (the default) self.modulator is None and forward is identical to
        # the unmodulated spectral conv.
        self._build_embedding(embed, n_params=n_params)
        self._build_modulator(mode_modulation)

    def _build_embedding(self, embed: Optional[dict], n_params: int = 1) -> None:
        self._k_grid_cache = {}
        if embed is None:
            self.embed_config = None
            return

        # Copy so resolved defaults don't leak back into the caller's dict.
        self.embed_config = dict(embed)

        embed_dim = int(self.embed_config.get("dim", 32))
        alpha = self.embed_config.get("alpha", -2.0)
        r = self.embed_config.get("r", 10000.0)
        type_t = self.embed_config.get("type_t", "sinusoidal")
        type_k = self.embed_config.get("type_k", "power")

        self.embed_config["dim"] = embed_dim
        self.embed_config["alpha"] = alpha
        self.embed_config["r"] = r
        self.embed_config["type_t"] = type_t
        self.embed_config["type_k"] = type_k

        self.embed_dim = embed_dim
        self.n_params = n_params

        if type_t == "power":
            self.register_buffer("t_powers", torch.linspace(alpha, 0.0, embed_dim))
        elif type_t == "sinusoidal":
            indices = torch.arange(0, embed_dim // 2, dtype=torch.float32)
            self.register_buffer("t_inv_freqs", r ** (-2.0 * indices / embed_dim))
        else:
            raise ValueError(f"Unknown embed['type_t']: {type_t!r}")

        if type_k == "power":
            self.register_buffer("k_powers", torch.linspace(alpha, 0.0, embed_dim))
        elif type_k == "sinusoidal":
            indices = torch.arange(0, embed_dim // 2, dtype=torch.float32)
            self.register_buffer("k_inv_freqs", r ** (-2.0 * indices / embed_dim))
        else:
            raise ValueError(f"Unknown embed['type_k']: {type_k!r}")

    def _build_modulator(self, mode_modulation: Optional[dict]) -> None:
        self.mode_modulation_config = mode_modulation
        if mode_modulation is None or not mode_modulation.get("enabled", True):
            self.modulator = None
            return

        if self.embed_config is None:
            raise ValueError(
                "mode_modulation is enabled but `embed` is None. "
                "Both must be provided to enable mode modulation."
            )

        self.modulation_type = mode_modulation.get("type")
        self.modulation_hidden_channels = mode_modulation.get("hidden_channels", 64)
        self.modulation_full_res = mode_modulation.get("full_res", False)

        # Input to modulator: D features for t plus n_dims * D for k.
        in_features = self.embed_dim * (self.order + self.n_params)

        if self.modulation_type in ("real", "polar"):
            mod_out_channels = self.in_channels
        elif self.modulation_type == "complex":
            mod_out_channels = self.in_channels * 2
        else:
            raise ValueError(
                f"Unknown mode_modulation['type']: {self.modulation_type!r}"
            )

        self.modulator = ChannelMLP(
            in_channels=in_features,
            out_channels=mod_out_channels,
            hidden_channels=self.modulation_hidden_channels,
            n_dim=self.order,
        )

    def embed_t(self, t: torch.Tensor, shape: Tuple[int, ...]) -> torch.Tensor:
        """Embed scalar time ``t`` and broadcast over a spatial shape.

        Parameters
        ----------
        t : torch.Tensor
            Shape ``(B, 1)``.
        shape : tuple of int
            Spatial shape to broadcast to.

        Returns
        -------
        torch.Tensor of shape ``(B, D, *shape)``.
        """
        embed_type = self.embed_config["type_t"]
        batch_size = t.shape[0]

        embeds = []
        for p in range(self.n_params):
            tp = t[:, p: p + 1]  # (B, 1)
            if embed_type == "power":
                # Power embedding is only defined for positive t (it raises t to
                # a negative-to-zero range of exponents). Fail loudly rather than
                # silently zeroing the embedding for t <= 0. Sinusoidal is valid
                # for any real t, so it is intentionally left unguarded.
                if not torch.all(tp > 0):
                    raise ValueError(
                        "embed['type_t']='power' requires t > 0; "
                        f"got t[:, {p}].min()={tp.min().item()}."
                    )
                tp_embed = tp ** self.t_powers.unsqueeze(0)
            else:  # sinusoidal
                tp_scaled = tp * self.t_inv_freqs.unsqueeze(0)
                tp_embed = torch.cat([torch.sin(tp_scaled), torch.cos(tp_scaled)], dim=-1)
            embeds.append(tp_embed)

        t_embed = torch.cat(embeds, dim=-1)
        return t_embed.reshape(batch_size, -1, *([1] * len(shape))).expand(
            batch_size, -1, *shape
        )

    def embed_k(self, shape: Tuple[int, ...], device=None) -> torch.Tensor:
        """Embed the per-axis frequency-mode index grid for the kept modes.

        Parameters
        ----------
        shape : tuple of int
            Shape of the kept-mode grid; for real FFT the last axis is
            ``S_N // 2 + 1``.
        device : torch.device or None
            Device for the returned tensor.

        Returns
        -------
        torch.Tensor of shape ``(1, n_dims * D, *shape)``.
        """
        embed_type = self.embed_config["type_k"]
        n_dims = len(shape)

        cache_key = (tuple(shape), device)
        if cache_key not in self._k_grid_cache:
            k_ranges = []
            for i, Si in enumerate(shape):
                if i < n_dims - 1:
                    modes_i = Si // 2
                    k_ranges.append(torch.arange(-modes_i, Si - modes_i, device=device))
                else:
                    k_ranges.append(torch.arange(0, Si, device=device))
            k_grid = torch.stack(
                torch.meshgrid(*k_ranges, indexing="ij"), dim=0
            ).unsqueeze(0)
            self._k_grid_cache[cache_key] = k_grid
        else:
            k_grid = self._k_grid_cache[cache_key]

        if embed_type == "power":
            sign = torch.sign(k_grid)
            k_embed = sign.unsqueeze(2) * (
                k_grid.abs().clamp_min(1.0).unsqueeze(2)
                ** self.k_powers.view(1, 1, -1, *([1] * n_dims))
            )
        else:  # 'sinusoidal'
            k_scaled = k_grid.unsqueeze(2) * self.k_inv_freqs.view(
                1, 1, -1, *([1] * n_dims)
            )
            k_embed = torch.cat([torch.sin(k_scaled), torch.cos(k_scaled)], dim=2)

        return k_embed.reshape(1, -1, *shape)

    def _modulation_factor(
        self, t_feature: torch.Tensor, k_feature: torch.Tensor
    ) -> torch.Tensor:
        batch_size = t_feature.shape[0]
        spatial_shape = t_feature.shape[2:]

        combined = torch.cat(
            [t_feature, k_feature.expand(batch_size, -1, *spatial_shape)],
            dim=1,
        )

        if self.modulation_type == "real":
            return self.modulator(combined)
        if self.modulation_type == "complex":
            mlp_out = self.modulator(combined)
            # Complex modulator output was sized to 2 * in_channels in
            # `_build_modulator`; the first half is the real part and the
            # second half is the imaginary part of a per-mode complex
            # multiplier. The factor multiplies the kept FFT input, which
            # has `in_channels` channels, so the split width must be
            # `in_channels`.
            return torch.complex(
                mlp_out[:, : self.in_channels, ...],
                mlp_out[:, self.in_channels:, ...],
            )
        # 'polar'
        theta = self.modulator(combined)
        return torch.exp(1j * theta)

    def clear_cache(self) -> None:
        """Drop any cached frequency-mode index grids."""
        self._k_grid_cache.clear()

    def transform(self, x, output_shape=None):
        in_shape = list(x.shape[2:])

        if self.resolution_scaling_factor is not None and output_shape is None:
            out_shape = tuple(
                round(s * r)
                for (s, r) in zip(in_shape, self.resolution_scaling_factor)
            )
        elif output_shape is not None:
            out_shape = output_shape
        else:
            out_shape = in_shape

        if in_shape == out_shape:
            return x
        else:
            return resample(x, 1.0, list(range(2, x.ndim)), output_shape=out_shape)

    @property
    def n_modes(self):
        return self._n_modes

    @n_modes.setter
    def n_modes(self, n_modes):
        # Should happen for 1D FNO only
        n_modes = [n_modes] if isinstance(n_modes, int) else list(n_modes)
        # the real FFT is skew-symmetric, so the last mode has a redundancy if our data is real in space
        # As a design choice we do the operation here to avoid users dealing with the +1
        # if we use the full FFT we cannot cut off information from the last mode
        if not self.complex_data:
            n_modes[-1] = n_modes[-1] // 2 + 1
        self._n_modes = n_modes

    def forward(
        self,
        x: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        output_shape: Optional[Tuple[int]] = None,
    ):
        """Generic forward pass for the Factorized Spectral Conv.

        Parameters
        ----------
        x : torch.Tensor
            input activation of size (batch_size, channels, d1, ..., dN)
        t : torch.Tensor, optional
            Scalar time of shape ``(B, 1)``. Required when mode modulation
            is enabled (``mode_modulation`` set at construction); ignored
            otherwise.

        Returns
        -------
        tensorized_spectral_conv(x)
        """
        batchsize, channels, *mode_sizes = x.shape

        fft_size = list(mode_sizes)
        if not self.complex_data:
            fft_size[-1] = (
                fft_size[-1] // 2 + 1
            )  # Redundant last coefficient in real spatial data
        fft_dims = list(range(-self.order, 0))

        if self.fno_block_precision == "half":
            x = x.half()

        if self.complex_data:
            x = torch.fft.fftn(x, norm=self.fft_norm, dim=fft_dims)
            dims_to_fft_shift = fft_dims
        else:
            x = torch.fft.rfftn(x, dim=fft_dims, norm=self.fft_norm)
            # When x is real in spatial domain, the last half of the last dim is redundant.
            # See :ref:`fft_shift_explanation` for discussion of the FFT shift.
            dims_to_fft_shift = fft_dims[:-1]

        # Sanitize the gradient that flows into the forward transform's backward
        # so MKL's DFTI never sees non-contiguous strides (see
        # _ContiguousBackward). No-op in the forward pass.
        x = _ContiguousBackward.apply(x)

        if self.order > 1:
            x = torch.fft.fftshift(x, dim=dims_to_fft_shift)

        if self.fno_block_precision == "mixed":
            # if 'mixed', the above fft runs in full precision, but the
            # following operations run at half precision
            x = x.chalf()

        if self.fno_block_precision in ["half", "mixed"]:
            out_dtype = torch.chalf
        else:
            out_dtype = torch.cfloat
        out_fft = torch.zeros(
            [batchsize, self.out_channels, *fft_size], device=x.device, dtype=out_dtype
        )

        # if current modes are less than max, start indexing modes closer to the center of the weight tensor
        starts = [
            (max_modes - min(size, n_mode))
            for (size, n_mode, max_modes) in zip(
                fft_size, self.n_modes, self.max_n_modes
            )
        ]
        # if contraction is separable, weights have shape (channels, modes_x, ...)
        # otherwise they have shape (in_channels, out_channels, modes_x, ...)
        slices_w = [slice(None)] if self.separable else [slice(None), slice(None)]
        if self.complex_data:
            slices_w += [
                slice(start // 2, -start // 2) if start else slice(start, None)
                for start in starts
            ]
        else:
            # The last mode already has redundant half removed in real FFT
            slices_w += [
                slice(start // 2, -start // 2) if start else slice(start, None)
                for start in starts[:-1]
            ]
            slices_w += [slice(None, -starts[-1]) if starts[-1] else slice(None)]

        slices_w = tuple(slices_w)
        weight = self.weight[slices_w]

        ### Pick the first n_modes modes of FFT signal along each dim

        # if separable conv, weight tensor only has one channel dim; otherwise drop first two dims (in_channels, out_channels)
        weight_start_idx = 1 if self.separable else 2
        slices_x = [slice(None), slice(None)]  # Batch_size, channels

        for all_modes, kept_modes in zip(
            fft_size, list(weight.shape[weight_start_idx:])
        ):
            # After fft-shift, the 0th frequency is located at n // 2 in each direction
            # We select n_modes modes around the 0th frequency (kept at index n//2) by grabbing indices
            # n//2 - n_modes//2  to  n//2 + n_modes//2       if n_modes is even
            # n//2 - n_modes//2  to  n//2 + n_modes//2 + 1   if n_modes is odd
            center = all_modes // 2
            negative_freqs = kept_modes // 2
            positive_freqs = kept_modes // 2 + kept_modes % 2

            # this slice represents the desired indices along each dim
            slices_x += [slice(center - negative_freqs, center + positive_freqs)]

        if weight.shape[-1] < fft_size[-1]:
            slices_x[-1] = slice(None, weight.shape[-1])
        else:
            slices_x[-1] = slice(None)

        slices_x = tuple(slices_x)

        # Optional (t, k)-modulation: multiply the kept spectral
        # coefficients by a learned multiplier before contraction.
        #
        # Note on Hermitian symmetry: for real-valued spatial data the
        # modulation factor is in general *not* Hermitian (e.g. a polar
        # `exp(1j*theta)` factor with real theta produces non-conjugate
        # values across positive/negative frequencies). The Hermitian
        # symmetry enforcement below (zeroing imag at DC and Nyquist)
        # therefore clips the modulation contribution at those two bins
        # to keep the irfft output real. This is intentional and matches
        # the design: the modulator can shape mid-band frequencies freely
        # but the DC/Nyquist response stays real-valued.
        if self.modulator is not None:
            if t is None:
                raise ValueError(
                    "SpectralConv has mode_modulation enabled; `t` must be provided."
                )
            weight_start_idx = 1 if self.separable else 2
            kept_shape = tuple(weight.shape[weight_start_idx:])
            t_embed = self.embed_t(t, shape=kept_shape)
            k_embed = self.embed_k(shape=kept_shape, device=x.device)
            mod_factor = self._modulation_factor(t_embed, k_embed)
            x_kept = x[slices_x] * mod_factor
        else:
            x_kept = x[slices_x]

        out_fft[slices_x] = self._contract(x_kept, weight, separable=self.separable)

        if self.resolution_scaling_factor is not None and output_shape is None:
            mode_sizes = tuple(
                round(s * r)
                for (s, r) in zip(mode_sizes, self.resolution_scaling_factor)
            )

        if output_shape is not None:
            mode_sizes = output_shape

        if self.order > 1:
            out_fft = torch.fft.ifftshift(out_fft, dim=fft_dims[:-1])

        # Inverse FFT
        if self.complex_data:
            # For complex data, we can use ifftn.
            x = torch.fft.ifftn(out_fft, s=mode_sizes, dim=fft_dims, norm=self.fft_norm)

        else:
            # For real spatial data we need an irfftn. cuFFT can produce
            # line artifacts at DC / Nyquist on certain GPUs and input sizes
            # when calling irfftn directly, so when running on CUDA with
            # `enforce_hermitian_symmetry=True` we split the transform:
            # ifftn over the leading (n-1) dims, explicit Hermitian
            # enforcement on the last axis, then irfft on the last dim.
            if self.enforce_hermitian_symmetry and out_fft.is_cuda:
                out_fft = torch.fft.ifftn(
                    out_fft, s=mode_sizes[:-1], dim=fft_dims[:-1], norm=self.fft_norm
                )
                # Zero imag at DC (and Nyquist if the last spatial size is
                # even) via an out-of-place multiplicative mask, so the
                # saved tensor used by irfft's backward is not mutated.
                last_size = mode_sizes[-1]
                mask = torch.ones(
                    out_fft.shape[-1], dtype=out_fft.real.dtype, device=out_fft.device
                )
                mask[0] = 0.0
                if last_size % 2 == 0:
                    mask[-1] = 0.0
                mask = mask.view(*([1] * (out_fft.ndim - 1)), -1)
                out_fft = torch.complex(out_fft.real, out_fft.imag * mask)

                x = torch.fft.irfft(
                    out_fft, n=last_size, dim=fft_dims[-1], norm=self.fft_norm
                )
            else:
                x = torch.fft.irfftn(
                    out_fft, s=mode_sizes, dim=fft_dims, norm=self.fft_norm
                )

        # Sanitize the gradient that flows into the inverse transform's
        # backward. Without this, `.sum()` / broadcast backward delivers a
        # zero-stride gradient that makes MKL's DFTI raise "Inconsistent
        # configuration parameters" (see _ContiguousBackward).
        x = _ContiguousBackward.apply(x)

        if self.bias is not None:
            x = x + self.bias

        return x
