from typing import List, Optional, Tuple, Union

import torch
from torch import nn
from torch_harmonics import RealSHT, InverseRealSHT

import tensorly as tl
from tensorly.plugins import use_opt_einsum
from tltorch.factorized_tensors.core import FactorizedTensor

from neuralop.utils import validate_scaling_factor
from .base_spectral_conv import BaseSpectralConv
from .channel_mlp import ChannelMLP

tl.set_backend("pytorch")
use_opt_einsum("optimal")

einsum_symbols = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _contract_dense(x, weight, separable=False, dhconv=True):
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
    if dhconv:
        weight_syms.pop()

    eq = "".join(x_syms) + "," + "".join(weight_syms) + "->" + "".join(out_syms)

    if not torch.is_tensor(weight):
        weight = weight.to_tensor()

    return tl.einsum(eq, x, weight)


def _contract_dense_separable(x, weight, separable=True, dhconv=False):
    if not separable:
        raise ValueError("This function is only for separable=True")
    if dhconv:
        return x * weight.unsqueeze(-1)
    return x * weight


def _contract_cp(x, cp_weight, separable=False, dhconv=True):
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

    if dhconv:
        factor_syms += [xs + rank_sym for xs in x_syms[2:-1]]  # x, y, ...
    else:
        factor_syms += [xs + rank_sym for xs in x_syms[2:]]  # x, y, ...

    eq = (
        x_syms + "," + rank_sym + "," + ",".join(factor_syms) + "->" + "".join(out_syms)
    )

    return tl.einsum(eq, x, cp_weight.weights, *cp_weight.factors)


def _contract_tucker(x, tucker_weight, separable=False, dhconv=False):
    order = tl.ndim(x)

    x_syms = str(einsum_symbols[:order])
    out_sym = einsum_symbols[order]
    out_syms = list(x_syms)
    if separable:
        core_syms = einsum_symbols[order + 1 : 2 * order]
        # factor_syms = [einsum_symbols[1]+core_syms[0]] #in only
        factor_syms = [xs + rs for (xs, rs) in zip(x_syms[1:], core_syms)]  # x, y, ...

    elif dhconv:
        core_syms = einsum_symbols[order + 1 : 2 * order]
        out_syms[1] = out_sym
        factor_syms = [
            einsum_symbols[1] + core_syms[0],
            out_sym + core_syms[1],
        ]  # out, in
        factor_syms += [
            xs + rs for (xs, rs) in zip(x_syms[2:], core_syms[2:])
        ]  # x, y, ...
    else:
        core_syms = einsum_symbols[order + 1 : 2 * order + 1]
        out_syms[1] = out_sym
        factor_syms = [
            einsum_symbols[1] + core_syms[0],
            out_sym + core_syms[1],
        ]  # out, in
        factor_syms += [
            xs + rs for (xs, rs) in zip(x_syms[2:], core_syms[2:])
        ]  # x, y, ...

    eq = (
        x_syms
        + ","
        + core_syms
        + ","
        + ",".join(factor_syms)
        + "->"
        + "".join(out_syms)
    )

    return tl.einsum(eq, x, tucker_weight.core, *tucker_weight.factors)


def _contract_tt(x, tt_weight, separable=False, dhconv=False):
    order = tl.ndim(x)

    x_syms = list(einsum_symbols[:order])
    weight_syms = list(x_syms[1:])  # no batch-size
    if not separable:
        weight_syms.insert(1, einsum_symbols[order])  # outputs
        out_syms = list(weight_syms)
        out_syms[0] = x_syms[0]
    else:
        out_syms = list(x_syms)

    if dhconv:
        weight_syms = weight_syms[:-1]  # no batch-size, no y dim

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

    return tl.einsum(eq, x, *tt_weight.factors)


def get_contract_fun(weight, implementation="reconstructed", separable=False):
    """Generic ND implementation of Fourier Spectral Conv contraction

    Parameters
    ----------
    weight : tensorly-torch's FactorizedTensor
    implementation : {'reconstructed', 'factorized'}, default is 'reconstructed'
        whether to reconstruct the weight and do a forward pass (reconstructed)
        or contract directly the factors of the factorized weight with the input
        (factorized)
    separable : bool
        whether to use the separable implementation of contraction. This arg is
        only checked when `implementation=reconstructed`.

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
            f'Got implementation={implementation}, expected "reconstructed" or '
            f'"factorized"'
        )


Number = Union[int, float]


class SHT(nn.Module):
    """A wrapper for the Spherical Harmonics transform.

    Allows to call it with an interface similar to that of FFT
    """

    def __init__(self, dtype=torch.float32, device=None):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self._SHT_cache = nn.ModuleDict()
        self._iSHT_cache = nn.ModuleDict()

    def sht(self, x, s=None, norm="ortho", grid="equiangular"):
        *_, height, width = x.shape  # height = latitude, width = longitude
        if s is None:
            if grid == "equiangular":
                modes_width = height // 2
            else:
                modes_width = height
            modes_height = height
        else:
            modes_height, modes_width = s

        cache_key = f"{height}_{width}_{modes_height}_{modes_width}_{norm}_{grid}"

        try:
            sht = self._SHT_cache[cache_key]
        except KeyError:
            sht = (
                RealSHT(
                    nlat=height,
                    nlon=width,
                    lmax=modes_height,
                    mmax=modes_width,
                    grid=grid,
                    norm=norm,
                )
                .to(device=x.device)
                .to(dtype=self.dtype)
            )
            self._SHT_cache[cache_key] = sht

        return sht(x)

    def isht(self, x, s=None, norm="ortho", grid="equiangular"):
        *_, modes_height, modes_width = x.shape  # height = latitude, width = longitude
        if s is None:
            if grid == "equiangular":
                width = modes_width * 2
            else:
                width = modes_width
            height = modes_height
        else:
            height, width = s

        cache_key = f"{height}_{width}_{modes_height}_{modes_width}_{norm}_{grid}"

        try:
            isht = self._iSHT_cache[cache_key]
        except KeyError:
            isht = (
                InverseRealSHT(
                    nlat=height,
                    nlon=width,
                    lmax=modes_height,
                    mmax=modes_width,
                    grid=grid,
                    norm=norm,
                )
                .to(device=x.device)
                .to(dtype=self.dtype)
            )
            self._iSHT_cache[cache_key] = isht

        return isht(x)


class SphericalConv(BaseSpectralConv):
    """Spherical Convolution for the SFNO.

    It is implemented as described in [1]_.

    Parameters
    ----------
    sht_norm : str, {'ortho'}
    sht_grids : str or str list, default is "equiangular", {"equiangular", "legendre-gauss"}
                * If str, the same grid is used for all layers
                * If list, should have n_layers + 1 values, corresponding to the input and output grid of each layer
                  e.g. for 1 layer, ["input_grid", "output_grid"]

    embed : dict or None, optional
        Configuration for the scalar-time and harmonic-mode embeddings used
        by the optional ``(t, l, m)`` mode-modulation pathway. Required when
        ``mode_modulation`` is provided. See :class:`SpectralConv` for the
        dict schema; ``type_k`` is applied to the harmonic indices.
        By default ``None``; no mode modulation is applied.
    mode_modulation : dict or None, optional
        Configuration for the optional per-mode modulation MLP. Adds two
        spherical-specific keys on top of the schema described in
        :class:`SpectralConv`:

        - ``share_m`` : bool, default ``True``. If ``True`` the modulation
          depends only on degree ``l`` and is shared across orders ``m``,
          matching :class:`SphericalConv`'s weight sharing across ``m``.
        - ``pre_modulate`` : bool, default ``True``. If ``True`` apply the
          modulation factor before the contraction (acts on ``in_channels``);
          otherwise apply it after (acts on ``out_channels``).

        By default ``None``; no mode modulation is applied.

    See SpectralConv for full list of other parameters.

    References
    ----------
    .. [1] Spherical Fourier Neural Operators: Learning Stable Dynamics on the Sphere,
           Boris Bonev, Thorsten Kurth, Christian Hundt, Jaideep Pathak, Maximilian Baust, Karthik Kashinath, Anima Anandkumar,
           ICML 2023.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        n_modes,
        max_n_modes=None,
        bias=True,
        separable=False,
        resolution_scaling_factor: Optional[Union[Number, List[Number]]] = None,
        fno_block_precision="full",
        rank=0.5,
        factorization="cp",
        implementation="reconstructed",
        fixed_rank_modes=False,
        joint_factorization=False,
        decomposition_kwargs=dict(),
        init_std="auto",
        sht_norm="ortho",
        sht_grids="equiangular",
        device=None,
        dtype=torch.float32,
        complex_data=False,  # dummy param until we unify dtype interface
        embed: Optional[dict] = None,
        mode_modulation: Optional[dict] = None,
    ):
        super().__init__(dtype=dtype, device=device)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.joint_factorization = joint_factorization

        if isinstance(n_modes, int):
            n_modes = [n_modes]
        self._n_modes = n_modes
        self.order = len(n_modes)

        if max_n_modes is None:
            max_n_modes = self.n_modes
        elif isinstance(max_n_modes, int):
            max_n_modes = [max_n_modes]
        self.max_n_modes = max_n_modes

        self.rank = rank
        self.factorization = factorization
        self.implementation = implementation

        self.resolution_scaling_factor: Union[None, List[List[float]]] = (
            validate_scaling_factor(resolution_scaling_factor, self.order)
        )

        if init_std == "auto":
            init_std = (2 / (in_channels + out_channels)) ** 0.5

        if isinstance(fixed_rank_modes, bool):
            if fixed_rank_modes:
                # If bool, keep the number of layers fixed
                fixed_rank_modes = [0]
            else:
                fixed_rank_modes = None

        # Make sure we are using a Complex Factorized Tensor to parametrize the conv
        if factorization is None:
            factorization = "Dense"  # No factorization
        if not factorization.lower().startswith("complex"):
            factorization = f"Complex{factorization}"

        if separable:
            if in_channels != out_channels:
                raise ValueError(
                    "To use separable Fourier Conv, in_channels must be equal "
                    f"to out_channels, but got in_channels={in_channels} "
                    f"and out_channels={out_channels}",
                )
            weight_shape = (in_channels, *self.n_modes[:-1])
        else:
            weight_shape = (in_channels, out_channels, *self.n_modes[:-1])
        self.separable = separable
        self.weight = FactorizedTensor.new(
            weight_shape,
            rank=self.rank,
            factorization=factorization,
            fixed_rank_modes=fixed_rank_modes,
            **decomposition_kwargs,
        )
        self.weight.normal_(0, init_std)

        self._contract = get_contract_fun(
            self.weight, implementation=implementation, separable=separable
        )

        if bias:
            self.bias = nn.Parameter(
                init_std
                * torch.randn(*(tuple([self.out_channels]) + (1,) * self.order))
            )
        else:
            self.bias = None

        self.sht_norm = sht_norm
        if isinstance(sht_grids, str):
            sht_grids = [sht_grids] * 2
        self.sht_grids = sht_grids
        self.sht_handle = SHT(dtype=self.dtype, device=self.device)

        # Optional (t, l, m)-modulation pathway. When both dicts are None
        # (the default) self.modulator is None and forward is identical to
        # the unmodulated spherical conv.
        self._build_embedding(embed)
        self._build_modulator(mode_modulation)

    def _build_embedding(self, embed: Optional[dict]) -> None:
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
        self.share_m = bool(mode_modulation.get("share_m", True))
        self.pre_modulate = bool(mode_modulation.get("pre_modulate", True))

        # Input to modulator: D features for t, plus k features for one axis
        # (share_m=True) or both axes (share_m=False).
        n_k_axes = 1 if self.share_m else self.order
        in_features = self.embed_dim * (1 + n_k_axes)

        # With pre_modulate=True the modulation factor acts on the input to
        # the contraction (in_channels); otherwise on the output (out_channels).
        self.mod_target_channels = (
            self.in_channels if self.pre_modulate else self.out_channels
        )

        if self.modulation_type in ("real", "polar"):
            mod_out_channels = self.mod_target_channels
        elif self.modulation_type == "complex":
            mod_out_channels = self.mod_target_channels * 2
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
        """Embed scalar time ``t`` and broadcast over a spectral shape."""
        embed_type = self.embed_config["type_t"]
        batch_size = t.shape[0]

        if embed_type == "power":
            # Power embedding is only defined for positive t (it raises t to a
            # negative-to-zero range of exponents). Fail loudly rather than
            # silently zeroing the embedding for t <= 0.
            if not torch.all(t > 0):
                raise ValueError(
                    "embed['type_t']='power' requires t > 0; "
                    f"got t.min()={t.min().item()}."
                )
            t_embed = t ** self.t_powers.unsqueeze(0)
        else:  # 'sinusoidal'
            t_scaled = t * self.t_inv_freqs.unsqueeze(0)
            t_embed = torch.cat([torch.sin(t_scaled), torch.cos(t_scaled)], dim=-1)

        return t_embed.reshape(batch_size, -1, *([1] * len(shape))).expand(
            batch_size, -1, *shape
        )

    def embed_k(self, shape: Tuple[int, ...], device=None) -> torch.Tensor:
        """Embed harmonic-mode indices for the kept ``(l, m)`` coefficients.

        Returns
        -------
        torch.Tensor of shape ``(1, n_k_axes * D, L, M)`` where
        ``n_k_axes`` is 1 if ``share_m`` else 2.
        """
        embed_type = self.embed_config["type_k"]
        n_dims = len(shape)
        cache_key = (tuple(shape), bool(self.share_m), device)

        if cache_key not in self._k_grid_cache:
            if self.share_m:
                # Single l-axis grid, broadcast across m.
                l_grid = (
                    torch.arange(0, shape[0], device=device)
                    .reshape(1, 1, shape[0], 1)
                    .expand(1, 1, shape[0], shape[1])
                )
                k_grid = l_grid
            else:
                k_ranges = [
                    torch.arange(0, shape[0], device=device),
                    torch.arange(0, shape[1], device=device),
                ]
                k_grid = torch.stack(
                    torch.meshgrid(*k_ranges, indexing="ij"), dim=0
                ).unsqueeze(0)
            self._k_grid_cache[cache_key] = k_grid
        else:
            k_grid = self._k_grid_cache[cache_key]

        if embed_type == "power":
            # (l, m) indices are non-negative; sign is always +1 here.
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
            # Complex modulator output was sized to 2 * mod_target_channels in
            # `_build_modulator`. The split width is `mod_target_channels`,
            # which equals `in_channels` when `pre_modulate=True` (the
            # multiplier acts on the pre-contraction input) and
            # `out_channels` when `pre_modulate=False` (acts on the
            # post-contraction output).
            n = self.mod_target_channels
            return torch.complex(mlp_out[:, :n, ...], mlp_out[:, n:, ...])
        # 'polar'
        theta = self.modulator(combined)
        return torch.exp(1j * theta)

    def clear_cache(self) -> None:
        """Drop any cached harmonic-mode grids."""
        self._k_grid_cache.clear()

    def transform(self, x, output_shape=None):
        *_, in_height, in_width = x.shape

        if self.resolution_scaling_factor is not None and output_shape is None:
            height = round(in_height * self.resolution_scaling_factor[0])
            width = round(in_width * self.resolution_scaling_factor[1])
        elif output_shape is not None:
            height, width = output_shape[0], output_shape[1]
        else:
            height, width = in_height, in_width

        # Return the identity if the resolution and grid of the input and output are the same
        if ((in_height, in_width) == (height, width)) and (
            self.sht_grids[0] == self.sht_grids[1]
        ):
            return x
        else:
            coefs = self.sht_handle.sht(
                x, s=self.n_modes, norm=self.sht_norm, grid=self.sht_grids[0]
            )
            return self.sht_handle.isht(
                coefs, s=(height, width), norm=self.sht_norm, grid=self.sht_grids[1]
            )

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
        batchsize, channels, height, width = x.shape

        if self.resolution_scaling_factor is not None and output_shape is None:
            scaling_factors = self.resolution_scaling_factor
            height = round(height * scaling_factors[0])
            width = round(width * scaling_factors[1])
        elif output_shape is not None:
            height, width = output_shape[0], output_shape[1]

        out_fft = self.sht_handle.sht(
            x,
            s=(self.n_modes[0], self.n_modes[1] // 2),
            norm=self.sht_norm,
            grid=self.sht_grids[0],
        )

        if self.modulator is None:
            out_fft = self._contract(
                out_fft[:, :, : self.n_modes[0], : self.n_modes[1] // 2],
                self.weight[:, :, : self.n_modes[0]],
                separable=self.separable,
                dhconv=True,
            )
        else:
            if t is None:
                raise ValueError(
                    "SphericalConv has mode_modulation enabled; `t` must be provided."
                )

            # Recent torch-harmonics applies triangular truncation that can
            # return fewer modes than requested. Clamp to the actual SHT
            # output shape so the modulation factor and weight slice all
            # share the same spectral dimensions before contraction.
            modes_height = min(self.n_modes[0], out_fft.shape[-2])
            modes_width = min(self.n_modes[1] // 2, out_fft.shape[-1])
            out_fft = out_fft[:, :, :modes_height, :modes_width]

            kept_shape = (modes_height, modes_width)
            t_embed = self.embed_t(t, shape=kept_shape)
            k_embed = self.embed_k(shape=kept_shape, device=x.device)
            mod_factor = self._modulation_factor(t_embed, k_embed)

            if self.pre_modulate:
                out_fft = out_fft * mod_factor
            out_fft = self._contract(
                out_fft,
                self.weight[:, :, :modes_height],
                separable=self.separable,
                dhconv=True,
            )
            if not self.pre_modulate:
                out_fft = out_fft * mod_factor

        x = self.sht_handle.isht(
            out_fft, s=(height, width), norm=self.sht_norm, grid=self.sht_grids[1]
        )

        if self.bias is not None:
            x = x + self.bias

        return x

    @property
    def n_modes(self):
        return self._n_modes

    @n_modes.setter
    def n_modes(self, n_modes):
        if isinstance(n_modes, int):  # Should happen for 1D FNO only
            n_modes = [n_modes]
        else:
            n_modes = list(n_modes)
        self._n_modes = n_modes
