"""Klein-Gordon Spectral Convolution Layer

A physics-constrained spectral convolution layer that encodes the
Klein-Gordon dispersion relation as a learnable spectral filter.

Instead of learning arbitrary complex weights in Fourier space
(as in the standard FNO SpectralConv), this layer parameterizes the
spectral filter via the exact solution operator of the Klein-Gordon
equation:

    H(k) = exp(-i T sqrt(c^2 |k|^2 + chi^2))

with per-channel learnable (T, c, chi) parameters, plus per-mode
learnable amplitude weights. This makes it dramatically more
parameter-efficient for wave-type (hyperbolic) PDEs while retaining
enough expressiveness for practical learning.

Like the standard SpectralConv, this layer operates only on the
first ``n_modes`` low-frequency modes and zeros out higher
frequencies (the skip connection in FNOBlocks handles the residual).
For multi-dimensional inputs, mode truncation uses centered frequency
windows via ``fftshift`` (matching SpectralConv behavior).

Mathematical background
-----------------------
The Klein-Gordon equation d^2u/dt^2 = c^2 nabla^2 u - chi^2 u has
the dispersion relation omega^2 = c^2 |k|^2 + chi^2. Its Green's
function in d dimensions is the Matern kernel family [1]_:

- Matern(nu=0.5) = exp(-r/l)            ... 1D KG Green's function
- Matern(nu=1.5) = (1+sqrt(3)r/l)e^(-sqrt(3)r/l)  ... 3D KG
- Matern(nu -> inf) = exp(-r^2/2l^2)    ... RBF (diffusion limit)

References
----------
.. [1] Whittle, P. "On stationary processes in the plane" (1954).
    Biometrika, 41(3-4), 434-449.

.. [2] Rasmussen, C.E. & Williams, C.K.I. "Gaussian Processes for
    Machine Learning" (2006). MIT Press. Chapter 4.

.. [3] Li, Z. et al. "Fourier Neural Operator for Parametric Partial
    Differential Equations" (2021). ICLR 2021.
"""

import warnings
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn

from ..utils import validate_scaling_factor
from .base_spectral_conv import BaseSpectralConv

Number = Union[int, float]


class KGSpectralConv(BaseSpectralConv):
    """Klein-Gordon Spectral Convolution

    Applies a physics-constrained spectral filter based on the Klein-Gordon
    dispersion relation, with per-mode learnable amplitudes.

    The spectral weight for output channel ``o`` at wavenumber ``k`` is::

        W_o(k) = alpha_o(k) * exp(-i T_o sqrt(c_o^2 |k|^2 + chi_o^2))

    where ``alpha_o(k)`` is a learnable complex amplitude per mode, and
    ``(T_o, c_o, chi_o)`` are per-channel dispersion parameters.

    This is a drop-in replacement for
    :class:`~neuralop.layers.spectral_convolution.SpectralConv`. It
    matches SpectralConv's mode-truncation behavior: centered frequency
    windows via ``fftshift`` for multi-D inputs, rFFT-aware last-dimension
    handling, and ``max_n_modes`` support for incremental training.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    n_modes : int or tuple of int
        Number of Fourier modes to retain along each spatial dimension.
        For real-valued data, the last dimension is automatically adjusted
        to ``n // 2 + 1`` to match rFFT output size (same as SpectralConv).
    max_n_modes : int or tuple of int or None, optional
        Maximum number of modes for parameter allocation (for incremental
        training). If None, uses ``n_modes``. Default: None.
    init_T : float, optional
        Initial propagation time. Default: 1.0.
    init_c : float, optional
        Initial wave speed. Default: 1.0.
    init_chi : float, optional
        Initial mass/dispersion. Default: 0.1.
    bias : bool, optional
        If True, add a learnable bias. Default: True.
    complex_data : bool, optional
        If True, input is complex in the spatial domain. Default: False.
    fft_norm : str, optional
        FFT normalization mode. Default: ``'forward'``.
    resolution_scaling_factor : float or list of float or None, optional
        Resolution scaling for super-resolution workflows. Applied in
        both ``transform()`` and ``forward()``. Default: None.
    fno_block_precision : str or None, optional
        Precision mode. Currently only ``'full'`` is supported; non-default
        values emit a warning. Default: None.
    device : torch.device or None, optional
        Device for parameters. Default: None.

    Examples
    --------
    >>> layer = KGSpectralConv(in_channels=3, out_channels=3, n_modes=(16,))
    >>> x = torch.randn(4, 3, 64)
    >>> y = layer(x)
    >>> y.shape
    torch.Size([4, 3, 64])

    >>> layer_2d = KGSpectralConv(in_channels=1, out_channels=1, n_modes=(16, 16))
    >>> x2d = torch.randn(2, 1, 64, 64)
    >>> y2d = layer_2d(x2d)
    >>> y2d.shape
    torch.Size([2, 1, 64, 64])
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_modes: Union[int, Tuple[int, ...]],
        init_T: float = 1.0,
        init_c: float = 1.0,
        init_chi: float = 0.1,
        bias: bool = True,
        complex_data: bool = False,
        fft_norm: str = "forward",
        device=None,
        # Backward compat: ignored old arg
        per_channel=None,
        # FNOBlocks compatibility kwargs
        max_n_modes=None,
        rank=None,
        factorization=None,
        implementation=None,
        separable=None,
        resolution_scaling_factor=None,
        fno_block_precision=None,
        fixed_rank_modes=None,
        decomposition_kwargs=None,
        init_std=None,
        **kwargs,
    ):
        super().__init__(device=device)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.complex_data = complex_data
        self.fft_norm = fft_norm
        self.fno_block_precision = fno_block_precision

        if isinstance(n_modes, int):
            n_modes = [n_modes]
        self.order = len(n_modes)

        # n_modes setter adjusts last dim for rFFT (matches SpectralConv)
        self.n_modes = n_modes

        # max_n_modes for alpha allocation (matches SpectralConv pattern)
        if max_n_modes is not None:
            if isinstance(max_n_modes, int):
                max_n_modes = [max_n_modes] * self.order
            else:
                max_n_modes = list(max_n_modes)
            if not self.complex_data:
                max_n_modes[-1] = max_n_modes[-1] // 2 + 1
            self.max_n_modes = max_n_modes
        else:
            self.max_n_modes = list(self._n_modes)

        # resolution_scaling_factor (matches SpectralConv)
        self.resolution_scaling_factor: Union[
            None, List[float]
        ] = validate_scaling_factor(resolution_scaling_factor, self.order)

        # Warn about unsupported precision modes
        if fno_block_precision is not None and fno_block_precision != "full":
            warnings.warn(
                f"KGSpectralConv does not yet support fno_block_precision="
                f"'{fno_block_precision}'. Falling back to full precision. "
                f"For mixed/half precision, use SpectralConv.",
                stacklevel=2,
            )

        # Per-channel KG dispersion parameters (log-space for positivity)
        self.log_T = nn.Parameter(
            torch.full(
                (out_channels,), float(np.log(max(init_T, 1e-8))), device=device
            )
        )
        self.log_c = nn.Parameter(
            torch.full(
                (out_channels,), float(np.log(max(init_c, 1e-8))), device=device
            )
        )
        self.log_chi = nn.Parameter(
            torch.full(
                (out_channels,), float(np.log(max(init_chi, 1e-8))), device=device
            )
        )

        # Per-mode learnable complex amplitude: (out_channels, *max_n_modes)
        alpha_shape = (out_channels, *self.max_n_modes)
        self.alpha_real = nn.Parameter(torch.ones(alpha_shape, device=device))
        self.alpha_imag = nn.Parameter(torch.zeros(alpha_shape, device=device))

        # Channel mixing (mode-independent): (in_channels, out_channels)
        fan_std = (2 / (in_channels + out_channels)) ** 0.5
        self.channel_weight = nn.Parameter(
            fan_std * torch.randn(in_channels, out_channels, device=device)
        )

        if bias:
            self.bias = nn.Parameter(
                fan_std
                * torch.randn(*((out_channels,) + (1,) * self.order), device=device)
            )
        else:
            self.bias = None

    def transform(self, x, output_shape=None):
        """Transform input for skip connections (identity or resample).

        Matches SpectralConv: when ``output_shape`` is not given, applies
        ``resolution_scaling_factor`` if set.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        output_shape : tuple of int or None
            Target spatial shape.
        """
        in_shape = list(x.shape[2:])
        if self.resolution_scaling_factor is not None and output_shape is None:
            out_shape = tuple(
                round(s * r)
                for s, r in zip(in_shape, self.resolution_scaling_factor)
            )
        elif output_shape is not None:
            out_shape = output_shape
        else:
            out_shape = tuple(in_shape)

        if list(out_shape) == in_shape:
            return x
        from .resample import resample

        return resample(
            x, 1.0, list(range(2, x.ndim)), output_shape=list(out_shape)
        )

    @property
    def n_modes(self):
        return self._n_modes

    @n_modes.setter
    def n_modes(self, n_modes):
        if isinstance(n_modes, int):
            n_modes = [n_modes]
        else:
            n_modes = list(n_modes)
        # Match SpectralConv: auto-adjust last dim for rFFT redundancy
        if not self.complex_data:
            n_modes[-1] = n_modes[-1] // 2 + 1
        self._n_modes = n_modes

    def _compute_kg_filter(self, kept_sizes, fft_sizes, spatial_sizes, shifted_dims):
        """Compute the KG spectral filter on the truncated frequency grid.

        Builds correct wavenumber grids accounting for ``fftshift`` on
        multi-D dims and non-shifted rFFT last dimension.

        Parameters
        ----------
        kept_sizes : list of int
            Number of modes kept per dimension.
        fft_sizes : list of int
            Full FFT output sizes per dimension.
        spatial_sizes : list of int
            Full spatial dimensions of the input.
        shifted_dims : set of int
            Spatial dim indices (0-based) that were fftshift'd.

        Returns
        -------
        H : torch.Tensor
            Complex spectral filter of shape ``(out_channels, *kept_sizes)``.
        """
        T = torch.exp(self.log_T)
        c = torch.exp(self.log_c)
        chi = torch.exp(self.log_chi)

        freq_components = []
        for i, (kept, fft_sz, sp_sz) in enumerate(
            zip(kept_sizes, fft_sizes, spatial_sizes)
        ):
            if i in shifted_dims:
                # Centered window: frequencies symmetric around DC
                freq_idx = torch.arange(kept, device=T.device, dtype=torch.float32)
                freq_idx = freq_idx - (kept // 2)
            else:
                # Non-shifted (rFFT last dim): 0, 1, ..., kept-1
                freq_idx = torch.arange(kept, device=T.device, dtype=torch.float32)
            freqs = freq_idx * (2 * np.pi / sp_sz)
            freq_components.append(freqs)

        grids = torch.meshgrid(*freq_components, indexing="ij")
        k_squared = sum(g**2 for g in grids)  # |k|^2

        ndim = len(kept_sizes)
        shape = (-1,) + (1,) * ndim
        omega = torch.sqrt(
            c.view(shape) ** 2 * k_squared.unsqueeze(0) + chi.view(shape) ** 2
        )

        phase = -T.view(shape) * omega
        H = torch.complex(torch.cos(phase), torch.sin(phase))

        # Slice alpha from max_n_modes to current kept_sizes
        # (symmetric cropping matching SpectralConv for incremental training)
        alpha = torch.complex(self.alpha_real, self.alpha_imag)
        starts = [
            max_nm - kept for max_nm, kept in zip(self.max_n_modes, kept_sizes)
        ]
        alpha_slices = [slice(None)]  # out_channels dim
        for s in starts:
            if s:
                alpha_slices.append(slice(s // 2, -s // 2))
            else:
                alpha_slices.append(slice(None))
        alpha_trunc = alpha[tuple(alpha_slices)]
        H = alpha_trunc * H

        return H

    @property
    def n_kernel_params(self):
        """Number of learnable parameters in the spectral kernel."""
        n_kg = self.log_T.numel() + self.log_c.numel() + self.log_chi.numel()
        n_alpha = self.alpha_real.numel() + self.alpha_imag.numel()
        return n_kg + n_alpha

    @property
    def n_total_params(self):
        """Total learnable parameters including channel mixing and bias."""
        total = self.n_kernel_params + self.channel_weight.numel()
        if self.bias is not None:
            total += self.bias.numel()
        return total

    def forward(
        self,
        x: torch.Tensor,
        output_shape: Optional[Tuple[int, ...]] = None,
    ) -> torch.Tensor:
        """Apply the Klein-Gordon spectral filter.

        Matches SpectralConv mode-truncation: uses centered frequency
        windows via ``fftshift`` for multi-D inputs, and rFFT-aware
        last-dimension handling.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(batch, in_channels, n1, ..., nd)``.
        output_shape : tuple of int or None, optional
            Target spatial dimensions for the output.

        Returns
        -------
        torch.Tensor
            Output of shape ``(batch, out_channels, n1, ..., nd)``.
        """
        batchsize, channels, *spatial_sizes = x.shape
        fft_dims = list(range(-self.order, 0))

        # Forward FFT
        if self.complex_data:
            x_hat = torch.fft.fftn(x, norm=self.fft_norm, dim=fft_dims)
            dims_to_shift = fft_dims
        else:
            x_hat = torch.fft.rfftn(x, norm=self.fft_norm, dim=fft_dims)
            dims_to_shift = fft_dims[:-1]  # Skip last dim (rFFT)

        # fftshift for centered mode selection (2D+)
        if self.order > 1 and len(dims_to_shift) > 0:
            x_hat = torch.fft.fftshift(x_hat, dim=dims_to_shift)

        fft_sizes = list(x_hat.shape[2:])

        # Track which spatial dims (0-indexed) were shifted
        shifted_dims = set()
        if self.order > 1:
            if self.complex_data:
                shifted_dims = set(range(self.order))
            else:
                shifted_dims = set(range(self.order - 1))

        # Allocate zero-padded output in Fourier space
        out_hat = torch.zeros(
            batchsize,
            self.out_channels,
            *fft_sizes,
            device=x.device,
            dtype=torch.cfloat,
        )

        # Mode truncation
        kept_sizes = [min(nm, fs) for nm, fs in zip(self._n_modes, fft_sizes)]

        # Build slices: centered window for shifted dims, low-freq for others
        mode_slices = []
        for i, (kept, fft_sz) in enumerate(zip(kept_sizes, fft_sizes)):
            if i in shifted_dims:
                center = fft_sz // 2
                neg = kept // 2
                pos = kept // 2 + kept % 2
                mode_slices.append(slice(center - neg, center + pos))
            else:
                mode_slices.append(slice(0, kept))

        full_slices = tuple([slice(None), slice(None)] + mode_slices)

        # Extract low-frequency modes from input
        x_low = x_hat[full_slices]

        # Channel mixing: (B, C_in, *k) @ (C_in, C_out) -> (B, C_out, *k)
        w = self.channel_weight.to(x_low.dtype)
        out_low = torch.einsum("bi...,io->bo...", x_low, w)

        # Compute and apply KG spectral filter (complex, per-channel)
        H = self._compute_kg_filter(
            kept_sizes, fft_sizes, spatial_sizes, shifted_dims
        )
        out_low = out_low * H.unsqueeze(0)  # broadcast over batch

        # Place back into zero-padded output
        out_hat[full_slices] = out_low

        # ifftshift before inverse FFT
        if self.order > 1 and len(dims_to_shift) > 0:
            out_hat = torch.fft.ifftshift(out_hat, dim=dims_to_shift)

        # Determine output size
        out_sizes = list(spatial_sizes)
        if self.resolution_scaling_factor is not None and output_shape is None:
            out_sizes = [
                round(s * r)
                for s, r in zip(out_sizes, self.resolution_scaling_factor)
            ]
        if output_shape is not None:
            out_sizes = list(output_shape)

        # Inverse FFT
        if self.complex_data:
            x = torch.fft.ifftn(
                out_hat, s=out_sizes, dim=fft_dims, norm=self.fft_norm
            )
        else:
            x = torch.fft.irfftn(
                out_hat, s=out_sizes, dim=fft_dims, norm=self.fft_norm
            )

        if self.bias is not None:
            x = x + self.bias

        return x

    def extra_repr(self) -> str:
        T = torch.exp(self.log_T).detach()
        c = torch.exp(self.log_c).detach()
        chi = torch.exp(self.log_chi).detach()
        return (
            f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"n_modes={self._n_modes}, max_n_modes={self.max_n_modes}, "
            f"T_range=[{T.min().item():.3g}, {T.max().item():.3g}], "
            f"c_range=[{c.min().item():.3g}, {c.max().item():.3g}], "
            f"chi_range=[{chi.min().item():.3g}, {chi.max().item():.3g}], "
            f"kernel_params={self.n_kernel_params}, "
            f"total_params={self.n_total_params}"
        )
