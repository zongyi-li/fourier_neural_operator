"""Fourier transform backends for spectral convolution layers."""

from dataclasses import dataclass
from abc import ABC, abstractmethod
import warnings

import torch
import numpy as np

from .index_sets import HyperRectangleIndexSet, RadialIndexSet
from .resample import resample


def radius_from_n_modes(index_set, n_modes):
    if isinstance(index_set, RadialIndexSet):
        return index_set.radius_from_n_modes(n_modes)
    return None


@dataclass
class FourierSelection:
    """Indexing plan for selected Fourier coefficients."""
    x_index: tuple
    weight_index: tuple


@dataclass
class FourierTransformState:
    mode_sizes: tuple
    fft_size: tuple
    fft_dims: tuple
    dims_to_fft_shift: tuple


class SpectralTransform(ABC):
    def __init__(
            self,
            order,
            complex_data,
            fft_norm,
    ):
        self.order = order
        self.complex_data = complex_data
        self.fft_norm = fft_norm

    @abstractmethod
    def forward_transform(self, x):
        pass

    @abstractmethod
    def inverse_transform(self, out_fft, mode_sizes, state):
        pass

    @abstractmethod
    def resize_fft(self, x_fft, output_shape, index_set, true_n_modes, state):
        pass

    @abstractmethod
    def transform(self, x, output_shape, index_set, true_n_modes):
        pass

    @abstractmethod
    def weight_shape(self, index_set, max_n_modes):
        pass

    @abstractmethod
    def selection(self,
        index_set,
        fft_size,
        max_n_modes,
        true_n_modes,
        separable=False,
        device=None,
    ):
        pass

    @property
    def data_dim(self):
        return self.order

class RegularGridFFT(SpectralTransform):
    """
    Regular-grid Fourier transform backend used by the default FNO.

    This code was factored out of the SpectralConv class.
    References
    ----------
    .. [1] Li, Z. et al. "Fourier Neural Operator for Parametric Partial Differential
        Equations" (2021). ICLR 2021, https://arxiv.org/pdf/2010.08895.
    """

    def __init__(
        self,
        order, # dimensionality of the FFT
        complex_data=False,
        fft_norm="forward",
        enforce_hermitian_symmetry=True,
    ):
        super().__init__(order=order, complex_data=complex_data, fft_norm=fft_norm)
        self.enforce_hermitian_symmetry = enforce_hermitian_symmetry
        self._selection_cache = {}

    def forward_transform(self, x):
        mode_sizes = tuple(x.shape[2:]) # input is (batch, channel_in, *xshape)
        fft_size = list(mode_sizes)
        if not self.complex_data:
            fft_size[-1] = fft_size[-1] // 2 + 1
        fft_dims = tuple(range(-self.order, 0))

        x = x.contiguous()
        if self.complex_data:
            x_fft = torch.fft.fftn(x, norm=self.fft_norm, dim=fft_dims)
            dims_to_fft_shift = fft_dims
        else:
            x_fft = torch.fft.rfftn(x, norm=self.fft_norm, dim=fft_dims)
            dims_to_fft_shift = fft_dims[:-1]

        if dims_to_fft_shift:
            x_fft = torch.fft.fftshift(x_fft, dim=dims_to_fft_shift)

        state = FourierTransformState(
            mode_sizes=mode_sizes,
            fft_size=tuple(fft_size),
            fft_dims=fft_dims,
            dims_to_fft_shift=tuple(dims_to_fft_shift),
        )
        return x_fft, state

    def inverse_transform(self, out_fft, mode_sizes, state):
        if state.dims_to_fft_shift:
            out_fft = torch.fft.ifftshift(out_fft, dim=state.dims_to_fft_shift)

        if self.complex_data:
            return torch.fft.ifftn(
                out_fft, s=mode_sizes, dim=state.fft_dims, norm=self.fft_norm
            )

        if self.enforce_hermitian_symmetry and self.order > 1:
            out_fft = torch.fft.ifftn(
                out_fft,
                s=mode_sizes[:-1],
                dim=state.fft_dims[:-1],
                norm=self.fft_norm,
            )
            out_fft[..., 0].imag.zero_()
            if mode_sizes[-1] % 2 == 0:
                out_fft[..., -1].imag.zero_()
            return torch.fft.irfft(
                out_fft, n=mode_sizes[-1], dim=state.fft_dims[-1], norm=self.fft_norm
            )

        return torch.fft.irfftn(
            out_fft, s=mode_sizes, dim=state.fft_dims, norm=self.fft_norm
        )

    def resize_fft(self, x_fft, output_shape, index_set, true_n_modes, state):
        return x_fft, tuple(output_shape), state

    def transform(self, x, output_shape, index_set, true_n_modes):
        return resample(
            x,
            1.0,
            list(range(2, x.ndim)),
            output_shape=tuple(output_shape),
        )

    def weight_shape(self, index_set, max_n_modes):
        if isinstance(index_set, HyperRectangleIndexSet):
            return tuple(self._fft_storage_n_modes(max_n_modes))
        return (index_set.n_modes,)

    def _fft_storage_n_modes(self, n_modes):
        n_modes = list(n_modes)
        if not self.complex_data:
            n_modes[-1] = n_modes[-1] // 2 + 1
        return n_modes

    def selection(
        self,
        index_set,
        fft_size,
        max_n_modes,
        true_n_modes,
        separable=False,
        device=None,
    ):
        active_radius = radius_from_n_modes(index_set, true_n_modes)
        cache_key = (
            device,
            id(index_set),
            tuple(fft_size),
            tuple(max_n_modes),
            active_radius,
            self.complex_data,
            separable,
        )
        cached = self._selection_cache.get(cache_key)
        if cached is not None:
            return cached

        if isinstance(index_set, HyperRectangleIndexSet):
            selection = self._hyperrectangle_selection(
                index_set=index_set,
                fft_size=fft_size,
                max_n_modes=max_n_modes,
                true_n_modes=true_n_modes,
                separable=separable,
            )
        else:
            selection = self._indexed_selection(
                index_set=index_set,
                fft_size=fft_size,
                active_radius=active_radius,
                separable=separable,
                device=device,
            )
        self._selection_cache[cache_key] = selection
        return selection

    def _indexed_selection(
        self, index_set, fft_size, active_radius=None, separable=False, device=None
    ):
        modes = index_set.modes(device=device, radius=active_radius)
        weight_indices = torch.arange(modes.shape[0], device=device)

        if not self.complex_data:
            keep = modes[:, -1] >= 0
            modes = modes[keep]
            weight_indices = weight_indices[keep]

        fft_size_tensor = torch.as_tensor(fft_size, device=device, dtype=torch.long)
        fft_indices = modes.clone()
        if self.complex_data:
            fft_indices = fft_indices + fft_size_tensor // 2
        else:
            fft_indices[:, :-1] = fft_indices[:, :-1] + fft_size_tensor[:-1] // 2

        keep = torch.ones(fft_indices.shape[0], device=device, dtype=torch.bool)
        for dim, size in enumerate(fft_size):
            keep &= (fft_indices[:, dim] >= 0) & (fft_indices[:, dim] < size)

        if separable:
            weight_index = (slice(None), weight_indices[keep])
        else:
            weight_index = (slice(None), slice(None), weight_indices[keep])
        x_index = (slice(None), slice(None), *fft_indices[keep].unbind(dim=1))

        return FourierSelection(x_index=x_index, weight_index=weight_index)

    def _hyperrectangle_selection(
        self,
        index_set,
        fft_size,
        max_n_modes,
        true_n_modes,
        separable=False,
    ):
        fft_max_n_modes = self._fft_storage_n_modes(max_n_modes)
        n_modes_per_dim = index_set.n_modes_per_dim_for_radius(
            radius_from_n_modes(index_set, true_n_modes),
            index_set.weights,
        )
        n_modes = self._fft_storage_n_modes(n_modes_per_dim)
        starts = [
            (fft_max_modes - min(size, n_mode))
            for (size, n_mode, fft_max_modes) in zip(
                fft_size, n_modes, fft_max_n_modes
            )
        ]

        if separable:
            slices_w = [slice(None)]
        else:
            slices_w = [slice(None), slice(None)]

        if self.complex_data:
            slices_w += [
                slice(start // 2, -start // 2) if start else slice(start, None)
                for start in starts
            ]
        else:
            slices_w += [
                slice(start // 2, -start // 2) if start else slice(start, None)
                for start in starts[:-1]
            ]
            slices_w += [slice(None, -starts[-1]) if starts[-1] else slice(None)]

        kept_modes = [
            min(size, n_mode, max_modes)
            for (size, n_mode, max_modes) in zip(
                fft_size, n_modes, fft_max_n_modes
            )
        ]
        slices_x = [slice(None), slice(None)]
        for all_modes, kept in zip(fft_size, kept_modes):
            center = all_modes // 2
            negative_freqs = kept // 2
            positive_freqs = kept // 2 + kept % 2
            slices_x += [slice(center - negative_freqs, center + positive_freqs)]

        if not self.complex_data and kept_modes[-1] < fft_size[-1]:
            slices_x[-1] = slice(None, kept_modes[-1])
        elif not self.complex_data:
            slices_x[-1] = slice(None)

        return FourierSelection(
            x_index=tuple(slices_x), weight_index=tuple(slices_w)
        )

class Rank1LatticeFFT(SpectralTransform):
    """Rank-1 lattice transform backend.

    The data transform is one-dimensional, while the index set still contains
    multi-dimensional Fourier modes. A mode k is stored at coefficient
    ``dot(k, z) mod n`` in the one-dimensional FFT.

    References
    ----------
    .. [1] Dilen, J., Keller, A., Kuo, F. Y., Nuyens, D. "Fourier Neural Operators
        with Rank-1 Lattice Points and Hyperbolic Cross" (2026).
        https://arxiv.org/abs/2606.08871.
    """

    def __init__(self, n, z, complex_data=False, fft_norm="forward"):
        super().__init__(order=1, complex_data=complex_data, fft_norm=fft_norm)
        self.n = int(n)
        self.z = torch.as_tensor(z, dtype=torch.long)
        self._selection_cache = {}

    def forward_transform(self, x):
        mode_sizes = tuple(x.shape[2:])
        if len(mode_sizes) != 1:
            raise ValueError(
                "Rank1LatticeFFT expects input shaped as "
                "(batch, channel, x) where x is a one-dimensional tensor "
                "with scalar field values given in the order of the lattice points"
            )

        fft_dims = (-1,)
        if self.complex_data:
            x_fft = torch.fft.fft(x, norm=self.fft_norm, dim=-1)
            fft_size = (mode_sizes[-1],)
        else:
            x_fft = torch.fft.rfft(x, norm=self.fft_norm, dim=-1)
            fft_size = (mode_sizes[-1] // 2 + 1,)

        state = FourierTransformState(
            mode_sizes=mode_sizes,
            fft_size=fft_size,
            fft_dims=fft_dims,
            dims_to_fft_shift=(),
        )
        return x_fft, state

    def inverse_transform(self, out_fft, mode_sizes, state):
        if self.complex_data:
            return torch.fft.ifft(out_fft, n=mode_sizes[-1], norm=self.fft_norm, dim=-1)
        return torch.fft.irfft(out_fft, n=mode_sizes[-1], norm=self.fft_norm, dim=-1)

    def _coefficients(self, modes, fft_size, device, n=None):
        if n is None:
            n = self.n
        z_int = self.z.cpu().numpy().astype(object)   # Python arbitrary precision integers
        modes_int = modes.cpu().numpy().astype(object)
        coefficients = np.sum(modes_int * z_int, axis=1) % n
        coefficients = torch.as_tensor(coefficients.astype(np.int64), device=device, dtype=torch.long)

        if len(torch.unique(coefficients)) != coefficients.numel():
            warnings.warn("Aliasing is taking place in rank-1 lattice coefficients.")

        if not self.complex_data:
            coefficients = torch.where(
                coefficients <= ((n - 1) // 2),
                coefficients,
                n - coefficients,
            )

        keep = coefficients < fft_size[0]
        return coefficients[keep], keep

    def _fft_size_from_n(self, n):
        if self.complex_data:
            return (n,)
        return (n // 2 + 1,)

    def resize_fft(self, x_fft, output_shape, index_set, true_n_modes, state):
        if isinstance(output_shape, int):
            output_shape = (output_shape,)
        if len(output_shape) != 1:
            raise ValueError("Rank1LatticeFFT output_shape must be one-dimensional")

        n_in = state.mode_sizes[-1]
        n_out = output_shape[-1]
        if n_in == n_out:
            return x_fft, state.mode_sizes, state

        active_radius = radius_from_n_modes(index_set, true_n_modes)
        modes = index_set.modes(device=x_fft.device, radius=active_radius)
        source_coefficients, keep_source = self._coefficients(
            modes, state.fft_size, x_fft.device, n=n_in
        )
        target_fft_size = self._fft_size_from_n(n_out)
        target_coefficients, keep_target = self._coefficients(
            modes, target_fft_size, x_fft.device, n=n_out
        )
        source_coefficients = source_coefficients[
            keep_target[keep_source]
        ]
        target_coefficients = target_coefficients[
            keep_source[keep_target]
        ]

        selected = x_fft[..., source_coefficients]
        out_fft = torch.zeros(
            *x_fft.shape[:-1],
            target_fft_size[0],
            device=x_fft.device,
            dtype=x_fft.dtype,
        )
        scatter_index = target_coefficients.reshape(
            *((1,) * (out_fft.ndim - 1)), -1
        ).expand(*out_fft.shape[:-1], -1)
        out_fft = torch.scatter(out_fft, -1, scatter_index, selected)
        target_state = FourierTransformState(
            mode_sizes=(n_out,),
            fft_size=target_fft_size,
            fft_dims=state.fft_dims,
            dims_to_fft_shift=state.dims_to_fft_shift,
        )
        return out_fft, target_state.mode_sizes, target_state

    def transform(
        self,
        x,
        output_shape,
        index_set,
        true_n_modes,
    ):
        x_fft, state = self.forward_transform(x)
        x_fft, mode_sizes, state = self.resize_fft(
            x_fft,
            output_shape=output_shape,
            index_set=index_set,
            true_n_modes=true_n_modes,
            state=state,
        )
        return self.inverse_transform(x_fft, mode_sizes, state)

    def weight_shape(self, index_set, max_n_modes):
        if isinstance(index_set, HyperRectangleIndexSet):
            return tuple(max_n_modes)
        return (index_set.n_modes,)

    def selection(
        self,
        index_set,
        fft_size,
        max_n_modes,
        true_n_modes,
        separable=False,
        device=None,
    ):
        active_radius = radius_from_n_modes(index_set, true_n_modes)
        modes = index_set.modes(device=device, radius=active_radius)
        cache_key = (
            device,
            tuple(fft_size),
            id(index_set),
            modes.shape[0],
            tuple(max_n_modes),
            active_radius,
            self.complex_data,
            separable,
        )
        cached = self._selection_cache.get(cache_key)
        if cached is not None:
            return cached

        coefficients, keep = self._coefficients(modes, fft_size, device)
        if isinstance(index_set, HyperRectangleIndexSet):
            max_n_modes_tensor = torch.as_tensor(
                max_n_modes, device=device, dtype=torch.long
            )
            weight_coordinates = modes[keep] + max_n_modes_tensor // 2
            if separable:
                weight_index = (slice(None), *weight_coordinates.unbind(dim=1))
            else:
                weight_index = (
                    slice(None),
                    slice(None),
                    *weight_coordinates.unbind(dim=1),
                )
        else:
            weight_indices = torch.arange(modes.shape[0], device=device)[keep]
            if separable:
                weight_index = (slice(None), weight_indices)
            else:
                weight_index = (slice(None), slice(None), weight_indices)
        x_index = (slice(None), slice(None), coefficients)
        selection = FourierSelection(
            x_index=x_index,
            weight_index=weight_index,
        )
        self._selection_cache[cache_key] = selection
        return selection
