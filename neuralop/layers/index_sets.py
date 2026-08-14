"""Fourier index sets used by spectral convolution backends."""

from math import ceil, prod
from typing import Optional, Sequence

import torch


class IndexSet:
    """Base class for index sets."""

    @property
    def n_dim(self):
        """Dimensionality of the index set."""
        # Default implementation assumes modes have been computed and returns shape[1].
        return self.modes().shape[1]

    @property
    def n_modes(self):
        """Number of modes in the index set."""
        # Default implementation assumes modes have been computed and returns shape[0].
        return self.modes().shape[0]

    def modes(self, device=None, radius=None):
        """
        Return the Fourier modes in the index set.

        This could be pre-computed, or generated on demand.
        This iterates over all the Fourier modes in the index set.
        We do not use this for RegularGridFFT and a real-valued field since then
        we can skip the Fourier modes in the last dimension that have negative mode
        by the conjugate symmetry of real-valued functions.

        Parameters
        ----------
        device : torch.device or None, optional
            Device on which to return the mode tensor.
        radius : int or float or None, optional
            Active radius to select from a nested radius-based index set. If
            None, return all modes stored by the index set. Index sets without a
            radius structure may ignore this argument.

        Returns
        -------
        torch.Tensor
            Integer tensor of shape ``(n_modes, n_dim)``.
        """
        raise NotImplementedError

class RadialIndexSet(IndexSet):

    @property
    def radius(self):
        return self._radius

    @radius.setter
    def radius(self, value):
        self._radius = value

    def radius_from_n_modes(self, n_modes):
        """Return the radius represented by ``n_modes[0]``.

        For legacy NeuralOp compatibility, ``n_modes`` is an ``n_dim``-
        dimensional tuple or list, but only ``n_modes[0]`` is used.
        """
        return None


class ExplicitIndexSet(IndexSet):
    """Index set defined by a given list of Fourier modes.

    Parameters
    ----------
    modes : torch.Tensor or Sequence[Sequence[int]]
        Integer tensor of shape ``(n_modes, n_dim)``.
    """

    def __init__(self, modes):
        modes = torch.as_tensor(modes, dtype=torch.long)
        if modes.ndim != 2:
            raise ValueError("modes must have shape (n_modes, n_dim)")
        self._modes = modes

    def modes(self, device=None, radius=None):
        if radius is not None:
            raise ValueError("ExplicitIndexSet does not support radius selection")
        return self._modes.to(device=device)


class HyperRectangleIndexSet(RadialIndexSet):
    """Hyperrectangular index set with product weights.

    With all weights equal to 1, ``radius`` determines the same number of
    modes in every direction.
    For example, ``radius=2`` and ``n_dim=2`` gives four modes per dimension,
    ``{-2, -1, 0, 1} x {-2, -1, 0, 1}``.
    Taking a half-integer radius gives an odd number of modes: ``radius=1.5`` gives
    ``{-1, 0, 1} x {-1, 0, 1}``.
    In general, the one-dimensional range with ``M = round(2 * radius)`` modes is
    ``{-M // 2, ..., M // 2 + M % 2 - 1}``, and the hyperrectangle is the
    Cartesian product of these ranges.

    Parameters
    ----------
    radius : int
        Radius used to derive the number of modes along each Fourier dimension.
        The active number of modes in each dimension is ``round(2 * radius * weight[j])``.
    n_dim : int
        Number of dimensions.
    weights : Sequence[float], optional
        Per-dimension weights. The active number of modes in each dimension is
        ``round(2 * radius * weight[j])``.

    References
    ----------
    .. [1] Li, Z. et al. "Fourier Neural Operator for Parametric Partial Differential
        Equations" (2021). ICLR 2021, https://arxiv.org/pdf/2010.08895.
    """

    def __init__(
        self,
        radius,
        n_dim,
        weights: Optional[Sequence[float]] = None,
    ):
        if weights is None:
            weights = [1.0] * n_dim
        elif len(weights) != n_dim:
            raise ValueError("weights must have length n_dim")
        self.weights = tuple(float(weight) for weight in weights)
        self._n_dim = n_dim
        self._set_radius(radius) # Also sets n_modes_per_dim

    def _set_radius(self, radius):
        # Internally we still keep n_modes_per_dim (as in the legacy interface)
        self.radius = radius
        self.n_modes_per_dim = self.n_modes_per_dim_for_radius(radius, self.weights)
        self._n_modes = prod(self.n_modes_per_dim)
        self._ranges = self.mode_ranges_from_n_modes_per_dim(self.n_modes_per_dim)

    @classmethod
    def from_n_modes_per_dim(cls, n_modes_per_dim):
        """Construct from (true) Fourier extents.

        ``n_modes_per_dim`` is the ``true_n_modes``. For real-valued fields,
        pass ``M`` rather than ``M // 2 + 1`` in the last dimension.

        Currently used by ``SpectralConv`` when constructing the default
        HyperRectangleIndexSet, not directly by the spectral transform backends.
        """
        n_modes_per_dim = tuple(int(n_mode) for n_mode in n_modes_per_dim)
        radius = n_modes_per_dim[0] / 2
        weights = [1.0]
        weights.extend(
            float(n_mode) / float(n_modes_per_dim[0])
            for n_mode in n_modes_per_dim[1:]
        )
        return cls(radius=radius, n_dim=len(n_modes_per_dim), weights=weights)

    @staticmethod
    def n_modes_per_dim_for_radius(radius, weights):
        """Return hyperrectangle extents for given ``radius`` and ``weights``.

        Used by ``RegularGridFFT`` to build regular-grid slices for the active
        hyperrectangle of given radius and weights.
        """
        return tuple(
            max(1, int(round(2 * radius * weight))) for weight in weights
        )

    @staticmethod
    def mode_ranges_from_n_modes_per_dim(n_modes_per_dim):
        """Return integer mode ranges for each extent.

        Used internally by ``HyperRectangleIndexSet``. Currently not called
        directly by ``RegularGridFFT`` or ``Rank1LatticeFFT``.
        """
        ranges = []
        for n_mode in n_modes_per_dim:
            negative_freqs = n_mode // 2
            positive_freqs = n_mode // 2 + n_mode % 2
            ranges.append((int(-negative_freqs), int(positive_freqs)))
        return tuple(ranges)

    @property
    def n_dim(self):
        """Dimensionality of the hyperrectangle."""
        return self._n_dim

    @property
    def n_modes(self):
        """Total number of modes in the hyperrectangle."""
        return self._n_modes

    @property
    def mode_ranges(self):
        """Stored integer ranges for the hyperrectangle."""
        return self._ranges

    def modes(self, device=None, radius=None):
        """Return the hyperrectangle modes, optionally at an active radius.

        Used by ``Rank1LatticeFFT``. ``RegularGridFFT`` uses
        ``n_modes_per_dim`` and ``n_modes_per_dim_for_radius`` instead so it can
        construct slice-based selections.
        """
        if radius is None:
            n_modes_per_dim = self.n_modes_per_dim
        else:
            n_modes_per_dim = self.n_modes_per_dim_for_radius(radius, self.weights)
        ranges = [
            torch.arange(start, stop, dtype=torch.long, device=device)
            for start, stop in self.mode_ranges_from_n_modes_per_dim(n_modes_per_dim)
        ]
        grids = torch.meshgrid(*ranges, indexing="ij")
        return torch.stack(grids, dim=-1).reshape(-1, len(n_modes_per_dim))

    def radius_from_n_modes(self, n_modes):
        """Return the radius radius represented by ``n_modes[0]``.

        For legacy NeuralOp compatibility, ``n_modes`` is an ``n_dim``-
        dimensional tuple or list, but only ``n_modes[0]`` is used. The
        active radius is ``n_modes[0] / (2 * weights[0])``.
        """
        return n_modes[0] / (2 * self.weights[0])


class HyperbolicCrossIndexSet(RadialIndexSet, ExplicitIndexSet):
    """Hyperbolic cross index set with product weights.

    Modes satisfy
    ``prod(abs(k[j]) ** beta / weights[j] for k[j] != 0) <= radius``.
    Modes are ordered by integer radius blocks: all modes with radius <= 1,
    then the new modes with radius <= 2, and so on. ``radius_starts[r]`` gives
    the first index in ``modes()`` whose integer radius is ``r``.

    ``beta=0`` is treated as the exceptional hyperrectangle case. Otherwise
    ``beta`` must be at least 1.


    References
    ----------
    .. [1] Dilen, J., Keller, A., Kuo, F. Y., Nuyens, D. "Fourier Neural Operators
        with Rank-1 Lattice Points and Hyperbolic Cross" (2026).
        https://arxiv.org/abs/2606.08871.
    """

    def __init__(
        self,
        radius: int,
        n_dim: int,
        weights: Optional[Sequence[float]] = None,
        beta: float = 1.0,
    ):
        if weights is None:
            weights = [1.0] * n_dim
        else:
            weights = list(weights)
            if len(weights) != n_dim:
                raise ValueError("weights must have length n_dim")
        if beta != 0 and beta < 1:
            raise ValueError("beta must be at least 1 (or exceptionally zero)")
        self.weights = tuple(float(weight) for weight in weights)
        self.beta = float(beta)
        self._n_dim = n_dim
        self.set_radius(radius)

    @classmethod
    def from_n_modes_per_dim(cls, n_modes_per_dim, beta: float = 1.0):
        """Construct from (true) Fourier extents.

        ``n_modes_per_dim`` is the ``true_n_modes``. For real-valued fields,
        pass ``M`` rather than ``M // 2 + 1`` in the last dimension.
        """
        n_modes_per_dim = tuple(int(n_mode) for n_mode in n_modes_per_dim)
        radius = cls._radius_from_n_modes_per_dim(n_modes_per_dim[0], beta)
        weights = [1.0]
        if beta == 0:
            weights.extend(
                float(n_mode - 1) / float(n_modes_per_dim[0] - 1)
                for n_mode in n_modes_per_dim[1:]
            )
        else:
            weights.extend(
                (((n_mode - 1) / 2) ** beta) / radius
                for n_mode in n_modes_per_dim[1:]
            )
        return cls(radius=radius, n_dim=len(n_modes_per_dim), weights=weights, beta=beta)

    @staticmethod
    def _radius_from_n_modes_per_dim(n_modes_per_dim, beta):
        half_width = (n_modes_per_dim - 1) / 2
        if beta == 0:
            return half_width
        return half_width**beta

    def set_radius(self, radius):
        """Update the stored radius.

        Currently not called by ``RegularGridFFT`` or ``Rank1LatticeFFT``.
        """
        self.radius = radius
        self._modes, self.radius_starts = self._compute_modes(radius)

    def _compute_modes(self, radius):
        """Compute modes sorted by integer radius blocks.

        Internal helper used by ``set_radius``. Currently not called directly by
        ``RegularGridFFT`` or ``Rank1LatticeFFT``.
        """
        if self.beta == 0:
            max_modes = [int(radius * weight) for weight in self.weights]
            ranges = [
                torch.arange(-max_mode, max_mode + 1, dtype=torch.long)
                for max_mode in max_modes
            ]
            max_integer_radius = int(ceil(radius))
        else:
            max_mode = ceil((radius * max(self.weights)) ** (1 / self.beta))
            ranges = [torch.arange(-max_mode, max_mode + 1, dtype=torch.long)] * self._n_dim
            max_integer_radius = int(ceil(radius))
        n_dim = self._n_dim
        grids = torch.meshgrid(*ranges, indexing="ij")
        modes = torch.stack(grids, dim=-1).reshape(-1, n_dim)

        weights_tensor = torch.as_tensor(self.weights, dtype=torch.float64)
        if self.beta == 0:
            scaled_modes = torch.abs(modes).to(torch.float64) / weights_tensor
            mode_radii = torch.max(scaled_modes, dim=1).values
        else:
            abs_modes = torch.abs(modes).to(torch.float64)
            scaled_modes = abs_modes**self.beta / weights_tensor
            scaled_modes = torch.where(abs_modes == 0, 1, scaled_modes)
            mode_radii = torch.prod(scaled_modes, dim=1)
        criterion = mode_radii <= radius
        modes = modes[criterion]
        mode_radii = mode_radii[criterion]

        integer_radii = torch.ceil(mode_radii).to(torch.long)
        radius_starts = [0] * (max_integer_radius + 2)
        ordered_modes = []
        start = 0
        for integer_radius in range(max_integer_radius + 1):
            radius_starts[integer_radius] = start
            block = modes[integer_radii == integer_radius]
            if block.numel():
                order = torch.argsort(torch.sum(torch.abs(block), dim=1))
                block = block[order]
                ordered_modes.append(block)
                start += block.shape[0]
        radius_starts[max_integer_radius + 1] = start

        if ordered_modes:
            modes = torch.cat(ordered_modes, dim=0)
        else:
            modes = modes.reshape(0, n_dim)
        return modes, radius_starts

    def modes(self, device=None, radius=None):
        """Return stored hyperbolic-cross modes, optionally truncated by radius."""
        if radius is None:
            modes = self._modes
        else:
            radius = int(radius)
            if radius >= len(self.radius_starts) - 1:
                modes = self._modes
            else:
                modes = self._modes[: self.radius_starts[radius + 1]]
        return modes.to(device=device)

    def radius_from_n_modes(self, n_modes):
        """Return the radius radius represented by ``n_modes[0]``.

        For legacy NeuralOp compatibility, ``n_modes`` is an ``n_dim``-
        dimensional tuple or list, but only ``n_modes[0]`` is used. The
        active radius is ``((n_modes[0] - 1) / 2) ** beta / weights[0]``.
        """
        half_width = (n_modes[0] - 1) / 2
        if self.beta == 0:
            return half_width / self.weights[0]
        return half_width**self.beta / self.weights[0]
