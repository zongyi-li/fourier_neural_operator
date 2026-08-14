"""Convenience helpers for Fourier neural operators on rank-1 lattices.

References
----------
.. [1] Dilen, J., Keller, A., Kuo, F. Y., Nuyens, D. "Fourier Neural Operators
    with Rank-1 Lattice Points and Hyperbolic Cross" (2026).
    https://arxiv.org/abs/2606.08871.
"""

from .fno import FNO
from ..layers.embeddings import LatticeEmbedding
from ..layers.index_sets import HyperbolicCrossIndexSet
from ..layers.lattice import (
    lattice_to_regular_grid,
    rank1_lattice_points,
    regular_grid_to_lattice,
)
from ..layers.spectral_transforms import Rank1LatticeFFT


def LatticeFNO(
    n,
    z,
    n_modes,
    in_channels,
    out_channels,
    hidden_channels,
    index_set=None,
    positional_embedding=True,
    complex_data=False,
    **kwargs,
):
    """Construct an ``FNO`` configured for rank-1 lattice data.

    Data is stored on a one-dimensional lattice-point axis of length ``n``,
    while ``n_modes`` and ``index_set`` describe the multi-dimensional Fourier
    index set associated with the generating vector ``z``.
    """
    if index_set is None:
        index_set = HyperbolicCrossIndexSet.from_n_modes_per_dim(n_modes)

    if positional_embedding is True:
        positional_embedding = LatticeEmbedding(in_channels=in_channels, z=z)
    elif positional_embedding is False:
        positional_embedding = None

    return FNO(
        n_modes=n_modes,
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=hidden_channels,
        index_set=index_set,
        spectral_transform=Rank1LatticeFFT(n=n, z=z, complex_data=complex_data),
        positional_embedding=positional_embedding,
        complex_data=complex_data,
        **kwargs,
    )
