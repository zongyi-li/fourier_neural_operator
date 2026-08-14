"""Utilities for rank-1 lattice point sets."""

from itertools import product
from math import prod

import torch
import numpy as np


def rank1_lattice_points(z, n, device=None, dtype=None):
    """Return rank-1 lattice points ``(j * z mod n) / n``."""
    if isinstance(z, torch.Tensor):
        z_array = z.cpu().numpy().astype(object)
    else:
        z_array = np.array(z, dtype=object)

    n_int = int(n)
    lattice = (np.outer(np.arange(n_int, dtype=object), z_array) % n_int) / n
    if dtype is None:
        dtype = torch.get_default_dtype()
    return torch.as_tensor(lattice.astype(np.float64), dtype=dtype, device=device)


def _coordinate_dtype(dtype):
    if dtype in (torch.float64, torch.complex128):
        return torch.float64
    return torch.float32


def regular_grid_to_lattice(x, z, n=None):
    """Sample regular-grid data at rank-1 lattice points.

    ``x`` is expected to have shape ``(batch, channels, *grid_shape)``. The
    number of grid dimensions is inferred from ``len(z)``. Periodic multilinear
    interpolation is used.
    """
    z = torch.as_tensor(z, device=x.device, dtype=torch.long)
    n_dim = z.numel()
    if x.ndim < n_dim + 2:
        raise ValueError("x must have shape (batch, channels, *grid_shape)")
    grid_shape = tuple(x.shape[-n_dim:])
    if n is None:
        n = prod(grid_shape)

    points = rank1_lattice_points(
        z,
        n,
        device=x.device,
        dtype=_coordinate_dtype(x.dtype),
    )
    scaled = points * torch.as_tensor(grid_shape, device=x.device, dtype=points.dtype)
    lower = torch.floor(scaled).to(torch.long)
    frac = scaled - lower.to(points.dtype)
    flat_x = x.reshape(*x.shape[:-n_dim], -1)
    out = torch.zeros(*x.shape[:-n_dim], n, device=x.device, dtype=x.dtype)

    strides = torch.as_tensor(
        [prod(grid_shape[dim + 1 :]) for dim in range(n_dim)],
        device=x.device,
        dtype=torch.long,
    )
    for corner in product((0, 1), repeat=n_dim):
        corner = torch.as_tensor(corner, device=x.device, dtype=torch.long)
        indices = torch.remainder(lower + corner, torch.as_tensor(grid_shape, device=x.device))
        flat_indices = torch.sum(indices * strides, dim=1)
        weights = torch.prod(
            torch.where(corner.bool(), frac, 1 - frac),
            dim=1,
        )
        values = torch.gather(
            flat_x,
            -1,
            flat_indices.reshape(*((1,) * (flat_x.ndim - 1)), n).expand(*flat_x.shape[:-1], n),
        )
        out = out + values * weights.reshape(*((1,) * (out.ndim - 1)), n)
    return out


def lattice_to_regular_grid(x, z, output_shape):
    """Place lattice data on a regular grid by nearest-cell averaging.

    This is a simple utility for visualization and coarse conversion. It is not
    an inverse of ``regular_grid_to_lattice`` unless the lattice points cover the
    requested grid cells appropriately.
    """
    z = torch.as_tensor(z, device=x.device, dtype=torch.long)
    output_shape = tuple(output_shape)
    if len(output_shape) != z.numel():
        raise ValueError("output_shape must have length len(z)")
    n = x.shape[-1]
    points = rank1_lattice_points(
        z,
        n,
        device=x.device,
        dtype=_coordinate_dtype(x.dtype),
    )
    indices = torch.floor(
        points * torch.as_tensor(output_shape, device=x.device, dtype=points.dtype)
    ).to(torch.long)
    indices = torch.remainder(indices, torch.as_tensor(output_shape, device=x.device))
    strides = torch.as_tensor(
        [prod(output_shape[dim + 1 :]) for dim in range(len(output_shape))],
        device=x.device,
        dtype=torch.long,
    )
    flat_indices = torch.sum(indices * strides, dim=1)
    flat_size = prod(output_shape)

    out = torch.zeros(*x.shape[:-1], flat_size, device=x.device, dtype=x.dtype)
    scatter_index = flat_indices.reshape(*((1,) * (x.ndim - 1)), n).expand(*x.shape)
    out = torch.scatter_add(out, -1, scatter_index, x)

    counts = torch.zeros(flat_size, device=x.device, dtype=_coordinate_dtype(x.dtype))
    counts = torch.scatter_add(
        counts,
        0,
        flat_indices,
        torch.ones(n, device=x.device, dtype=counts.dtype),
    )
    out = torch.where(counts > 0, out / counts.clamp_min(1), out)
    return out.reshape(*x.shape[:-1], *output_shape)
