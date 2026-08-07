from typing import Literal, Optional, Sequence, Union

import torch
import torch.nn as nn


class ResolutionInvariantReadout(nn.Module):
    """Resolution-invariant readout for neural operator field outputs.

    Maps tensors of shape ``(B, C, *spatial)`` to ``(B, out_dim)`` by
    spatially pooling the field and applying a learned head.  The pooling
    mode is either a plain mean (``reduce="mean"``) or a physical integral
    approximation (``reduce="integral"``) that scales the mean by the domain
    volume so that the result is independent of grid resolution.

    Parameters
    ----------
    in_channels : int
        Number of channels in the input field ``(B, C, *spatial)``.
    out_dim : int
        Number of output dimensions.
    reduce : {"mean", "integral"}, optional
        Spatial reduction mode.  ``"mean"`` computes the spatial mean.
        ``"integral"`` multiplies the mean by the domain volume to
        approximate the physical integral:
        ``mean(field) * prod(measure_per_dim) ≈ ∫ f dⁿx``.
        Default is ``"mean"``.
    measure_per_dim : float or sequence of float, optional
        Physical domain measure per spatial dimension, used only when
        ``reduce="integral"``.  If a scalar, the domain volume is
        ``measure_per_dim ** n_spatial_dims``.  If a sequence, its length
        must match the number of spatial dimensions and the domain volume is
        the product of the entries.  Defaults to ``1.0``.

        For non-cubic domains pass a sequence with one entry per dimension,
        e.g. ``measure_per_dim=[Lx, Ly, Lz]`` for a box of side-lengths
        *Lx*, *Ly*, *Lz* in Å (or any consistent length unit).
    head : {"linear", "mlp"}, optional
        Projection head applied after pooling.  ``"linear"`` uses a single
        ``nn.Linear``; ``"mlp"`` uses a two-layer MLP with an activation.
        Default is ``"linear"``.
    mlp_hidden_dim : int, optional
        Hidden width of the MLP head.  Defaults to ``in_channels`` when
        ``head="mlp"``.
    activation : nn.Module, optional
        Activation function for the MLP head.  Defaults to ``nn.GELU()``.
    """

    def __init__(
        self,
        in_channels: int,
        out_dim: int,
        reduce: Literal["mean", "integral"] = "mean",
        measure_per_dim: Optional[Union[float, Sequence[float]]] = None,
        head: Literal["linear", "mlp"] = "linear",
        mlp_hidden_dim: Optional[int] = None,
        activation: Optional[nn.Module] = None,
    ):
        super().__init__()

        if reduce not in {"mean", "integral"}:
            raise ValueError(f"reduce must be 'mean' or 'integral', got {reduce}.")
        if head not in {"linear", "mlp"}:
            raise ValueError(f"head must be 'linear' or 'mlp', got {head}.")

        self.in_channels = in_channels
        self.out_dim = out_dim
        self.reduce = reduce

        if measure_per_dim is None:
            measure_per_dim = 1.0

        # Store the raw measure values as a buffer so they move with the module
        # and are saved in checkpoints. _scalar_mode selects between
        # value**n_dims (scalar) and per-dimension products (sequence).
        # Copy caller-provided sequences so later external mutation cannot
        # invalidate the cached values or dimension checks.
        if isinstance(measure_per_dim, (float, int)):
            self.measure_per_dim = float(measure_per_dim)
            self._scalar_mode = True
            self.register_buffer(
                "_measure_buffer",
                torch.tensor(self.measure_per_dim, dtype=torch.float32),
            )
        else:
            values = list(measure_per_dim)
            self.measure_per_dim = values
            self._scalar_mode = False
            self.register_buffer(
                "_measure_buffer",
                torch.tensor(values, dtype=torch.float32),
            )

        if head == "linear":
            self.head = nn.Linear(in_channels, out_dim)
        else:
            if mlp_hidden_dim is None:
                mlp_hidden_dim = in_channels
            if activation is None:
                activation = nn.GELU()
            self.head = nn.Sequential(
                nn.Linear(in_channels, mlp_hidden_dim),
                activation,
                nn.Linear(mlp_hidden_dim, out_dim),
            )

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, out_dim={self.out_dim}, "
            f"reduce={self.reduce!r}, measure_per_dim={self.measure_per_dim}"
        )

    def _measure_product(
        self, n_spatial_dims: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        if self._scalar_mode:
            value = self._measure_buffer.to(device=device, dtype=dtype)
            return value.pow(n_spatial_dims)
        else:
            # Sequence case: validate length, then multiply the per-dimension measures.
            if len(self.measure_per_dim) != n_spatial_dims:
                raise ValueError(
                    f"measure_per_dim has length {len(self.measure_per_dim)},"
                    f" expected {n_spatial_dims}."
                )
            return self._measure_buffer.to(device=device, dtype=dtype).prod()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim < 3:
            raise ValueError(
                f"Expected input with shape (B, C, *spatial), got {tuple(x.shape)}"
            )
        if x.is_complex():
            raise ValueError(
                "ResolutionInvariantReadout does not support complex inputs. "
                "Convert to real first, e.g. x.real, x.abs(), or x.float()."
            )

        spatial_dims = tuple(range(2, x.ndim))
        reduced = x.mean(dim=spatial_dims)

        if self.reduce == "integral":
            reduced = reduced * self._measure_product(
                n_spatial_dims=len(spatial_dims), device=x.device, dtype=x.dtype
            )

        return self.head(reduced)
