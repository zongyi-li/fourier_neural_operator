import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Callable
import math

from ..layers.channel_mlp import ChannelMLP
from ..layers.mwno_block import MWNOBlock


class MWNO(nn.Module):
    """N-Dimensional Multiwavelet Neural Operator. The MWNO learns a mapping between
    spaces of functions using multiwavelet bases to represent operators in localized,
    multiscale domains, as described in [1].

    By leveraging orthogonality, compact support, and multi-resolution properties of
    multiwavelets, it effectively captures both global structures and local irregularities.
    This makes it particularly well-suited for learning operators from data with high
    fluctuations and supports accurate modeling of complex dynamical systems governed
    by partial differential equations.

    Main Parameters
    ---------------
    n_modes : int
        Number of Fourier modes retained along each transformed axis (isotropic cutoff).
        Must be a positive integer; list/tuple forms are not supported.
    n_dim : int
        Spatial dimensionality: ``1``, ``2``, or ``3`` (sets input layout and wavelet grid).
    in_channels : int
        Number of input function channels. Determined by the problem.
        For scalar fields: in_channels=1. For vector fields: in_channels=n_components.
    out_channels : int
        Number of output function channels. Determined by the problem.
        Set to 1 for scalar outputs (will auto-squeeze).
    k : int, optional
        Wavelet basis size (number of polynomial basis functions). Default: 4
    c : int, optional
        Number of parallel wavelet channels. Default: 16, Recommended values: 16
        Increases model capacity: total wavelet features = c * k^n_dim.
    n_layers : int, optional
        Number of MWNOBlock transformation layers. Default: 3
        More layers = deeper hierarchical processing.

    Other parameters
    ----------------
    L : int, optional
        Number of coarsest decomposition levels to skip. Default: 0
        Reduces computation by stopping wavelet decomposition early.
        L=0: Full decomposition to coarsest scale. L=1: Stop 1 level before coarsest.
        Must satisfy ``L < num_scales`` where ``num_scales = floor(log2(grid))`` on the
        wavelet axis; validated once per model ``forward`` (``ValueError`` if violated).
    lifting_channels : int, optional
        Hidden dimension for lifting layer. Default: 0
        If 0: Direct linear lifting (fast).
        If > 0: Two-layer MLP (in → lifting_channels → wavelet_space).
    projection_channels : int, optional
        Hidden width in the projection ``ChannelMLP`` (two-layer, ReLU between).
        Default: 128. If 0, uses 128 (same as default).
    base : str, optional
        Polynomial basis for wavelet construction. Default: "legendre"
        Options: "legendre" (uniform weighting, general purpose),
        "chebyshev" (better for boundary-dominated problems).
    initializer : callable, optional
        Custom weight initialization function.
        Applied to ``Linear`` and ``Conv1d`` (projection ``ChannelMLP``) if provided.
        Signature: initializer(weight_tensor) -> None. Default: None
    check_spatial_resolution_once : bool, optional
        If True (default), check power-of-two grid sizes and tensor rank only on the
        first ``forward`` call, then skip (typical training loop). If False, skip
        these checks in ``forward`` (caller must ensure valid grids). Call
        ``reset_spatial_resolution_check()`` if the spatial shape changes and you
        want validation to run again on the next ``forward``.

    Notes
    -----
    - Wavelet grids along decomposed axes must use sizes that are powers of 2;
      by default the first ``forward`` raises ``ValueError`` if not (see Input Shapes).
    - For 3D, only first two dimensions are decomposed (time preserved)
    - ReLU activation applied between MWNO layers (except after last layer)
    - Output channel dimension auto-squeezed if out_channels=1

    Input Shapes:
        - 1D: (batch, n_points, in_channels) where n_points must be a power of 2
        - 2D: (batch, height, width, in_channels) where height, width must be powers of 2
        - 3D: (batch, height, width, time, in_channels) where height, width must be powers of 2

    Output Shapes:
        Same spatial dimensions as input, with out_channels feature dimension.
        If out_channels=1, the channel dimension is squeezed.

    Examples
    --------

    >>> from neuralop.models import MWNO
    >>> # 1D time series operator
    >>> model_1d = MWNO(n_modes=16, n_dim=1, in_channels=1, out_channels=1)
    >>> x = torch.randn(32, 256, 1)  # (batch, time, channels)
    >>> y = model_1d(x)  # (32, 256) - channel squeezed

    >>> # 2D image-to-image operator
    >>> model_2d = MWNO(n_modes=16, n_dim=2, in_channels=3, out_channels=1)
    >>> x = torch.randn(16, 64, 64, 3)  # (batch, H, W, channels)
    >>> y = model_2d(x)  # (16, 64, 64) - channel squeezed

    References
    ----------
    .. [1] :

    Gupta, G., Xiao, X. and Bogdan, P., 2021.
        Multiwavelet-based operator learning for differential equations.
        Advances in Neural Information Processing Systems, 34, pp.24048-24062.

    @article{gupta2021multiwavelet,
        title={Multiwavelet-based operator learning for differential equations},
        author={Gupta, Gaurav and Xiao, Xiongye and Bogdan, Paul},
        journal={Advances in neural information processing systems},
        volume={34},
        pages={24048--24062},
        year={2021}
    }

    .. [2] :

    Xiao, Xiongye, et al. "Coupled multiwavelet operator learning for coupled differential equations."
        The Eleventh International Conference on Learning Representations. 2022.

    @inproceedings{xiao2023coupled,
    title={Coupled Multiwavelet Operator Learning for Coupled Differential Equations},
    author={Xiongye Xiao and Defu Cao and Ruochen Yang and Gaurav Gupta and Gengshuo Liu and Chenzhong Yin and Radu Balan and Paul Bogdan},
    booktitle={The Eleventh International Conference on Learning Representations },
    year={2023},
    url={https://openreview.net/forum?id=kIo_C6QmMOM}

    """


    def __init__(
            self,
            n_modes: int,
            n_dim: int,
            in_channels: int = 1,
            out_channels: int = 1,
            k: int = 4,
            c: int = 16,
            n_layers: int = 3,
            L: int = 0,
            lifting_channels: int = 0,
            projection_channels: int = 128,
            base: str = 'legendre',
            initializer: Optional[Callable] = None,
            check_spatial_resolution_once: bool = True,
    ):
        super().__init__()

        if not isinstance(n_modes, int):
            raise TypeError(
                "n_modes must be a positive int (same mode cutoff in every transformed "
                f"direction). Got {type(n_modes).__name__}: {n_modes!r}."
            )
        if n_modes < 1:
            raise ValueError(f"n_modes must be >= 1, got {n_modes}")
        if n_dim not in (1, 2, 3):
            raise ValueError(f"n_dim must be 1, 2, or 3, got {n_dim}")

        self.n_modes = n_modes
        self.n_dim = n_dim

        # Store configuration
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k = k
        self.c = c
        self.n_layers = n_layers
        self.L = L
        self.base = base

        # Calculate channel multiplier for wavelet representation
        # 1D: c * k coefficients, 2D/3D: c * k^2 coefficients
        if self.n_dim == 1:
            channel_multiplier = c * k
        else:
            channel_multiplier = c * k ** 2

        self.channel_multiplier = channel_multiplier

        # Build network layers
        self.lifting = self._build_lifting_layer(
            in_channels, channel_multiplier, lifting_channels
        )

        self.mwno_layers = nn.ModuleList([
            MWNOBlock(
                k=k,
                alpha=n_modes,
                L=L,
                c=c,
                base=base,
                n_dim=self.n_dim,
                initializer=initializer
            )
            for _ in range(n_layers)
        ])

        self.projection = self._build_projection_layer(
            channel_multiplier, out_channels, projection_channels
        )

        # Apply custom initialization if provided
        if initializer is not None:
            self._reset_parameters(initializer)

        self.check_spatial_resolution_once = check_spatial_resolution_once
        self._spatial_resolution_checked = False

    def reset_spatial_resolution_check(self) -> None:
        """Re-enable spatial grid validation on the next ``forward`` (e.g. new resolution)."""
        self._spatial_resolution_checked = False

    def _build_lifting_layer(
            self,
            in_channels: int,
            channel_multiplier: int,
            lifting_channels: int
    ) -> nn.Module:
        """
        Build the lifting layer that embeds inputs into wavelet space.

        The lifting layer transforms from physical input space to the
        higher-dimensional wavelet coefficient space where MWNO operates.

        Parameters
        ----------
        in_channels : int
            Input feature dimension
        channel_multiplier : int
            Output dimension (c * k^n_dim)
        lifting_channels : int
            Hidden dimension (0 for direct linear projection)

        Returns
        -------
        nn.Module
            Lifting layer (Linear or Sequential MLP)
        """
        if lifting_channels > 0:
            # Two-layer MLP with nonlinearity
            return nn.Sequential(
                nn.Linear(in_channels, lifting_channels),
                nn.ReLU(inplace=True),
                nn.Linear(lifting_channels, channel_multiplier)
            )
        else:
            # Direct linear projection
            return nn.Linear(in_channels, channel_multiplier)

    def _build_projection_layer(
            self,
            channel_multiplier: int,
            out_channels: int,
            projection_channels: int
    ) -> nn.Module:
        """
        Build the projection layer that maps from wavelet space to output.

        The projection layer transforms from the wavelet coefficient space
        back to the physical output space.

        Parameters
        ----------
        channel_multiplier : int
            Input dimension (c * k^n_dim)
        out_channels : int
            Output feature dimension
        projection_channels : int
            Requested hidden width; 0 is treated as 128.

        Returns
        -------
        nn.Module
            ``ChannelMLP`` (two Conv1d layers with ReLU), matching other models in the library.
        """

        hidden = projection_channels if projection_channels > 0 else 128
        return ChannelMLP(
            in_channels=channel_multiplier,
            out_channels=out_channels,
            hidden_channels=hidden,
            n_layers=2,
            n_dim=self.n_dim,
            non_linearity=F.relu,
            dropout=0.0,
        )

    def _reset_parameters(self, initializer: Callable) -> None:
        """
        Apply custom initialization to Linear and Conv1d layers (lifting / projection).

        Parameters
        ----------
        initializer : callable
            Initialization function with signature: initializer(weight) -> None
        """
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d)):
                initializer(module.weight)

    @staticmethod
    def _is_power_of_2(n: int) -> bool:
        return n > 0 and (n & (n - 1)) == 0

    def _validate_spatial_resolution(self, x: torch.Tensor) -> None:
        """Require decomposed spatial sizes to be powers of 2; raise ValueError otherwise."""
        expected_ndim = self.n_dim + 2
        if x.ndim != expected_ndim:
            raise ValueError(
                f"Expected {expected_ndim}D input (batch, *spatial, channels) for n_dim={self.n_dim}, "
                f"got {x.ndim}D tensor with shape {tuple(x.shape)}."
            )

        if self.n_dim == 1:
            n_points = x.shape[1]
            if not self._is_power_of_2(n_points):
                raise ValueError(
                    f"MWNO 1D requires n_points to be a power of 2, got n_points={n_points} "
                    f"(shape {tuple(x.shape)})."
                )
        elif self.n_dim == 2:
            height, width = x.shape[1], x.shape[2]
            if not self._is_power_of_2(height) or not self._is_power_of_2(width):
                raise ValueError(
                    f"MWNO 2D requires height and width to be powers of 2, "
                    f"got height={height}, width={width} (shape {tuple(x.shape)})."
                )
        else:
            height, width = x.shape[1], x.shape[2]
            if not self._is_power_of_2(height) or not self._is_power_of_2(width):
                raise ValueError(
                    f"MWNO 3D requires height and width (wavelet axes) to be powers of 2, "
                    f"got height={height}, width={width} (shape {tuple(x.shape)}). "
                    "The time dimension is not checked."
                )

    def _validate_L_vs_num_scales(self, x: torch.Tensor) -> None:
        """``L < num_scales`` with ``num_scales = floor(log2(spatial))`` on the wavelet axis (matches MWNOBlock)."""
        if self.n_dim == 1:
            n_points = x.shape[1]
            num_scales = math.floor(math.log2(n_points))
        else:
            nx = x.shape[1]
            num_scales = math.floor(math.log2(nx))
        if self.L >= num_scales:
            raise ValueError(
                f"L ({self.L}) must be less than num_scales ({num_scales}), where "
                f"num_scales = floor(log2(spatial_size)) along the wavelet-decomposed axis "
                f"(input shape {tuple(x.shape)})."
            )

    def _reshape_to_wavelet_format(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reshape lifted features to wavelet coefficient format.
        Splits the last dimension (c*k^n_dim) into (c, k^n_dim).
        """
        # Determine basis size: k for 1D, k^2 for 2D/3D
        basis = self.k if self.n_dim == 1 else self.k ** 2
        expected_channels = self.c * basis

        # Verify channel dimension matches
        assert x.shape[-1] == expected_channels, (
            f"Lifting produced {x.shape[-1]} channels, "
            f"expected {expected_channels} (c={self.c} × basis={basis})"
        )

        # Use unflatten to split the last dimension
        return x.unflatten(-1, (self.c, basis))

    def _reshape_from_wavelet_format(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reshape from wavelet coefficient format back to standard format.

        Flattens the (wavelet_channels, basis_functions) dimensions back
        into a single channel dimension.

        Parameters
        ----------
        x : torch.Tensor
            Wavelet format (see _reshape_to_wavelet_format)

        Returns
        -------
        torch.Tensor
            Standard format: (*spatial, c*k^n_dim)
        """
        # Collapse the last two dimensions (c, basis_functions) -> (channels)
        return x.flatten(-2, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Multiwavelet Neural Operator.

        Processing pipeline:
        1. (First ``forward`` only, if enabled) validate spatial grid and tensor rank
        2. Validate ``L < num_scales`` for the input grid (once per ``forward``)
        3. Lift to wavelet space: input → high-dimensional wavelet coefficients
        4. Apply MWNO layers with ReLU activation (except last layer)
        5. Project back to output space: wavelet coefficients → output
        6. Squeeze channel dimension if out_channels=1

        Parameters
        ----------
        x : torch.Tensor
            Input function values
            - 1D: (batch, n_points, in_channels)
            - 2D: (batch, height, width, in_channels)
            - 3D: (batch, height, width, time, in_channels)

            Requirements:
            - Spatial dimensions must be powers of 2
            - n_points, height, width ∈ {2, 4, 8, 16, 32, 64, 128, ...}
            - When checks run, violations raise ``ValueError`` (see ``check_spatial_resolution_once``).

        Returns
        -------
        torch.Tensor
            Output function values
            - 1D: (batch, n_points, out_channels) or (batch, n_points) if out_channels=1
            - 2D: (batch, height, width, out_channels) or (batch, height, width) if out_channels=1
            - 3D: (batch, height, width, time, out_channels) or (batch, height, width, time) if out_channels=1

        Examples
        --------
        >>> model = MWNO(n_modes=12, n_dim=2, in_channels=1, out_channels=1)
        >>> x = torch.randn(16, 64, 64, 1)
        >>> y = model(x)  # shape: (16, 64, 64) - channel squeezed
        """
        if self.check_spatial_resolution_once and not self._spatial_resolution_checked:
            self._validate_spatial_resolution(x)
            self._spatial_resolution_checked = True

        self._validate_L_vs_num_scales(x)

        # Lift to wavelet space
        x = self.lifting(x)

        # Reshape to wavelet coefficient format
        x = self._reshape_to_wavelet_format(x)

        # Apply MWNO transformation layers
        for layer_idx, layer in enumerate(self.mwno_layers):
            x = layer(x)
            # Apply ReLU between layers (but not after last layer)
            if layer_idx < self.n_layers - 1:
                x = F.relu(x)

        # Reshape back to standard format
        x = self._reshape_from_wavelet_format(x)

        # Project to output space (ChannelMLP expects batch, channels, *spatial)
        x = x.movedim(-1, 1)
        x = self.projection(x)
        x = x.movedim(1, -1)

        # Validate output channels
        assert x.shape[-1] == self.out_channels, (
            f"Expected {self.out_channels} output channels, "
            f"but got {x.shape[-1]}. This is an internal error."
        )

        # Squeeze channel dimension for scalar outputs
        if self.out_channels == 1:
            return x.squeeze(-1)

        return x