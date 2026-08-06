from typing import List, Union

import torch
from torch import nn
import torch.nn.functional as F

from .channel_mlp import ChannelMLP
from .complex import CGELU, ctanh, ComplexValued
from .normalization_layers import AdaIN, InstanceNorm, BatchNorm
from .skip_connections import skip_connection
from .spectral_convolution import SpectralConv
from ..utils import validate_scaling_factor


Number = Union[int, float]


def _make_conv_bias(in_channels, out_channels, n_dim, kernel_size):
    """Create the local convolutional bias term used alongside SpectralConv."""
    if kernel_size < 1:
        raise ValueError(f"conv_bias_kernel must be >= 1, got {kernel_size}")

    if kernel_size == 1:
        return skip_connection(
            in_channels,
            out_channels,
            skip_type="linear",
            n_dim=n_dim,
        )

    if n_dim > 3:
        raise NotImplementedError(
            "conv_bias_kernel > 1 is only implemented for 1D, 2D, and 3D FNO blocks."
        )

    conv_module = getattr(nn, f"Conv{n_dim}d")
    return conv_module(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        padding="same",
        bias=False,
    )


class FNOBlocks(nn.Module):
    """FNOBlocks implements a sequence of Fourier layers.
    
    The Fourier layers are first described in [1]_, and the exact implementation details 
    of the Fourier layer architecture are discussed in [2]_.

    Parameters
    ----------
    in_channels : int
        Number of input channels to Fourier layers
    out_channels : int
        Number of output channels after Fourier layers
    n_modes : int or List[int]
        Number of modes to keep along each dimension in frequency space.
        Can either be specified as an int (for all dimensions) or an iterable
        with one number per dimension
    resolution_scaling_factor : Optional[Union[Number, List[Number]]], optional
        Factor by which to scale outputs for super-resolution, by default None
    n_layers : int, optional
        Number of Fourier layers to apply in sequence, by default 1
    max_n_modes : int or List[int], optional
        Maximum number of modes to keep along each dimension, by default None
    fno_block_precision : str, optional
        Floating point precision to use for computations. Options: "full", "half", "mixed", by default "full"
    use_channel_mlp : bool, optional
        Whether to use an MLP layer after each FNO block, by default True
    channel_mlp_dropout : float, optional
        Dropout parameter for self.channel_mlp, by default 0
    channel_mlp_expansion : float, optional
        Expansion parameter for self.channel_mlp, by default 0.5
    non_linearity : torch.nn.F module, optional
        Nonlinear activation function to use between layers, by default F.gelu
    stabilizer : Literal["tanh"], optional
        Stabilizing module to use between certain layers. Options: "tanh", None, by default None
    norm : Literal["ada_in", "group_norm", "instance_norm", "batch_norm"], optional
        Normalization layer to use. Options: "ada_in", "group_norm", "instance_norm", "batch_norm", None, by default None
    norm_groups : int, optional
        Number of groups for GroupNorm, by default 1
    ada_in_features : int, optional
        Number of features for adaptive instance norm above, by default None
    preactivation : bool, optional
        Whether to call forward pass with pre-activation, by default False
        If True, call nonlinear activation and norm before Fourier convolution
        If False, call activation and norms after Fourier convolutions
    fno_skip : str, optional
        Module to use for FNO skip connections. Options: "linear", "soft-gating", "identity", None, by default "linear"
        If None, no skip connection is added. See layers.skip_connections for more details
    conv_bias_kernel : int, optional
        Kernel size for the local convolutional bias term used when ``fno_skip="linear"``.
        ``1`` preserves the default pointwise bias term, while larger kernels add a
        local convolution alongside the global spectral convolution. Default: 1
    channel_mlp_skip : str, optional
        Module to use for ChannelMLP skip connections. Options: "linear", "soft-gating", "identity", None, by default "soft-gating"
        If None, no skip connection is added. See layers.skip_connections for more details

    Other Parameters
    ----------------
    complex_data : bool, optional
        Whether the FNO's data takes on complex values in space, by default False
    separable : bool, optional
        Separable parameter for SpectralConv, by default False
    factorization : str, optional
        Factorization parameter for SpectralConv. Options: "tucker", "cp", "tt", None, by default None
    rank : float, optional
        Rank parameter for SpectralConv, by default 1.0
    conv_module : BaseConv, optional
        Module to use for convolutions in FNO block, by default SpectralConv
    joint_factorization : bool, optional
        Whether to factorize all spectralConv weights as one tensor, by default False
    fixed_rank_modes : bool, optional
        Fixed_rank_modes parameter for SpectralConv, by default False
    implementation : str, optional
        Implementation parameter for SpectralConv. Options: "factorized", "reconstructed", by default "factorized"
    decomposition_kwargs : dict, optional
        Kwargs for tensor decomposition in SpectralConv, by default dict()
    enforce_hermitian_symmetry : bool, optional
        Whether to enforce Hermitian symmetry conditions when performing inverse FFT
        for real-valued data. Only used when ``conv_module`` is :class:`SpectralConv`
        or a subclass; ignored otherwise. When True, explicitly enforces that the 0th
        frequency and Nyquist frequency are real-valued before calling irfft. When False,
        relies on cuFFT's irfftn to handle symmetry automatically, which may fail on
        certain GPUs or input sizes, causing line artifacts. By default True.

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
        resolution_scaling_factor=None,
        n_layers=1,
        max_n_modes=None,
        fno_block_precision="full",
        use_channel_mlp=True,
        channel_mlp_dropout=0,
        channel_mlp_expansion=0.5,
        non_linearity=F.gelu,
        stabilizer=None,
        norm=None,
        norm_groups=1,
        ada_in_features=None,
        preactivation=False,
        fno_skip="linear",
        conv_bias_kernel=1,
        channel_mlp_skip="soft-gating",
        complex_data=False,
        separable=False,
        factorization=None,
        rank=1.0,
        conv_module=SpectralConv,
        fixed_rank_modes=False,
        implementation="factorized",
        decomposition_kwargs=dict(),
        enforce_hermitian_symmetry=True,
    ):
        super().__init__()
        if isinstance(n_modes, int):
            n_modes = [n_modes]
        self._n_modes = n_modes
        self.n_dim = len(n_modes)

        self.resolution_scaling_factor: Union[
            None, List[List[float]]
        ] = validate_scaling_factor(resolution_scaling_factor, self.n_dim, n_layers)

        self.max_n_modes = max_n_modes
        self.fno_block_precision = fno_block_precision
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_layers = n_layers
        self.stabilizer = stabilizer
        self.rank = rank
        self.factorization = factorization
        self.fixed_rank_modes = fixed_rank_modes
        self.decomposition_kwargs = decomposition_kwargs
        self.fno_skip = fno_skip
        self.conv_bias_kernel = conv_bias_kernel
        self.channel_mlp_skip = channel_mlp_skip
        self.complex_data = complex_data

        self.use_channel_mlp = use_channel_mlp
        self.channel_mlp_expansion = channel_mlp_expansion
        self.channel_mlp_dropout = channel_mlp_dropout
        self.implementation = implementation
        self.separable = separable
        self.preactivation = preactivation
        self.ada_in_features = ada_in_features
        self.enforce_hermitian_symmetry = enforce_hermitian_symmetry

        # apply real nonlin if data is real, otherwise CGELU
        if self.complex_data:
            self.non_linearity = CGELU
        else:
            self.non_linearity = non_linearity

        # One conv per layer. Only resolution_scaling_factor varies by layer index
        self.convs = nn.ModuleList(
            [
                conv_module(
                    self.in_channels,
                    self.out_channels,
                    self.n_modes,
                    # Per-layer scaling for super-resolution, or None if disabled
                    resolution_scaling_factor=(
                        self.resolution_scaling_factor[i]
                        if resolution_scaling_factor is not None
                        else None
                    ),
                    max_n_modes=max_n_modes,
                    rank=rank,
                    fixed_rank_modes=fixed_rank_modes,
                    implementation=implementation,
                    separable=separable,
                    factorization=factorization,
                    fno_block_precision=fno_block_precision,
                    decomposition_kwargs=decomposition_kwargs,
                    complex_data=complex_data,
                    # Only SpectralConv (and subclasses) accept enforce_hermitian_symmetry. Others ignore it
                    **(
                        {"enforce_hermitian_symmetry": enforce_hermitian_symmetry}
                        if issubclass(conv_module, SpectralConv)
                        else {}
                    ),
                )
                for i in range(n_layers)
            ]
        )

        if fno_skip is not None:
            if fno_skip.lower() == "linear":
                # FourCastNet v3-like local + global convolutions: the spectral
                # path is global, while larger kernels give this bias term locality.
                fno_skip_modules = [
                    _make_conv_bias(
                        self.in_channels,
                        self.out_channels,
                        self.n_dim,
                        conv_bias_kernel,
                    )
                    for _ in range(n_layers)
                ]
            else:
                if conv_bias_kernel != 1:
                    raise ValueError(
                        "conv_bias_kernel can only differ from 1 when fno_skip='linear'."
                    )
                fno_skip_modules = [
                    skip_connection(
                        self.in_channels,
                        self.out_channels,
                        skip_type=fno_skip,
                        n_dim=self.n_dim,
                    )
                    for _ in range(n_layers)
                ]
            self.fno_skips = nn.ModuleList(fno_skip_modules)
        else:
            if conv_bias_kernel != 1:
                raise ValueError(
                    "conv_bias_kernel can only differ from 1 when fno_skip='linear'."
                )
            self.fno_skips = None
        if self.complex_data and self.fno_skips is not None:
            self.fno_skips = nn.ModuleList([ComplexValued(x) for x in self.fno_skips])

        if self.use_channel_mlp:
            self.channel_mlp = nn.ModuleList(
                [
                    ChannelMLP(
                        in_channels=self.out_channels,
                        hidden_channels=round(self.out_channels * channel_mlp_expansion),
                        dropout=channel_mlp_dropout,
                        n_dim=self.n_dim,
                    )
                    for _ in range(n_layers)
                ]
            )
            if self.complex_data:
                self.channel_mlp = nn.ModuleList(
                    [ComplexValued(x) for x in self.channel_mlp]
                )
            if channel_mlp_skip is not None:
                self.channel_mlp_skips = nn.ModuleList(
                    [
                        skip_connection(
                            self.in_channels,
                            self.out_channels,
                            skip_type=channel_mlp_skip,
                            n_dim=self.n_dim,
                        )
                        for _ in range(n_layers)
                    ]
                )
            else:
                self.channel_mlp_skips = None
            if self.complex_data and self.channel_mlp_skips is not None:
                self.channel_mlp_skips = nn.ModuleList(
                    [ComplexValued(x) for x in self.channel_mlp_skips]
                )

        # Each block has 2 norms when using ChannelMLP (one after FNO conv, one after MLP),
        # but only 1 when channel MLP is disabled.
        self.n_norms = 2 if self.use_channel_mlp else 1
        if norm is None:
            self.norm = None
        elif norm == "instance_norm":
            self.norm = nn.ModuleList(
                [InstanceNorm() for _ in range(n_layers * self.n_norms)]
            )
        elif norm == "group_norm":
            self.norm = nn.ModuleList(
                [
                    nn.GroupNorm(num_groups=norm_groups, num_channels=self.out_channels)
                    for _ in range(n_layers * self.n_norms)
                ]
            )

        elif norm == "batch_norm":
            self.norm = nn.ModuleList(
                [
                    BatchNorm(n_dim=self.n_dim, num_features=self.out_channels)
                    for _ in range(n_layers * self.n_norms)
                ]
            )

        elif norm == "ada_in":
            self.norm = nn.ModuleList(
                [
                    AdaIN(ada_in_features, out_channels)
                    for _ in range(n_layers * self.n_norms)
                ]
            )
        else:
            raise ValueError(
                f"Got norm={norm} but expected None or one of "
                "[instance_norm, group_norm, batch_norm, ada_in]"
            )

        if self.complex_data and self.norm is not None:
            self.norm = nn.ModuleList([ComplexValued(x) for x in self.norm])

    def set_ada_in_embeddings(self, *embeddings):
        """Sets the embeddings of each Ada-IN norm layers

        Parameters
        ----------
        embeddings : tensor or list of tensor
            if a single embedding is given, it will be used for each norm layer
            otherwise, each embedding will be used for the corresponding norm layer
        """
        if self.norm is not None:
            if len(embeddings) == 1:
                for norm in self.norm:
                    norm.set_embedding(embeddings[0])
            else:
                for norm, embedding in zip(self.norm, embeddings):
                    norm.set_embedding(embedding)

    def forward(self, x, index=0, output_shape=None):
        if self.preactivation:
            return self.forward_with_preactivation(x, index, output_shape)
        else:
            return self.forward_with_postactivation(x, index, output_shape)

    def forward_with_postactivation(self, x, index=0, output_shape=None):
        if self.fno_skips is not None:
            x_skip_fno = self.fno_skips[index](x)
            x_skip_fno = self.convs[index].transform(x_skip_fno, output_shape=output_shape)

        if self.use_channel_mlp and self.channel_mlp_skips is not None:
            x_skip_channel_mlp = self.channel_mlp_skips[index](x)
            x_skip_channel_mlp = self.convs[index].transform(x_skip_channel_mlp, output_shape=output_shape)

        if self.stabilizer == "tanh":
            if self.complex_data:
                x = ctanh(x)
            else:
                x = torch.tanh(x)

        x_fno = self.convs[index](x, output_shape=output_shape)

        if self.norm is not None:
            x_fno = self.norm[self.n_norms * index](x_fno)

        x = x_fno + x_skip_fno if self.fno_skips is not None else x_fno

        if index < (self.n_layers - 1):
            x = self.non_linearity(x)

        if self.use_channel_mlp:
            if self.channel_mlp_skips is not None:
                x = self.channel_mlp[index](x) + x_skip_channel_mlp
            else:
                x = self.channel_mlp[index](x)

            if self.norm is not None:
                x = self.norm[self.n_norms * index + 1](x)

            if index < (self.n_layers - 1):
                x = self.non_linearity(x)

        return x

    def forward_with_preactivation(self, x, index=0, output_shape=None):
        # Apply non-linear activation (and norm)
        # before this block's convolution/forward pass:
        x = self.non_linearity(x)

        if self.norm is not None:
            x = self.norm[self.n_norms * index](x)

        if self.fno_skips is not None:
            x_skip_fno = self.fno_skips[index](x)
            x_skip_fno = self.convs[index].transform(x_skip_fno, output_shape=output_shape)

        if self.use_channel_mlp and self.channel_mlp_skips is not None:
            x_skip_channel_mlp = self.channel_mlp_skips[index](x)
            x_skip_channel_mlp = self.convs[index].transform(x_skip_channel_mlp, output_shape=output_shape)

        if self.stabilizer == "tanh":
            if self.complex_data:
                x = ctanh(x)
            else:
                x = torch.tanh(x)

        x_fno = self.convs[index](x, output_shape=output_shape)

        x = x_fno + x_skip_fno if self.fno_skips is not None else x_fno

        if index < (self.n_layers - 1):
            x = self.non_linearity(x)

        if self.use_channel_mlp:
            if self.norm is not None:
                x = self.norm[self.n_norms * index + 1](x)

            if self.channel_mlp_skips is not None:
                x = self.channel_mlp[index](x) + x_skip_channel_mlp
            else:
                x = self.channel_mlp[index](x)

        return x

    @property
    def n_modes(self):
        return self._n_modes

    @n_modes.setter
    def n_modes(self, n_modes):
        for i in range(self.n_layers):
            self.convs[i].n_modes = n_modes
        self._n_modes = n_modes

    def get_block(self, indices):
        """Returns a sub-FNO Block layer from the jointly parametrized main block

        The parametrization of an FNOBlock layer is shared with the main one.
        """
        if self.n_layers == 1:
            raise ValueError(
                "A single layer is parametrized, directly use the main class."
            )

        return SubModule(self, indices)

    def __getitem__(self, indices):
        return self.get_block(indices)


class SubModule(nn.Module):
    """Class representing one of the sub_module from the mother joint module

    Notes
    -----
    This relies on the fact that nn.Parameters are not duplicated:
    if the same nn.Parameter is assigned to multiple modules,
    they all point to the same data, which is shared.
    """

    def __init__(self, main_module, indices):
        super().__init__()
        self.main_module = main_module
        self.indices = indices

    def forward(self, x):
        return self.main_module.forward(x, self.indices)
