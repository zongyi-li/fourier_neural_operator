from functools import partialmethod
from typing import Tuple, List, Optional, Union, Literal

Number = Union[float, int]

import torch
import torch.nn as nn
import torch.nn.functional as F

# Set warning filter to show each warning only once
import warnings

warnings.filterwarnings("once", category=UserWarning)


from ..layers.embeddings import GridEmbeddingND, GridEmbedding2D, ConditioningEmbedding
from ..layers.spectral_convolution import SpectralConv
from ..layers.padding import DomainPadding
from ..layers.fno_block import FNOBlocks
from ..layers.channel_mlp import ChannelMLP
from ..layers.complex import ComplexValued
from .base_model import BaseModel


class FNO(BaseModel, name="FNO"):
    """N-Dimensional Fourier Neural Operator. The FNO learns a mapping between
    spaces of functions discretized over regular grids using Fourier convolutions,
    as described in [1]_.

    The key component of an FNO is its SpectralConv layer (see
    ``neuralop.layers.spectral_convolution``), which is similar to a standard CNN
    conv layer but operates in the frequency domain.

    For a deeper dive into the FNO architecture, refer to :ref:`fno_intro`.

    Parameters
    ----------
    n_modes : Tuple[int, ...]
        Number of modes to keep in Fourier Layer, along each dimension.
        The dimensionality of the FNO is inferred from len(n_modes).
        n_modes must be larger enough but smaller than max_resolution//2 (Nyquist frequency)
    in_channels : int
        Number of channels in input function. Determined by the problem.
    out_channels : int
        Number of channels in output function. Determined by the problem.
    hidden_channels : int
        Width of the FNO (i.e. number of channels).
        This significantly affects the number of parameters of the FNO.
        Good starting point can be 64, and then increased if more expressivity is needed.
        Update lifting_channel_ratio and projection_channel_ratio accordingly since they are proportional to hidden_channels.
    n_layers : int, optional
        Number of Fourier Layers. Default: 4

    Other Parameters
    ----------------
    lifting_channel_ratio : Number, optional
        Ratio of lifting channels to hidden_channels.
        The number of lifting channels in the lifting block of the FNO is
        lifting_channel_ratio * hidden_channels (e.g. default 2 * hidden_channels).
    projection_channel_ratio : Number, optional
        Ratio of projection channels to hidden_channels.
        The number of projection channels in the projection block of the FNO is
        projection_channel_ratio * hidden_channels (e.g. default 2 * hidden_channels).
    positional_embedding : Union[str, nn.Module], optional
        Positional embedding to apply to last channels of raw input
        before being passed through the FNO.
        Options:
        - "grid": Appends a grid positional embedding with default settings to the last channels of raw input.
          Assumes the inputs are discretized over a grid with entry [0,0,...] at the origin and side lengths of 1.
        - GridEmbeddingND: Uses this module directly (see :mod:`neuralop.embeddings.GridEmbeddingND` for details).
        - GridEmbedding2D: Uses this module directly for 2D cases.
        - None: Does nothing.
        Default: "grid"
    non_linearity : nn.Module, optional
        Non-Linear activation function module to use. Default: F.gelu
    norm : Literal["ada_in", "group_norm", "instance_norm"], optional
        Normalization layer to use. Options: "ada_in", "group_norm", "instance_norm", None. Default: None
    norm_groups : int, optional
        Number of groups for GroupNorm, by default 1
    complex_data : bool, optional
        Whether the data is complex-valued. If True, initializes complex-valued modules. Default: False
    use_channel_mlp : bool, optional
        Whether to use an MLP layer after each FNO block. Default: True
    channel_mlp_dropout : float, optional
        Dropout parameter for ChannelMLP in FNO Block. Default: 0
    channel_mlp_expansion : float, optional
        Expansion parameter for ChannelMLP in FNO Block. Default: 0.5
    channel_mlp_skip : Literal["linear", "identity", "soft-gating", None], optional
        Type of skip connection to use in channel-mixing mlp. Options: "linear", "identity", "soft-gating", None.
        Default: "soft-gating"
    fno_skip : Literal["linear", "identity", "soft-gating", None], optional
        Type of skip connection to use in FNO layers. Options: "linear", "identity", "soft-gating", None.
        Default: "linear"
    resolution_scaling_factor : Union[Number, List[Number]], optional
        Layer-wise factor by which to scale the domain resolution of function.
        Options:
        - None: No scaling
        - Single number n: Scales resolution by n at each layer
        - List of numbers [n_0, n_1,...]: Scales layer i's resolution by n_i
        Default: None
    domain_padding : Union[Number, List[Number]], optional
        Percentage of padding to use.
        Options:
        - None: No padding
        - Single number: Percentage of padding to use along all dimensions
        - List of numbers [p1, p2, ..., pN]: Percentage of padding along each dimension
        Default: None
    fno_block_precision : str, optional
        Precision mode in which to perform spectral convolution.
        Options: "full", "half", "mixed". Default: "full". Default: "full"
    stabilizer : str, optional
        Whether to use a stabilizer in FNO block. Options: "tanh", None. Default: None.
        stabilizer greatly improves performance in the case `fno_block_precision='mixed'`.
    max_n_modes : Tuple[int, ...], optional
        Maximum number of modes to use in Fourier domain during training.
        None means that all the n_modes are used.
        Tuple of integers: Incrementally increase the number of modes during training.
        This can be updated dynamically during training.
    factorization : str, optional
        Tensor factorization of the FNO layer weights to use.
        Options: "None", "Tucker", "CP", "TT"
        Other factorization methods supported by tltorch. Default: None
    rank : float, optional
        Tensor rank to use in factorization. Default: 1.0
        Set to float <1.0 when using TFNO (i.e. when factorization is not None).
        A TFNO with rank 0.1 has roughly 10% of the parameters of a dense FNO.
    fixed_rank_modes : bool, optional
        Whether to not factorize certain modes. Default: False
    implementation : str, optional
        Implementation method for factorized tensors.
        Options: "factorized", "reconstructed". Default: "factorized"
    decomposition_kwargs : dict, optional
        Extra kwargs for tensor decomposition (see `tltorch.FactorizedTensor`). Default: {}
    separable : bool, optional
        Whether to use a separable spectral convolution. Default: False
    preactivation : bool, optional
        Whether to compute FNO forward pass with resnet-style preactivation. Default: False
    conv_module : nn.Module, optional
        Module to use for FNOBlock's convolutions. Default: SpectralConv
    enforce_hermitian_symmetry : bool, optional
        Whether to enforce Hermitian symmetry conditions when performing inverse FFT
        for real-valued data. Only used when ``conv_module`` is :class:`SpectralConv`
        or a subclass; ignored otherwise. When True, explicitly enforces that the 0th
        frequency and Nyquist frequency are real-valued before calling irfft. When False,
        relies on cuFFT's irfftn to handle symmetry automatically, which may fail on
        certain GPUs or input sizes, causing line artifacts. By default True.

    Examples
    --------
    >>> from neuralop.models import FNO
    >>> model = FNO(n_modes=(12,12), in_channels=1, out_channels=1, hidden_channels=64)
    >>> model
    FNO(
    (positional_embedding): GridEmbeddingND()
    (fno_blocks): FNOBlocks(
        (convs): SpectralConv(
        (weight): ModuleList(
            (0-3): 4 x DenseTensor(shape=torch.Size([64, 64, 12, 7]), rank=None)
        )
        )
            ... torch.nn.Module printout truncated ...

    References
    ----------
    .. [1] :

    Li, Z. et al. "Fourier Neural Operator for Parametric Partial Differential
        Equations" (2021). ICLR 2021, https://arxiv.org/pdf/2010.08895.

    """

    def __init__(
        self,
        n_modes: Tuple[int, ...],
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        n_layers: int = 4,
        lifting_channel_ratio: Number = 2,
        projection_channel_ratio: Number = 2,
        positional_embedding: Union[str, nn.Module] = "grid",
        non_linearity: nn.Module = F.gelu,
        norm: Literal["ada_in", "group_norm", "instance_norm"] = None,
        norm_groups: int = 1,
        complex_data: bool = False,
        use_channel_mlp: bool = True,
        channel_mlp_dropout: float = 0,
        channel_mlp_expansion: float = 0.5,
        channel_mlp_skip: Literal["linear", "identity", "soft-gating", None] = "soft-gating",
        fno_skip: Literal["linear", "identity", "soft-gating", None] = "linear",
        resolution_scaling_factor: Union[Number, List[Number]] = None,
        domain_padding: Union[Number, List[Number]] = None,
        fno_block_precision: str = "full",
        stabilizer: str = None,
        max_n_modes: Tuple[int, ...] = None,
        factorization: str = None,
        rank: float = 1.0,
        fixed_rank_modes: bool = False,
        implementation: str = "factorized",
        decomposition_kwargs: dict = None,
        separable: bool = False,
        preactivation: bool = False,
        conv_module: nn.Module = SpectralConv,
        enforce_hermitian_symmetry: bool = True,
        conditioning_embedding: Optional[nn.Module] = None,
        mode_modulation: Optional[dict] = None,
        norm_modulation: Optional[dict] = None,
    ):
        if decomposition_kwargs is None:
            decomposition_kwargs = {}
        super().__init__()
        self.n_dim = len(n_modes)

        # Optional conditioning. Time/parameter conditioning is active iff a
        # ConditioningEmbedding is supplied AND at least one modulation dict is
        # enabled. The embedding is computed once here in forward and threaded
        # down; the layers only consume precomputed tensors.
        self.conditioning_embedding = conditioning_embedding
        self.n_params = (
            conditioning_embedding.n_params if conditioning_embedding is not None else 1
        )
        cond_embed_dim = (
            conditioning_embedding.out_dim if conditioning_embedding is not None else None
        )
        self._mode_mod_enabled = (
            mode_modulation is not None and mode_modulation.get("enabled", True)
        )
        self._norm_mod_enabled = (
            norm_modulation is not None and norm_modulation.get("enabled", True)
        )
        self._time_conditioned = self._mode_mod_enabled or self._norm_mod_enabled
        if self._time_conditioned and conditioning_embedding is None:
            raise ValueError(
                "mode_modulation/norm_modulation is enabled but "
                "`conditioning_embedding` is None; pass a ConditioningEmbedding."
            )
        if preactivation and self._norm_mod_enabled:
            raise ValueError(
                "norm_modulation is only supported with preactivation=False; "
                "got preactivation=True."
            )

        # n_modes is a special property - see the class' property for underlying mechanism
        # When updated, change should be reflected in fno blocks
        self._n_modes = n_modes

        self.hidden_channels = hidden_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_layers = n_layers

        # init lifting and projection channels using ratios w.r.t hidden channels
        self.lifting_channel_ratio = lifting_channel_ratio
        self.lifting_channels = int(lifting_channel_ratio * self.hidden_channels)

        self.projection_channel_ratio = projection_channel_ratio
        self.projection_channels = int(projection_channel_ratio * self.hidden_channels)

        self.non_linearity = non_linearity
        self.rank = rank
        self.factorization = factorization
        self.fixed_rank_modes = fixed_rank_modes
        self.decomposition_kwargs = decomposition_kwargs
        self.fno_skip = (fno_skip,)
        self.channel_mlp_skip = (channel_mlp_skip,)
        self.implementation = implementation
        self.separable = separable
        self.preactivation = preactivation
        self.complex_data = complex_data
        self.fno_block_precision = fno_block_precision

        ## Positional embedding
        if positional_embedding == "grid":
            spatial_grid_boundaries = [[0.0, 1.0]] * self.n_dim
            self.positional_embedding = GridEmbeddingND(
                in_channels=self.in_channels,
                dim=self.n_dim,
                grid_boundaries=spatial_grid_boundaries,
            )
        elif isinstance(positional_embedding, GridEmbedding2D):
            if self.n_dim == 2:
                self.positional_embedding = positional_embedding
            else:
                raise ValueError(
                    f"Error: expected {self.n_dim}-d positional embeddings, got {positional_embedding}"
                )
        elif isinstance(positional_embedding, GridEmbeddingND):
            self.positional_embedding = positional_embedding
        elif positional_embedding is None:
            self.positional_embedding = None
        else:
            raise ValueError(
                f"Error: tried to instantiate FNO positional embedding with {positional_embedding},\
                              expected one of 'grid', GridEmbeddingND"
            )

        ## Domain padding
        if domain_padding is not None and (
            (isinstance(domain_padding, list) and sum(domain_padding) > 0)
            or (isinstance(domain_padding, (float, int)) and domain_padding > 0)
        ):
            self.domain_padding = DomainPadding(
                domain_padding=domain_padding,
                resolution_scaling_factor=resolution_scaling_factor,
            )
        else:
            self.domain_padding = None

        ## Resolution scaling factor
        if resolution_scaling_factor is not None:
            if isinstance(resolution_scaling_factor, (float, int)):
                resolution_scaling_factor = [resolution_scaling_factor] * self.n_layers
        self.resolution_scaling_factor = resolution_scaling_factor

        ## FNO blocks
        self.fno_blocks = FNOBlocks(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            n_modes=self.n_modes,
            resolution_scaling_factor=resolution_scaling_factor,
            use_channel_mlp=use_channel_mlp,
            channel_mlp_dropout=channel_mlp_dropout,
            channel_mlp_expansion=channel_mlp_expansion,
            non_linearity=non_linearity,
            stabilizer=stabilizer,
            norm=norm,
            norm_groups=norm_groups,
            preactivation=preactivation,
            fno_skip=fno_skip,
            channel_mlp_skip=channel_mlp_skip,
            complex_data=complex_data,
            max_n_modes=max_n_modes,
            fno_block_precision=fno_block_precision,
            rank=rank,
            fixed_rank_modes=fixed_rank_modes,
            implementation=implementation,
            separable=separable,
            factorization=factorization,
            decomposition_kwargs=decomposition_kwargs,
            conv_module=conv_module,
            n_layers=n_layers,
            enforce_hermitian_symmetry=enforce_hermitian_symmetry,
            mode_modulation=mode_modulation if self._mode_mod_enabled else None,
            cond_embed_dim=cond_embed_dim,
        )

        # FiLM (norm-modulation) heads live here, not in the block: their width
        # depends on the block's channel count, so building them in the FNO
        # keeps FNOBlocks as a pure applies-affine consumer of `mod_params`.
        self._build_norm_modulator(
            norm_modulation, cond_embed_dim, n_layers, hidden_channels
        )

        ## Lifting layer
        # if adding a positional embedding, add those channels to lifting
        lifting_in_channels = self.in_channels
        if self.positional_embedding is not None:
            lifting_in_channels += self.n_dim
        self.lifting = ChannelMLP(
            in_channels=lifting_in_channels,
            out_channels=self.hidden_channels,
            hidden_channels=self.lifting_channels,
            n_layers=2,
            n_dim=self.n_dim,
            non_linearity=non_linearity,
        )
        if self.complex_data:
            self.lifting = ComplexValued(self.lifting)

        ## Projection layer
        self.projection = ChannelMLP(
            in_channels=self.hidden_channels,
            out_channels=out_channels,
            hidden_channels=self.projection_channels,
            n_layers=2,
            n_dim=self.n_dim,
            non_linearity=non_linearity,
        )
        if self.complex_data:
            self.projection = ComplexValued(self.projection)

    def _build_norm_modulator(
        self, norm_modulation, cond_embed_dim, n_layers, out_channels
    ):
        """Build the per-layer FiLM heads that map the embedding `e` to the
        scale/shift/gate params applied around each block's norm layers.

        Each head is a ChannelMLP (cond_embed_dim -> total_out_dim).
        Disabled specs are omitted from the output, so the head is sized to
        exactly the enabled params.
        """
        self.norm_modulator = None
        self._mod_slices = {}
        if not self._norm_mod_enabled:
            return

        modulate1 = bool(norm_modulation.get("modulate1", True))
        modulate1_gate = bool(norm_modulation.get("modulate1_gate", True))
        modulate2 = bool(norm_modulation.get("modulate2", True))
        modulate2_gate = bool(norm_modulation.get("modulate2_gate", True))
        hidden_channels = int(norm_modulation.get("hidden_channels", 64))

        # Spec for the head output: name, channel count, enabled flag.
        specs = [
            ("scale1", out_channels, modulate1),
            ("shift1", out_channels, modulate1),
            ("gate1", 1, modulate1_gate),
            ("scale2", out_channels, modulate2),
            ("shift2", out_channels, modulate2),
            ("gate2", 1, modulate2_gate),
        ]
        offset = 0
        for name, dim, enabled in specs:
            if enabled:
                self._mod_slices[name] = slice(offset, offset + dim)
                offset += dim

        total_out_dim = offset
        if total_out_dim == 0:
            return

        self.norm_modulator = nn.ModuleList(
            [
                ChannelMLP(
                    in_channels=cond_embed_dim,
                    out_channels=total_out_dim,
                    hidden_channels=hidden_channels,
                    n_dim=1,
                )
                for _ in range(n_layers)
            ]
        )

    def _film_params(self, e, layer_idx):
        """Return the FiLM `mod_params` dict for a layer from embedding `e`.

        Each value is shaped (B, dim, 1, ...) with self.n_dim trailing
        singleton dims so it broadcasts over spatial axes. Disabled specs are
        absent from the returned dict.
        """
        if self.norm_modulator is None:
            return None
        raw = self.norm_modulator[layer_idx](e.unsqueeze(-1))  # (B, total_out, 1)
        spatial = (1,) * self.n_dim
        params = {}
        for name, sl in self._mod_slices.items():
            v = raw[:, sl]
            params[name] = v.reshape(v.shape[0], v.shape[1], *spatial)
        return params

    def _prepare_conditioning(self, x, t):
        """Validate/normalize `t` to shape (B, n_params) and embed it.

        Returns the embedding e of shape (B, out_dim), or None when
        the model is not time-conditioned.
        """
        if not self._time_conditioned:
            return None
        if t is None:
            t = torch.ones(
                x.shape[0], self.n_params, dtype=x.dtype, device=x.device
            )
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, dtype=x.dtype, device=x.device)
        if t.ndim == 0:
            t = t.expand(x.shape[0], 1)
        elif t.ndim == 1:
            if t.shape[0] == self.n_params:
                t = t.unsqueeze(0).expand(x.shape[0], self.n_params)
            elif t.shape[0] == x.shape[0]:
                t = t.unsqueeze(-1)
        if t.ndim != 2 or t.shape[1] != self.n_params or t.shape[0] != x.shape[0]:
            raise ValueError(
                f"t must have shape ({x.shape[0]}, {self.n_params}); got tensor "
                f"with shape {tuple(t.shape)} for x batch {x.shape[0]}."
            )
        return self.conditioning_embedding(t)

    def forward(self, x, output_shape=None, t=None, **kwargs):
        """FNO's forward pass

        1. Applies optional positional encoding

        2. Sends inputs through a lifting layer to a high-dimensional latent space

        3. Applies optional domain padding to high-dimensional intermediate function representation

        4. Applies `n_layers` Fourier/FNO layers in sequence (SpectralConvolution + skip connections, nonlinearity)

        5. If domain padding was applied, domain padding is removed

        6. Projection of intermediate function representation to the output channels

        Parameters
        ----------
        x : tensor
            input tensor

        output_shape : {tuple, tuple list, None}, default is None
            Gives the option of specifying the exact output shape for odd shaped inputs.

            * If None, don't specify an output shape

            * If tuple, specifies the output-shape of the **last** FNO Block

            * If tuple list, specifies the exact output-shape of each FNO Block

        t : float, int, or torch.Tensor, optional
            Conditioning parameter(s), used only when the model was built with
            a conditioning_embedding and a modulation dict. Accepted as a
            Python scalar, a 0-d tensor, a 1-d tensor ((B,) or
            (n_params,)), or a (B, n_params) tensor. Ignored when the
            model is not conditioned; defaults to ones when omitted.
        """
        if kwargs:
            warnings.warn(
                f"FNO.forward() received unexpected keyword arguments: {list(kwargs.keys())}. "
                "These arguments will be ignored.",
                UserWarning,
                stacklevel=2,
            )

        if output_shape is None:
            output_shape = [None] * self.n_layers
        elif isinstance(output_shape, tuple):
            output_shape = [None] * (self.n_layers - 1) + [output_shape]

        # Compute the conditioning embedding once; threaded to every block.
        e = self._prepare_conditioning(x, t)

        # append spatial pos embedding if set
        if self.positional_embedding is not None:
            x = self.positional_embedding(x)

        x = self.lifting(x)

        if self.domain_padding is not None:
            x = self.domain_padding.pad(x)

        for layer_idx in range(self.n_layers):
            mod_params = self._film_params(e, layer_idx) if e is not None else None
            x = self.fno_blocks(
                x, layer_idx, output_shape=output_shape[layer_idx],
                mode_embedding=e, mod_params=mod_params,
            )

        if self.domain_padding is not None:
            x = self.domain_padding.unpad(x)

        x = self.projection(x)

        return x

    @property
    def n_modes(self):
        return self._n_modes

    @n_modes.setter
    def n_modes(self, n_modes):
        self.fno_blocks.n_modes = n_modes
        self._n_modes = n_modes

    def clear_all_caches(self):
        """Clear cached frequency-mode grids in all spectral conv submodules.

        Spectral convs cache the phi_k grid per (shape, device) when mode
        modulation is enabled. Call this after changing input resolution or
        moving the model to a new device so the grids are rebuilt.
        """
        for module in self.modules():
            if hasattr(module, "clear_cache"):
                module.clear_cache()


def partialclass(new_name, cls, *args, **kwargs):
    """Create a new class with different default values

    See the Spherical FNO class in neuralop/models/sfno.py for an example.

    Notes
    -----
    An obvious alternative would be to use functools.partial
    >>> new_class = partial(cls, **kwargs)

    The issue is twofold:
    1. the class doesn't have a name, so one would have to set it explicitly:
    >>> new_class.__name__ = new_name

    2. the new class will be a functools object and one cannot inherit from it.

    Instead, here, we define dynamically a new class, inheriting from the existing one.
    """
    __init__ = partialmethod(cls.__init__, *args, **kwargs)
    return type(
        new_name,
        (cls,),
        {
            "__init__": __init__,
            "__doc__": cls.__doc__,
            "forward": cls.forward,
        },
    )


class TFNO(FNO):
    """Tucker Tensorized Fourier Neural Operator (TFNO).

    TFNO is an FNO with Tucker factorization enabled by default.

    It uses Tucker factorization of the weights, making the forward pass efficient by contracting
    directly with the factors of the decomposition.

    This results in a fraction of the parameters of an equivalent dense FNO.

    Parameters
    ----------
    factorization : str, optional
        Tensor factorization method, by default "Tucker"
    rank : float, optional
        Tensor rank for factorization, by default 0.1.
        A TFNO with rank 0.1 has roughly 10% of the parameters of a dense FNO.

    All other parameters are inherited from FNO with identical defaults.
    See FNO class docstring for the complete parameter list.

    Examples
    --------
    >>> from neuralop.models import TFNO
    >>> # Create a TFNO model with default Tucker factorization
    >>> model = TFNO(n_modes=(12, 12), in_channels=1, out_channels=1, hidden_channels=64)
    >>>
    >>> # Equivalent FNO model with explicit factorization:
    >>> model = FNO(n_modes=(12, 12), in_channels=1, out_channels=1, hidden_channels=64,
    ...             factorization="Tucker", rank=0.1)
    """

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("factorization", "Tucker")
        kwargs.setdefault("rank", 0.1)
        super().__init__(*args, **kwargs)


# Default mode-modulation config used by the time-conditioned convenience
# classes. Magnitude-preserving polar modulation, ~identity at init.
_T_EMB_DEFAULT_MODE_MOD = {
    "enabled": True,
    "type": "polar",
    "hidden_channels": 64,
    "k_embed_dim": 32,
    "type_k": "power",
}


class t_emb_FNO(FNO):
    """Time/parameter-conditioned Fourier Neural Operator.

    An :class:`FNO` pre-wired for conditioning: it builds a default
    :class:`~neuralop.layers.embeddings.ConditioningEmbedding` and enables
    polar mode modulation, so forward(x, t=...) conditions out of the box.
    Supports vector-valued conditioning via n_params: pass t of shape
    (B, n_params).

    Parameters
    ----------
    n_params : int, optional
        Number of conditioning parameters. Default 1 (scalar conditioning).
    embed_dim : int, optional
        Per-parameter embedding width. Default 32.
    type_t : {'sinusoidal', 'power'}, optional
        Conditioning feature map. Default 'sinusoidal' (valid for any real
        t; 'power' requires t > 0).

    Any of conditioning_embedding, mode_modulation, or
    norm_modulation passed explicitly override the defaults.

    Examples
    --------
    >>> from neuralop.models import t_emb_FNO
    >>> import torch
    >>> # scalar conditioning
    >>> model = t_emb_FNO(n_modes=(12, 12), in_channels=1, out_channels=1,
    ...                   hidden_channels=64)
    >>> y = model(torch.randn(2, 1, 16, 16), t=0.5)
    >>> # vector conditioning: t of shape (B, 5)
    >>> model = t_emb_FNO(n_modes=(12, 12), in_channels=1, out_channels=1,
    ...                   hidden_channels=64, n_params=5)
    >>> y = model(torch.randn(8, 1, 16, 16), t=torch.rand(8, 5))
    """

    def __init__(self, *args, n_params: int = 1, embed_dim: int = 32,
                 type_t: str = "sinusoidal", **kwargs):
        kwargs.setdefault("mode_modulation", _T_EMB_DEFAULT_MODE_MOD.copy())
        if kwargs.get("conditioning_embedding") is None:
            kwargs["conditioning_embedding"] = ConditioningEmbedding(
                embed_dim=embed_dim, n_params=n_params, type_t=type_t
            )
        super().__init__(*args, **kwargs)


class t_emb_TFNO(t_emb_FNO):
    """Time-conditioned Tucker-factorized FNO.

    Combines :class:`t_emb_FNO` with :class:`TFNO`'s defaults
    (factorization='Tucker', rank=0.1).
    """

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("factorization", "Tucker")
        kwargs.setdefault("rank", 0.1)
        super().__init__(*args, **kwargs)
