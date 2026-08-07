from functools import partial
import torch
import torch.nn.functional as F
import time

# Set warning filter to show each warning only once
import warnings

warnings.filterwarnings("once", category=UserWarning)


from .base_model import BaseModel

from ..layers.channel_mlp import ChannelMLP
from ..layers.embeddings import SinusoidalEmbedding
from ..layers.padding import DomainPadding
from ..layers.fno_block import FNOBlocks
from ..layers.spectral_convolution import SpectralConv
from ..layers.gno_block import GNOBlock
from ..layers.gno_weighting_functions import dispatch_weighting_fn


class GINO(BaseModel):
    """
    GINO: Geometry-informed Neural Operator - learns a mapping between
    functions presented over arbitrary coordinate meshes. The model carries
    global integration through spectral convolution layers in an intermediate
    latent space, as described in [1]_. Optionally enables a weighted output
    GNO for use in a Mollified Graph Neural Operator scheme, as introduced in [2]_.

    Parameters
    ----------
    in_channels : int
        Feature dimension of input points. Determined by the problem.
    out_channels : int
        Feature dimension of output points. Determined by the problem.
    fno_n_modes : tuple, optional
        Number of modes along each dimension to use in FNO. Default: (16, 16, 16)
        Must be larger enough but smaller than max_resolution//2 (Nyquist frequency) on the latent grid
    fno_hidden_channels : int, optional
        Hidden channels for use in FNO. Default: 64
    fno_n_layers : int, optional
        Number of layers in FNO. Default: 4
    latent_feature_channels : int, optional
        Number of channels in optional latent feature map to concatenate onto latent embeddings
        before the FNO's forward pass. Default: None
    projection_channel_ratio : int, optional
        Ratio of pointwise projection channels in the final ChannelMLP to fno_hidden_channels.
        The number of projection channels in the final ChannelMLP is computed by
        projection_channel_ratio * fno_hidden_channels (i.e. default 256). Default: 4
    gno_coord_dim : int, optional
        Geometric dimension of input/output queries. Determined by the problem. Default: 3
    in_gno_radius : float, optional
        Radius in input space for GNO neighbor search. Default: 0.033
        Larger radius means more neighboors so more global interactions, but larger computational cost.
    out_gno_radius : float, optional
        Radius in output space for GNO neighbor search. Default: 0.033
        Larger radius means more neighboors so more global interactions, but larger computational cost.
    gno_weighting_function : Literal["half_cos", "bump", "quartic", "quadr", "octic"], optional
        Choice of weighting function to use in the output GNO for Mollified Graph Neural Operator-based models.
        See neuralop.layers.gno_weighting_functions for more details. Default: None
    gno_weight_function_scale : float, optional
        Factor by which to scale weights from GNO weighting function.
        If gno_weighting_function is None, this is not used. Default: 1
    in_gno_transform_type : str, optional
        Transform type parameter for input GNO. Default: "linear"
        See neuralop.layers.gno_block for more details.
        See neuralop.layers.gno_block for more details.
    out_gno_transform_type : str, optional
        Transform type parameter for output GNO.
        Options: "linear", "nonlinear", "nonlinear_kernelonly". Default: "linear"
        See neuralop.layers.gno_block for more details.
        Type of optional sinusoidal positional embedding to use in input GNOBlock. Default: "transformer"
        Type of optional sinusoidal positional embedding to use in input GNOBlock.
        Options: "transformer", "nerf". Default: "transformer"
        Type of optional sinusoidal positional embedding to use in output GNOBlock. Default: "transformer"
    fno_in_channels : int, optional
        Number of input channels for FNO. Default: 3
    fno_lifting_channel_ratio : int, optional
        Ratio of lifting channels to fno_hidden_channels.
        The number of lifting channels in the lifting block of the FNO is
        fno_lifting_channel_ratio * hidden_channels (i.e. default 128). Default: 2
    gno_embed_channels : int, optional
        Dimension of optional per-channel embedding to use in GNOBlock. Default: 32
    gno_embed_max_positions : int, optional
        Max positions of optional per-channel embedding to use in GNOBlock.
        If gno_pos_embed_type != 'transformer', this is not used. Default: 10000
    in_gno_channel_mlp_hidden_layers : list, optional
        Widths of hidden layers in input GNO. Default: [80, 80, 80]
    out_gno_channel_mlp_hidden_layers : list, optional
        Widths of hidden layers in output GNO. Default: [512, 256]
    gno_channel_mlp_non_linearity : nn.Module, optional
        Nonlinearity to use in GNO ChannelMLP. Default: F.gelu
    gno_use_open3d : bool, optional
        Whether to use Open3D neighbor search.
        If False, uses pure-PyTorch fallback neighbor search. Default: True
    gno_use_torch_scatter : bool, optional
        Whether to use torch-scatter to perform grouped reductions in the IntegralTransform.
        If False, uses native Python reduction in neuralop.layers.segment_csr. Default: True

        .. warning::
            torch-scatter is an optional dependency that conflicts with the newest versions of PyTorch,
            so you must handle the conflict explicitly in your environment. See :ref:`torch_scatter_dependency`
            for more information.
    out_gno_tanh : bool, optional
        Whether to use tanh to stabilize outputs of the output GNO. Default: False
    fno_resolution_scaling_factor : float, optional
        Factor by which to scale output of FNO. Default: None
    fno_block_precision : str, optional
        Data precision to compute within FNO block. Options: "full", "half", "mixed". Default: "full"
    fno_use_channel_mlp : bool, optional
        Whether to use a ChannelMLP layer after each FNO block. Default: True
    fno_channel_mlp_dropout : float, optional
        Dropout parameter of above ChannelMLP. Default: 0
    fno_channel_mlp_expansion : float, optional
        Expansion parameter of above ChannelMLP. Default: 0.5
    fno_non_linearity : nn.Module, optional
        Nonlinear activation function between each FNO layer. Default: F.gelu
    fno_stabilizer : nn.Module, optional
        By default None, otherwise tanh is used before FFT in the FNO block. Default: None
    fno_norm : str, optional
        Normalization layer to use in FNO. Options: "ada_in", "group_norm", "instance_norm", None. Default: None
    fno_norm_groups : int, optional
        Number of groups for GroupNorm in FNO, by default 1
    fno_ada_in_features : int, optional
        If an adaptive mesh is used, number of channels of its positional embedding.
        If None, adaptive mesh embedding is not used. Default: 4
    fno_ada_in_dim : int, optional
        Dimensions of above FNO adaptive mesh. Default: 1
    fno_preactivation : bool, optional
        Whether to use ResNet-style preactivation. Default: False
    fno_skip : str, optional
        Type of skip connection to use. Options: "linear", "identity", "soft-gating", None. Default: "linear"
    fno_channel_mlp_skip : str, optional
        Type of skip connection to use in the FNO.
        Options: "linear", "identity", "soft-gating", None. Default: "soft-gating"
    fno_separable : bool, optional
        Whether to use a separable spectral convolution. Default: False
    fno_factorization : str, optional
        Tensor factorization of the parameters weight to use.
        Options: "tucker", "tt", "cp", None. Default: None
    fno_rank : float, optional
        Rank of the tensor factorization of the Fourier weights. Default: 1.0
        Set to float <1.0 when using TFNO (i.e. when factorization is not None).
        A TFNO with rank 0.1 has roughly 10% of the parameters of a dense FNO.
    fno_fixed_rank_modes : bool, optional
        Whether to not factorize certain modes. Default: False
    fno_implementation : str, optional
        If factorization is not None, forward mode to use.
        Options: "reconstructed", "factorized". Default: "factorized"
    fno_decomposition_kwargs : dict, optional
        Additional parameters to pass to the tensor decomposition. Default: {}
    fno_domain_padding : float | None, defaults to None
            If not None, percentage of padding to use.
    fno_conv_module : nn.Module, optional
        Spectral convolution module to use. Default: SpectralConv
    fno_enforce_hermitian_symmetry : bool, optional
        Whether to enforce Hermitian symmetry conditions when performing inverse FFT
        for real-valued data in the FNO branch. Only used in :class:`SpectralConv`;
        ignored otherwise. When True, explicitly enforces that the 0th frequency and
        Nyquist frequency are real-valued before calling irfft. When False, relies on
        cuFFT's irfftn to handle symmetry automatically, which may fail on certain
        GPUs or input sizes, causing line artifacts. By default True.

    References
    ----------
    .. [1] : Li, Z., Kovachki, N., Choy, C., Li, B., Kossaifi, J., Otta, S.,
        Nabian, M., Stadler, M., Hundt, C., Azizzadenesheli, K., Anandkumar, A. (2023)
        Geometry-Informed Neural Operator for Large-Scale 3D PDEs. NeurIPS 2023,
        https://proceedings.neurips.cc/paper_files/paper/2023/hash/70518ea42831f02afc3a2828993935ad-Abstract-Conference.html
    .. [2] : Lin, R. et al. Placeholder reference for Mollified Graph Neural Operators.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        latent_feature_channels=None,
        projection_channel_ratio=4,
        gno_coord_dim=3,
        in_gno_radius=0.033,
        out_gno_radius=0.033,
        in_gno_transform_type="linear",
        out_gno_transform_type="linear",
        gno_weighting_function=None,
        gno_weight_function_scale=1,
        in_gno_pos_embed_type="transformer",
        out_gno_pos_embed_type="transformer",
        fno_in_channels=3,
        fno_n_modes=(16, 16, 16),
        fno_hidden_channels=64,
        fno_lifting_channel_ratio=2,
        fno_n_layers=4,
        # Other GNO Params
        gno_embed_channels=32,
        gno_embed_max_positions=10000,
        in_gno_channel_mlp_hidden_layers=[80, 80, 80],
        out_gno_channel_mlp_hidden_layers=[512, 256],
        gno_channel_mlp_non_linearity=F.gelu,
        gno_use_open3d=True,
        gno_use_torch_scatter=True,
        out_gno_tanh=None,
        # Other FNO Params
        fno_resolution_scaling_factor=None,
        fno_block_precision="full",
        fno_use_channel_mlp=True,
        fno_channel_mlp_dropout=0,
        fno_channel_mlp_expansion=0.5,
        fno_non_linearity=F.gelu,
        fno_stabilizer=None,
        fno_norm=None,
        fno_norm_groups=1,
        fno_ada_in_features=4,
        fno_ada_in_dim=1,
        fno_preactivation=False,
        fno_skip="linear",
        fno_channel_mlp_skip="soft-gating",
        fno_separable=False,
        fno_factorization=None,
        fno_rank=1.0,
        fno_fixed_rank_modes=False,
        fno_implementation="factorized",
        fno_decomposition_kwargs=dict(),
        fno_domain_padding=None,
        fno_conv_module=SpectralConv,
        fno_enforce_hermitian_symmetry=True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.latent_feature_channels = latent_feature_channels
        self.gno_coord_dim = gno_coord_dim
        self.fno_hidden_channels = fno_hidden_channels

        self.lifting_channels = fno_lifting_channel_ratio * fno_hidden_channels

        # If the input GNO performs a nonlinear kernel, the GNO's output
        # features must be the same dimension as its input.
        # otherwise the kernel's MLP will perform a lifting operation to
        # lift the inputs to ``fno_in_channels`` channels
        if in_gno_transform_type in ["nonlinear", "nonlinear_kernelonly"]:
            in_gno_out_channels = self.in_channels
        else:
            in_gno_out_channels = fno_in_channels

        # The actual input channels to the FNO are computed here.
        self.fno_in_channels = in_gno_out_channels

        if latent_feature_channels is not None:
            self.fno_in_channels += latent_feature_channels

        if self.gno_coord_dim != 3 and gno_use_open3d:
            warnings.warn(
                f'GNO expects {self.gno_coord_dim}-d data but Open3d expects 3-d data',
                UserWarning,
                stacklevel=2,
            )
            gno_use_open3d = False

        self.in_coord_dim = len(fno_n_modes)
        self.gno_out_coord_dim = len(fno_n_modes) # gno output and fno will use same dimensions
        if self.in_coord_dim != self.gno_coord_dim:
            warnings.warn(
                f'FNO expects {self.in_coord_dim}-d data while input GNO expects {self.gno_coord_dim}-d data',
                UserWarning,
                stacklevel=2,
            )

        self.in_coord_dim_forward_order = list(range(self.in_coord_dim))

        # tensor indices starting at 2 to permute everything after channel and batch dims
        self.in_coord_dim_reverse_order = [j + 2 for j in self.in_coord_dim_forward_order]

        self.fno_norm = fno_norm
        if self.fno_norm == "ada_in":
            if fno_ada_in_features is not None and out_gno_pos_embed_type is not None:
                self.adain_pos_embed = SinusoidalEmbedding(
                    in_channels=fno_ada_in_dim,
                    num_frequencies=fno_ada_in_features,
                    max_positions=10000,
                    embedding_type=out_gno_pos_embed_type,
                )
                self.ada_in_dim = self.adain_pos_embed.out_channels
            else:
                self.ada_in_dim = fno_ada_in_dim
                self.adain_pos_embed = None
        else:
            self.adain_pos_embed = None
            self.ada_in_dim = None

        self.in_gno_radius = in_gno_radius
        self.out_gno_radius = out_gno_radius

        self.out_gno_tanh = out_gno_tanh

        ### input GNO
        # input to the first GNO ChannelMLP: `x` pos encoding,
        # `y` (integrand) pos encoding, potentially `f_y`

        self.gno_in = GNOBlock(
            in_channels=in_channels,
            out_channels=in_gno_out_channels,
            coord_dim=self.gno_coord_dim,
            pos_embedding_type=in_gno_pos_embed_type,
            pos_embedding_channels=gno_embed_channels,
            pos_embedding_max_positions=gno_embed_max_positions,
            radius=in_gno_radius,
            reduction="mean",
            weighting_fn=None,
            channel_mlp_layers=in_gno_channel_mlp_hidden_layers,
            channel_mlp_non_linearity=gno_channel_mlp_non_linearity,
            transform_type=in_gno_transform_type,
            use_torch_scatter_reduce=gno_use_torch_scatter,
            use_open3d_neighbor_search=gno_use_open3d,
        )

        ### Lifting layer before FNOBlocks
        self.lifting = ChannelMLP(
            in_channels=self.fno_in_channels,
            hidden_channels=self.lifting_channels,
            out_channels=fno_hidden_channels,
            n_layers=2,
        )  # CHANGED RECENTLY FOR THIS PAPER

        # Domain padding
        if fno_domain_padding is not None and (
            (isinstance(fno_domain_padding, list) and sum(fno_domain_padding) > 0)
            or (isinstance(fno_domain_padding, (float, int)) and fno_domain_padding > 0)
        ):
            self.domain_padding = DomainPadding(
                domain_padding=fno_domain_padding
            )
        else:
            self.domain_padding = None

        ### FNOBlocks in latent space
        # input: `in_p` intermediate embeddings,
        # possibly concatenated feature channels `latent_features`
        self.fno_blocks = FNOBlocks(
            n_modes=fno_n_modes,
            in_channels=fno_hidden_channels,
            out_channels=fno_hidden_channels,
            n_layers=fno_n_layers,
            resolution_scaling_factor=fno_resolution_scaling_factor,
            fno_block_precision=fno_block_precision,
            use_channel_mlp=fno_use_channel_mlp,
            channel_mlp_expansion=fno_channel_mlp_expansion,
            channel_mlp_dropout=fno_channel_mlp_dropout,
            non_linearity=fno_non_linearity,
            stabilizer=fno_stabilizer,
            norm=fno_norm,
            norm_groups=fno_norm_groups,
            ada_in_features=self.ada_in_dim,
            preactivation=fno_preactivation,
            fno_skip=fno_skip,
            channel_mlp_skip=fno_channel_mlp_skip,
            separable=fno_separable,
            factorization=fno_factorization,
            rank=fno_rank,
            fixed_rank_modes=fno_fixed_rank_modes,
            implementation=fno_implementation,
            decomposition_kwargs=fno_decomposition_kwargs,
            conv_module=fno_conv_module,
            enforce_hermitian_symmetry=fno_enforce_hermitian_symmetry,
        )

        ### output GNO
        if gno_weighting_function is not None:  # sq radius**2?
            weight_fn = dispatch_weighting_fn(
                gno_weighting_function,
                sq_radius=out_gno_radius**2,
                scale=gno_weight_function_scale,
            )
        else:
            weight_fn = None
        self.gno_out = GNOBlock(
            in_channels=fno_hidden_channels,  # number of channels in f_y
            out_channels=fno_hidden_channels,
            coord_dim=self.gno_coord_dim,
            radius=self.out_gno_radius,
            reduction="sum",
            weighting_fn=weight_fn,
            pos_embedding_type=out_gno_pos_embed_type,
            pos_embedding_channels=gno_embed_channels,
            pos_embedding_max_positions=gno_embed_max_positions,
            channel_mlp_layers=out_gno_channel_mlp_hidden_layers,
            channel_mlp_non_linearity=gno_channel_mlp_non_linearity,
            transform_type=out_gno_transform_type,
            use_torch_scatter_reduce=gno_use_torch_scatter,
            use_open3d_neighbor_search=gno_use_open3d,
        )

        projection_channels = projection_channel_ratio * fno_hidden_channels
        self.projection = ChannelMLP(
            in_channels=fno_hidden_channels,
            out_channels=self.out_channels,
            hidden_channels=projection_channels,
            n_layers=2,
            n_dim=1,
            non_linearity=fno_non_linearity,
        )

    # returns: (fno_hidden_channels, n_1, n_2, ...)
    def latent_embedding(self, in_p, ada_in=None):
        # in_p : (batch, n_1 , ... , n_k, in_channels + k)
        # ada_in : (fno_ada_in_dim, )

        # permute (b, n_1, ..., n_k, c) -> (b,c, n_1,...n_k)
        in_p = in_p.permute(0, len(in_p.shape) - 1, *list(range(1,len(in_p.shape)-1)))
        #Update Ada IN embedding    
        if ada_in is not None:
            if ada_in.ndim == 2:
                ada_in = ada_in.squeeze(0)
            if self.adain_pos_embed is not None:
                ada_in_embed = self.adain_pos_embed(ada_in.unsqueeze(0)).squeeze(0)
            else:
                ada_in_embed = ada_in
            if self.fno_norm == "ada_in":
                self.fno_blocks.set_ada_in_embeddings(ada_in_embed)

        # Apply FNO blocks
        in_p = self.lifting(in_p)
        if self.domain_padding is not None:
            in_p = self.domain_padding.pad(in_p)

        for idx in range(self.fno_blocks.n_layers):
            in_p = self.fno_blocks(in_p, idx)

        if self.domain_padding is not None:
            in_p = self.domain_padding.unpad(in_p)

        return in_p

    def forward(
        self,
        input_geom,
        latent_queries,
        output_queries,
        x=None,
        latent_features=None,
        ada_in=None,
        **kwargs,
    ):
        if kwargs:
            warnings.warn(
                f"GINO.forward() received unexpected keyword arguments: {list(kwargs.keys())}. "
                "These arguments will be ignored.",
                UserWarning,
                stacklevel=2,
            )

        """The GINO's forward call:
        Input GNO --> FNOBlocks --> output GNO + projection to output queries.

        .. note ::
            GINO currently supports batching **only in cases where the geometry of
            inputs and outputs is shared across the entire batch**. Inputs can have a batch dim
            in ``x`` and ``latent_features``, but it must be shared for both. 

        Parameters
        ----------
        input_geom : torch.Tensor
            input domain coordinate mesh
            shape (1, n_in, gno_coord_dim)
        latent_queries : torch.Tensor
            latent geometry on which to compute FNO latent embeddings
            a grid on [0,1] x [0,1] x ....
            shape (1, n_gridpts_1, .... n_gridpts_n, gno_coord_dim)
        output_queries : torch.Tensor | dict[torch.Tensor]
            points at which to query the final GNO layer to get output.

            shape (1, n_out, gno_coord_dim) per tensor.

            * if a tensor, the model will output a tensor. 

            * if a dict of tensors, the model will return a dict of outputs, so
            that ``output[key]`` corresponds to the model queried at 
            ``output_queries[key]``. 
        x : torch.Tensor, optional
            input function a defined on the input domain `input_geom`
            shape (batch, n_in, in_channels). Default None
        latent_features : torch.Tensor, optional
            optional feature map to concatenate onto latent embedding
            before being passed into the latent FNO, default None
            if `latent_feature_channels` is set, must be passed
        ada_in : torch.Tensor, optional
            adaptive scalar instance parameter, defaults to None

        Returns
        -------
        out : torch.Tensor | dict[torch.Tensor]
            Function over the output query coordinates
            * tensor if if ``output_queries`` is a tensor
            * dict if if ``output_queries`` is a dict
        """

        # Ensure input functions on the input geom and latent geom
        # have compatible batch sizes
        if x is None:
            batch_size = 1
        else:
            batch_size = x.shape[0]

        if latent_features is not None:
            assert (
                self.latent_feature_channels is not None
            ), "if passing latent features, latent_feature_channels must be set."
            assert latent_features.shape[-1] == self.latent_feature_channels

            # batch, n_gridpts_1, .... n_gridpts_n, gno_coord_dim
            assert (
                latent_features.ndim == self.gno_coord_dim + 2
            ), f"Latent features must be of shape (batch, n_gridpts_1, ...n_gridpts_n, gno_coord_dim), got {latent_features.shape}"
            # latent features must have the same shape (except channels) as latent_queries
            if latent_features.shape[0] != batch_size:
                if latent_features.shape[0] == 1:
                    latent_features = latent_features.repeat(batch_size, *[1]*(latent_features.ndim-1))

        input_geom = input_geom.squeeze(0)
        latent_queries = latent_queries.squeeze(0)

        # Pass through input GNOBlock
        in_p = self.gno_in(
            y=input_geom, x=latent_queries.view((-1, latent_queries.shape[-1])), f_y=x
        )

        grid_shape = latent_queries.shape[:-1]  # disregard positional encoding dim

        # shape (batch_size, grid1, ...gridn, -1)
        in_p = in_p.view((batch_size, *grid_shape, -1))

        if latent_features is not None:
            in_p = torch.cat((in_p, latent_features), dim=-1)
        # take apply fno in latent space
        latent_embed = self.latent_embedding(in_p=in_p, ada_in=ada_in)

        # Integrate latent space to output queries
        # latent_embed shape (b, c, n_1, n_2, ..., n_k)
        batch_size = latent_embed.shape[0]
        # permute to (b, n_1, n_2, ...n_k, c)
        # then reshape to (b, n_1 * n_2 * ...n_k, out_channels)
        latent_embed = latent_embed.permute(0, *self.in_coord_dim_reverse_order, 1).reshape(batch_size, -1, self.fno_hidden_channels)

        if self.out_gno_tanh in ["latent_embed", "both"]:
            latent_embed = torch.tanh(latent_embed)

        # integrate over the latent space
        # if output queries is a dict, query the output gno separately
        # with each tensor of query points
        if isinstance(output_queries, dict):
            out = {}
            for key, out_p in output_queries.items():
                out_p = out_p.squeeze(0)

                sub_output = self.gno_out(
                    y=latent_queries.reshape((-1, latent_queries.shape[-1])),
                    x=out_p,
                    f_y=latent_embed,
                )
                sub_output = sub_output.permute(0, 2, 1)

                # Project pointwise to out channels
                # (b, n_in, out_channels)
                sub_output = self.projection(sub_output).permute(0, 2, 1)

                out[key] = sub_output
        else:
            output_queries = output_queries.squeeze(0)

            # latent queries is of shape (d_1 x d_2 x... d_n x n), reshape to n_out x n
            out = self.gno_out(
                y=latent_queries.reshape((-1, latent_queries.shape[-1])),
                x=output_queries,
                f_y=latent_embed,
            )
            out = out.permute(0, 2, 1)

            # Project pointwise to out channels
            # (b, n_in, out_channels)
            out = self.projection(out).permute(0, 2, 1)

        return out
