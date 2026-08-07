import torch
import torch.nn.functional as F

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


class FNOGNO(BaseModel, name="FNOGNO"):
    """FNOGNO: Fourier/Geometry Neural Operator - maps from a regular N-d grid to an arbitrary query point cloud.

    Parameters
    ----------
    in_channels : int
        Number of input channels. Determined by the problem.
    out_channels : int
        Number of output channels. Determined by the problem.
    fno_n_modes : tuple, optional
        Number of modes to keep along each spectral dimension of FNO block.
        Must be larger enough but smaller than max_resolution//2 (Nyquist frequency). Default: (16, 16, 16)
    fno_hidden_channels : int, optional
        Number of hidden channels of FNO block. Default: 64
    fno_n_layers : int, optional
        Number of FNO layers in the block. Default: 4

    projection_channel_ratio : int, optional
        Ratio of pointwise projection channels in the final ChannelMLP to fno_hidden_channels.
        The number of projection channels in the final ChannelMLP is computed by
        projection_channel_ratio * fno_hidden_channels (i.e. default 256). Default: 4
    gno_coord_dim : int, optional
        Dimension of coordinate space where GNO is computed. Determined by the problem. Default: 3
    gno_pos_embed_type : Literal["transformer", "nerf"], optional
        Type of optional sinusoidal positional embedding to use in GNOBlock. Default: "transformer"
    gno_radius : float, optional
        Radius parameter to construct graph. Default: 0.033
        Larger radius means more neighboors so more global interactions, but larger computational cost.
    gno_transform_type : str, optional
        Type of kernel integral transform to apply in GNO.
        Kernel k(x,y): parameterized as ChannelMLP MLP integrated over a neighborhood of x.

        Options:
        - "linear_kernelonly": Integrand is k(x, y)
        - "linear": Integrand is k(x, y) * f(y)
        - "nonlinear_kernelonly": Integrand is k(x, y, f(y))
        - "nonlinear": Integrand is k(x, y, f(y)) * f(y)
        Default: "linear"
    gno_weighting_function : Literal["half_cos", "bump", "quartic", "quadr", "octic"], optional
        Choice of weighting function to use in the output GNO for Mollified Graph Neural Operator-based models.
        See neuralop.layers.gno_weighting_functions for more details. Default: None
    gno_weight_function_scale : float, optional
        Factor by which to scale weights from GNO weighting function. If gno_weighting_function is None, this is not used. Default: 1
    gno_embed_channels : int, optional
        Dimension of optional per-channel embedding to use in GNOBlock. Default: 32
    gno_embed_max_positions : int, optional
        Max positions of optional per-channel embedding to use in GNOBlock. If gno_pos_embed_type != 'transformer', value is unused. Default: 10000
    gno_channel_mlp_hidden_layers : list, optional
        Dimension of hidden ChannelMLP layers of GNO. Default: [512, 256]
    gno_channel_mlp_non_linearity : nn.Module, optional
        Nonlinear activation function between layers. Default: F.gelu
    gno_use_open3d : bool, optional
        Whether to use Open3D functionality. If False, uses simple fallback neighbor search. Default: True
    gno_use_torch_scatter : bool, optional
        Whether to use torch-scatter to perform grouped reductions in the IntegralTransform.
        If False, uses native Python reduction in neuralop.layers.segment_csr.

        .. warning::
            torch-scatter is an optional dependency that conflicts with the newest versions of PyTorch,
            so you must handle the conflict explicitly in your environment. See :ref:`torch_scatter_dependency`
            for more information.

        Default: True
    gno_batched : bool, optional
        Whether to use IntegralTransform/GNO layer in "batched" mode. If False, sets batched=False. Default: False
    fno_lifting_channel_ratio : int, optional
        Ratio of lifting channels to FNO hidden channels. Default: 4
    fno_resolution_scaling_factor : float, optional
        Factor by which to rescale output predictions in the original domain. Default: None
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
        If an adaptive mesh is used, number of channels of its positional embedding. Default: None
    fno_ada_in_dim : int, optional
        Dimensions of above FNO adaptive mesh. Default: 1
    fno_preactivation : bool, optional
        Whether to use ResNet-style preactivation. Default: False
    fno_skip : str, optional
        Type of skip connection to use. Options: "linear", "identity", "soft-gating", None. Default: "linear"
    fno_channel_mlp_skip : str, optional
        Type of skip connection to use in the FNO.

        Options:
        - "linear": Conv layer
        - "soft-gating": Weights the channels of the input
        - "identity": nn.Identity
        - None: No skip connection

        Default: "soft-gating"
    fno_separable : bool, optional
        Whether to use a depthwise separable spectral convolution. Default: False
    fno_factorization : str, optional
        Tensor factorization of the parameters weight to use. Options: "tucker", "tt", "cp", None. Default: None
    fno_rank : float, optional
        Rank of the tensor factorization of the Fourier weights. Default: 1.0
    fno_fixed_rank_modes : bool, optional
        Whether to not factorize certain modes. Default: False
    fno_implementation : str, optional
        If factorization is not None, forward mode to use.

        Options:
        - "reconstructed": The full weight tensor is reconstructed from the factorization and used for the forward pass
        - "factorized": The input is directly contracted with the factors of the decomposition

        Default: "factorized"
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
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        projection_channel_ratio=4,
        gno_coord_dim=3,
        gno_pos_embed_type="transformer",
        gno_transform_type="linear",
        fno_n_modes=(16, 16, 16),
        fno_hidden_channels=64,
        fno_lifting_channel_ratio=4,
        fno_n_layers=4,
        # Other GNO params
        gno_embed_channels=32,
        gno_embed_max_positions=10000,
        gno_radius=0.033,
        gno_weighting_function=None,
        gno_weight_function_scale=1.0,
        gno_channel_mlp_hidden_layers=[512, 256],
        gno_channel_mlp_non_linearity=F.gelu,
        gno_use_open3d=True,
        gno_use_torch_scatter=True,
        gno_batched=False,
        # Other FNO params
        fno_resolution_scaling_factor=None,
        fno_block_precision="full",
        fno_use_channel_mlp=True,
        fno_channel_mlp_dropout=0,
        fno_channel_mlp_expansion=0.5,
        fno_non_linearity=F.gelu,
        fno_stabilizer=None,
        fno_norm=None,
        fno_norm_groups=1,
        fno_ada_in_features=None,
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

        self.gno_coord_dim = gno_coord_dim
        if self.gno_coord_dim != 3 and gno_use_open3d:
            warnings.warn(
                f"GNO expects {self.gno_coord_dim}-d data but Open3d expects 3-d data",
                UserWarning,
                stacklevel=2,
            )

        self.in_coord_dim = len(fno_n_modes)
        if self.in_coord_dim != self.gno_coord_dim:
            warnings.warn(
                f"FNO expects {self.in_coord_dim}-d data while GNO expects {self.gno_coord_dim}-d data",
                UserWarning,
                stacklevel=2,
            )

        # these lists contain the interior dimensions of the input
        # in order to reshape without explicitly providing dims
        self.in_coord_dim_forward_order = list(range(self.in_coord_dim))
        self.in_coord_dim_reverse_order = [
            j + 1 for j in self.in_coord_dim_forward_order
        ]

        self.gno_batched = gno_batched  # used in forward call to GNO

        # if batched, we must account for the extra batch dim
        # which causes previous dims to be incremented by 1
        if self.gno_batched:
            self.in_coord_dim_forward_order = [
                j + 1 for j in self.in_coord_dim_forward_order
            ]
            self.in_coord_dim_reverse_order = [
                j + 1 for j in self.in_coord_dim_reverse_order
            ]

        if fno_norm == "ada_in":
            if fno_ada_in_features is not None:
                self.adain_pos_embed = SinusoidalEmbedding(
                    in_channels=fno_ada_in_dim,
                    num_frequencies=fno_ada_in_features,
                    embedding_type="transformer",
                )
                # if ada_in positional embedding is provided, set the input dimension
                # of the ada_in norm to the output channels of positional embedding
                self.ada_in_dim = self.adain_pos_embed.out_channels
            else:
                self.ada_in_dim = fno_ada_in_dim
        else:
            self.adain_pos_embed = None
            self.ada_in_dim = None

        # Create lifting for FNOBlock separately
        fno_lifting_channels = fno_lifting_channel_ratio * fno_hidden_channels
        self.lifting = ChannelMLP(
            in_channels=in_channels + self.in_coord_dim,
            hidden_channels=fno_lifting_channels,
            out_channels=fno_hidden_channels,
            n_layers=3,
        )

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

        self.fno_hidden_channels = fno_hidden_channels
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

        self.gno_radius = gno_radius

        if gno_weighting_function is not None:
            weight_fn = dispatch_weighting_fn(
                gno_weighting_function,
                sq_radius=gno_radius**2,
                scale=gno_weight_function_scale,
            )
        else:
            weight_fn = None

        self.gno = GNOBlock(
            in_channels=fno_hidden_channels,
            out_channels=fno_hidden_channels,
            radius=gno_radius,
            weighting_fn=weight_fn,
            coord_dim=self.gno_coord_dim,
            pos_embedding_type=gno_pos_embed_type,
            pos_embedding_channels=gno_embed_channels,
            pos_embedding_max_positions=gno_embed_max_positions,
            channel_mlp_layers=gno_channel_mlp_hidden_layers,
            channel_mlp_non_linearity=gno_channel_mlp_non_linearity,
            transform_type=gno_transform_type,
            use_open3d_neighbor_search=gno_use_open3d,
            use_torch_scatter_reduce=gno_use_torch_scatter,
        )

        projection_channels = projection_channel_ratio * fno_hidden_channels
        self.projection = ChannelMLP(
            in_channels=fno_hidden_channels,
            out_channels=out_channels,
            hidden_channels=projection_channels,
            n_layers=2,
            n_dim=1,
            non_linearity=fno_non_linearity,
        )

    # out_p : (n_out, gno_coord_dim)
    # in_p : (n_1, n_2, ..., n_k, k)
    # if batched shape is the same because this is just geometry
    # that remains constant across the entire batch
    # f : (n_1, n_2, ..., n_k,  in_channels)
    # if batched, (b, n_1, n_2, ..., n_k,  in_channels)
    # ada_in : (fno_ada_in_dim, )

    # returns: (fno_hidden_channels, n_1, n_2, ...)
    def latent_embedding(self, in_p, f, ada_in=None):
        if self.gno_batched:
            batch_size = f.shape[0]
            # repeat in_p along the batch dimension for latent embedding
            in_p = in_p.repeat([batch_size] + [1] * (in_p.ndim))
        in_p = torch.cat((f, in_p), dim=-1)

        if self.gno_batched:
            # shape: (b, k, n_1, n_2, ... n_k)
            in_p = in_p.permute(0, -1, *self.in_coord_dim_forward_order)
        else:
            in_p = in_p.permute(-1, *self.in_coord_dim_forward_order).unsqueeze(0)

        # Update Ada IN embedding
        if ada_in is not None:
            if self.adain_pos_embed is not None:
                ada_in_embed = self.adain_pos_embed(ada_in.unsqueeze(0)).squeeze(0)
            else:
                ada_in_embed = ada_in

            self.fno_blocks.set_ada_in_embeddings(ada_in_embed)

        # Apply FNO blocks

        in_p = self.lifting(in_p)
        if self.domain_padding is not None:
            in_p = self.domain_padding.pad(in_p)

        for layer_idx in range(self.fno_blocks.n_layers):
            in_p = self.fno_blocks(in_p, layer_idx)

        if self.domain_padding is not None:
            in_p = self.domain_padding.unpad(in_p)

        if self.gno_batched:
            return in_p
        else:
            return in_p.squeeze(0)

    def integrate_latent(self, in_p, out_p, latent_embed):
        """
        Compute integration region for each output point
        """

        # (n_1*n_2*..., fno_hidden_channels)
        # if batched, (b, n1*n2*..., fno_hidden_channels)

        if self.gno_batched:
            batch_size = latent_embed.shape[0]
            latent_embed = latent_embed.permute(
                0, *self.in_coord_dim_reverse_order, 1
            ).reshape((batch_size, -1, self.fno_hidden_channels))
        else:
            latent_embed = latent_embed.permute(
                *self.in_coord_dim_reverse_order, 0
            ).reshape((-1, self.fno_hidden_channels))

        # (n_out, fno_hidden_channels)

        out = self.gno(
            y=in_p.reshape(-1, in_p.shape[-1]),
            x=out_p,
            f_y=latent_embed,
        )

        # if self.gno is variable and not batched
        if out.ndim == 2:
            out = out.unsqueeze(0)
        out = out.permute(0, 2, 1)  # b, c, n_out

        # Project pointwise to out channels
        out = self.projection(out)

        if self.gno_batched:
            out = out.permute(0, 2, 1)
        else:
            out = out.squeeze(0).permute(1, 0)

        return out

    def forward(self, in_p, out_p, f, ada_in=None, **kwargs):
        if kwargs:
            warnings.warn(
                f"FNOGNO.forward() received unexpected keyword arguments: {list(kwargs.keys())}. "
                "These arguments will be ignored.",
                UserWarning,
                stacklevel=2,
            )

        # Compute latent space embedding
        latent_embed = self.latent_embedding(in_p=in_p, f=f, ada_in=ada_in)
        # Integrate latent space
        out = self.integrate_latent(in_p=in_p, out_p=out_p, latent_embed=latent_embed)

        return out
