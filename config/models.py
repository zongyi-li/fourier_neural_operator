from typing import List, Literal, Optional, Any
from zencfg import ConfigBase


class ModelConfig(ConfigBase):
    arch: str
    data_channels: int
    out_channels: int


class FNOConfig(ModelConfig):
    model_arch: str = "fno"
    data_channels: int
    out_channels: int
    n_modes: List[int]
    hidden_channels: int
    lifting_channel_ratio: int = 2
    projection_channel_ratio: int = 4
    n_layers: int = 4
    domain_padding: float = 0.0
    norm: str = "None"
    fno_skip: str = "linear"
    implementation: str = "factorized"
    use_channel_mlp: bool = True
    channel_mlp_expansion: float = 0.5
    channel_mlp_dropout: float = 0
    separable: bool = False
    factorization: str = "None"
    rank: float = 1.0
    fixed_rank_modes: bool = False
    stabilizer: str = "None"


class SimpleFNOConfig(FNOConfig):
    """
    SimpleFNOConfig: a basic FNO config that provides access to only
    the most important FNO parameters.
    """

    data_channels: int
    out_channels: int
    n_modes: List[int]
    hidden_channels: int
    projection_channel_ratio: int


class FNO_Small2d(SimpleFNOConfig):
    """
    FNO_Small2d: a basic, small FNO for 2d problems.
    """

    data_channels: int = 1
    out_channels: int = 1
    n_modes: List[int] = [16, 16]
    hidden_channels: int = 24
    projection_channel_ratio: int = 2


class FNO_Medium2d(SimpleFNOConfig):
    """
    FNO_Medium2d: a basic, medium-sized FNO for 2d problems.
    """

    data_channels: int = 1
    out_channels: int = 1
    n_modes: List[int] = [64, 64]
    hidden_channels: int = 64
    projection_channel_ratio: int = 4


class FNO_Large2d(SimpleFNOConfig):
    """
    FNO_Large2d: a large FNO for 2d problems.
    """

    data_channels: int = 1
    out_channels: int = 1
    n_modes: List[int] = [64, 64]
    hidden_channels: int = 128
    projection_channel_ratio: int = 2


class FNO_Huge2d(SimpleFNOConfig):
    """
    FNO_Huge2d: a giant FNO for 2d problems.

    Note that this will likely be too large for most single-GPU local settings,
    and should be paired with tensor factorization, low-rank/sparse optimization
    via TensorGRaD, or some other means of reducing memory overhead.
    """

    data_channels: int = 1
    out_channels: int = 1
    n_modes: List[int] = [128, 128]
    hidden_channels: int = 128
    projection_channel_ratio: int = 2


class FNO_Medium3d(SimpleFNOConfig):
    """
    FNO_Medium3d: a medium FNO for 3d problems.
    """

    data_channels: int = 1
    out_channels: int = 1
    n_modes: List[int] = [32, 32, 32]
    hidden_channels: int = 64
    projection_channel_ratio: int = 4


class FNOGNOConfig(ModelConfig):
    model_arch: str = "fnogno"
    data_channels: int
    out_channels: int
    gno_coord_dim: int
    gno_embed_channels: int
    gno_radius: float
    gno_transform_type: str
    fno_n_modes: List[int]
    fno_hidden_channels: int
    fno_use_channel_mlp: bool = True
    fno_norm: str = "instance_norm"
    fno_ada_in_features: int = 32
    fno_factorization: str = "tucker"
    fno_rank: float = 1.0
    fno_domain_padding: float = 0.125
    fno_use_channel_mlp: bool = True
    fno_channel_mlp_expansion: float = 1.0
    fno_resolution_scaling_factor: int = 1


class FNOGNO_Small3d(FNOGNOConfig):
    data_channels: int = 1
    out_channels: int = 1
    gno_coord_dim: int = 3
    gno_embed_channels: int = 16
    gno_radius: float = 0.033
    gno_transform_type: str = "linear"
    fno_n_modes: List[int] = [16, 16, 16]
    fno_hidden_channels: int = 64
    fno_use_channel_mlp: bool = True
    fno_rank: float = 0.4
    fno_domain_padding: float = 0.125


class GINOConfig(ModelConfig):
    model_arch: str = "gino"  # all GINO configs must use this
    data_channels: int
    out_channels: int
    latent_feature_channels: Optional[int] = None
    gno_coord_dim: int = 3
    gno_embed_channels: int = 16
    gno_radius: float = 0.033
    in_gno_transform_type: str = "linear"
    out_gno_transform_type: str = "linear"
    gno_pos_embed_type: str = "nerf"
    gno_weighting_function: Optional[str] = None
    gno_weight_function_scale: Optional[float] = None
    fno_n_modes: List[int] = [16, 16, 16]
    fno_hidden_channels: int = 64
    fno_use_channel_mlp: bool = True
    fno_norm: str = "instance_norm"
    fno_ada_in_features: int = 32
    fno_factorization: str = "tucker"
    fno_rank: float = 0.4
    fno_domain_padding: float = 0.125
    fno_channel_mlp_expansion: float = 1.0
    fno_resolution_scaling_factor: int = 1


class GINO_Small3d(GINOConfig):
    data_channels: int = 0
    out_channels: int = 1
    latent_feature_channels: Optional[int] = 1
    gno_coord_dim: int = 3
    gno_embed_channels: int = 16
    gno_radius: float = 0.033


class GINO_Poisson2d(GINOConfig):
    data_channels: int = 3
    out_channels: int = 1
    projection_channel_ratio: int = 4
    gno_coord_dim: int = 2
    in_gno_pos_embed_type: Optional[Any] = None
    out_gno_pos_embed_type: str = "transformer"
    gno_embed_channels: int = 16
    gno_embed_max_positions: int = 600
    gno_use_torch_scatter: bool = True
    in_gno_radius: float = 0.16
    out_gno_radius: float = 0.175
    in_gno_transform_type: str = "linear"
    out_gno_transform_type: str = "linear"
    gno_reduction: str = "mean"
    gno_weighting_function: str = "half_cos"
    gno_weight_function_scale: float = 0.030625
    gno_use_open3d: bool = False
    in_gno_channel_mlp_hidden_layers: List[int] = [256, 512, 256]
    out_gno_channel_mlp_hidden_layers: List[int] = [512, 1024, 512]
    in_gno_tanh: Optional[Any] = None
    out_gno_tanh: Optional[Any] = None
    fno_n_modes: List[int] = [20, 20]
    fno_hidden_channels: int = 64
    fno_lifting_channel_ratio: int = 4
    fno_n_layers: int = 4
    fno_use_channel_mlp: bool = True
    fno_channel_mlp_expansion: float = 0.5
    fno_norm: str = "group_norm"
    fno_ada_in_features: int = 8
    fno_factorization: Optional[Any] = None
    fno_rank: float = 0.8
    fno_domain_padding: float = 0.0


class OTNOConfig(ModelConfig):
    model_arch: str = "otno"
    n_modes: List[int]
    hidden_channels: int
    data_channels: int = 9
    out_channels: int = 1
    lifting_channel_ratio: int = 3
    projection_channel_ratio: int = 3
    norm: str = "group_norm"
    use_channel_mlp: bool = True
    channel_mlp_expansion: float = 1.0
    domain_padding: float = 0.125
    factorization: str = "tucker"
    rank: float = 0.4


class OTNO_Small2d(OTNOConfig):
    n_modes: List[int] = [16, 16]
    hidden_channels: int = 64
    data_channels: int = 9
    out_channels: int = 1
