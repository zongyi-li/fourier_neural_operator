import torch
import torch.nn as nn

from neuralop.models import FNO


class OTNO(FNO):
    """
    Optimal Transport Neural Operator

    The architecture is described in [1]_.

    OTNO integrates optimal transport (OT) into operator learning for partial
    differential equations (PDEs) on complex geometries.

    Parameters
    ----------
    All arguments are the same as :class:`neuralop.models.FNO`.

    See the FNO documentation for detailed descriptions.

    References
    ----------
    .. [1] Li, X., Li, Z., Kovachki, N., & Anandkumar, A. "Geometric Operator
        Learning with Optimal Transport" (2025). arXiv preprint arXiv:2507.20065.
        https://arxiv.org/pdf/2507.20065
    """

    def __init__(
        self,
        n_modes,
        hidden_channels,
        in_channels=4,
        out_channels=1,
        lifting_channel_ratio=2,
        projection_channel_ratio=2,
        n_layers=4,
        positional_embedding=None,
        use_channel_mlp=False,
        channel_mlp_expansion=None,
        channel_mlp_dropout=0,
        norm="group_norm",
        norm_groups=1,
        factorization=None,
        rank=1,
        domain_padding=None,
        **kwargs,
    ):
        if kwargs.get("readout") is not None:
            raise ValueError(
                "readout is not supported by OTNO. OTNO.forward indexes into a "
                "target mesh and returns output of shape (out_channels, n_target), "
                "which is incompatible with ResolutionInvariantReadout "
                "and similar heads that expect (B, C, *spatial) input."
            )
        super().__init__(
            n_modes=n_modes,
            hidden_channels=hidden_channels,
            in_channels=in_channels,
            out_channels=out_channels,
            lifting_channel_ratio=lifting_channel_ratio,
            projection_channel_ratio=projection_channel_ratio,
            n_layers=n_layers,
            positional_embedding=positional_embedding,
            use_channel_mlp=use_channel_mlp,
            channel_mlp_expansion=channel_mlp_expansion,
            channel_mlp_dropout=channel_mlp_dropout,
            norm=norm,
            norm_groups=norm_groups,
            factorization=factorization,
            rank=rank,
            domain_padding=domain_padding,
            **kwargs,
        )

    @property
    def device(self):
        """Returns the device that the model is on."""
        return self.device_indicator_param.device

    # x: (1, in_channels, n_s_sqrt, n_s_sqrt)
    #    Transport features containing concatenated data:
    #    - Source mesh coordinates X_s
    #    - Transported mesh coordinates T(X_s)
    #    - Additional features (e.g., normals)
    #    where n_s_sqrt * n_s_sqrt = n_s (total number of source vertices)
    #
    # ind_dec: (n_t,)
    #    Index decoder for mapping from latent grid to target mesh vertices
    #    where n_t is the number of target vertices
    def forward(self, x, ind_dec, **kwargs):
        """OTNO's forward pass"""
        # Append spatial positional embedding if set (mirrors FNO.forward)
        if self.positional_embedding is not None:
            x = self.positional_embedding(x)

        x = self.lifting(x)  # (1, hidden_channels, n_s_sqrt, n_s_sqrt)

        # Apply domain padding if specified
        if self.domain_padding is not None:
            x = self.domain_padding.pad(x)

        # Apply FNO layers
        for layer_idx in range(self.n_layers):
            x = self.fno_blocks(x, layer_idx)

        # Remove domain padding if applied
        if self.domain_padding is not None:
            x = self.domain_padding.unpad(x)  # (1, hidden_channels, n_s_sqrt, n_s_sqrt)

        # Use ind_dec to transport back to target mesh
        x = x.reshape(self.hidden_channels, -1).permute(1, 0)  # (n_s, hidden_channels)
        out = x[ind_dec].permute(1, 0)  # (hidden_channels, n_t)

        out = out.unsqueeze(0)
        out = self.projection(out)  # (1, out_channels, n_t)
        out = out.squeeze(0)  # (out_channels, n_t)
        return out
