import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple, Union, Literal, Optional

Number = Union[float, int]

from ..layers.spectral_convolution import SpectralConv
from ..layers.channel_mlp import LinearChannelMLP
from ..models.fno import FNO
from ..losses.differentiation import *


class FC_FNO(FNO):
    """Fourier Continuation based Fourier Neural Operator.

    FC_FNO extends the FNO architecture by extending non-periodic functions on bounded domains
    to periodic functions on larger intervals, enabling the use of spectral methods
    (FFT-based differentiation and FNO layers) while maintaining high precision, as described in [1]_.

    The workflow of FC_FNO is:
    1. Extend the input to a periodic function on a larger domain using Fourier Continuation
    2. Process through FNO layers
    3. Compute spectral derivatives on the extended domain if requested
    4. Restrict the output and derivatives back to the original domain

    The key component of an FC_FNO is the Fourier Continuation layer (see
    ``neuralop.layers.fourier_continuation``), which extends non-periodic functions to periodic
    ones on larger intervals, allowing the FNO to operate effectively and enabling accurate
    spectral derivatives via FFT-based differentiation.

    All arguments from :class:`neuralop.models.FNO` are also available here, with the exception of:
    - domain_padding (not supported in FC_FNO; Fourier Continuation handles domain extension)
    - complex_data (not supported in FC_FNO)
    - output_shape (not supported in FC_FNO)
    - resolution_scaling_factor (not supported in FC_FNO)
    See the FNO documentation for detailed descriptions of inherited parameters.

    Additional Parameters
    --------------------
    FC_object : Instantiated Fourier Continuation object. See :class:`neuralop.layers.fourier_continuation`
        for details. Determines how non-periodic functions are extended to periodic ones on larger intervals.
        Options:
        - FCGram: Uses pre-computed Gram matrices for continuation
        - FCLegendre: Uses Legendre polynomial basis for continuation
        The object adds ``n_additional_pts`` points to each spatial dimension to create the periodic extension.
    projection_nonlinearity : nn.Module, optional. Default: F.tanh
        Nonlinearity of the projection layer. Must be F.tanh or F.silu.
        This nonlinearity is used in the final projection layer and must be differentiable
        up to 2nd order for derivative computation via the chain rule.
    domain_lengths : Tuple[float, ...]
        Physical lengths of the input domain along each spatial dimension. Must be a tuple with length
        matching the number of spatial dimensions (len(n_modes)).
        This is used to compute the correct domain length for spectral differentiation on the extended domain.
        For example, for a 2D domain from [-2, 2] x [-2, 2], domain_lengths should be (4, 4).

    Notes
    -----
    - Input data must be of type torch.float64 to get high numerical precision in spectral differentiation.
    - Derivatives can be computed by passing ``derivs_to_compute`` to the forward pass. Valid options are:
      'dx', 'dxx', 'dy', 'dyy', 'dz', 'dzz' (where x, y, z represent the first, second, and third
      spatial dimensions, respectively). Derivatives are computed up to 2nd-order using spectral
      differentiation on the extended periodic domain, then restricted back to the original domain.
      The chain rule is applied through the projection layer to account for the nonlinearity.

    References
    ----------
    .. [1] Ganeshram, A., Maust H., Duruisseaux V., Li, Z., Wang Y., Leibovici D., Bruno O., Hou T., & Anandkumar, A.
        "FC-PINO: High Precision Physics-Informed Neural Operators via
        Fourier Continuation" (2025). arXiv preprint arXiv:2211.15960.
        https://arxiv.org/pdf/2211.15960
    """

    def __init__(
        self,
        n_modes: Tuple[int],
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        n_layers: int = 4,
        lifting_channel_ratio: Number = 2,
        projection_channel_ratio: Number = 2,
        positional_embedding: Union[str, nn.Module] = "grid",
        non_linearity: nn.Module = F.gelu,
        norm: Literal["ada_in", "group_norm", "instance_norm"] = None,
        use_channel_mlp: bool = True,
        channel_mlp_dropout: float = 0,
        channel_mlp_expansion: float = 0.5,
        channel_mlp_skip: Literal["linear", "identity", "soft-gating", None] = "soft-gating",
        fno_skip: Literal["linear", "identity", "soft-gating", None] = "linear",
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
        FC_object: Optional[nn.Module] = None,
        projection_nonlinearity: nn.Module = F.tanh,
        domain_lengths: Tuple[float, ...] = None,
    ):
        if FC_object is None:
            raise ValueError(
                "FC_object must be provided (e.g. FCGram or FCLegendre object)"
            )
        if domain_lengths is None:
            raise ValueError(
                "domain_lengths must be provided (e.g. (4, 4) for a 2D domain from [-2, 2] x [-2, 2])"
            )

        super().__init__(
            n_modes=n_modes,
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            n_layers=n_layers,
            lifting_channel_ratio=lifting_channel_ratio,
            projection_channel_ratio=projection_channel_ratio,
            positional_embedding=positional_embedding,
            non_linearity=non_linearity,
            norm=norm,
            use_channel_mlp=use_channel_mlp,
            channel_mlp_dropout=channel_mlp_dropout,
            channel_mlp_expansion=channel_mlp_expansion,
            channel_mlp_skip=channel_mlp_skip,
            fno_skip=fno_skip,
            fno_block_precision=fno_block_precision,
            stabilizer=stabilizer,
            max_n_modes=max_n_modes,
            factorization=factorization,
            rank=rank,
            fixed_rank_modes=fixed_rank_modes,
            implementation=implementation,
            decomposition_kwargs=decomposition_kwargs,
            separable=separable,
            preactivation=preactivation,
            conv_module=conv_module,
            enforce_hermitian_symmetry=enforce_hermitian_symmetry,
        )

        # Convert FNO block weights to complex double precision for higher precision
        for i in range(n_layers):
            self.fno_blocks.convs[i].weight = self.fno_blocks.convs[i].weight.to(torch.cdouble)

        self.FC_object = FC_object
        self.projection_nonlinearity = projection_nonlinearity
        self.domain_lengths = domain_lengths

        # Linear projection MLP
        self.projection = LinearChannelMLP(
            layers=[hidden_channels, self.projection_channels, out_channels],
            non_linearity=self.projection_nonlinearity,
        )


    def dQ_1D(self, X1, Dx_arr, Q1, Q2, derivs_to_compute):
        """Compute derivatives of projection layer output via chain rule.

        Parameters
        ----------
        X1 : torch.Tensor
            Output of Q1 (first projection layer). Shape: (b, x, m) before permute.
        Dx_arr : dict
            Spectral derivatives of pre-projection features. Keys: "dx", "dxx".
            Values: torch.Tensor of shape (b, i, x).
        Q1 : nn.Linear
            First linear layer of projection MLP.
        Q2 : nn.Linear
            Last linear layer of projection MLP.
        derivs_to_compute : list of str
            Derivatives to compute: "dx", "dxx".
        """

        # ----------------------------
        # dQ_1D EINSUM KEY
        # ----------------------------

        # Indices:
        #   b : batch
        #   i : input/feature channel index (hidden_channels before projection)
        #   m : intermediate channel index (projection hidden / Q1 out)
        #   o : output channel index (out_channels after projection)
        #   x : spatial grid index
        #
        # Tensors / shapes (consistent with permutes):
        #   X1   : (b, m, x)          [after X1 = X1.permute(0, 2, 1)]
        #   dP1  : (b, m, x)          a'(X1), where a represents the non-linear activation function
        #   dP2  : (b, m, x)          a''(X1)
        #   dW1  : (m, i)             Q1.weight
        #   dW2  : (o, m)             Q2.weight   (as used in dQ_1D)
        #   dx   : (b, i, x)          dx[V]
        #   dxx  : (b, i, x)          dxx[V]

        # ----------------------------
        # Einsums Used:
        #   dQ = einsum("mi,bmx,om->boix")               => (b, o, i, x)
        #       -> Jacobian wrt input features:  Q2 · diag(a'(X1)) · Q1
        #   Dx = einsum("boix,bix->box")                 => (b, o, x)
        #       -> Apply Jacobian to dx[V]:  (dQ)·dx
        #   W = einsum("mi,bix->bmx")                   => (b, m, x)
        #       -> Push dx[V] through Q1:  Q1·dx
        #   H2 = einsum("om,bmx->bomx")                  => (b, o, m, x)
        #       -> Q2-weighted a'' term:  Q2 ⊙ a''(X1)
        #   d2X_1 = einsum("bomx,bmx,bmx->box")          => (b, o, x)
        #       -> a'' contribution:  Q2·(a'' ⊙ (Q1·dx)^2)
        #   d2X_2 = einsum("boix,bix->box")              => (b, o, x)
        #       -> a'·dxx contribution:  (dQ)·dxx
        # ----------------------------

        Dx_out = []

        need_dx = "dx" in derivs_to_compute or "dxx" in derivs_to_compute
        if need_dx:
            dx = Dx_arr["dx"]
        if "dxx" in derivs_to_compute:
            dxx = Dx_arr["dxx"]

        X1 = X1.permute(0, 2, 1)

        dW1 = Q1.weight
        dW2 = Q2.weight

        # Compute first derivative of activation function: a'(X1)
        if self.projection_nonlinearity == F.tanh:
            dP1 = 1 / torch.cosh(X1) ** 2
        elif self.projection_nonlinearity == F.silu:
            dP1 = torch.sigmoid(X1) * (1 + X1 * (1 - torch.sigmoid(X1)))
        else:
            raise ValueError(
                f"Projection nonlinearity {self.projection_nonlinearity} not supported. Must be F.tanh or F.silu"
            )

        # Compute Jacobian: dQ/dV = Q2 · diag(a'(X1)) · Q1
        dQ = torch.einsum("mi,bmx,om->boix", dW1, dP1, dW2)

        # Compute first-order derivative: dQ/dx = (dQ/dV) · dV/dx
        if "dx" in derivs_to_compute:
            Dx = torch.einsum("boix,bix->box", dQ, dx)
            Dx_out.append(Dx)

        # Compute second derivative of activation function: a''(X1)
        if "dxx" in derivs_to_compute:
            if self.projection_nonlinearity == F.tanh:
                dP2 = -2 * dP1 * torch.tanh(X1)
            elif self.projection_nonlinearity == F.silu:
                dP2 = (torch.sigmoid(X1) * (1 - torch.sigmoid(X1)) * (2 + X1 * (1 - 2 * torch.sigmoid(X1))))
            else:
                raise ValueError(
                    f"Projection nonlinearity {self.projection_nonlinearity} not supported. Must be F.tanh or F.silu"
                )

            # Compute second-order derivative using chain rule:
            # d²Q/dx² = Q2·(a'' ⊙ (Q1·dx)²) + (dQ/dV)·d²V/dx²
            W = torch.einsum("mi,bix->bmx", dW1, dx)  # Q1·dx
            H2 = torch.einsum("om,bmx->bomx", dW2, dP2)  # Q2-weighted a''

            d2X_1 = torch.einsum("bomx,bmx,bmx->box", H2, W, W)  # a'' contribution
            d2X_2 = torch.einsum("boix,bix->box", dQ, dxx)  # a'·dxx contribution

            dxx_out = d2X_1 + d2X_2
            Dx_out.append(dxx_out)

        return Dx_out


    def dQ_2D(self, X1, Dx_arr, Q1, Q2, derivs_to_compute):
        """Compute derivatives of projection layer output via chain rule.

        Parameters
        ----------
        X1 : torch.Tensor
            Output of Q1 (first projection layer). Shape: (b, x, t, m) before permute.
        Dx_arr : dict
            Spectral derivatives of pre-projection features. Keys: "dx", "dy", "dxx", "dyy".
            Values: torch.Tensor of shape (b, i, t, x).
        Q1 : nn.Linear
            First linear layer of projection MLP.
        Q2 : nn.Linear
            Last linear layer of projection MLP.
        derivs_to_compute : list of str
            Derivatives to compute: "dx", "dy", "dxx", "dyy".
        """

        # ----------------------------
        # dQ_2D EINSUM KEY
        # ----------------------------
        # Indices:
        #   b : batch
        #   i : input/feature channel index (hidden_channels before projection)
        #   m : intermediate channel index (projection hidden / Q1 out)
        #   o : output channel index (out_channels after projection)
        #   x : first grid axis
        #   t : second grid axis
        #
        # Tensors / shapes (consistent with permutes):
        #   X1   : (b, m, x, t)       [after X1 = X1.permute(0, 3, 1, 2)]
        #   dP1  : (b, m, x, t)       a'(X1), where a represents the non-linear activation function
        #   dP2  : (b, m, x, t)       a''(X1)
        #   dW1  : (m, i)             Q1.weight
        #   dW2  : (m, o)             Q2.weight.t()
        #   wx   : (b, i, t, x)       dx[V]  (as provided in Dx_arr)
        #   wy   : (b, i, t, x)       dy[V]  (as provided in Dx_arr)
        #   wxx  : (b, i, t, x)       dxx[V]
        #   wyy  : (b, i, t, x)       dyy[V]

        # ----------------------------
        # Einsums used:
        #   dQ   = einsum("mi,bmtx,mo->boitx")             => (b, o, i, t, x)
        #       -> Jacobian wrt input features:  Q2 · diag(a'(X1)) · Q1
        #   wxQ  = einsum("boitx,bitx->botx")              => (b, o, t, x)
        #       -> Apply Jacobian to dx[V]:  (dQ)·dx
        #   wyQ  = einsum("boitx,bitx->botx")              => (b, o, t, x)
        #       -> Apply Jacobian to dy[V]:  (dQ)·dy
        #   H2   = einsum("mo,bmtx->bomtx")                => (b, o, m, t, x)
        #       -> Q2-weighted a'' term:  Q2 ⊙ a''(X1)
        #   wxx1 = einsum("bitx,mi,bomtx,mj,bjtx->botx")   => (b, o, t, x)
        #       -> a'' contribution:  Q2·(a'' ⊙ (Q1·dx)^2)
        #   wxx2 = einsum("boitx,bitx->botx")              => (b, o, t, x)
        #       -> a'·dxx contribution:  (dQ)·dxx
        #   wyy1 = einsum("bitx,mi,bomtx,mj,bjtx->botx")   => (b, o, t, x)
        #       -> a'' contribution:  Q2·(a'' ⊙ (Q1·dy)^2)
        #   wyy2 = einsum("boitx,bitx->botx")              => (b, o, t, x)
        #       -> a'·dyy contribution:  (dQ)·dyy
        # ----------------------------

        Dx_out = []
        need_dx = "dx" in derivs_to_compute or "dxx" in derivs_to_compute
        need_dy = "dy" in derivs_to_compute or "dyy" in derivs_to_compute
        if need_dx:
            wx = Dx_arr["dx"]
        if need_dy:
            wy = Dx_arr["dy"]

        X1 = X1.permute(0, 3, 1, 2)

        dW1 = Q1.weight

        # Compute first derivative of activation function: a'(X1)
        if self.projection_nonlinearity == F.tanh:
            dP1 = 1 / torch.cosh(X1) ** 2
        elif self.projection_nonlinearity == F.silu:
            dP1 = torch.sigmoid(X1) * (1 + X1 * (1 - torch.sigmoid(X1)))
        else:
            raise ValueError(
                f"Projection nonlinearity {self.projection_nonlinearity} not supported. Must be F.tanh or F.silu"
            )

        dW2 = Q2.weight.t()

        # Compute Jacobian: dQ/dV = Q2 · diag(a'(X1)) · Q1
        dQ = torch.einsum("mi,bmtx,mo->boitx", dW1, dP1, dW2)

        # Compute first-order derivatives: dQ/dx = (dQ/dV) · dV/dx, dQ/dy = (dQ/dV) · dV/dy
        if "dx" in derivs_to_compute:
            wxQ = torch.einsum("boitx,bitx->botx", dQ, wx)
            Dx_out.append(wxQ)

        if "dy" in derivs_to_compute:
            wyQ = torch.einsum("boitx,bitx->botx", dQ, wy)
            Dx_out.append(wyQ)

        # Compute second-order derivatives using chain rule
        if "dxx" in derivs_to_compute or "dyy" in derivs_to_compute:
            # Compute second derivative of activation function: a''(X1)
            if self.projection_nonlinearity == F.tanh:
                dP2 = -2 * dP1 * torch.tanh(X1)
            elif self.projection_nonlinearity == F.silu:
                dP2 = (torch.sigmoid(X1) * (1 - torch.sigmoid(X1)) * (2 + X1 * (1 - 2 * torch.sigmoid(X1))))
            else:
                raise ValueError(
                    f"Projection nonlinearity {self.projection_nonlinearity} not supported. Must be F.tanh or F.silu"
                )

            # Q2-weighted a'' term: Q2 ⊙ a''(X1)
            H2 = torch.einsum("mo,bmtx->bomtx", dW2, dP2)

        if "dxx" in derivs_to_compute:
            wxx = Dx_arr["dxx"]
            # d²Q/dx² = Q2·(a'' ⊙ (Q1·dx)²) + (dQ/dV)·d²V/dx²
            wxx1 = torch.einsum("bitx,mi,bomtx,mj,bjtx->botx", wx, dW1, H2, dW1, wx)  # a'' contribution
            wxx2 = torch.einsum("boitx,bitx->botx", dQ, wxx)  # a'·dxx contribution
            wxxQ = wxx1 + wxx2
            Dx_out.append(wxxQ)

        if "dyy" in derivs_to_compute:
            wyy = Dx_arr["dyy"]
            # d²Q/dy² = Q2·(a'' ⊙ (Q1·dy)²) + (dQ/dV)·d²V/dy²
            wyy1 = torch.einsum("bitx,mi,bomtx,mj,bjtx->botx", wy, dW1, H2, dW1, wy)  # a'' contribution
            wyy2 = torch.einsum("boitx,bitx->botx", dQ, wyy)  # a'·dyy contribution
            wyyQ = wyy1 + wyy2
            Dx_out.append(wyyQ)

        return Dx_out


    def dQ_3D(self, X1, Dx_arr, Q1, Q2, derivs_to_compute):
        """Compute derivatives of projection layer output via chain rule.

        Parameters
        ----------
        X1 : torch.Tensor
            Output of Q1 (first projection layer). Shape: (b, t, x, z, c) before permute.
        Dx_arr : dict
            Spectral derivatives of pre-projection features. Keys: "dx", "dy", "dz", "dxx", "dyy", "dzz".
            Values: torch.Tensor of shape (b, i, t, x, z).
        Q1 : nn.Linear
            First linear layer of projection MLP.
        Q2 : nn.Linear
            Last linear layer of projection MLP.
        derivs_to_compute : list of str
            Derivatives to compute: "dx", "dy", "dz", "dxx", "dyy", "dzz".
        """

        # ----------------------------
        # dQ_3D EINSUM KEY
        # ----------------------------

        # Indices:
        #   b : batch
        #   i : input/feature channel index (hidden_channels before projection)
        #   c : intermediate channel index (projection hidden / Q1 out)
        #   o : output channel index (out_channels after projection)
        #   t : grid axis 1
        #   x : grid axis 2
        #   z : grid axis 3

        # Tensors / shapes (consistent with permutes):
        #   X1   : (b, c, t, x, z)    [after X1 = X1.permute(0, 4, 1, 2, 3)]
        #   dP1  : (b, c, t, x, z)    a'(X1)
        #   dP2  : (b, c, t, x, z)    a''(X1)
        #   dW1  : (c, i)             Q1.weight
        #   dW2  : (c, o)             Q2.weight.t()
        #   dx   : (b, i, t, x, z)    dx[V]
        #   dy   : (b, i, t, x, z)    dy[V]
        #   dz   : (b, i, t, x, z)    dz[V]
        #   dxx  : (b, i, t, x, z)    dxx[V]
        #   dyy  : (b, i, t, x, z)    dyy[V]
        #   dzz  : (b, i, t, x, z)    dzz[V]

        # ----------------------------
        # dQ_3D Einsums used:
        #   dQ   = einsum("ci,bctxz,co->boitxz")                 => (b, o, i, t, x, z)
        #       -> Jacobian wrt input features:  Q2 · diag(a'(X1)) · Q1
        #   wxQ  = einsum("boitxz,bitxz->botxz")                 => (b, o, t, x, z)
        #       -> Apply Jacobian to dx[V]:  (dQ)·dx
        #   wyQ  = einsum("boitxz,bitxz->botxz")                 => (b, o, t, x, z)
        #       -> Apply Jacobian to dy[V]:  (dQ)·dy
        #   wzQ  = einsum("boitxz,bitxz->botxz")                 => (b, o, t, x, z)
        #       -> Apply Jacobian to dz[V]:  (dQ)·dz
        #   H2   = einsum("co,bctxz->bcotxz")                    => (b, c, o, t, x, z)
        #       -> Q2-weighted a'' term:  Q2 ⊙ a''(X1)
        #   wxx1 = einsum("bitxz,ci,bcotxz,cj,bjtxz->botxz")     => (b, o, t, x, z)
        #       -> a'' contribution:  Q2·(a'' ⊙ (Q1·dx)^2)
        #   wxx2 = einsum("boitxz,bitxz->botxz")                 => (b, o, t, x, z)
        #       -> a'·dxx contribution:  (dQ)·dxx
        #   wyy1 = einsum("bitxz,ci,bcotxz,cj,bjtxz->botxz")     => (b, o, t, x, z)
        #       -> a'' contribution:  Q2·(a'' ⊙ (Q1·dy)^2)
        #   wyy2 = einsum("boitxz,bitxz->botxz")                 => (b, o, t, x, z)
        #       -> a'·dyy contribution:  (dQ)·dyy
        #   wzz1 = einsum("bitxz,ci,bcotxz,cj,bjtxz->botxz")     => (b, o, t, x, z)
        #       -> a'' contribution:  Q2·(a'' ⊙ (Q1·dz)^2)
        #   wzz2 = einsum("boitxz,bitxz->botxz")                 => (b, o, t, x, z)
        #       -> a'·dzz contribution:  (dQ)·dzz
        # ----------------------------

        Dx_out = []

        need_dx = "dx" in derivs_to_compute or "dxx" in derivs_to_compute
        need_dy = "dy" in derivs_to_compute or "dyy" in derivs_to_compute
        need_dz = "dz" in derivs_to_compute or "dzz" in derivs_to_compute

        dx = Dx_arr["dx"] if need_dx else None
        dy = Dx_arr["dy"] if need_dy else None
        dz = Dx_arr["dz"] if need_dz else None

        dxx = Dx_arr["dxx"] if "dxx" in derivs_to_compute else None
        dyy = Dx_arr["dyy"] if "dyy" in derivs_to_compute else None
        dzz = Dx_arr["dzz"] if "dzz" in derivs_to_compute else None

        X1 = X1.permute(0, 4, 1, 2, 3)

        dW1 = Q1.weight
        dW2 = Q2.weight.t()

        # Compute first derivative of activation function: a'(X1)
        if self.projection_nonlinearity == F.tanh:
            dP1 = 1 / torch.cosh(X1) ** 2  # (B, C, T, X, Z)
        elif self.projection_nonlinearity == F.silu:
            dP1 = torch.sigmoid(X1) * (1 + X1 * (1 - torch.sigmoid(X1)))
        else:
            raise ValueError(
                f"Projection nonlinearity {self.projection_nonlinearity} not supported. Must be F.tanh or F.silu"
            )

        # Compute Jacobian: dQ/dV = Q2 · diag(a'(X1)) · Q1
        dQ = torch.einsum("ci, bctxz, co -> boitxz", dW1, dP1, dW2)

        # Compute first-order derivatives: dQ/dx = (dQ/dV) · dV/dx, etc.
        if "dx" in derivs_to_compute:
            wxQ = torch.einsum("boitxz,bitxz->botxz", dQ, dx)
            Dx_out.append(wxQ)

        if "dy" in derivs_to_compute:
            wyQ = torch.einsum("boitxz,bitxz->botxz", dQ, dy)
            Dx_out.append(wyQ)

        if "dz" in derivs_to_compute:
            wzQ = torch.einsum("boitxz,bitxz->botxz", dQ, dz)
            Dx_out.append(wzQ)

        # Compute second-order derivatives using chain rule
        if (
            "dxx" in derivs_to_compute
            or "dyy" in derivs_to_compute
            or "dzz" in derivs_to_compute
        ):
            # Compute second derivative of activation function: a''(X1)
            if self.projection_nonlinearity == F.tanh:
                dP2 = -2 * dP1 * torch.tanh(X1)
            elif self.projection_nonlinearity == F.silu:
                dP2 = (torch.sigmoid(X1) * (1 - torch.sigmoid(X1)) * (2 + X1 * (1 - 2 * torch.sigmoid(X1))))
            else:
                raise ValueError(
                    f"Projection nonlinearity {self.projection_nonlinearity} not supported. Must be F.tanh or F.silu"
                )

            # Q2-weighted a'' term: Q2 ⊙ a''(X1)
            H2 = torch.einsum("co,bctxz->bcotxz", dW2, dP2)

        if "dxx" in derivs_to_compute:
            # d²Q/dx² = Q2·(a'' ⊙ (Q1·dx)²) + (dQ/dV)·d²V/dx²
            wxx1 = torch.einsum("bitxz,ci,bcotxz,cj,bjtxz->botxz", dx, dW1, H2, dW1, dx)  # a'' contribution
            wxx2 = torch.einsum("boitxz,bitxz->botxz", dQ, dxx)  # a'·dxx contribution
            wxxQ = wxx1 + wxx2
            Dx_out.append(wxxQ)

        if "dyy" in derivs_to_compute:
            # d²Q/dy² = Q2·(a'' ⊙ (Q1·dy)²) + (dQ/dV)·d²V/dy²
            wyy1 = torch.einsum("bitxz,ci,bcotxz,cj,bjtxz->botxz", dy, dW1, H2, dW1, dy)  # a'' contribution)
            wyy2 = torch.einsum("boitxz,bitxz->botxz", dQ, dyy)  # a'·dyy contribution
            wyyQ = wyy1 + wyy2
            Dx_out.append(wyyQ)

        if "dzz" in derivs_to_compute:
            # d²Q/dz² = Q2·(a'' ⊙ (Q1·dz)²) + (dQ/dV)·d²V/dz²
            wzz1 = torch.einsum("bitxz,ci,bcotxz,cj,bjtxz->botxz", dz, dW1, H2, dW1, dz)  # a'' contribution
            wzz2 = torch.einsum("boitxz,bitxz->botxz", dQ, dzz)  # a'·dzz contribution
            wzzQ = wzz1 + wzz2
            Dx_out.append(wzzQ)

        return Dx_out


    def _compute_derivatives(self, x, derivs_to_compute, x_res, y_res=None, z_res=None):
        """Compute derivatives of the output with respect to input coordinates.

        Parameters
        ----------
        x : torch.Tensor
            Features on extended periodic domain.
        derivs_to_compute : list of str
            List of derivative strings to compute.
        x_res : int
            Original x resolution before extension.
        y_res : int, optional
            Original y resolution before extension (for 2D/3D).
        z_res : int, optional
            Original z resolution before extension (for 3D).

        Returns
        -------
        tuple
            (x_restricted, Dx_arr) where:
            - x_restricted: Features restricted to original domain
            - Dx_arr: Dictionary of computed derivatives
        """

        Dx_arr = {}
        deriv_tuples = []

        if self.n_dim == 1:
            # Compute new domain length accounting for Fourier Continuation extension
            new_Lengths = (self.domain_lengths[0] * (x_res + self.FC_object.n_additional_pts) / x_res)
            FourierDiff1d = FourierDiff(dim=self.n_dim, L=new_Lengths)

            dx_tuple = 1
            if "dx" in derivs_to_compute:
                deriv_tuples.append(dx_tuple)
            if "dxx" in derivs_to_compute:
                dxx_tuple = 2
                deriv_tuples.append(dxx_tuple)
            if "dxx" in derivs_to_compute and "dx" not in derivs_to_compute:
                deriv_tuples.append(dx_tuple)

            # Compute spectral derivatives on extended domain
            deriv_array = FourierDiff1d.compute_multiple_derivatives(x, derivatives=deriv_tuples)

            # Restrict derivatives back to original domain
            for i, deriv_tuple in enumerate(deriv_tuples):
                if deriv_tuple == (1):
                    Dx_arr["dx"] = self.FC_object.restrict(deriv_array[i], 1)
                elif deriv_tuple == (2):
                    Dx_arr["dxx"] = self.FC_object.restrict(deriv_array[i], 1)

        elif self.n_dim == 2:
            # Compute new domain lengths accounting for Fourier Continuation extension
            new_Lengths = (
                self.domain_lengths[0] * (x_res + self.FC_object.n_additional_pts) / x_res,
                self.domain_lengths[1] * (y_res + self.FC_object.n_additional_pts) / y_res
            )
            FourierDiff2d = FourierDiff(dim=self.n_dim, L=new_Lengths)

            dy_tuple = (0, 1)
            dx_tuple = (1, 0)
            if "dy" in derivs_to_compute:
                deriv_tuples.append(dy_tuple)
            if "dx" in derivs_to_compute:
                deriv_tuples.append(dx_tuple)
            if "dyy" in derivs_to_compute:
                dyy_tuple = (0, 2)
                deriv_tuples.append(dyy_tuple)
            if "dxx" in derivs_to_compute:
                dxx_tuple = (2, 0)
                deriv_tuples.append(dxx_tuple)
            if "dxx" in derivs_to_compute and "dx" not in derivs_to_compute:
                deriv_tuples.append(dx_tuple)
            if "dyy" in derivs_to_compute and "dy" not in derivs_to_compute:
                deriv_tuples.append(dy_tuple)

            # Compute spectral derivatives on extended domain
            deriv_array = FourierDiff2d.compute_multiple_derivatives(x, derivatives=deriv_tuples)

            # Restrict derivatives back to original domain
            for i, deriv_tuple in enumerate(deriv_tuples):
                if deriv_tuple == (0, 1):
                    Dx_arr["dy"] = self.FC_object.restrict(deriv_array[i], 2)
                elif deriv_tuple == (1, 0):
                    Dx_arr["dx"] = self.FC_object.restrict(deriv_array[i], 2)
                elif deriv_tuple == (0, 2):
                    Dx_arr["dyy"] = self.FC_object.restrict(deriv_array[i], 2)
                elif deriv_tuple == (2, 0):
                    Dx_arr["dxx"] = self.FC_object.restrict(deriv_array[i], 2)

        elif self.n_dim == 3:
            # Compute new domain lengths accounting for Fourier Continuation extension
            new_Lengths = (
                self.domain_lengths[0] * (x_res + self.FC_object.n_additional_pts) / x_res,
                self.domain_lengths[1] * (y_res + self.FC_object.n_additional_pts) / y_res,
                self.domain_lengths[2] * (z_res + self.FC_object.n_additional_pts) / z_res
            )
            FourierDiff3d = FourierDiff(dim=self.n_dim, L=new_Lengths)

            dz_tuple = (0, 0, 1)
            dy_tuple = (0, 1, 0)
            dx_tuple = (1, 0, 0)
            if "dz" in derivs_to_compute:
                deriv_tuples.append(dz_tuple)
            if "dy" in derivs_to_compute:
                deriv_tuples.append(dy_tuple)
            if "dx" in derivs_to_compute:
                deriv_tuples.append(dx_tuple)
            if "dxx" in derivs_to_compute:
                dxx_tuple = (2, 0, 0)
                deriv_tuples.append(dxx_tuple)
            if "dyy" in derivs_to_compute:
                dyy_tuple = (0, 2, 0)
                deriv_tuples.append(dyy_tuple)
            if "dzz" in derivs_to_compute:
                dzz_tuple = (0, 0, 2)
                deriv_tuples.append(dzz_tuple)
            if "dxx" in derivs_to_compute and "dx" not in derivs_to_compute:
                deriv_tuples.append(dx_tuple)
            if "dyy" in derivs_to_compute and "dy" not in derivs_to_compute:
                deriv_tuples.append(dy_tuple)
            if "dzz" in derivs_to_compute and "dz" not in derivs_to_compute:
                deriv_tuples.append(dz_tuple)

            # Compute spectral derivatives on extended domain
            deriv_array = FourierDiff3d.compute_multiple_derivatives(x, derivatives=deriv_tuples)

            # Restrict derivatives back to original domain
            for i, deriv_tuple in enumerate(deriv_tuples):
                if deriv_tuple == (0, 0, 1):
                    Dx_arr["dz"] = self.FC_object.restrict(deriv_array[i], 3)
                elif deriv_tuple == (0, 1, 0):
                    Dx_arr["dy"] = self.FC_object.restrict(deriv_array[i], 3)
                elif deriv_tuple == (1, 0, 0):
                    Dx_arr["dx"] = self.FC_object.restrict(deriv_array[i], 3)
                elif deriv_tuple == (2, 0, 0):
                    Dx_arr["dxx"] = self.FC_object.restrict(deriv_array[i], 3)
                elif deriv_tuple == (0, 2, 0):
                    Dx_arr["dyy"] = self.FC_object.restrict(deriv_array[i], 3)
                elif deriv_tuple == (0, 0, 2):
                    Dx_arr["dzz"] = self.FC_object.restrict(deriv_array[i], 3)

        # Get projection layers for chain rule computation
        Q1 = self.projection.fcs[0]  # first Linear layer
        Q2 = self.projection.fcs[-1]  # last Linear layer

        # Restrict extended features back to original domain
        x = self.FC_object.restrict(x, self.n_dim)

        # Apply chain rule through projection layer to get derivatives w.r.t. input
        if self.n_dim == 1:
            X1 = Q1(x.transpose(1, 2))
            Dx_arr = self.dQ_1D(X1, Dx_arr, Q1, Q2, derivs_to_compute)
        elif self.n_dim == 2:
            X1 = Q1(x.permute(0, 2, 3, 1))
            Dx_arr = self.dQ_2D(X1, Dx_arr, Q1, Q2, derivs_to_compute)
        elif self.n_dim == 3:
            X1 = Q1(x.permute(0, 2, 3, 4, 1))
            Dx_arr = self.dQ_3D(X1, Dx_arr, Q1, Q2, derivs_to_compute)
        else:
            raise ValueError(f"Error: expected 1, 2, or 3 dimensions, got {self.n_dim}")

        return x, Dx_arr


    def forward(self, x, derivs_to_compute=None, **kwargs):
        """Forward pass of FC_FNO.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (b, c, *spatial_dims), dtype torch.float64.
        derivs_to_compute : list of str, optional
            Derivatives to compute: 1D: ["dx", "dxx"], 2D: ["dx", "dy", "dxx", "dyy"],
            3D: ["dx", "dy", "dz", "dxx", "dyy", "dzz"]. If None, returns only output.

        Returns
        -------
        torch.Tensor or tuple
            If derivs_to_compute is None: output tensor of shape (b, out_channels, *spatial_dims).
            Otherwise: tuple (output, derivatives_dict) where derivatives_dict contains
            requested derivatives, each of shape (b, out_channels, *spatial_dims).


        The forward pass follows this workflow:
        - Validate input dtype (must be float64)
        - Apply positional embedding if specified
        - Extend input to periodic domain using Fourier Continuation
        - Lift input to hidden channels
        - Process through FNO blocks
        - If derivatives requested:
          * Compute spectral derivatives on extended domain
          * Restrict derivatives back to original domain
          * Apply chain rule through projection layer to get derivatives
        - Restrict output back to original domain
        - Apply projection layer to get final output
        """

        if x.dtype != torch.float64:
            raise ValueError("Input must be of type torch.float64")

        # Validate derivs_to_compute
        if derivs_to_compute is not None:
            if not isinstance(derivs_to_compute, (list, tuple, set)):
                raise ValueError("derivs_to_compute must be a list of strings")

            derivs_to_compute = [d.lower() for d in list(derivs_to_compute)]

            if not derivs_to_compute:
                derivs_to_compute = None
            else:
                # Define allowed derivatives for each dimension
                allowed_derivatives_map = {
                    1: {"dx", "dxx"},
                    2: {"dx", "dxx", "dy", "dyy"},
                    3: {"dx", "dxx", "dy", "dyy", "dz", "dzz"},
                }

                if self.n_dim not in allowed_derivatives_map:
                    raise ValueError(
                        f"Unsupported dimension: {self.n_dim}. Expected 1, 2, or 3."
                    )

                allowed_derivatives = allowed_derivatives_map[self.n_dim]
                invalid_derivs = set(derivs_to_compute) - allowed_derivatives
                if invalid_derivs:
                    raise ValueError(
                        f"Unsupported derivatives requested: {sorted(invalid_derivs)}. "
                        f"Allowed for {self.n_dim}D FC_FNO: {sorted(allowed_derivatives)}."
                    )

        # Apply positional embedding if specified
        if self.positional_embedding is not None:
            x = self.positional_embedding(x)

        # Extend input to periodic domain using Fourier Continuation
        # This adds n_additional_pts to each spatial dimension to create a periodic extension
        y_res = None
        z_res = None
        if self.n_dim == 1:
            __, __, x_res = x.shape
            x = self.FC_object(x, dim=1)
        elif self.n_dim == 2:
            __, __, x_res, y_res = x.shape
            x = self.FC_object(x, dim=2)
        elif self.n_dim == 3:
            __, __, x_res, y_res, z_res = x.shape
            x = self.FC_object(x, dim=3)
        else:
            raise ValueError(f"Error: expected 1, 2, or 3 dimensions, got {self.n_dim}")

        # Lift input to hidden channels
        x = self.lifting(x)

        # Process through FNO blocks on extended periodic domain
        for layer_idx in range(self.n_layers):
            x = self.fno_blocks(x, layer_idx, output_shape=None)

        # Compute derivatives if requested
        if derivs_to_compute is not None:
            x, Dx_arr = self._compute_derivatives(x, derivs_to_compute, x_res, y_res, z_res)
        else:
            # Restrict output back to original domain
            x = self.FC_object.restrict(x, self.n_dim)
            Dx_arr = None

        # Apply projection layer and restore original tensor layout
        if self.n_dim == 1:
            x = self.projection(x.transpose(1, 2)).transpose(1, 2)
        elif self.n_dim == 2:
            x = self.projection(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        elif self.n_dim == 3:
            x = self.projection(x.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)

        # Return output and derivatives if computed, otherwise just the output
        if Dx_arr is not None:
            return x, Dx_arr
        else:
            return x

    @property
    def n_modes(self):
        return self._n_modes

    @n_modes.setter
    def n_modes(self, n_modes):
        self.fno_blocks.n_modes = n_modes
        self._n_modes = n_modes
