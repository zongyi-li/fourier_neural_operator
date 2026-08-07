import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Union, Sequence
from typing import List, Optional, Tuple, Union
from .resample import resample
from ..utils import validate_scaling_factor
from .shape_enforcer import ShapeEnforcer
from .base_spectral_conv import BaseSpectralConv
from neuralop.layers.embeddings import regular_grid_nd
import tensorly as tl
from math import prod as math_prod  # Use math.prod for Python 3.8+

Number = Union[int, float]

# Use optimal einsum path if available (optional)
try:
    from tensorly.plugins import use_opt_einsum

    use_opt_einsum("optimal")
    einsum_symbols = (
        "acdefghjklmnpqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"  # Except b, i and o
    )
except ImportError:
    print("Tensorly or opt_einsum not found. Using default torch.einsum.")
    einsum_symbols = "acdefghjklmnpqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _compute_dt_nd(
    shape: Tuple[int, ...],
    start_points: Optional[List[float]] = None,
    end_points: Optional[List[float]] = None,
    device: torch.device = torch.device("cpu"),
):
    """
    Compute uniform spacing (dt) and grid points for each dimension N-D.
    """
    dim = len(shape)

    # Set default start and end points if not provided
    if start_points is None:
        start_points = [0.0] * dim
    if end_points is None:
        end_points = [1.0] * dim

    # Validate lengths
    if len(start_points) != dim or len(end_points) != dim:
        raise ValueError(
            f"Start/end points length ({len(start_points)}/{len(end_points)})"
            f" must match number of dimensions ({dim})."
        )

    grid_coords = []
    dt_list = []

    for i in range(dim):
        """
        Create grid points for dimension i
        Note: linspace includes start and end, linspace(start, end, steps) gives steps points.
        fftfreq expects N samples. If shape[i] is the number of samples,
        the domain length is (end - start) * (shape[i] / (shape[i]-1)) if using linspace directly.
        Let's assume shape[i] is the number of points, and calculate dt based on that.
        Alternative: regular_grid_nd
        """

        # Using linspace to define points:
        coords_i = torch.linspace(
            start_points[i], end_points[i], steps=shape[i], device=device
        )
        grid_coords.append(coords_i)

        # Calculate dt: (total length) / (number of intervals)
        if shape[i] > 1:
            dt = (end_points[i] - start_points[i]) / (shape[i] - 1)
        else:
            dt = (
                end_points[i] - start_points[i]
            )  # Or 1.0, or raise error? Let's use the length.
        dt_list.append(dt)

    return dt_list, grid_coords


class SpectralConvLaplace(BaseSpectralConv):
    """Implements N-Dimensional Laplace-based Spectral Convolution using pole-residue.
    described in [1]_

    Generalizes the 1D, 2D, and 3D implementations to N spatial dimensions.

    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    n_modes : tuple of int
        Number of modes (poles/residues) for EACH dimension (M1, M2, ..., MN)
    complex_data : bool, optional
        Currently ignored, assumes complex weights and uses complex FFT internally.
        By default False.
    bias : bool, default is True
        Whether to include a bias term
    steady_state_sign : int, optional
        Sign multiplier for the steady-state term calculation (+1 or -1).
        Defaults to (-1)**(order + 1), based on observed pattern in 1D/2D/3D.
    init_std : str or float, default is "auto"
        Standard deviation for weight initialization
    device : torch.device, optional
        Device for computation.
    linspace_steps : tuple of int, optional
        Number of steps (grid points) for each dimension. If None, inferred from input x.
    linspace_startpoints : list of float, optional
        Domain start points for each dimension. Defaults to 0.0 for all.
    linspace_endpoints : list of float, optional
        Domain end points for each dimension. Defaults to 1.0 for all.

    .. [1] : Cao, Q., Goswami, S., and Karniadakis, G. E. "LNO: Laplace Neural Operator
        for Solving Differential Equations" (2023). arXiv preprint arXiv:2303.10528,
        https://arxiv.org/pdf/2303.10528.

        LNO can handle non-periodic signals by incorporating the pole-residue
        relationship between input and output space, offering great
        interpretability and generalizability.

        Examples
        --------
        - High accuracy in ODEs such as the Duffing oscillator and Lorenz system.
        - Strong performance in PDEs like the diffusion equation and
        reaction-diffusion system.

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_modes: Tuple[int, ...],
        bias: bool = False,
        steady_state_sign: Optional[int] = None,
        init_std: Union[str, float] = "auto",
        device: Optional[torch.device] = None,
        linspace_steps: Optional[Tuple[int, ...]] = None,
        linspace_startpoints: Optional[List[float]] = None,
        linspace_endpoints: Optional[List[float]] = None,
    ):
        super(SpectralConvLaplace, self).__init__(device=device)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_modes = n_modes
        self.order = len(n_modes)  # Number of spatial dimensions

        if self.order == 0:
            raise ValueError("n_modes cannot be empty for SpectralConvLaplace")

        # Store grid parameters for domain specification
        self.linspace_steps = linspace_steps
        self.linspace_startpoints = linspace_startpoints
        self.linspace_endpoints = linspace_endpoints

        # Determine sign convention for steady state part
        if steady_state_sign is None:
            self.steady_state_sign = (-1) ** (self.order)
            print(
                f"SpectralConvLaplace Info: Using default steady_state_sign = {self.steady_state_sign}"
            )
        elif steady_state_sign in [-1, 1]:
            self.steady_state_sign = steady_state_sign
        else:
            raise ValueError("steady_state_sign must be +1, -1, or None (for default)")

        # Calculate sizes for poles and residues
        self._num_poles_per_dim = list(self.n_modes)
        self._total_num_poles = sum(self._num_poles_per_dim)
        self._num_residues = math_prod(self._num_poles_per_dim)

        total_weight_dim = self._total_num_poles + self._num_residues

        # Initialize single weight tensor combining poles and residues
        # Shape: (in_channels, out_channels, total_poles + num_residues)
        if init_std == "auto":
            std = 1 / (in_channels * out_channels)
        else:
            std = init_std

        # Initialize weights as complex
        self.weight = nn.Parameter(
            std
            * torch.rand(
                in_channels, out_channels, total_weight_dim, dtype=torch.cfloat
            )
        )

        if bias:
            # Bias added in spatial domain, shape (out_channels, 1, ..., 1)
            self.bias = nn.Parameter(
                std * torch.randn(out_channels, *(1,) * self.order)
            )
        else:
            self.register_parameter("bias", None)

        self.shape_enforcer = ShapeEnforcer()

    def _extract_poles_residues(self) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Extracts pole tensors (one per dim) and the residue tensor from self.weight."""
        poles_list = []
        current_idx = 0
        for dim in range(self.order):
            n_poles_d = self._num_poles_per_dim[dim]
            poles_d = self.weight[:, :, current_idx : current_idx + n_poles_d]
            poles_list.append(poles_d)
            current_idx += n_poles_d

        # Remaining part is the residue tensor
        residues_flat = self.weight[:, :, current_idx:]

        # Reshape residues to (in_channels, out_channels, M1, M2, ..., MN)
        residue_shape = (self.in_channels, self.out_channels, *self.n_modes)
        if residues_flat.shape[-1] != self._num_residues:
            raise ValueError(
                f"Internal error: Expected {self._num_residues} residues, found {residues_flat.shape[-1]} in weight tensor slice."
            )
        residues = residues_flat.view(residue_shape)

        return poles_list, residues

    def _get_einsum_indices(self) -> dict:
        """Generates standard indices for einsum based on self.order."""
        if self.order > len(einsum_symbols) // 2:
            raise ValueError(
                f"Order {self.order} is too high for default einsum symbols."
            )

        indices = {"batch": "b", "in": "i", "out": "o"}
        # Use p, q, r... for pole dimensions
        indices["poles"] = list(einsum_symbols[: self.order])
        # Use x, y, z... for spatial/frequency dimensions
        indices["spatial"] = list(einsum_symbols[self.order : 2 * self.order])
        # Use k, l, m... if needed for intermediate steps
        indices["intermediate"] = list(einsum_symbols[2 * self.order : 3 * self.order])
        return indices

    def forward(
        self, x: torch.Tensor, output_shape: Optional[Tuple[int, ...]] = None
    ) -> torch.Tensor:
        """Forward pass for N-D Laplace spectral convolution."""
        batchsize, channels, *mode_sizes = x.shape

        if len(mode_sizes) != self.order:
            raise ValueError(
                f"Input tensor has {len(mode_sizes)} spatial dimensions, "
                f"but layer was configured for order {self.order}."
            )

        # Use input shape for grid if not specified in init
        current_linspace_steps = (
            self.linspace_steps
            if self.linspace_steps is not None
            else tuple(mode_sizes)
        )
        if len(current_linspace_steps) != self.order:
            raise ValueError(
                f"Linspace steps {current_linspace_steps} don't match order {self.order}."
            )

        # Compute grid properties (dt per dim, grid coordinates per dim)
        dt_list, grid_coords_list = _compute_dt_nd(
            shape=current_linspace_steps,
            start_points=self.linspace_startpoints,
            end_points=self.linspace_endpoints,
            device=x.device,
        )

        # --- Frequency Domain Calculation ---
        fft_dims = list(range(-self.order, 0))  # Dimensions to apply FFT over

        # Calculate Laplace domain variables (s = jw) for each dimension
        lambdas_list = []
        for d in range(self.order):
            # Ensure dt is positive, handle case of single point dimension
            dt_d = dt_list[d] if dt_list[d] > 1e-9 else 1.0  # Avoid division by zero
            freqs = torch.fft.fftfreq(mode_sizes[d], d=dt_d, device=x.device)
            lambda_d = freqs * (2 * np.pi * 1j)  # s = j * omega
            lambdas_list.append(lambda_d)

        # Transform input to frequency domain
        alpha = torch.fft.fftn(x, dim=fft_dims)  # Without normalization

        # alpha shape: (batch, in_channels, freq1, freq2, ..., freqN)

        # --- Pole-Residue Calculation ---
        poles_list, residues = self._extract_poles_residues()
        # poles_list[d]: (in_channels, out_channels, n_modes_d)
        # residues: (in_channels, out_channels, n_modes_0, ..., n_modes_{N-1})

        # Get standard indices for einsum
        idx = self._get_einsum_indices()
        pole_indices = idx["poles"]  # p, q, r...
        spatial_indices = idx["spatial"]  # x, y, z...

        # --- Denominator Calculation using Einsum  ---

        # 1. Calculate difference terms: diff_d = lambda_d - pole_d
        diff_terms = []
        for d in range(self.order):
            lambda_d = lambdas_list[d]  # Shape: (freq_d,) -> mode_sizes[d]
            pole_d = poles_list[
                d
            ]  # Shape: (i, o, poles_d) -> (in_c, out_c, n_modes[d])

            # Reshape lambda_d for subtraction: target (1, 1, 1, freq_d)
            lambda_reshaped_for_sub = lambda_d.view(
                1, 1, 1, mode_sizes[d]
            )  # similar to lambda_d.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

            # Reshape pole_d for subtraction: target (i, o, poles_d, 1)
            pole_reshaped_for_sub = pole_d.view(
                self.in_channels, self.out_channels, self.n_modes[d], 1
            )

            # Calculate diff_d = lambda_d - pole_d using broadcasting
            # Result shape: (i, o, poles_d, freq_d)
            # diff_d = lambda_reshaped_for_sub - pole_reshaped_for_sub
            diff_d = torch.sub(lambda_reshaped_for_sub, pole_reshaped_for_sub)
            diff_terms.append(diff_d)

        # 2. Generate einsum string for combining difference terms
        # Example 2D: iopx,ioqy->iopqxy
        # Example 3D: iopx,ioqy,iorz->iopqrxyz
        input_strs = ",".join(
            [
                f"{idx['in']}{idx['out']}{idx['poles'][d]}{idx['spatial'][d]}"
                for d in range(self.order)
            ]
        )
        output_str = (
            f"{idx['in']}{idx['out']}{''.join(idx['poles'])}{''.join(idx['spatial'])}"
        )
        einsum_denom_str = f"{input_strs}->{output_str}"

        # 3. Compute the full denominator via einsum
        # Output shape: (i, o, poles0..N, freqs0..N)
        full_denominator = torch.einsum(einsum_denom_str, *diff_terms)

        # --- End of Einsum Denominator Calculation ---

        # Reshape residues for division (add spatial dimensions)
        # residues shape: (i, o, poles0..N)
        # target shape:   (i, o, poles0..N, 1...1) (order ones)
        residues_reshaped = residues.view(
            list(residues.shape) + [1] * self.order
        )  # same shape as full_denominator

        # Calculate Hw = Residues / Denominator

        Hw = torch.div(residues_reshaped, full_denominator)  # Element-wise division
        # Hw shape: (i, o, poles0, ..., polesN, freqs0, ..., freqsN)

        # Determine Pk (term used for steady-state calculation)
        Pk = self.steady_state_sign * Hw

        # --- Compute Output Components using Einsum ---

        # Einsum for Transient part (output_residue1)
        # Sum over input channels (i) and spatial frequencies (x, y, z...)
        # alpha: bixyz...  Hw: iopqr...xyz... -> output1: bopqr...xyz...
        alpha_indices = idx["batch"] + idx["in"] + "".join(spatial_indices)
        hw_indices = (
            idx["in"] + idx["out"] + "".join(pole_indices) + "".join(spatial_indices)
        )
        out1_indices = (
            idx["batch"] + idx["out"] + "".join(pole_indices) + "".join(spatial_indices)
        )
        einsum_str_1 = f"{alpha_indices},{hw_indices}->{out1_indices}"

        output_residue1 = torch.einsum(einsum_str_1, alpha, Hw)
        # output_residue1 shape: (b, o, poles0, ..., polesN, freqs0, ..., freqsN)

        # Einsum for Steady-state part coefficient (output_residue2)
        # Sum over input channels (i) and spatial frequencies (x, y, z...)
        # alpha: bixyz...  Pk: iopqr...xyz... -> output2: bopqr...
        pk_indices = hw_indices  # Pk has same indices as Hw
        out2_indices = idx["batch"] + idx["out"] + "".join(pole_indices)
        einsum_str_2 = f"{alpha_indices},{pk_indices}->{out2_indices}"

        output_residue2 = torch.einsum(einsum_str_2, alpha, Pk)
        # output_residue2 shape: (b, o, poles0, ..., polesN)

        # --- Inverse Transform (Transient) ---
        # Need to sum output_residue1 over pole dimensions before IFFT
        # output1: bopqr...xyz... -> Sum over p,q,r... -> boxyz...

        # Build summation dims for poles
        sum_dims = tuple(
            range(2, 2 + self.order)
        )  # Dimensions corresponding to p, q, r...
        out1_summed = torch.sum(output_residue1, dim=sum_dims)
        # out1_summed shape: (b, o, freqs0, ..., freqsN)

        # Determine output spatial size for IFFT
        if output_shape is None:
            output_spatial_shape = tuple(mode_sizes)
        else:
            output_spatial_shape = tuple(output_shape)

        # Apply IFFT
        x1 = torch.fft.ifftn(out1_summed, s=output_spatial_shape, dim=fft_dims)

        # If input was real, take real part (assuming output should also be real)
        if not x.is_complex():  # Check if input was complex
            x1 = torch.real(x1)

        # --- Time/Spatial Domain Calculation (Steady State) ---

        # Calculate exponential terms: exp(pole_d * grid_coord_d)
        exp_terms = []
        for d in range(self.order):
            pole_d = poles_list[d]  # Shape: (i, o, poles_d)
            # Ensure grid coords are complex if poles are complex
            grid_d = grid_coords_list[d].to(pole_d.dtype)  # Shape: (spatial_d,)

            # Einsum: iop, x -> iopx (for dim d)
            pole_idx_d = idx["poles"][d]
            spatial_idx_d = idx["spatial"][d]
            einsum_exp_d = f"{idx['in']}{idx['out']}{pole_idx_d},{spatial_idx_d}->{idx['in']}{idx['out']}{pole_idx_d}{spatial_idx_d}"
            term_d = torch.exp(torch.einsum(einsum_exp_d, pole_d, grid_d))
            # term_d shape: (i, o, poles_d, spatial_d)
            exp_terms.append(term_d)

        # Combine exponential terms across dimensions using einsum
        # Example 3D: iopx, ioqy, iorz -> iopqrxyz
        exp_inputs_str = ",".join(
            [
                f"{idx['in']}{idx['out']}{idx['poles'][d]}{idx['spatial'][d]}"
                for d in range(self.order)
            ]
        )
        exp_output_str = (
            idx["in"] + idx["out"] + "".join(idx["poles"]) + "".join(idx["spatial"])
        )
        einsum_exp_comb = f"{exp_inputs_str}->{exp_output_str}"

        combined_exp = torch.einsum(einsum_exp_comb, *exp_terms)
        # combined_exp shape: (i, o, poles0..N, spatial0..N)

        # Calculate steady state x2 using einsum
        # output2: bopqr... combined_exp: iopqr...xyz... -> x2: boxyz...
        # Sum over i, p, q, r...
        out2_indices_x2 = idx["batch"] + idx["in"] + "".join(idx["poles"])
        comb_exp_indices_x2 = (
            idx["in"] + idx["out"] + "".join(idx["poles"]) + "".join(idx["spatial"])
        )
        x2_indices = idx["batch"] + idx["out"] + "".join(idx["spatial"])
        einsum_x2_str = f"{out2_indices_x2},{comb_exp_indices_x2}->{x2_indices}"

        x2 = torch.einsum(einsum_x2_str, output_residue2, combined_exp)
        # x2 shape: (b, o, spatial0, ..., spatialN)

        # Normalize steady-state term (as seen in original code)
        norm_factor = math_prod(
            output_spatial_shape
        )  # Use output shape for normalization
        if norm_factor > 0:
            x2 = x2 / norm_factor

        # Take real part if needed
        if not x.is_complex():
            x2 = torch.real(x2)

        if x2.shape[2:] != output_spatial_shape:
            x2 = self.shape_enforcer(x2, output_shape=output_spatial_shape)

        # --- Combine and Finalize ---
        x = x1 + x2

        if self.bias is not None:
            # Reshape bias to broadcast correctly: (1, out_channels, 1, ..., 1)
            bias_shape = [1, self.out_channels] + [1] * self.order
            x = x + self.bias.view(bias_shape)

        # Ensure final output shape matches expectation via resampling/enforcer
        if tuple(x.shape[2:]) != output_spatial_shape:
            x = self.shape_enforcer(x, output_shape=output_spatial_shape)

        return x
