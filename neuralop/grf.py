import torch
import numpy as np
from scipy.special import poch

import sys

sys.path.append("./neuralop")
from dct import idct, idct_2d, idct_3d

# from .dct import idct, idct_2d, idct_3d


def get_fixed_coords(Ln1, Ln2):
    """
    Generates a flattened grid of coordinates.

    Args:
        Ln1 (int): Number of points in the first dimension.
        Ln2 (int): Number of points in the second dimension.

    Returns:
        torch.Tensor: A tensor of shape (Ln1*Ln2, 2) with grid coordinates.
    """
    xs = torch.linspace(0, 1, steps=Ln1 + 1)[0:-1]
    ys = torch.linspace(0, 1, steps=Ln2 + 1)[0:-1]
    xx, yy = torch.meshgrid(xs, ys, indexing="xy")
    coords = torch.cat([yy.reshape(-1, 1), xx.reshape(-1, 1)], dim=-1)
    return coords


class GRFSampler(object):
    """Base class for Gaussian Random Field samplers."""

    def sample(self, N):
        raise NotImplementedError()


class MaternKernelSampler(GRFSampler):
    @torch.no_grad()
    def __init__(
        self,
        in_channels,
        Ln1,
        Ln2,
        scale,
        nu,
        normalize_std=False,
        boundary_condition="none",
        device=None,
    ):
        """
        Gaussian random field sampler for a Matérn covariance function, implemented
        using spectral methods with PyTorch acceleration.

        The Matérn covariance function in the spatial domain is:
            C_ν(d) = σ² × (2^(1-ν)/Γ(ν)) × (√(2ν) × d/ρ)^ν × K_ν(√(2ν) × d/ρ)

        Where:
        - d is the distance between points
        - ρ (rho) is the length scale parameter
        - ν (nu) is the smoothness parameter
        - K_ν is the modified Bessel function of the second kind
        - σ² is the variance

        The spectral density in the frequency domain is:
            S(f) = σ² × (2^n × π^(n/2) × Γ(ν + n/2) × (2ν)^ν) / (Γ(ν) × ρ^(2ν)) × (2ν/ρ² + 4π²f²)^(-(ν + n/2))

        Args:
            in_channels (int): Number of input channels for the samples.
            Ln1, Ln2 (int): Grid dimensions.
            scale (float): The length scale `l` of the Matérn kernel.
                                 Controls the correlation distance.
            nu (float): The smoothness parameter `ν` of the Matérn kernel.
                       As ν → ∞, it approaches the RBF kernel. Must be > 0.
            normalize_std (bool): If True, normalize output to have unit standard deviation.
            boundary_condition (str): One of "zero-neumann", "periodic", or "none".
            device (torch.device): The PyTorch device to use.
        """
        self.in_channels = in_channels
        self.Ln1 = Ln1
        self.Ln2 = Ln2
        self.device = device
        self.normalize_std = normalize_std
        self.boundary_condition = boundary_condition

        # Validate parameters
        if nu <= 0:
            raise ValueError("Smoothness parameter nu must be positive.")
        if scale <= 0:
            raise ValueError("Length scale must be positive.")

        # Store parameters
        self.scale = scale
        self.nu = nu

        if boundary_condition == "zero-neumann":
            if Ln1 != Ln2:
                raise ValueError("MaternKernelSampler with 'zero-neumann' currently supports square grids only")
            self.Ln = Ln1
            self._setup_zero_neumann()
        elif boundary_condition == "periodic":
            self._setup_periodic()
        elif boundary_condition == "none":
            self._setup_none()
        else:
            raise ValueError("boundary_condition must be one of 'zero-neumann', 'periodic', or 'none'")

    @staticmethod
    def _normalize_to_unit_std(tensor):
        """Normalizes a tensor to have a standard deviation of 1."""
        current_std = tensor.std()
        if current_std > 0:  # Avoid division by zero
            return tensor / current_std
        return tensor

    def _setup_zero_neumann(self):
        """Sets up spectral coefficients for zero-Neumann boundary condition using IDCT."""
        k = torch.arange(self.Ln, device=self.device)
        K1, K2 = torch.meshgrid(k, k, indexing="ij")

        # Calculate parameters directly from scale and nu
        D = 2.0  # spatial dimension
        alpha = self.nu + D / 2.0
        tau = np.sqrt(2 * self.nu) / self.scale

        # Define the (square root of) eigenvalues of the covariance operator
        C = (torch.pi**2) * (torch.square(K1) + torch.square(K2)) + tau**2
        C = torch.pow(C, -alpha / 2.0)

        # Calculate the constant factor from Matérn spectral density
        gamma_ratio = poch(self.nu, D / 2)  # gamma(self.nu + n/2) / gamma(self.nu)
        const_factor = np.sqrt((2**D * np.pi ** (D / 2) * gamma_ratio * (2 * self.nu) ** self.nu) / (self.scale ** (2 * self.nu)))

        C = const_factor * C
        self.coeff = C

    def _setup_periodic(self):
        """Sets up spectral coefficients for periodic boundary using rfft frequencies."""
        freq1 = torch.fft.fftfreq(self.Ln1, 1 / self.Ln1, device=self.device)
        freq2 = torch.fft.rfftfreq(self.Ln2, 1 / self.Ln2, device=self.device)  # Note: rfftfreq
        K1, K2 = torch.meshgrid(freq1, freq2, indexing="ij")

        # Calculate parameters directly from scale and nu
        D = 2.0  # spatial dimension
        alpha = self.nu + D / 2.0
        tau = np.sqrt(2 * self.nu) / self.scale

        # Define the (square root of) eigenvalues of the covariance operator
        C = (2 * torch.pi) ** 2 * (torch.square(K1) + torch.square(K2)) + tau**2
        C = torch.pow(C, -alpha / 2.0)

        # Calculate the constant factor from Matérn spectral density
        gamma_ratio = poch(self.nu, D / 2)  # gamma(self.nu + n/2) / gamma(self.nu)
        const_factor = np.sqrt((2**D * np.pi ** (D / 2) * gamma_ratio * (2 * self.nu) ** self.nu) / (self.scale ** (2 * self.nu)))

        C = const_factor * C
        self.coeff_rfft = C

    def _setup_none(self):
        """Sets up a nested sampler on a larger grid to simulate no boundary condition."""
        # This works by generating a larger periodic field and cropping the center,
        # which mitigates wrap-around effects from the periodic boundary.
        self.nested_sampler = MaternKernelSampler(
            in_channels=self.in_channels,
            Ln1=2 * self.Ln1,
            Ln2=2 * self.Ln2,
            scale=self.scale / 2,  # Corrected scale for larger grid
            nu=self.nu,
            normalize_std=False,  # Normalization is handled after cropping
            boundary_condition="periodic",
            device=self.device,
        )

    @torch.no_grad()
    def sample(self, N):
        """
        Generates N samples of the GRF using the specified boundary condition.

        Returns:
            torch.Tensor: A tensor of shape (N, in_channels, Ln1, Ln2).
        """
        if self.boundary_condition == "zero-neumann":
            return self._sample_zero_neumann(N)
        elif self.boundary_condition == "periodic":
            return self._sample_periodic(N)
        else:  # "none"
            return self._sample_none(N)

    def _sample_zero_neumann(self, N):
        """Samples using IDCT for zero-Neumann boundary."""
        xr = torch.randn(N, self.in_channels, self.Ln, self.Ln, device=self.device)

        # Apply spectral filter (coefficients)
        L = self.coeff[None, None, :, :] * xr
        L = self.Ln * L

        # Transform to real domain using IDCT
        result = idct_2d(L, norm="ortho")

        if self.normalize_std:
            result = self._normalize_to_unit_std(result)
        return result

    def _sample_periodic(self, N):
        """Samples using IFFT for periodic boundary."""
        # Shape: (N, in_channels, Ln1, Ln2 // 2 + 1)
        xr = torch.randn(N, self.in_channels, self.Ln1, self.Ln2 // 2 + 1, 2, device=self.device)
        xr = torch.view_as_complex(xr) / np.sqrt(2)  # standard complex noise

        # Apply spectral filter
        L = self.coeff_rfft[None, None, :, :] * xr

        # Special handling for DC and Nyquist frequencies
        L[:, :, 0, 0] = L[:, :, 0, 0].real * np.sqrt(2)  # DC must be real
        if self.Ln2 % 2 == 0:
            L[:, :, :, -1] = L[:, :, :, -1].real * np.sqrt(2)  # Nyquist must be real
        if self.Ln1 % 2 == 0:
            L[:, :, self.Ln1 // 2, 0] = L[:, :, self.Ln1 // 2, 0].real * np.sqrt(2)
            if self.Ln2 % 2 == 0:
                L[:, :, self.Ln1 // 2, -1] = L[:, :, self.Ln1 // 2, -1].real * np.sqrt(2)

        # Transform to real domain using IRFFT2
        result = torch.fft.irfft2(L, s=(self.Ln1, self.Ln2), dim=(-2, -1), norm="forward")

        if self.normalize_std:
            result = self._normalize_to_unit_std(result)
        return result

    def _sample_none(self, N):
        """Samples by cropping from a larger periodic field."""
        large_samples = self.nested_sampler.sample(N)

        # Crop the central region to get the desired grid size
        start1 = self.Ln1 // 2
        end1 = start1 + self.Ln1
        start2 = self.Ln2 // 2
        end2 = start2 + self.Ln2
        result = large_samples[:, :, start1:end1, start2:end2]

        if self.normalize_std:
            result = self._normalize_to_unit_std(result)
        return result


class RBFKernelSpectralSampler(GRFSampler):
    """
    Efficient sampler for a Gaussian Random Field with a Radial Basis Function (RBF) kernel
    using spectral methods. This implementation avoids the memory-intensive Cholesky decomposition
    used in RBFKernelSampler and instead uses FFT-based spectral methods for efficient sampling.

    The RBF (Gaussian) covariance kernel is:
        C(d) = σ² * exp(-d² / (2 * ℓ²))
    where d is the Euclidean distance and ℓ is the length scale parameter.

    The spectral density in the frequency domain is:
        S(f) = σ² * (2π)^(n/2) * ℓ^n * exp(-2π² * ℓ² * |f|²)
    where n is the spatial dimension (2 for 2D) and |f| is the magnitude of the frequency vector.
    """

    @torch.no_grad()
    def __init__(
        self,
        in_channels,
        Ln1,
        Ln2,
        scale=1.0,
        normalize_std=False,
        boundary_condition="none",
        device=None,
    ):
        """
        Initialize the RBF Kernel Spectral Sampler.

        Args:
            in_channels (int): Number of input channels for the samples.
            Ln1, Ln2 (int): Grid dimensions.
            scale (float): The length scale `ℓ` of the RBF kernel.
                          Controls the correlation distance. Default: 1.0.
            normalize_std (bool): If True, normalize output to have unit standard deviation.
            boundary_condition (str): One of "periodic" or "none".
            device (torch.device): The PyTorch device to use.
        """
        self.in_channels = in_channels
        self.Ln1 = Ln1
        self.Ln2 = Ln2
        self.device = device
        self.normalize_std = normalize_std
        self.boundary_condition = boundary_condition

        # Validate parameters
        if scale <= 0:
            raise ValueError("Length scale must be positive.")

        # Store parameters
        self.scale = scale

        if boundary_condition == "periodic":
            self._setup_periodic()
        elif boundary_condition == "none":
            self._setup_none()
        else:
            raise ValueError("boundary_condition must be one of 'periodic' or 'none'")

    @staticmethod
    def _normalize_to_unit_std(tensor):
        """Normalizes a tensor to have a standard deviation of 1."""
        current_std = tensor.std()
        if current_std > 0:  # Avoid division by zero
            return tensor / current_std
        return tensor

    def _setup_periodic(self):
        """Sets up spectral coefficients for periodic boundary using rfft frequencies."""
        freq1 = torch.fft.fftfreq(self.Ln1, 1 / self.Ln1, device=self.device)
        freq2 = torch.fft.rfftfreq(self.Ln2, 1 / self.Ln2, device=self.device)  # Note: rfftfreq
        K1, K2 = torch.meshgrid(freq1, freq2, indexing="ij")

        # Spatial dimension
        D = 2

        # Frequency magnitude squared
        freq_mag_sq = K1**2 + K2**2

        # RBF spectral density for periodic frequencies
        C = (2 * torch.pi) ** (D / 2) * self.scale**D * torch.exp(-2 * torch.pi**2 * self.scale**2 * freq_mag_sq)

        # Take square root for sampling
        self.coeff_rfft = torch.sqrt(C)

    def _setup_none(self):
        """Sets up a nested sampler on a larger grid to simulate no boundary condition."""
        # This works by generating a larger periodic field and cropping the center,
        # which mitigates wrap-around effects from the periodic boundary.
        self.nested_sampler = RBFKernelSpectralSampler(
            in_channels=self.in_channels,
            Ln1=2 * self.Ln1,
            Ln2=2 * self.Ln2,
            scale=self.scale,  # Keep same scale for consistent correlation structure
            normalize_std=False,  # Normalization is handled after cropping
            boundary_condition="periodic",
            device=self.device,
        )

    @torch.no_grad()
    def sample(self, N):
        """
        Generates N samples of the GRF using the specified boundary condition.

        Args:
            N (int): The number of samples to generate.

        Returns:
            torch.Tensor: A tensor of shape (N, in_channels, Ln1, Ln2).
        """
        if self.boundary_condition == "periodic":
            return self._sample_periodic(N)
        else:  # "none"
            return self._sample_none(N)

    def _sample_periodic(self, N):
        """Samples using IFFT for periodic boundary."""
        # Shape: (N, in_channels, Ln1, Ln2 // 2 + 1)
        xr = torch.randn(N, self.in_channels, self.Ln1, self.Ln2 // 2 + 1, 2, device=self.device)
        xr = torch.view_as_complex(xr) / np.sqrt(2)  # standard complex noise

        # Apply spectral filter
        L = self.coeff_rfft[None, None, :, :] * xr

        # Special handling for DC and Nyquist frequencies
        L[:, :, 0, 0] = L[:, :, 0, 0].real * np.sqrt(2)  # DC must be real
        if self.Ln2 % 2 == 0:
            L[:, :, :, -1] = L[:, :, :, -1].real * np.sqrt(2)  # Nyquist must be real
        if self.Ln1 % 2 == 0:
            L[:, :, self.Ln1 // 2, 0] = L[:, :, self.Ln1 // 2, 0].real * np.sqrt(2)
            if self.Ln2 % 2 == 0:
                L[:, :, self.Ln1 // 2, -1] = L[:, :, self.Ln1 // 2, -1].real * np.sqrt(2)

        # Transform to real domain using IRFFT2
        result = torch.fft.irfft2(L, s=(self.Ln1, self.Ln2), dim=(-2, -1), norm="forward")

        if self.normalize_std:
            result = self._normalize_to_unit_std(result)
        return result

    def _sample_none(self, N):
        """Samples by cropping from a larger periodic field."""
        large_samples = self.nested_sampler.sample(N)

        # Crop the central region to get the desired grid size
        start1 = self.Ln1 // 2
        end1 = start1 + self.Ln1
        start2 = self.Ln2 // 2
        end2 = start2 + self.Ln2
        result = large_samples[:, :, start1:end1, start2:end2]

        if self.normalize_std:
            result = self._normalize_to_unit_std(result)
        return result


class ExponentialKernelSampler(GRFSampler):
    """
    Sampler for a Gaussian Random Field with an Exponential kernel using FFT-based method.
    This implementation follows the efficient circulant embedding approach for fast generation
    on regular grids, avoiding the memory-intensive Cholesky decomposition.

    The exponential covariance kernel is:
        C(d) = sigma^2 * exp(-d / L)
    where d is the Euclidean distance and L is the correlation length.
    """

    @torch.no_grad()
    def __init__(self, in_channels, Ln1, Ln2, corr_length=10.0, device=None):
        """
        Initialize the Exponential Kernel Sampler.

        Args:
            in_channels (int): Number of input channels for the samples.
            Ln1, Ln2 (int): Grid dimensions (Ln1 x Ln2).
            corr_length (float): Correlation length L. Default: 10.0.
            device (torch.device): The PyTorch device to use.
        """
        self.in_channels = in_channels
        self.Ln1 = Ln1
        self.Ln2 = Ln2
        self.device = device
        self.corr_length = corr_length

        # Validate parameters
        if corr_length <= 0:
            raise ValueError("Correlation length must be positive.")

        # Pre-compute the circulant embedding and eigenvalues
        self._setup_circulant_embedding()

    def _setup_circulant_embedding(self):
        """
        Set up the circulant embedding for efficient FFT-based sampling.
        This creates a larger periodic domain that makes the covariance matrix circulant.
        """
        # Size for the circulant embedding
        # M = 2 * grid_size - 2
        self.M1 = 2 * self.Ln1 - 2
        self.M2 = 2 * self.Ln2 - 2

        # Create coordinate grids for distance calculation
        grid_I, grid_J = torch.meshgrid(torch.arange(self.M1, device=self.device), torch.arange(self.M2, device=self.device), indexing="ij")

        # Calculate distances using the circulant property
        dist_I = torch.minimum(grid_I, self.M1 - grid_I)
        dist_J = torch.minimum(grid_J, self.M2 - grid_J)
        Dist = torch.sqrt(dist_I.float() ** 2 + dist_J.float() ** 2)

        # Apply the exponential kernel to get covariance values
        C = torch.exp(-Dist / self.corr_length)

        # Compute eigenvalues using FFT2
        # The eigenvalues of a circulant matrix are the FFT of its first row
        lambda_vals = torch.fft.fft2(C)

        # Eigenvalues must be real and non-negative for a valid covariance matrix
        # Handle numerical errors by taking real part and enforcing non-negativity
        lambda_vals = torch.real(lambda_vals)
        lambda_vals = torch.clamp(lambda_vals, min=0.0)

        # Store the square root of eigenvalues for sampling
        self.sqrt_eigenvals = torch.sqrt(lambda_vals)

    @torch.no_grad()
    def sample(self, N):
        """
        Generate N samples from the Gaussian Random Field with exponential kernel.

        Args:
            N (int): The number of samples to generate.

        Returns:
            torch.Tensor: A tensor of shape (N, in_channels, Ln1, Ln2).
        """
        # Generate all noise at once for efficiency
        # Shape: (N, in_channels, M1, M2) for real and imaginary parts
        noise_real = torch.randn(N, self.in_channels, self.M1, self.M2, device=self.device)
        noise_imag = torch.randn(N, self.in_channels, self.M1, self.M2, device=self.device)
        noise = torch.complex(noise_real, noise_imag)

        # Create the field in frequency domain by multiplying noise with sqrt of eigenvalues
        # f_hat = sqrt(lambda) * noise / sqrt(M*M)
        f_hat = self.sqrt_eigenvals[None, None, :, :] * noise * np.sqrt(self.M1 * self.M2)

        # Transform back to spatial domain using IFFT2
        grf_full = torch.fft.ifft2(f_hat, dim=(-2, -1))

        # Take real part (should be real, but numerical precision might introduce small imaginary parts)
        grf_full = torch.real(grf_full)

        # Crop to the original grid size (top-left corner)
        # Shape: (N, in_channels, Ln1, Ln2)
        samples = grf_full[:, :, : self.Ln1, : self.Ln2]

        return samples
