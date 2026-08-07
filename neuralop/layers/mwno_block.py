import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Tuple, Union, Literal
import math
from functools import partial
from scipy.special import eval_legendre
from numpy.polynomial.legendre import Legendre
from numpy.polynomial.chebyshev import Chebyshev
from numpy.polynomial.polynomial import Polynomial as PowerPolynomial


class WaveletUtils:
    """
    Unified wavelet utility class containing wavelet transform functions for all dimensions
    """

    @staticmethod
    def legendreDer(k, x):
        """
        Compute the derivative of Legendre polynomial using the recurrence relation.
        
        The derivative of P_k(x) can be expressed as a sum of lower-order Legendre polynomials:
        P'_k(x) = (2k+1)P_{k-1}(x) + (2(k-2)+1)P_{k-3}(x) + ... 
        
        Parameters
        ----------
        k : int
            Order of the Legendre polynomial
        x : array_like
            Points at which to evaluate the derivative
            
        Returns
        -------
        ndarray
            Derivative values with shape x.shape
        """

        def _legendre(k, x):
            return (2*k+1) * eval_legendre(k, x)
        out = 0
        for i in np.arange(k-1,-1,-2):
            out += _legendre(i, x)
        return out

    @staticmethod
    def phi_(phi_coeffs, x, lower_bound=0, upper_bound=1):
        """
        Evaluate a polynomial basis function with compact support.
        
        This function evaluates a polynomial defined by coefficients phi_c,
        but sets the value to zero outside the interval [lower_bound, upper_bound].
        
        Parameters
        ----------
        phi_coeffs : array_like
            Polynomial coefficients, ordered from lowest to highest degree.
            Shape depends on the wavelet construction:
            - 1D wavelets: shape (k,)
            - 2D wavelets: shape (k,) or (k, k) for tensor products
            - 3D wavelets: shape (k,) or (k, k, k) for tensor products
        x : array_like
            Points at which to evaluate the polynomial.
            - Can be 1D array of shape (n,) for evaluating at n points
            - Can be 2D array of shape (m, n) for broadcasting
            - Can be 3D array of shape (m, n, p) for 3D evaluation
            - Generally any shape supported by numpy polynomial evaluation
        lower_bound : float, optional
            Lower bound of the support interval, default 0
        upper_bound : float, optional
            Upper bound of the support interval, default 1
            
        Returns
        -------
        ndarray
        Polynomial values at x, with shape x.shape. Values outside
        [lower_bound, upper_bound] are set to zero.
        """
        mask = np.logical_or(x<lower_bound, x>upper_bound) * 1.0
        return np.polynomial.polynomial.Polynomial(phi_coeffs)(x) * (1-mask)

    @staticmethod
    def _gram_schmidt_wavelet_coeffs(
        scaling_coeffs: np.ndarray, scaling_2x_coeffs: np.ndarray, k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Multiwavelet Gram–Schmidt in power basis (same recipe for Legendre / Chebyshev rows).
        """
        wavelet_left_coeffs = np.zeros((k, k))
        wavelet_right_coeffs = np.zeros((k, k))

        for basis_idx in range(k):
            wavelet_left_coeffs[basis_idx, :] = scaling_2x_coeffs[basis_idx, :]

            for scaling_idx in range(k):
                a = scaling_2x_coeffs[basis_idx, :basis_idx + 1]
                b = scaling_coeffs[scaling_idx, :scaling_idx + 1]
                product_poly = np.convolve(a, b)
                product_poly[np.abs(product_poly) < 1e-8] = 0

                projection = (product_poly * 1 / (np.arange(len(product_poly)) + 1) *
                              np.power(0.5, 1 + np.arange(len(product_poly)))).sum()

                wavelet_left_coeffs[basis_idx, :] -= projection * scaling_coeffs[scaling_idx, :]
                wavelet_right_coeffs[basis_idx, :] -= projection * scaling_coeffs[scaling_idx, :]

            for prev_wavelet_idx in range(basis_idx):
                a = scaling_2x_coeffs[basis_idx, :basis_idx + 1]
                b = wavelet_left_coeffs[prev_wavelet_idx, :]
                product_poly = np.convolve(a, b)
                product_poly[np.abs(product_poly) < 1e-8] = 0

                projection = (product_poly * 1 / (np.arange(len(product_poly)) + 1) *
                              np.power(0.5, 1 + np.arange(len(product_poly)))).sum()

                wavelet_left_coeffs[basis_idx, :] -= projection * wavelet_left_coeffs[prev_wavelet_idx, :]
                wavelet_right_coeffs[basis_idx, :] -= projection * wavelet_left_coeffs[prev_wavelet_idx, :]

            a = wavelet_left_coeffs[basis_idx, :]
            product_poly = np.convolve(a, a)
            product_poly[np.abs(product_poly) < 1e-8] = 0
            norm_squared_left = (product_poly * 1 / (np.arange(len(product_poly)) + 1) *
                                 np.power(0.5, 1 + np.arange(len(product_poly)))).sum()

            a = wavelet_right_coeffs[basis_idx, :]
            product_poly = np.convolve(a, a)
            product_poly[np.abs(product_poly) < 1e-8] = 0
            norm_squared_right = (product_poly * 1 / (np.arange(len(product_poly)) + 1) *
                                  (1 - np.power(0.5, 1 + np.arange(len(product_poly))))).sum()

            norm = np.sqrt(norm_squared_left + norm_squared_right)
            if norm < 1e-14:
                wavelet_left_coeffs[basis_idx, :] = 0
                wavelet_right_coeffs[basis_idx, :] = 0
            else:
                wavelet_left_coeffs[basis_idx, :] /= norm
                wavelet_right_coeffs[basis_idx, :] /= norm

            wavelet_left_coeffs[np.abs(wavelet_left_coeffs) < 1e-8] = 0
            wavelet_right_coeffs[np.abs(wavelet_right_coeffs) < 1e-8] = 0

        return wavelet_left_coeffs, wavelet_right_coeffs

    @staticmethod
    def _chebyshev_coeff_matrix_to_power_basis(coeff_matrix: np.ndarray, k: int) -> np.ndarray:
        """Convert each row from Chebyshev-domain coeffs to power (Polynomial) coeffs."""
        out = np.zeros((k, k))
        for i in range(k):
            row = coeff_matrix[i, :]
            inds = np.where(np.abs(row) > 1e-15)[0]
            if inds.size == 0:
                continue
            nz = inds[-1] + 1
            cheb = Chebyshev(row[:nz])
            pw = cheb.convert(kind=PowerPolynomial)
            coef = np.asarray(pw.coef, dtype=float)
            out[i, :len(coef)] = coef
        return out

    @classmethod
    def get_phi_psi(cls, k, base):
        """
        Generate scaling and wavelet basis functions for multi-resolution analysis.

        This creates a hierarchical basis for function approximation using wavelets.
        Think of it like building a family of functions that can represent any signal
        at multiple levels of detail - coarse features (scaling functions) and fine
        details (wavelet functions).

        How it works:
        1. Start with orthogonal polynomials (Legendre or Chebyshev) on [0,1]
        2. These become our "scaling functions" (phi) - they capture smooth, low-frequency content
        3. Create "wavelet functions" (psi) through orthogonalization - they capture details/edges
        4. Wavelets are split into two halves: psi1 for [0, 0.5] and psi2 for [0.5, 1]

        The result is a complete basis where:
        - phi functions = "blur" or "smooth approximation" at a given scale
        - psi functions = "details" or "differences" between scales

        Parameters
        ----------
        k : int
            Number of basis functions (polynomial order + 1).
            - k=2: Linear approximation (simplest)
            - k=3-4: Good balance for most applications
            - k>4: Smoother wavelets, more computational cost
            Higher k means you can represent smoother functions more accurately.

        base : str
            Polynomial family to use as foundation:
            - 'legendre': Uniform weight, good general-purpose choice
            - 'chebyshev': Clustered near boundaries, good for edge phenomena

        Returns
        -------
        phi : list of k callable functions
            Scaling functions φ_0, φ_1, ..., φ_{k-1}
            Each phi[i] takes array input on [0,1] and returns same-shaped array.
            These represent smooth approximations at different polynomial orders.

        psi1 : list of k callable functions
            Wavelet functions for left half-interval [0, 0.5]
            Each psi1[i] captures detail/difference information in the left half.

        psi2 : list of k callable functions
            Wavelet functions for right half-interval [0.5, 1]
            Each psi2[i] captures detail/difference information in the right half.

        Notes
        -----
        For multidimensional problems, use tensor products:
        - 2D: phi[i](x) * phi[j](y) gives a 2D basis function
        - 3D: phi[i](x) * phi[j](y) * phi[k](z) gives a 3D basis function

        The wavelets are constructed to be orthogonal to:
        1. All scaling functions (so they represent "new" information)
        2. All previous wavelets (so each adds independent detail)
        """
        # Coefficient matrices: rows = basis function index, cols = polynomial coefficients
        scaling_coeffs = np.zeros((k, k))  # phi(x) coefficients
        scaling_2x_coeffs = np.zeros((k, k))  # phi(2x) coefficients (for orthogonalization)

        if base == 'legendre':
            # Build normalized Legendre polynomials transformed to [0,1]
            for basis_idx in range(k):
                # Create Legendre polynomial P_n(x) and map from [-1,1] to [0,1] via x -> 2x-1
                legendre_poly = Legendre.basis(basis_idx)

                # Transform domain: [0,1] -> [-1,1] via 2x - 1
                # Normalization: sqrt(2n+1) ensures orthonormality on [0,1]
                poly_on_unit_interval = legendre_poly(Legendre([-1, 2]))  # Compose: P_n(2x - 1)
                coeffs = poly_on_unit_interval.coef * np.sqrt(2 * basis_idx + 1)
                scaling_coeffs[basis_idx, :len(coeffs)] = coeffs

                # Also need phi(2x) for wavelet construction
                # Transform: [0,1] -> [-1,1] via 4x - 1
                poly_stretched = legendre_poly(Legendre([-1, 4]))  # P_n(4x - 1)
                coeffs_2x = poly_stretched.coef * np.sqrt(2) * np.sqrt(2 * basis_idx + 1)
                scaling_2x_coeffs[basis_idx, :len(coeffs_2x)] = coeffs_2x

            wavelet_left_coeffs, wavelet_right_coeffs = cls._gram_schmidt_wavelet_coeffs(
                scaling_coeffs, scaling_2x_coeffs, k
            )

            # Create callable functions using numpy's polynomial evaluation
            # Note: np.polyval expects coefficients in descending order
            phi = [lambda x, c=scaling_coeffs[i, :]: np.polyval(c[::-1], x)
                   for i in range(k)]
            psi1 = [lambda x, c=wavelet_left_coeffs[i, :]: np.polyval(c[::-1], x)
                    for i in range(k)]
            psi2 = [lambda x, c=wavelet_right_coeffs[i, :]: np.polyval(c[::-1], x)
                    for i in range(k)]

        elif base == 'chebyshev':
            # Build normalized Chebyshev polynomials
            for basis_idx in range(k):
                if basis_idx == 0:
                    # T_0 = 1, special normalization
                    scaling_coeffs[basis_idx, 0] = np.sqrt(2 / np.pi)
                    scaling_2x_coeffs[basis_idx, 0] = np.sqrt(2 / np.pi) * np.sqrt(2)
                else:
                    # Create Chebyshev polynomial T_n(x)
                    cheb_poly = Chebyshev.basis(basis_idx)

                    # Transform to [0,1] domain
                    poly_on_unit_interval = cheb_poly(Chebyshev([-1, 2]))
                    coeffs = poly_on_unit_interval.coef * 2 / np.sqrt(np.pi)
                    scaling_coeffs[basis_idx, :len(coeffs)] = coeffs

                    # Stretched version
                    poly_stretched = cheb_poly(Chebyshev([-1, 4]))
                    coeffs_2x = poly_stretched.coef * np.sqrt(2) * 2 / np.sqrt(np.pi)
                    scaling_2x_coeffs[basis_idx, :len(coeffs_2x)] = coeffs_2x

            # Gram–Schmidt uses power-basis convolution; convert Chebyshev rows first.
            scaling_coeffs = cls._chebyshev_coeff_matrix_to_power_basis(scaling_coeffs, k)
            scaling_2x_coeffs = cls._chebyshev_coeff_matrix_to_power_basis(scaling_2x_coeffs, k)
            wavelet_left_coeffs, wavelet_right_coeffs = cls._gram_schmidt_wavelet_coeffs(
                scaling_coeffs, scaling_2x_coeffs, k
            )

            phi = [partial(cls.phi_, scaling_coeffs[i, :]) for i in range(k)]
            psi1 = [
                partial(cls.phi_, wavelet_left_coeffs[i, :], lower_bound=0, upper_bound=0.5)
                for i in range(k)
            ]
            psi2 = [
                partial(cls.phi_, wavelet_right_coeffs[i, :], lower_bound=0.5, upper_bound=1)
                for i in range(k)
            ]

        return phi, psi1, psi2

    @classmethod
    def get_filter(cls, base, k):
        """
        Compute filter bank coefficients for fast wavelet transform.

        This generates the matrices needed for the Fast Wavelet Transform (FWT), which
        efficiently decomposes a function into coarse (low-frequency) and detail
        (high-frequency) components at multiple scales.

        Parameters
        ----------
        base : str
            Polynomial basis type:
            - 'legendre': Standard choice for uniform problems
            - 'chebyshev': Better for boundary layers or oscillatory behavior

        k : int
            Number of basis functions (polynomial order + 1)
            Must match the k used in get_phi_psi()

        Returns
        -------
        h_0 : ndarray of shape (k, k)
            Low-pass decomposition filter for even-indexed coefficients
            Maps fine-scale coefficients → coarse-scale approximation (even part)

        h_1 : ndarray of shape (k, k)
            Low-pass decomposition filter for odd-indexed coefficients
            Maps fine-scale coefficients → coarse-scale approximation (odd part)

        g_0 : ndarray of shape (k, k)
            High-pass decomposition filter for even-indexed coefficients
            Maps fine-scale coefficients → detail coefficients (even part)

        g_1 : ndarray of shape (k, k)
            High-pass decomposition filter for odd-indexed coefficients
            Maps fine-scale coefficients → detail coefficients (odd part)

        phi_0 : ndarray of shape (k, k)
            Reconstruction operator for even samples
            Used to rebuild fine-scale function from coarse + detail (even part)

        phi_1 : ndarray of shape (k, k)
            Reconstruction operator for odd samples
            Used to rebuild fine-scale function from coarse + detail (odd part)

        Notes
        -----
        The filters form a perfect reconstruction filter bank:
        - Decomposition (analysis): Split signal into coarse + detail
        - Reconstruction (synthesis): Perfectly recover original from coarse + detail

        For 2D/3D problems, apply these filters dimension-by-dimension (separable transform).
        """

        def construct_full_wavelet(psi_left, psi_right, basis_idx, x_values):
            """
            Combine left and right wavelets into single function.

            Wavelets have compact support split at x=0.5:
            - Use psi_left for x <= 0.5
            - Use psi_right for x > 0.5
            """
            left_mask = (x_values <= 0.5).astype(float)
            return (psi_left[basis_idx](x_values) * left_mask +
                    psi_right[basis_idx](x_values) * (1 - left_mask))

        if base not in ['legendre', 'chebyshev']:
            raise ValueError(f"Base '{base}' not supported. Use 'legendre' or 'chebyshev'.")

        # Initialize filter matrices
        h_0 = np.zeros((k, k))  # Low-pass filter, even samples
        h_1 = np.zeros((k, k))  # Low-pass filter, odd samples
        g_0 = np.zeros((k, k))  # High-pass filter, even samples
        g_1 = np.zeros((k, k))  # High-pass filter, odd samples
        phi_0 = np.zeros((k, k))  # Reconstruction, even samples
        phi_1 = np.zeros((k, k))  # Reconstruction, odd samples

        # Get basis functions
        scaling_funcs, wavelet_left_funcs, wavelet_right_funcs = cls.get_phi_psi(k, base)

        if base == 'legendre':
            # Use Gauss-Legendre quadrature for accurate integration
            # Get quadrature points (roots of Legendre polynomial)
            legendre_poly = Legendre.basis(k)
            quadrature_roots_std = legendre_poly.roots()  # Roots on [-1, 1]
            quadrature_points = (quadrature_roots_std + 1) / 2  # Map to [0, 1]

            # Compute quadrature weights
            # For Gauss-Legendre: w_i = 2 / [(1-x_i^2) * (P'_k(x_i))^2]
            quadrature_weights = (1 / k / cls.legendreDer(k, 2 * quadrature_points - 1) /
                                  eval_legendre(k - 1, 2 * quadrature_points - 1))

            # Compute filter coefficients via weighted inner products
            for row_idx in range(k):
                for col_idx in range(k):
                    # Two-scale relation at even positions: x/2
                    h_0[row_idx, col_idx] = (1 / np.sqrt(2) *
                                             (quadrature_weights *
                                              scaling_funcs[row_idx](quadrature_points / 2) *
                                              scaling_funcs[col_idx](quadrature_points)).sum())

                    g_0[row_idx, col_idx] = (1 / np.sqrt(2) *
                                             (quadrature_weights *
                                              construct_full_wavelet(wavelet_left_funcs, wavelet_right_funcs,
                                                                     row_idx, quadrature_points / 2) *
                                              scaling_funcs[col_idx](quadrature_points)).sum())

                    # Two-scale relation at odd positions: (x+1)/2
                    h_1[row_idx, col_idx] = (1 / np.sqrt(2) *
                                             (quadrature_weights *
                                              scaling_funcs[row_idx]((quadrature_points + 1) / 2) *
                                              scaling_funcs[col_idx](quadrature_points)).sum())

                    g_1[row_idx, col_idx] = (1 / np.sqrt(2) *
                                             (quadrature_weights *
                                              construct_full_wavelet(wavelet_left_funcs, wavelet_right_funcs,
                                                                     row_idx, (quadrature_points + 1) / 2) *
                                              scaling_funcs[col_idx](quadrature_points)).sum())

            # For Legendre, reconstruction is simple (orthonormal basis)
            phi_0 = np.eye(k)
            phi_1 = np.eye(k)

        elif base == 'chebyshev':
            # Use Chebyshev nodes for quadrature
            num_quad_points = 2 * k  # Oversample for accuracy
            cheb_poly = Chebyshev.basis(num_quad_points)
            quadrature_roots_std = cheb_poly.roots()  # Roots on [-1, 1]
            quadrature_points = (quadrature_roots_std + 1) / 2  # Map to [0, 1]

            # Chebyshev quadrature weights (equal weights)
            quadrature_weight = np.pi / num_quad_points / 2

            # Compute filters
            for row_idx in range(k):
                for col_idx in range(k):
                    # Decomposition filters
                    h_0[row_idx, col_idx] = (1 / np.sqrt(2) *
                                             (quadrature_weight *
                                              scaling_funcs[row_idx](quadrature_points / 2) *
                                              scaling_funcs[col_idx](quadrature_points)).sum())

                    g_0[row_idx, col_idx] = (1 / np.sqrt(2) *
                                             (quadrature_weight *
                                              construct_full_wavelet(wavelet_left_funcs, wavelet_right_funcs,
                                                                     row_idx, quadrature_points / 2) *
                                              scaling_funcs[col_idx](quadrature_points)).sum())

                    h_1[row_idx, col_idx] = (1 / np.sqrt(2) *
                                             (quadrature_weight *
                                              scaling_funcs[row_idx]((quadrature_points + 1) / 2) *
                                              scaling_funcs[col_idx](quadrature_points)).sum())

                    g_1[row_idx, col_idx] = (1 / np.sqrt(2) *
                                             (quadrature_weight *
                                              construct_full_wavelet(wavelet_left_funcs, wavelet_right_funcs,
                                                                     row_idx, (quadrature_points + 1) / 2) *
                                              scaling_funcs[col_idx](quadrature_points)).sum())

                    # Reconstruction matrices (Chebyshev basis not orthonormal)
                    phi_0[row_idx, col_idx] = (quadrature_weight *
                                               scaling_funcs[row_idx](2 * quadrature_points) *
                                               scaling_funcs[col_idx](2 * quadrature_points)).sum() * 2

                    phi_1[row_idx, col_idx] = (quadrature_weight *
                                               scaling_funcs[row_idx](2 * quadrature_points - 1) *
                                               scaling_funcs[col_idx](2 * quadrature_points - 1)).sum() * 2

            # Clean up numerical noise
            phi_0[np.abs(phi_0) < 1e-8] = 0
            phi_1[np.abs(phi_1) < 1e-8] = 0

        # Clean up numerical noise in all filters
        for filter_matrix in [h_0, h_1, g_0, g_1]:
            filter_matrix[np.abs(filter_matrix) < 1e-8] = 0

        return h_0, h_1, g_0, g_1, phi_0, phi_1


class SparseKernel(nn.Module):
    """
    Sparse Spatial Convolution Kernel for Multiwavelet Networks.

    This layer applies learned spatial convolutions to wavelet coefficients.
    It's "sparse" in the sense that it uses small convolutional kernels (3x3)
    rather than dense connections, making it computationally efficient.

    Use case: Apply spatial mixing to wavelet detail coefficients, capturing
    local patterns and textures in the spatial domain.

    The layer automatically handles 1D, 2D, and 3D inputs by:
    1. Reshaping input to (batch, channels, *spatial_dims)
    2. Applying appropriate Conv{1,2,3}d layer
    3. Projecting back to original channel structure
    4. Reshaping to original format

    Parameters
    ----------
    k : int
        Wavelet basis size (number of basis functions per dimension)

    alpha : int
        Hidden dimension size for the intermediate convolution.
        Controls the expressiveness of the spatial mixing.
        Typical value: 128

    c : int, default=1
        Number of wavelet channels.
        Total input channels = c * k^n_dim

    n_dim : int, default=1
        Spatial dimensionality (1, 2, or 3)

    Attributes
    ----------
    conv : nn.Sequential
        Convolutional block: Conv → ReLU
        Uses Conv1d/Conv2d/Conv3d depending on n_dim

    output_proj : nn.Linear
        Projects from hidden dimension back to input channels

    Input/Output Shapes
    -------------------
    1D: (batch, n_points, channels, k)
    2D: (batch, height, width, channels, k²)
    3D: (batch, height, width, time, channels, k²)

    Examples
    --------
    >>> # 2D spatial convolution for wavelets
    >>> layer = SparseKernel(k=3, alpha=128, c=1, n_dim=2)
    >>> x = torch.randn(16, 32, 32, 1, 9)  # (B, H, W, c, k²)
    >>> output = layer(x)
    >>> assert output.shape == x.shape

    """

    def __init__(self, k, alpha, c=1, n_dim=1, **kwargs):
        super().__init__()
        self.k = k
        self.alpha = alpha  # Hidden dimension
        self.c = c
        self.n_dim = n_dim

        # Calculate total input channels
        if n_dim == 1:
            in_channels = c * k
        elif n_dim in [2, 3]:
            in_channels = c * k ** 2
        else:
            raise ValueError(f"Unsupported dimension: {n_dim}. Must be 1, 2, or 3.")

        self.in_channels = in_channels

        # Build dimension-appropriate convolution
        self.conv = self._build_conv_block(in_channels, alpha)
        self.output_proj = nn.Linear(alpha, in_channels)

    def _build_conv_block(self, in_channels, hidden_dim):
        """
        Build convolutional block matching the input dimensionality.

        Uses 3x3(x3) kernels with padding to preserve spatial dimensions.
        ReLU activation for nonlinearity.
        """
        if self.n_dim == 1:
            return nn.Sequential(
                nn.Conv1d(in_channels, hidden_dim, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True)
            )
        elif self.n_dim == 2:
            return nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True)
            )
        elif self.n_dim == 3:
            return nn.Sequential(
                nn.Conv3d(in_channels, hidden_dim, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True)
            )

    def _reshape_for_conv(self, x):
        """
        Reshape from wavelet format to standard convolution format.

        Wavelet format: (*spatial, channels, k^n_dim)
        Conv format: (batch, channels, *spatial)

        Returns
        -------
        x_reshaped : torch.Tensor
            Reshaped for convolution
        original_shape : tuple
            Original shape for restoring later
        """
        original_shape = x.shape

        if self.n_dim == 1:
            # (B, N, c, k) -> (B, c*k, N)
            batch_size, n_points, num_channels, k = x.shape
            # move (c,k) next to batch: (B, c, k, N), then merge c and k
            x = x.permute(0, 2, 3, 1).contiguous().reshape(batch_size, num_channels * k, n_points)

        elif self.n_dim == 2:
            # (B, Nx, Ny, c, k²) -> (B, c*k², Nx, Ny)
            batch_size, nx, ny, num_channels, k_sq = x.shape
            x = x.permute(0, 3, 4, 1, 2).contiguous().reshape(batch_size, num_channels * k_sq, nx, ny)

        elif self.n_dim == 3:
            # (B, Nx, Ny, T, c, k²) -> (B, c*k², Nx, Ny, T)
            batch_size, nx, ny, time_steps, num_channels, k_sq = x.shape
            x = x.permute(0, 4, 5, 1, 2, 3).contiguous().reshape(batch_size, num_channels * k_sq, nx, ny, time_steps)

        return x, original_shape

    def _reshape_from_conv(self, x, original_shape):
        """
        Reshape from convolution format back to wavelet format.

        Conv format: (batch, channels, *spatial)
        Wavelet format: (*spatial, channels, k^n_dim)
        """
        if self.n_dim == 1:
            # (B, hidden, N) → (B, N, hidden)
            x = x.permute(0, 2, 1)
        elif self.n_dim == 2:
            # (B, hidden, Nx, Ny) → (B, Nx, Ny, hidden)
            x = x.permute(0, 2, 3, 1)
        elif self.n_dim == 3:
            # (B, hidden, Nx, Ny, T) → (B, Nx, Ny, T, hidden)
            x = x.permute(0, 2, 3, 4, 1)

        return x

    def forward(self, x):
        """
        Apply spatial convolution to wavelet coefficients.

        Parameters
        ----------
        x : torch.Tensor
            Wavelet coefficients

        Returns
        -------
        torch.Tensor
            Spatially mixed coefficients, same shape as input
        """
        # Reshape for convolution
        x, original_shape = self._reshape_for_conv(x)

        # Apply convolution
        x = self.conv(x)

        # Reshape back
        x = self._reshape_from_conv(x, original_shape)

        # Project to original channels
        x = self.output_proj(x)

        # Restore exact original shape
        x = x.reshape(original_shape)

        return x


class SparseKernelFT(nn.Module):
    """
    Sparse Fourier Transform Kernel for Multiwavelet Networks.

    This layer operates in the frequency domain, applying learned transformations
    to a sparse subset of Fourier modes. This is computationally efficient and
    particularly effective for capturing global patterns.

    Parameters
    ----------
    k : int
        Wavelet basis size

    alpha : int
        Number of Fourier modes to retain per dimension.
        Controls frequency domain sparsity:
        - Small alpha: Very sparse, fast, may miss details
        - Large alpha: Less sparse, slower, more expressive

    c : int, default=1
        Number of wavelet channels

    n_dim : int, default=1
        Spatial dimensionality (1, 2, or 3)

    Attributes
    ----------
    weights : nn.Parameter (1D case)
        Complex-valued weights of shape (in_channels, out_channels, alpha)

    weights1, weights2, ... : nn.Parameter (2D/3D cases)
        Complex-valued weights for different frequency quadrants
        Handles both positive and negative frequencies

    output_proj : nn.Linear (2D/3D only)
        Final projection with nonlinearity

    Input/Output Shapes
    -------------------
    1D: (batch, n_points, channels, k)
    2D: (batch, height, width, channels, k²)
    3D: (batch, height, width, time, channels, k²)

    Examples
    --------
    >>> # 2D Fourier kernel
    >>> layer = SparseKernelFT(k=3, alpha=12, c=1, n_dim=2)
    >>> x = torch.randn(16, 64, 64, 1, 9)
    >>> output = layer(x)
    >>> assert output.shape == x.shape

    Notes
    -----
    - Uses rfft (real FFT) for efficiency since inputs are real-valued
    - For 2D/3D, we separately handle different frequency quadrants to account
      for the asymmetry in rfft output
    - Frequency indexing is carefully handled to avoid rfftshift issues
    """

    def __init__(self, k, alpha, c=1, n_dim=1, **kwargs):
        super().__init__()
        self.k = k
        self.alpha = alpha  # Number of modes to keep
        self.c = c
        self.n_dim = n_dim

        # Calculate input channels
        if n_dim == 1:
            in_channels = c * k
        elif n_dim in [2, 3]:
            in_channels = c * k ** 2
        else:
            raise ValueError(f"Unsupported dimension: {n_dim}. Must be 1, 2, or 3.")

        self.in_channels = in_channels

        # Initialize frequency domain weights
        self._init_frequency_weights(in_channels)

        # Output projection for 2D/3D
        if n_dim in [2, 3]:
            self.output_proj = nn.Linear(in_channels, in_channels)

    @staticmethod
    def _rfft_positive_modes(axis_len: int, alpha: int) -> int:
        """Modes along the last axis of ``rfft2`` / ``rfftn`` (non-negative freqs only)."""
        return min(int(alpha), axis_len // 2 + 1)

    @staticmethod
    def _full_fft_axis_symmetric_modes(n: int, alpha: int) -> Tuple[int, int]:
        """
        Sizes for ``[:m_pos]`` and ``[-m_neg:]`` along a full complex FFT axis so the two
        index ranges do not overlap (requires ``m_pos + m_neg <= n``; we use ``m_pos = m_neg``).

        For ``n <= 1`` only the DC bin exists: fill the low slice only and set ``m_neg = 0``
        (``tensor[-0:]`` would alias the whole axis).
        """
        a = int(alpha)
        if n <= 1:
            return min(a, max(n, 0)), 0
        m = min(a, n // 2)
        return m, m

    def _init_frequency_weights(self, in_channels):
        """
        Initialize learnable frequency domain weights.

        For 1D: Single weight matrix for positive frequencies
        For 2D: Two weight matrices for positive/negative frequencies in first dimension
        For 3D: Four weight matrices for all quadrants of first two dimensions

        The asymmetry comes from rfft only outputting positive frequencies in
        the last dimension, but full frequencies in other dimensions.
        """
        if self.n_dim == 1:
            # Simple case: only positive frequencies
            self.weights = nn.Parameter(
                torch.rand(in_channels, in_channels, self.alpha, dtype=torch.cfloat) /
                (in_channels * in_channels)
            )

        elif self.n_dim == 2:
            # 2D rfft2: First dim has both +/- frequencies, second dim only +
            # Need 2 weight matrices: one for positive first dim, one for negative
            self.weights1 = nn.Parameter(
                torch.zeros(in_channels, in_channels, self.alpha, self.alpha, dtype=torch.cfloat)
            )
            self.weights2 = nn.Parameter(
                torch.zeros(in_channels, in_channels, self.alpha, self.alpha, dtype=torch.cfloat)
            )
            nn.init.xavier_normal_(self.weights1)
            nn.init.xavier_normal_(self.weights2)

        elif self.n_dim == 3:
            # 3D rfftn: First two dims have +/- frequencies, third dim only +
            # Need 4 weight matrices for all combinations: (++, +-, -+, --)
            self.weights1 = nn.Parameter(
                torch.zeros(in_channels, in_channels, self.alpha, self.alpha, self.alpha,
                            dtype=torch.cfloat)
            )
            self.weights2 = nn.Parameter(
                torch.zeros(in_channels, in_channels, self.alpha, self.alpha, self.alpha,
                            dtype=torch.cfloat)
            )
            self.weights3 = nn.Parameter(
                torch.zeros(in_channels, in_channels, self.alpha, self.alpha, self.alpha,
                            dtype=torch.cfloat)
            )
            self.weights4 = nn.Parameter(
                torch.zeros(in_channels, in_channels, self.alpha, self.alpha, self.alpha,
                            dtype=torch.cfloat)
            )
            for weight in [self.weights1, self.weights2, self.weights3, self.weights4]:
                nn.init.xavier_normal_(weight)

    def _reshape_for_fft(self, x):
        """
        Reshape from wavelet format to FFT format.

        Wavelet format: (*spatial, channels, k^n_dim)
        FFT format: (batch, channels, *spatial)
        """
        original_shape = x.shape

        if self.n_dim == 1:
            # (B, N, c, k) -> (B, c*k, N)
            batch_size, n_points, num_channels, k = x.shape
            # (B, N, c, k) -> (B, c, k, N) -> (B, c*k, N)
            x = x.permute(0, 2, 3, 1).contiguous().reshape(batch_size, num_channels * k, n_points)
        elif self.n_dim == 2:
            # (B, Nx, Ny, c, k²) -> (B, c*k², Nx, Ny)
            batch_size, nx, ny, num_channels, k_sq = x.shape
            # (B, Nx, Ny, c, k²) -> (B, c, k², Nx, Ny) -> (B, c*k², Nx, Ny)
            x = x.permute(0, 3, 4, 1, 2).contiguous().reshape(batch_size, num_channels * k_sq, nx, ny)

        elif self.n_dim == 3:
            # (B, Nx, Ny, T, c, k²) -> (B, c*k², Nx, Ny, T)
            batch_size, nx, ny, time_steps, num_channels, k_sq = x.shape
            # (B, Nx, Ny, T, c, k²) -> (B, c, k², Nx, Ny, T) -> (B, c*k², Nx, Ny, T)
            x = x.permute(0, 4, 5, 1, 2, 3).contiguous().reshape(batch_size, num_channels * k_sq, nx, ny, time_steps)

        return x, original_shape

    def _reshape_from_fft(self, x, original_shape):
        """Reshape from FFT format back to wavelet format."""
        if self.n_dim == 1:
            x = x.permute(0, 2, 1)
        elif self.n_dim == 2:
            x = x.permute(0, 2, 3, 1)
        elif self.n_dim == 3:
            x = x.permute(0, 2, 3, 4, 1)
        return x

    def forward(self, x):
        """
        Apply sparse Fourier transformation.

        Process:
        1. Reshape to (batch, channels, *spatial)
        2. Apply real FFT
        3. Multiply selected frequencies by learned weights
        4. Apply inverse FFT
        5. Reshape back to original format

        Parameters
        ----------
        x : torch.Tensor
            Input wavelet coefficients

        Returns
        -------
        torch.Tensor
            Transformed coefficients in frequency domain, same shape as input
        """
        # Reshape for FFT
        x, original_shape = self._reshape_for_fft(x)

        # Apply dimension-specific Fourier transform
        if self.n_dim == 1:
            x = self._forward_1d(x, original_shape)
        elif self.n_dim == 2:
            x = self._forward_2d(x, original_shape)
        elif self.n_dim == 3:
            x = self._forward_3d(x, original_shape)

        return x

    def _forward_1d(self, x, original_shape):
        """
        1D Fourier sparse kernel.

        For 1D real FFT, output is [0, 1, 2, ..., N//2] (only positive frequencies).
        We only use the lowest `alpha` modes.
        """
        batch_size, channels, n_points = x.shape

        # Real FFT: (B, C, N) → (B, C, N//2+1)
        x_fft = torch.fft.rfft(x, dim=-1)

        # Determine how many modes to actually use
        num_modes = min(self.alpha, n_points // 2 + 1)

        # Initialize output in frequency domain
        output_fft = torch.zeros(batch_size, channels, n_points // 2 + 1,
                                 device=x.device, dtype=torch.cfloat)

        # Apply learned weights to lowest modes only
        # Einstein summation: batch, input_channel, freq → batch, output_channel, freq
        output_fft[:, :, :num_modes] = torch.einsum(
            "bix,iox->box",
            x_fft[:, :, :num_modes],
            self.weights[:, :, :num_modes]
        )

        # Inverse FFT back to spatial domain
        x = torch.fft.irfft(output_fft, n=n_points, dim=-1)

        # Reshape back to original format
        x = self._reshape_from_fft(x, original_shape)
        x = x.view(original_shape)

        return x

    def _forward_2d(self, x, original_shape):
        """
        2D Fourier sparse kernel with correct frequency indexing.

        For 2D real FFT:
        - First dimension: Full FFT with positive and negative frequencies
          Layout: [0, 1, ..., N//2-1, -N//2, ..., -1]
        - Second dimension: Real FFT with only positive frequencies
          Layout: [0, 1, 2, ..., M//2]

        We need to handle both positive and negative frequencies in the first
        dimension carefully. The low/high index blocks must not overlap (capped at
        ``nx // 2`` per side); the library's :class:`~neuralop.layers.spectral_convolution.SpectralConv`
        uses ``fftshift`` and related conventions for the same redundancy issue.
        """
        batch_size, channels, nx, ny = x.shape

        # 2D Real FFT: (B, C, Nx, Ny) → (B, C, Nx, Ny//2+1)
        x_fft = torch.fft.rfft2(x, dim=(-2, -1))

        # Along x the layout is a full length-nx axis; low/high slices [:m] and [-m:] overlap if
        # 2*m > nx (e.g. m = nx//2+1). Cap symmetric modes at nx//2 (see also library SpectralConv).
        mx_pos, mx_neg = self._full_fft_axis_symmetric_modes(nx, self.alpha)
        num_modes_y = self._rfft_positive_modes(ny, self.alpha)

        # Initialize output in frequency domain
        output_fft = torch.zeros(batch_size, channels, nx, ny // 2 + 1,
                                 device=x.device, dtype=torch.cfloat)

        # Low x-frequencies: [0, mx_pos)
        if mx_pos > 0 and num_modes_y > 0:
            output_fft[:, :, :mx_pos, :num_modes_y] = torch.einsum(
                "bixy,ioxy->boxy",
                x_fft[:, :, :mx_pos, :num_modes_y],
                self.weights1[:, :, :mx_pos, :num_modes_y]
            )

        # High x-frequencies (negative wavenumbers in FFT order); skip if mx_neg==0 (e.g. nx<=1)
        if mx_neg > 0 and num_modes_y > 0:
            output_fft[:, :, -mx_neg:, :num_modes_y] = torch.einsum(
                "bixy,ioxy->boxy",
                x_fft[:, :, -mx_neg:, :num_modes_y],
                self.weights2[:, :, :mx_neg, :num_modes_y]
            )

        # Inverse FFT back to spatial domain
        x = torch.fft.irfft2(output_fft, s=(nx, ny), dim=(-2, -1))

        # Reshape and apply nonlinearity + projection
        x = self._reshape_from_fft(x, original_shape)
        x = F.relu(x)
        x = self.output_proj(x)
        x = x.view(original_shape)

        return x

    def _forward_3d(self, x, original_shape):
        """
        3D Fourier sparse kernel with correct frequency indexing.

        For 3D real FFT:
        - First two dimensions: Full FFT with +/- frequencies
        - Third dimension: Real FFT with only + frequencies

        We need 4 weight matrices for all quadrants:
        - weights1: (+x, +y, +z)
        - weights2: (-x, +y, +z)
        - weights3: (+x, -y, +z)
        - weights4: (-x, -y, +z)

        Along ``x`` and ``y`` we cap symmetric mode counts so ``[:m]`` and ``[-m:]`` do not overlap;
        along ``z`` only non-negative rfft bins are used.
        """
        batch_size, channels, nx, ny, nz = x.shape

        # 3D Real FFT: (B, C, Nx, Ny, Nz) → (B, C, Nx, Ny, Nz//2+1)
        x_fft = torch.fft.rfftn(x, dim=(-3, -2, -1))

        mx_pos, mx_neg = self._full_fft_axis_symmetric_modes(nx, self.alpha)
        my_pos, my_neg = self._full_fft_axis_symmetric_modes(ny, self.alpha)
        num_modes_z = self._rfft_positive_modes(nz, self.alpha)

        # Initialize output
        output_fft = torch.zeros(batch_size, channels, nx, ny, nz // 2 + 1,
                                 device=x.device, dtype=torch.cfloat)

        # Quadrant 1: (+x, +y, +z)
        if mx_pos > 0 and my_pos > 0 and num_modes_z > 0:
            output_fft[:, :, :mx_pos, :my_pos, :num_modes_z] = torch.einsum(
                "bixyz,ioxyz->boxyz",
                x_fft[:, :, :mx_pos, :my_pos, :num_modes_z],
                self.weights1[:, :, :mx_pos, :my_pos, :num_modes_z]
            )

        # Quadrant 2: (-x, +y, +z)
        if mx_neg > 0 and my_pos > 0 and num_modes_z > 0:
            output_fft[:, :, -mx_neg:, :my_pos, :num_modes_z] = torch.einsum(
                "bixyz,ioxyz->boxyz",
                x_fft[:, :, -mx_neg:, :my_pos, :num_modes_z],
                self.weights2[:, :, :mx_neg, :my_pos, :num_modes_z]
            )

        # Quadrant 3: (+x, -y, +z)
        if mx_pos > 0 and my_neg > 0 and num_modes_z > 0:
            output_fft[:, :, :mx_pos, -my_neg:, :num_modes_z] = torch.einsum(
                "bixyz,ioxyz->boxyz",
                x_fft[:, :, :mx_pos, -my_neg:, :num_modes_z],
                self.weights3[:, :, :mx_pos, :my_neg, :num_modes_z]
            )

        # Quadrant 4: (-x, -y, +z)
        if mx_neg > 0 and my_neg > 0 and num_modes_z > 0:
            output_fft[:, :, -mx_neg:, -my_neg:, :num_modes_z] = torch.einsum(
                "bixyz,ioxyz->boxyz",
                x_fft[:, :, -mx_neg:, -my_neg:, :num_modes_z],
                self.weights4[:, :, :mx_neg, :my_neg, :num_modes_z]
            )

        # Inverse FFT
        x = torch.fft.irfftn(output_fft, s=(nx, ny, nz), dim=(-3, -2, -1))

        # Reshape and apply nonlinearity + projection
        x = self._reshape_from_fft(x, original_shape)
        x = F.relu(x)
        x = self.output_proj(x)
        x = x.view(original_shape)

        return x


class MWNOBlock(nn.Module):
    """
    Multiwavelet Neural Operator Core Z-transform Layer.

    This layer is the heart of the Multiwavelet Neural Operator (MWNO) architecture.
    It works like a hierarchical image pyramid, but with learnable transformations
    at each level.

    Parameters
    ----------
    k : int, default=3
        Wavelet basis size (polynomial order + 1).
        - k=3: Cubic approximation (good balance)
        - k=4: Quartic approximation (smoother, more computation)
        Determines how many coefficients represent each wavelet basis function.

    alpha : int, default=5
        Number of Fourier modes to retain in frequency domain kernels.
        Controls how much frequency information to keep:
        - Low alpha: Faster but less expressive
        - High alpha: More expressive but computationally expensive
        Think of it as "bandwidth" of the learnable frequency filter.

    L : int, default=0
        Number of coarsest decomposition levels to skip.
        Stops decomposition early to save computation:
        - L=0: Full decomposition (finest to coarsest)
        - L=1: Stop 1 level before coarsest
        - L=2: Stop 2 levels before coarsest
        Useful when very coarse scales aren't needed.
        Must satisfy ``L < num_scales`` with ``num_scales = floor(log2(spatial_size))`` on the
        wavelet axis. :class:`~neuralop.models.mwno.MWNO` checks this once per ``forward``;
        standalone ``MWNOBlock`` use must satisfy the inequality manually.

    c : int, default=1
        Number of channels in wavelet coefficients.
        Like feature channels in CNNs:
        - c=1: Minimal capacity
        - c>1: More expressive power (parallel wavelet representations)

    base : str, default='legendre'
        Polynomial basis for wavelet construction:
        - 'legendre': Uniform weight, general purpose
        - 'chebyshev': Better for boundary-heavy problems

    n_dim : int, default=1
        Spatial dimensionality of input data:
        - 1: Temporal signals (batch, time, channels, k)
        - 2: Spatial fields (batch, height, width, channels, k²)
        - 3: Spatio-temporal (batch, height, width, time, channels, k²)

    initializer : callable or None, default=None
        Custom weight initialization for the coarsest scale transform T0.
        If None, uses PyTorch's default Kaiming initialization.

    Attributes
    ----------
    A : SparseKernelFT
        Learnable Fourier-space kernel for detail coefficients.
        Captures high-frequency patterns (edges, oscillations).

    B : SparseKernelFT or SparseKernel
        Learnable kernel for approximation coefficients.
        - 1D: Fourier space (frequency domain)
        - 2D/3D: Spatial domain (more efficient for smooth features)

    C : SparseKernel
        Learnable spatial kernel for detail coefficient skip connections.
        Provides direct path for high-frequency information.

    T0 : nn.Linear
        Linear transformation at the coarsest scale.
        Processes the most compressed representation.

    ec_s, ec_d : torch.Tensor (buffers)
        Decomposition filters:
        - ec_s: Extract approximation (smooth, low-frequency)
        - ec_d: Extract details (edges, high-frequency)

    rc_* : torch.Tensor (buffers)
        Reconstruction filters for even/odd grid positions:
        - 1D: rc_e (even), rc_o (odd)
        - 2D/3D: rc_ee, rc_eo, rc_oe, rc_oo (all combinations)

    Methods
    -------
    forward(x)
        Main processing pipeline: decompose → transform → reconstruct

    wavelet_transform(x)
        Single-level decomposition: x → (detail, approximation)
        Splits signal into two half-resolution components

    even_odd_reconstruction(x)
        Single-level reconstruction: (detail, approximation) → x
        Interleaves even/odd samples to restore full resolution

    Input Shapes
    ------------
    1D: (batch, n_points, channels, k)
        where n_points must be a power of 2
        Example: (32, 256, 1, 3) for 32 signals of 256 time steps

    2D: (batch, height, width, channels, k²)
        where height, width must be powers of 2
        Example: (16, 64, 64, 1, 9) for 16 images of 64×64 with k=3

    3D: (batch, height, width, time, channels, k²)
        where height, width must be powers of 2 (time can be arbitrary)
        Example: (8, 32, 32, 20, 1, 9) for spatio-temporal field

    Output Shape
    ------------
    Same as input shape - the transformation is resolution-preserving.

    Examples
    --------
    >>> # 1D signal processing
    >>> layer_1d = MWNOBlock(k=3, alpha=5, L=0, c=1, n_dim=1)
    >>> signal = torch.randn(32, 128, 1, 3)
    >>> output = layer_1d(signal)
    >>> assert output.shape == signal.shape

    >>> # 2D image processing
    >>> layer_2d = MWNOBlock(k=4, alpha=8, L=1, c=2, n_dim=2)
    >>> image = torch.randn(16, 64, 64, 2, 16)
    >>> output = layer_2d(image)
    >>> assert output.shape == image.shape

    >>> # 3D spatio-temporal processing
    >>> layer_3d = MWNOBlock(k=3, alpha=5, L=0, c=1, n_dim=3)
    >>> field = torch.randn(8, 32, 32, 20, 1, 9)
    >>> output = layer_3d(field)
    >>> assert output.shape == field.shape

    Architecture:
    ------------
    Input (fine scale)
        ↓ wavelet_transform
    [Detail₀, Approximation₀]
        ↓ A(Detail₀) + B(Approx₀) → processed Detail₀
        ↓ C(Detail₀) → skip connection
        ↓ wavelet_transform on Approximation₀
    [Detail₁, Approximation₁]
        ↓ A(Detail₁) + B(Approx₁) → processed Detail₁
        ↓ C(Detail₁) → skip connection
        ↓ ... (repeat for multiple scales)
    [Detail_n, Approximation_n (coarsest)]
        ↓ T0(Approximation_n) → process coarsest scale
        ↓ reconstruction (combine all scales)
    Output (fine scale)

    Dimensionality Support:
    ----------------------
    - 1D: Time series, signals → wavelet along time axis
    - 2D: Images, spatial fields → wavelets along both x, y
    - 3D: Spatio-temporal data → wavelets along x, y only (preserves time/feature axis)

    Notes
    -----
    - Input spatial dimensions must be powers of 2 for wavelet decomposition
    - The number of decomposition levels is automatically determined: log₂(spatial_size) - L
    - Wavelet filters are precomputed and frozen (not learned)
    - Only the kernels A, B, C and transform T0 are learned during training
    - For 3D inputs, only the first two dimensions (Nx, Ny) are decomposed;
      the third dimension (T) is preserved as-is

    References
    ----------
    Based on "Multiwavelet-based Operator Learning for Differential Equations"
    by Gupta et al. (2021)
    """

    def __init__(self, k=3, alpha=5, L=0, c=1, base='legendre', n_dim=1, initializer=None, **kwargs):
        super().__init__()

        self.k = k  # Wavelet basis size
        self.L = L  # Levels to skip at coarsest scale
        self.n_dim = n_dim  # Spatial dimensionality

        # Get wavelet filter banks from utility
        h_0, h_1, g_0, g_1, phi_0, phi_1 = WaveletUtils.get_filter(base, k)

        # Compose decomposition and reconstruction filters
        # These transform between scales in the wavelet pyramid
        h_0_reconstructed = h_0 @ phi_0  # Low-pass decomposition → reconstruction
        g_0_reconstructed = g_0 @ phi_0  # High-pass decomposition → reconstruction
        h_1_reconstructed = h_1 @ phi_1  # Low-pass odd samples
        g_1_reconstructed = g_1 @ phi_1  # High-pass odd samples

        # Clean up numerical noise
        for matrix in [h_0_reconstructed, h_1_reconstructed, g_0_reconstructed, g_1_reconstructed]:
            matrix[np.abs(matrix) < 1e-8] = 0

        # Initialize learnable kernels
        # A: Fourier kernel for detail (high-frequency) coefficients
        self.A = SparseKernelFT(k, alpha, c, n_dim)

        # B: Kernel for approximation (low-frequency) coefficients
        # Use Fourier for 1D (efficient), spatial for 2D/3D (smooth features)
        self.B = SparseKernelFT(k, alpha, c, n_dim) if n_dim == 1 else SparseKernel(k, alpha, c, n_dim)

        # C: Spatial kernel for detail skip connections
        self.C = SparseKernel(k, alpha, c, n_dim)

        # T0: Linear transform at coarsest scale
        if n_dim == 1:
            self.T0 = nn.Linear(k, k)
        else:
            # For 2D/3D, need to handle all channels and basis functions
            self.T0 = nn.Linear(c * k ** 2, c * k ** 2)

        # Apply custom initialization if provided
        if initializer is not None:
            initializer(self.T0.weight)

        # Register wavelet filters as non-trainable buffers
        self._register_wavelet_filters(h_0, h_1, g_0, g_1,
                                       h_0_reconstructed, h_1_reconstructed,
                                       g_0_reconstructed, g_1_reconstructed)

    def _register_wavelet_filters(self, h_0, h_1, g_0, g_1, h_0r, h_1r, g_0r, g_1r):
        """
        Parameters
        ----------
        h_0, h_1 : ndarray
            Low-pass decomposition filters (even/odd)
        g_0, g_1 : ndarray
            High-pass decomposition filters (even/odd)
        h_0r, h_1r : ndarray
            Low-pass reconstruction filters (even/odd)
        g_0r, g_1r : ndarray
            High-pass reconstruction filters (even/odd)

        Register wavelet decomposition and reconstruction filters as buffers.

        Buffers are stored with the model but not trained. They're used for:
        - ec_s, ec_d: Decomposition (analysis) filters
        - rc_*: Reconstruction (synthesis) filters

        The filters are constructed using Kronecker products for 2D/3D to handle
        separable transforms (apply filter to each dimension independently).

        """
        if self.n_dim == 1:
            # 1D: Simple concatenation of even and odd filters
            # ec_s: [H0^T; H1^T] extracts approximation (smooth)
            # ec_d: [G0^T; G1^T] extracts detail (edges)
            self.register_buffer('ec_s', torch.Tensor(np.concatenate((h_0.T, h_1.T), axis=0)))
            self.register_buffer('ec_d', torch.Tensor(np.concatenate((g_0.T, g_1.T), axis=0)))

            # Reconstruction: recombine even and odd to restore full resolution
            self.register_buffer('rc_e', torch.Tensor(np.concatenate((h_0r, g_0r), axis=0)))
            self.register_buffer('rc_o', torch.Tensor(np.concatenate((h_1r, g_1r), axis=0)))

        elif self.n_dim >= 2:
            # 2D/3D: Use Kronecker products for separable 2D transforms
            # Kronecker product creates 2D filter from 1D filters:
            # kron(A, B) applies A to rows and B to columns

            # Decomposition filters for all 4 combinations: (even,even), (even,odd), (odd,even), (odd,odd)
            self.register_buffer('ec_s', torch.Tensor(
                np.concatenate((np.kron(h_0, h_0).T,  # both dimensions even
                                np.kron(h_0, h_1).T,  # x even, y odd
                                np.kron(h_1, h_0).T,  # x odd, y even
                                np.kron(h_1, h_1).T), axis=0)))  # both dimensions odd

            self.register_buffer('ec_d', torch.Tensor(
                np.concatenate((np.kron(g_0, g_0).T,
                                np.kron(g_0, g_1).T,
                                np.kron(g_1, g_0).T,
                                np.kron(g_1, g_1).T), axis=0)))

            # Reconstruction filters for all 4 parity combinations
            self.register_buffer('rc_ee',
                                 torch.Tensor(np.concatenate((np.kron(h_0r, h_0r), np.kron(g_0r, g_0r)), axis=0)))
            self.register_buffer('rc_eo',
                                 torch.Tensor(np.concatenate((np.kron(h_0r, h_1r), np.kron(g_0r, g_1r)), axis=0)))
            self.register_buffer('rc_oe',
                                 torch.Tensor(np.concatenate((np.kron(h_1r, h_0r), np.kron(g_1r, g_0r)), axis=0)))
            self.register_buffer('rc_oo',
                                 torch.Tensor(np.concatenate((np.kron(h_1r, h_1r), np.kron(g_1r, g_1r)), axis=0)))

    def wavelet_transform(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            Input at current scale
            - 1D: (batch, n_points, channels, k)
            - 2D: (batch, height, width, channels, k²)
            - 3D: (batch, height, width, time, channels, k²)

        Returns
        -------
        detail : torch.Tensor
            High-frequency detail coefficients (same size as approx)
            Contains edges, textures, rapid variations

        approximation : torch.Tensor
            Low-frequency approximation coefficients
            Contains smooth, slowly-varying features

        Perform one level of wavelet decomposition.

        This is the "analysis" step that splits the input into two components:
        1. Approximation (smooth, low-frequency): downsampled by 2
        2. Detail (edges, high-frequency): also downsampled by 2

        The decomposition works by:
        - Separating even and odd samples (downsampling)
        - Applying wavelet filters to extract smooth vs. detail information

        Think of it like creating a Gaussian pyramid (blur) and a Laplacian
        pyramid (edges) simultaneously, but with learnable/adaptive filters.

        Notes
        -----
        Both outputs have half the spatial resolution of input (due to downsampling).
        For 3D, only Nx and Ny are downsampled; time dimension is preserved.
        """
        if self.n_dim == 1:
            # 1D: Separate even [0::2] and odd [1::2] samples
            # Concatenate along last dimension for filter application
            downsampled = torch.cat([x[:, ::2, :, :],  # even samples
                                     x[:, 1::2, :, :]], -1)  # odd samples

        elif self.n_dim == 2:
            # 2D: Separate all 4 combinations of even/odd in x and y
            downsampled = torch.cat([x[:, ::2, ::2, :, :],  # even-even
                                     x[:, ::2, 1::2, :, :],  # even-odd (x even, y odd)
                                     x[:, 1::2, ::2, :, :],  # odd-even (x odd, y even)
                                     x[:, 1::2, 1::2, :, :]], -1)  # odd-odd

        elif self.n_dim == 3:
            # 3D: Only downsample spatial dims (Nx, Ny), preserve temporal dim (T)
            # x shape: (batch, Nx, Ny, T, channels, k²)
            downsampled = torch.cat([x[:, ::2, ::2, :, :, :],  # even-even
                                     x[:, ::2, 1::2, :, :, :],  # even-odd
                                     x[:, 1::2, ::2, :, :, :],  # odd-even
                                     x[:, 1::2, 1::2, :, :, :]], -1)  # odd-odd

        # Apply wavelet filters to extract detail and approximation
        detail = torch.matmul(downsampled, self.ec_d)  # High-pass filter
        approximation = torch.matmul(downsampled, self.ec_s)  # Low-pass filter

        return detail, approximation

    def even_odd_reconstruction(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            Combined coefficients [approximation; detail] at current scale
            Last dimension has size 2*k (1D) or 2*k² (2D/3D)
            - 1D: (batch, n_points//2, channels, 2*k)
            - 2D: (batch, height//2, width//2, channels, 2*k²)
            - 3D: (batch, height//2, width//2, time, channels, 2*k²)

        Returns
        -------
        torch.Tensor
            Reconstructed signal at double the spatial resolution
            - 1D: (batch, n_points, channels, k)
            - 2D: (batch, height, width, channels, k²)
            - 3D: (batch, height, width, time, channels, k²)

        Perform one level of wavelet reconstruction.

        This is the "synthesis" step that combines detail and approximation
        coefficients back into the full-resolution signal. It's the inverse
        of wavelet_transform().

        The reconstruction:
        1. Applies reconstruction filters to detail and approximation
        2. Upsamples by interleaving even and odd positions
        3. Produces output at double the input resolution

        Think of it like "unzipping" the compressed representation back to
        full resolution, intelligently placing values at even and odd positions.

        Notes
        -----
        The even/odd interleaving ensures smooth reconstruction without
        checkerboard artifacts. For 3D, time dimension remains unchanged.
        """
        if self.n_dim == 1:
            batch_size, n_coarse, num_channels, input_channels = x.shape
            assert input_channels == 2 * self.k, \
                f"Expected {2 * self.k} input channels, got {input_channels}"

            # Apply reconstruction filters
            x_even = torch.matmul(x, self.rc_e)  # Reconstruct even positions
            x_odd = torch.matmul(x, self.rc_o)  # Reconstruct odd positions

            # Interleave even and odd to restore full resolution
            result = torch.zeros(batch_size, n_coarse * 2, num_channels, self.k, device=x.device)
            result[:, ::2, :, :] = x_even  # Place at even indices
            result[:, 1::2, :, :] = x_odd  # Place at odd indices
            return result

        elif self.n_dim == 2:
            batch_size, nx_coarse, ny_coarse, num_channels, input_channels = x.shape
            assert input_channels == 2 * self.k ** 2, \
                f"Expected {2 * self.k ** 2} input channels, got {input_channels}"

            # Apply reconstruction filters for all 4 parity combinations
            x_even_even = torch.matmul(x, self.rc_ee)  # Even in both x and y
            x_even_odd = torch.matmul(x, self.rc_eo)  # Even in x, odd in y
            x_odd_even = torch.matmul(x, self.rc_oe)  # Odd in x, even in y
            x_odd_odd = torch.matmul(x, self.rc_oo)  # Odd in both x and y

            # Interleave in 2D checkerboard pattern
            result = torch.zeros(batch_size, nx_coarse * 2, ny_coarse * 2,
                                 num_channels, self.k ** 2, device=x.device)
            result[:, ::2, ::2, :, :] = x_even_even
            result[:, ::2, 1::2, :, :] = x_even_odd
            result[:, 1::2, ::2, :, :] = x_odd_even
            result[:, 1::2, 1::2, :, :] = x_odd_odd
            return result

        elif self.n_dim == 3:
            batch_size, nx_coarse, ny_coarse, time_steps, num_channels, input_channels = x.shape
            assert input_channels == 2 * self.k ** 2, \
                f"Expected {2 * self.k ** 2} input channels, got {input_channels}"

            # Apply reconstruction filters (same as 2D)
            x_even_even = torch.matmul(x, self.rc_ee)
            x_even_odd = torch.matmul(x, self.rc_eo)
            x_odd_even = torch.matmul(x, self.rc_oe)
            x_odd_odd = torch.matmul(x, self.rc_oo)

            # Interleave in spatial dimensions, preserve time dimension
            result = torch.zeros(batch_size, nx_coarse * 2, ny_coarse * 2, time_steps,
                                 num_channels, self.k ** 2, device=x.device)
            result[:, ::2, ::2, :, :, :] = x_even_even
            result[:, ::2, 1::2, :, :, :] = x_even_odd
            result[:, 1::2, ::2, :, :, :] = x_odd_even
            result[:, 1::2, 1::2, :, :, :] = x_odd_odd
            return result

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor with spatial dimensions as powers of 2
            - 1D: (batch, n_points, channels, k)
            - 2D: (batch, height, width, channels, k²)
            - 3D: (batch, height, width, time, channels, k²)

        Returns
        -------
        torch.Tensor
            Transformed output with same shape as input

                Apply multiwavelet transform: decompose → transform → reconstruct.

        This is the main processing pipeline that:
        1. Decomposes input into multiple scales (wavelet pyramid)
        2. Applies learnable transformations at each scale
        3. Reconstructs output from all scales

        Processing flow:
        ---------------
        Decomposition phase (coarse-to-fine):
            for each scale level:
                Split into (detail, approximation)
                Transform detail: A(detail) + B(approximation)
                Skip connection: C(detail)
                Continue with approximation at next coarser level

            Process coarsest scale with T0

        Reconstruction phase (fine-to-coarse):
            for each scale level:
                Add skip connection
                Combine with detail from decomposition
                Reconstruct to finer scale

        Notes
        -----
        - Spatial dimensions must be powers of 2 for clean decomposition
        - Number of scales = log₂(spatial_size) - L
        - Skip connections preserve high-frequency information
        - Uses TorchScript type annotations for compilation optimization
        """
        # Parse input shape and determine number of decomposition scales
        if self.n_dim == 1:
            batch_size, n_points, num_channels, input_channels = x.shape
            num_scales = math.floor(np.log2(n_points))
        elif self.n_dim == 2:
            batch_size, nx, ny, num_channels, input_channels = x.shape
            num_scales = math.floor(np.log2(nx))
        elif self.n_dim == 3:
            batch_size, nx, ny, time_steps, num_channels, input_channels = x.shape
            num_scales = math.floor(np.log2(nx))

        # Storage for detail coefficients and skip connections at each scale
        # Use TorchScript type annotations for JIT compilation
        detail_coeffs = torch.jit.annotate(List[Tensor], [])  # Transformed details: A(d) + B(s)
        skip_connections = torch.jit.annotate(List[Tensor], [])  # Skip paths: C(d)

        # ========== Decomposition Phase ==========
        # Progressively decompose into coarser scales
        for scale_idx in range(num_scales - self.L):
            # Split current scale into detail (high-freq) and approximation (low-freq)
            detail, approximation = self.wavelet_transform(x)

            # Apply learnable transformations
            # detail pathway: Fourier kernel on high-frequency + coupling with low-frequency
            detail_transformed = self.A(detail) + self.B(approximation)

            # Skip connection: spatial kernel on high-frequency (preserves edges)
            skip = self.C(detail)

            # Store for reconstruction phase
            detail_coeffs.append(detail_transformed)
            skip_connections.append(skip)

            # Continue decomposing the approximation (smoother component)
            x = approximation

        # ========== Coarsest Scale Processing ==========
        # At the coarsest scale, apply learned linear transformation
        if self.n_dim == 1:
            # 1D: Direct linear transform
            x = self.T0(x)
        elif self.n_dim == 2:
            # 2D: Flatten spatial dimensions, apply transform, reshape
            # x shape: (batch, 2^L, 2^L, channels, k²)
            batch_size, nx_coarse, ny_coarse, num_channels, input_channels = x.shape
            x_flattened = x.reshape(batch_size * nx_coarse * ny_coarse, num_channels * input_channels)
            x_flattened = self.T0(x_flattened)
            x = x_flattened.reshape(batch_size, nx_coarse, ny_coarse, num_channels, input_channels)
        elif self.n_dim == 3:
            # 3D: Flatten spatial AND temporal, apply transform, reshape
            # x shape: (batch, 2^L, 2^L, time, channels, k²)
            batch_size, nx_coarse, ny_coarse, time_steps, num_channels, input_channels = x.shape
            x_flattened = x.reshape(batch_size * nx_coarse * ny_coarse * time_steps,
                                    num_channels * input_channels)
            x_flattened = self.T0(x_flattened)
            x = x_flattened.reshape(batch_size, nx_coarse, ny_coarse, time_steps,
                                    num_channels, input_channels)

        # ========== Reconstruction Phase ==========
        # Progressively reconstruct from coarse to fine
        # Iterate in reverse order through the scales
        for scale_idx in range(num_scales - 1 - self.L, -1, -1):
            # Add skip connection (preserves high-frequency from decomposition)
            x = x + skip_connections[scale_idx]

            # Concatenate with transformed detail coefficients
            # This combines: approximation + detail = full signal at this scale
            x = torch.cat((x, detail_coeffs[scale_idx]), dim=-1)

            # Upsample to next finer scale using wavelet reconstruction
            x = self.even_odd_reconstruction(x)

        return x
