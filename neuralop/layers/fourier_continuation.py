import torch
import torch.nn as nn
import numpy as np
from numpy.polynomial.legendre import Legendre
import warnings
from pathlib import Path
import tensorly as tl


class FourierContinuation(nn.Module):
    """Base class for Fourier Continuation implementations.

    Fourier Continuation is a technique for extending non-periodic functions to periodic
    ones on larger domains, enabling the use of spectral methods for differentiation
    and solving PDEs. This base class provides the common interface and methods for
    different Fourier Continuation approaches (Legendre polynomials, Gram matrices, etc.).

    This allows spectral methods such as spectral differentiation to be applied to
    non-periodic functions by working on the extended periodic domain.
    """

    def __init__(self, d=5, n_additional_pts=50):
        """
        Initialize FourierContinuation with specified parameters.

        Parameters
        ----------
        d : int, optional
            Number of matching points on the left and right boundaries, by default 5
        n_additional_pts : int, optional
            Number of additional points to add for continuation, by default 50
        """
        super().__init__()

        self.d = d
        self.n_additional_pts = n_additional_pts

    def extend(self, x, dim):
        """
        Extend tensor along specified dimensions using Fourier Continuation.

        This method extends non-periodic functions to periodic ones by adding
        continuation points on both sides of the specified dimensions using
        TensorLy's multi_mode_dot for efficient multi-dimensional extension.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor to extend. Must have at least 2*d points along each
            dimension to be extended (for boundary value extraction).
        dim : int or tuple of ints,
            Dimensions along which to extend:
            - If int: extend along the last dim axes (e.g. dim=2 extends along the last 2 axes)
            - If tuple: extend along the specified axes (supports negative indexing)

        Returns
        -------
        torch.Tensor
            Extended tensor with additional points added along specified dimensions.
            Each extended dimension will have n_additional_pts more points than the input.
        """

        # Convert input dimension(s) to list of axes to extend along:
        if isinstance(dim, int):
            # If dim is an integer n, extend along the last n dimensions
            axes = list(range(-dim, 0))
        else:
            # If dim is a tuple, extend along those specific dimensions
            axes = list(dim)

        # Convert negative axes to positive indices
        axes = [a if a >= 0 else x.ndim + a for a in axes]

        # Create extension matrices for each axis
        # Each matrix maps input tensor along one axis to extended tensor
        extension_matrices = []  # List of torch tensors, each shape: (extended_size, original_size)
        modes = []  # List of axis indices to extend along

        for axis in axes:
            # Get the extension matrix for this axis
            ext_mat = self._get_extension_matrix_for_axis(x, axis)
            extension_matrices.append(ext_mat)
            modes.append(axis)

        # Use TensorLy's multi_mode_dot to apply all extensions simultaneously
        # Input: x (original shape), matrices (list of extension matrices), modes (axis indices)
        # Output: extended tensor with shape modified along specified axes
        return tl.tenalg.multi_mode_dot(x, extension_matrices, modes=modes)

    def _get_extension_matrix_for_axis(self, x, axis):
        """
        Get the extension matrix for a specific axis using Fourier continuation.

        This method creates a matrix that maps the input tensor along one axis
        to the extended tensor using the Fourier continuation approach.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        axis : int
            Axis for which to get the extension matrix

        Returns
        -------
        torch.Tensor
            Extension matrix for the specified axis
        """

        axis_size = x.shape[axis]  # Original size along the axis to extend
        extended_size = axis_size + self.n_additional_pts  # Final size after extension
        c = self.n_additional_pts // 2  # Number of continuation points on each side

        # Create the extension matrix, which maps the input to the extended output
        # Shape: (extended_size, axis_size) where extended_size = axis_size + n_additional_pts
        ext_mat = torch.zeros((extended_size, axis_size), dtype=x.dtype, device=x.device)
        
        # Place identity matrix in the middle (original values)
        ext_mat[c:c+axis_size, :] = torch.eye(axis_size, dtype=x.dtype, device=x.device)
        
        # Get the extension matrix for boundary points
        ext_mat_boundary = self.ext_mat.to(dtype=x.dtype, device=x.device)

        # Fill in the continuation regions
        if c > 0:
            # The extension matrix maps [right_bnd, left_bnd] to continuation values
            # where ext_mat_boundary has shape (n_additional_pts, 2*d)
            # and maps [right_bnd, left_bnd] to [left_cont, right_cont]

            # Left continuation: use the last c rows of the extension matrix
            # These map [right_bnd, left_bnd] to left continuation values
            ext_mat[:c, :self.d] = ext_mat_boundary[-c:, self.d:]  # right_bnd -> left_cont
            ext_mat[:c, axis_size-self.d:] = ext_mat_boundary[-c:, :self.d]  # left_bnd -> left_cont
            
            # Right continuation: use the first c rows of the extension matrix  
            # These map [right_bnd, left_bnd] to right continuation values
            ext_mat[-c:, :self.d] = ext_mat_boundary[:c, self.d:]  # right_bnd -> right_cont
            ext_mat[-c:, axis_size-self.d:] = ext_mat_boundary[:c, :self.d]  # left_bnd -> right_cont
        
        return ext_mat

    def forward(self, x, dim):
        """
        Forward pass that calls the extend method.

        This method extends non-periodic functions to periodic ones by adding
        continuation points on both sides of the specified dimensions.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor to extend. Must have at least 2*d points along each
            dimension to be extended (for boundary value extraction).
        dim : int or tuple of ints,
            Dimensions along which to extend:
            - If int: extend along the last dim axes (e.g. dim=2 extends along the last 2 axes)
            - If tuple: extend along the specified axes (supports negative indexing)

        Returns
        -------
        torch.Tensor
            Extended tensor with additional points added along specified dimensions.
            Each extended dimension will have n_additional_pts more points than the input.
        """
        return self.extend(x, dim)

    def restrict(self, x, dim):
        """
        Remove Fourier continuation extension points to restore original domain size.

        This method reverses the extension process by removing the continuation points
        that were added during Fourier continuation.

        It removes n_additional_pts//2 points from each side of the specified dimensions.

        Parameters
        ----------
        x : torch.Tensor
            Extended tensor from Fourier continuation. Must have been extended using
            the same n_additional_pts parameter as this instance.
        dim : int or tuple of ints
            Dimensions along which to restrict:
            - If int: restrict along the last dim axes (e.g. dim=2 restricts along the last 2 axes)
            - If tuple: restrict along the specified axes (supports negative indexing)

        Returns
        -------
        torch.Tensor
            Tensor with original domain size. Each restricted dimension will have
            n_additional_pts fewer points than the input (n_additional_pts//2 removed
            from each side).
        """

        # Convert input dimension(s) to list of axes to restrict along:
        if isinstance(dim, int):
            # If dim is an integer n, restrict along the last n dimensions
            axes = list(range(-dim, 0))
        else:
            # If dim is a tuple, restrict along those specific dimensions
            axes = list(dim)

        # Convert negative axes to positive for easier handling
        axes = [a if a >= 0 else x.ndim + a for a in axes]

        # Create slices to restrict along each axis
        c = self.n_additional_pts // 2  # Number of points to remove from each side
        slices = [slice(None)] * x.ndim

        for axis in axes:
            # For each axis to restrict, remove c points from each side
            slices[axis] = slice(c, -c)

        # Return restricted tensor with reduced size along specified axes
        return x[tuple(slices)]


class FCLegendre(FourierContinuation):
    """Fourier Continuation using Legendre polynomials.

    This class implements Fourier Continuation using Legendre polynomial basis functions
    to extend non-periodic functions to periodic ones on larger domains. The method works
    by fitting Legendre polynomials to boundary values and using them to compute
    continuation values that make the extended function periodic.

    Legendre polynomials P_n(x) are orthogonal polynomials with weight w=1 on [-1, 1]:
        ∫_{-1}^{1} P_j(x) P_k(x) dx = 2/(2j+1) δ_{jk}

    The extension process:
    1. Extract d boundary points from each end of the input signal
    2. Fit a linear combination of Legendre polynomials to these boundary values
    3. Use the fitted polynomials to compute continuation values
    4. Concatenate the continuation with the original signal

    """

    def __init__(self, d=5, n_additional_pts=50, rcond=1e-15):
        """
        Initialize FCLegendre with specified parameters.

        Parameters
        ----------
        d : int
            Number of matching points on the left and right boundaries
            Related to the degree of the Legendre polynomial basis used for extension (degree 2d-1).
            By default 6
        n_additional_pts : int
            Number of additional points to add for continuation, by default 50
            Lower values of d typically require more n_additional_pts points
        rcond : float, optional
            Cutoff for small singular values for the pseudoinverse of the extension matrix
            Singular values less than or equal to rcond * largest_singular_value are set to zero
            This can be adjusted for numerical stability, especially for larger values of d.
            A smaller value can lead to more stable results, but may also introduce numerical errors.
            By default 1e-15 (same as numpy default for np.linalg.pinv)
        """
        super().__init__(d, n_additional_pts)

        self.rcond = rcond
        self.compute_extension_matrix()

    def compute_extension_matrix(self):
        """
        Compute the extension matrix using Legendre polynomials.

        This method constructs the matrix that maps boundary values to continuation
        values using Legendre polynomial basis functions.
        """

        total_points = 2 * self.d + self.n_additional_pts

        # Use [-1,1] interval where Legendre polynomials are orthogonal
        a, b = -1.0, 1.0

        # Generate uniform grid on [-1,1] with total_points points
        h = (b - a) / (total_points - 1)
        full_grid = a + h * np.arange(total_points, dtype=np.float64)

        # Extract points for fitting (d points from each end)
        fit_grid = np.concatenate((full_grid[0 : self.d], full_grid[-self.d :]), 0)

        # Extract points for extension (middle points)
        extension_grid = full_grid[self.d : -self.d]

        # Construct normalized Legendre polynomials
        # numpy.polynomial.legendre.Legendre are orthogonal but not normalized on [-1,1]:
        #    ∫_{-1}^{1} P_j(x) P_k(x) dx = 2/(2j+1) δ_{jk}
        # Normalization can improve numerical stability for larger values of d
        I = np.eye(2 * self.d, dtype=np.float64)
        polynomials = [
            np.sqrt((2 * j + 1) / 2) * Legendre(I[j, :]) for j in range(2 * self.d)
        ]

        # Evaluate normalized polynomials on the fit and extension grids
        X = np.stack([P(fit_grid) for P in polynomials], axis=1)  # Fit grid evaluations
        Q = np.stack([P(extension_grid) for P in polynomials], axis=1)  # Extension grid evaluations

        # Compute extension matrix using pseudoinverse
        ext_mat = Q @ np.linalg.pinv(X, rcond=self.rcond)

        # Register matrices as persistent buffers for the module
        self.register_buffer("ext_mat", torch.from_numpy(ext_mat))
        self.register_buffer("ext_mat_T", self.ext_mat.T.clone())

        return self.ext_mat


class FCGram(FourierContinuation):
    """Fourier Continuation using Gram matrices.

    This class implements the FC-Gram algorithm described in Section 3.1 of:
    Amlani, F., & Bruno, O. P. (2016). An FC-based spectral solver for
    elastodynamic problems in general three-dimensional domains.
    Journal of Computational Physics, 307, 333-354.

    The algorithm uses pre-computed Gram matrices (ArQr, AlQl) to perform
    Fourier Continuation of discretized functions.

    If the required pre-computed FCGram matrices are not available, they
    can be computed using the MATLAB code from the FC_PINO repository
    https://github.com/neuraloperator/FC_PINO/tree/main/FC_Gram_Construction
    and then saved to the appropriate matrices_path directory in .npz format.

    The matrices are:
    - ArQr: Right boundary continuation matrix
    - AlQl: Left boundary continuation matrix

    The matrices are saved in the .npz format.

    """

    def __init__(self, d=5, n_additional_pts=50, matrices_path=None):
        """
        Initialize FCGram with specified parameters.

        Parameters
        ----------
        d : int
            Number of matching points on the left and right boundaries
            Degree of the Gram polynomial d
            Typically between 3 and 8 (precomputed matrices available for d in {2,3,4,5,6,7,8}).
            d=3,4,5,6 are typically good choices, by default 6.
        n_additional_pts : int
            Number of continuation points. By default 50.
            Adds n_additional_pts//2 points on each side of the input signal.
            Precomputed matrices available only for n_additional_pts = 50.
            Unlike for FCLegendre, it is usually not necessary to change this parameter.
        matrices_path : str or Path, optional
            Path to directory containing FCGram matrices.
            If None, uses the directory containing this file, by default None
        """
        super().__init__(d, n_additional_pts)

        if self.n_additional_pts % 2 == 1:
            warnings.warn("n_additional_pts must be even, rounding down.", UserWarning)
            self.n_additional_pts -= 1

        self.c = int(self.n_additional_pts // 2)

        if matrices_path is None:
            self.matrices_path = Path(__file__).parent / "fcgram_matrices"
        else:
            self.matrices_path = Path(matrices_path)

        # Load pre-computed FCGram matrices
        self.load_matrices()

    def _get_extension_matrix_for_axis(self, x, axis):
        """
        Get the extension matrix for a specific axis using FC-Gram algorithm.

        This method creates a matrix that maps the input tensor along one axis
        to the extended tensor using pre-computed Gram matrices.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        axis : int
            Axis for which to get the extension matrix

        Returns
        -------
        torch.Tensor
            Extension matrix for the specified axis
        """
        axis_size = x.shape[axis]
        extended_size = axis_size + self.n_additional_pts
        c = self.n_additional_pts // 2

        # Create the extension matrix
        ext_mat = torch.zeros((extended_size, axis_size), dtype=x.dtype, device=x.device)
        
        # Place identity matrix in the middle (original values)
        ext_mat[c:c+axis_size, :] = torch.eye(axis_size, dtype=x.dtype, device=x.device)
        
        # Get pre-computed Gram matrices
        AlQl = self.AlQl.to(dtype=x.dtype, device=x.device)
        ArQr = self.ArQr.to(dtype=x.dtype, device=x.device)

        # Fill in the continuation regions
        if c > 0:
            # Left continuation: map left boundary to left continuation
            ext_mat[:c, : self.d] = AlQl[:c, :]

            # Right continuation: map right boundary to right continuation
            ext_mat[-c:, axis_size - self.d :] = ArQr[:c, :]

        return ext_mat

    def load_matrices(self):
        """
        Load the pre-computed FCGram matrices from .npz files.

        This method loads the pre-computed Gram matrices required for the FC-Gram
        algorithm. The matrices are stored in .npz format and contain the optimized
        continuation matrices for the specified (d, c) parameter combination.

        The loaded matrices are:
        - ArQr: Right boundary continuation matrix
        - AlQl: Left boundary continuation matrix

        These matrices are registered as PyTorch buffers so they are automatically
        moved to the correct device (CPU/GPU) when the module is moved.

        Raises
        ------
        FileNotFoundError
            If the required .npz file is not found in the matrices_path directory.
            The file should be named 'FCGram_data_d{d}_c{c}.npz' where c = n_additional_pts // 2.

        Note
        ----
        If the required matrices are not available, they can be computed using the
        MATLAB code from the FC_PINO repository:
        https://github.com/neuraloperator/FC_PINO/tree/main/FC_Gram_Construction

        The computed matrices should be saved in .npz format in the matrices_path
        directory with the correct naming convention.
        """
        filepath = self.matrices_path / f"FCGram_data_d{self.d}_c{self.c}.npz"

        if not filepath.exists():
            raise FileNotFoundError(
                f"FCGram matrices not found at {filepath}. \n"
                f"Please ensure the .npz file exists with d={self.d}, c={self.c}."
            )

        # Load matrices from .npz file
        npz_data = np.load(str(filepath))

        # Extract matrices, convert to torch tensors, and register as buffers so they are moved to GPU
        self.register_buffer("ArQr", torch.from_numpy(npz_data["ArQr"]))
        self.register_buffer("AlQl", torch.from_numpy(npz_data["AlQl"]))
