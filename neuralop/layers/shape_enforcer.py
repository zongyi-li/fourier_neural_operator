import torch
import torch.nn.functional as F
from typing import Optional, Tuple, List


class ShapeEnforcer(torch.nn.Module):
    """
    A PyTorch module that enforces a specific output shape on the tensor
    by cropping or padding the spatial/temporal dimensions.

    This class is useful in scenarios where you need consistent output shapes
    after operations like spectral or wavelet transforms, where dimensions might
    not match the desired size naturally.

    Parameters
    ----------
    start_dim : int, optional
        The dimension index at which to start enforcing the shape.
        For example, with typical 2D data `[B, C, H, W]`, `start_dim=2`.
        For 1D data `[B, C, L]`, `start_dim=2`. Default is 2.

    Methods
    -------
    forward(x: torch.Tensor, output_shape: Optional[List[int]] = None) -> torch.Tensor
        Enforces the desired shape on the input tensor.
    """

    def __init__(self, start_dim: int = 2):
        """
        Initialize the ShapeEnforcer module.

        Parameters
        ----------
        start_dim : int, optional
            The dimension index at which to start enforcing the shape.
            Default is 2.
        """
        super(ShapeEnforcer, self).__init__()
        self.start_dim = start_dim

    def forward(
        self, x: torch.Tensor, output_shape: Optional[List[int]] = None
    ) -> torch.Tensor:
        """
        Forward method to enforce the desired shape on the input tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape `[B, C, D1, D2, ..., DN]`.
        output_shape : List[int], optional
            The desired shape of the tensor's spatial/temporal dimensions,
            e.g., `[D_out1, D_out2, ...]`. If None, the tensor is returned unchanged.

        Returns
        -------
        torch.Tensor
            Tensor with shapes from `self.start_dim` onward forced to match `output_shape`.
        """
        if output_shape is None:
            return x  # No shape enforcement required

        # -- 1) Crop if needed --
        # Build slicing for each dimension
        slices = [slice(None)] * self.start_dim  # keep [B, C, ...] as is
        for i, size_desired in enumerate(output_shape):
            dim_idx = self.start_dim + i
            # Skip if dim_idx is out of bounds
            if dim_idx >= len(x.shape):
                break

            current_size = x.shape[dim_idx]
            # If current is bigger, crop
            if current_size > size_desired:
                slices.append(slice(0, size_desired))
            else:
                slices.append(slice(0, current_size))

        # Add remaining dimensions as-is
        for i in range(self.start_dim + len(output_shape), len(x.shape)):
            slices.append(slice(None))

        x = x[tuple(slices)]

        # -- 2) Pad if needed --
        # Calculate which dimensions need padding
        pad_list = []
        for i in range(len(x.shape) - 1, self.start_dim - 1, -1):
            # If this dimension is within our output_shape range
            if self.start_dim <= i < self.start_dim + len(output_shape):
                output_idx = i - self.start_dim
                size_desired = output_shape[output_idx]
                current_size = x.shape[i]

                if current_size < size_desired:
                    pad_amount = size_desired - current_size
                    # Pad only on the "end" (right side)
                    pad_list.extend([0, pad_amount])
                else:
                    pad_list.extend([0, 0])
            else:
                # Dimensions outside our range of interest don't need padding
                pad_list.extend([0, 0])

        if any(pad_list):
            x = F.pad(x, pad_list, mode="constant", value=0.0)

        return x
