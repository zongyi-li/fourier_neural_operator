import numpy as np
import itertools
import torch
import torch.nn.functional as F


def resample(x, res_scale, axis, output_shape=None):
    """A module for generic n-dimentional interpolation.

    For 1D and 2D inputs, the ``resample`` function uses PyTorch’s built-in spatial interpolators
    for efficiency, applying linear interpolation for 1D data and bicubic interpolation for 2D data
    directly in the spatial domain.

    For 3D or higher-dimensional inputs, the ``resample`` function switches to a spectral interpolation
    method  based on the Fourier transform. The input is transformed into the frequency domain using a
    real n-dimensional FFT, which decomposes the signal into its frequency components. By resizing
    this frequency representation and then applying an inverse FFT, the function achieves smooth,
    alias-free interpolation that preserves the signal’s overall structure.

    For complex-valued inputs, the function uses a full FFT and inverse FFT along the resampled
    axes to preserve both real and imaginary components. Like the real-valued spectral path,
    this assumes the resampled axes are the final dimensions of ``x``.

    Parameters
    ----------
    x : torch.Tensor
            input activation of size (batch_size, channels, d1, ..., dN)
    res_scale: int or tuple
            Scaling factor along each of the dimensions in 'axis' parameter. If res_scale is scaler, then isotropic
            scaling is performed
    axis: axis or dimensions along which interpolation will be performed.
    output_shape : None or tuple[int]
    """

    if isinstance(res_scale, (float, int)):
        if axis is None:
            axis = list(range(2, x.ndim))
            res_scale = [res_scale] * len(axis)
        elif isinstance(axis, int):
            axis = [axis]
            res_scale = [res_scale]
        else:
            res_scale = [res_scale] * len(axis)
    else:
        assert len(res_scale) == len(axis), "leght of res_scale and axis are not same"

    old_size = x.shape[-len(axis) :]
    if output_shape is None:
        new_size = tuple([int(round(s * r)) for (s, r) in zip(old_size, res_scale)])
    else:
        new_size = output_shape

    if x.is_complex():
        X = torch.fft.fftn(x, norm="forward", dim=axis)

        new_fft_size = list(new_size)
        new_fft_size_c = [min(i, j) for (i, j) in zip(new_fft_size, X.shape[-len(axis) :])]
        out_fft = torch.zeros(
            [x.shape[0], x.shape[1], *new_fft_size], device=x.device, dtype=X.dtype
        )

        mode_indexing = [((None, (m + 1) // 2), (-(m // 2), None)) for m in new_fft_size_c]
        for boundaries in itertools.product(*mode_indexing):
            idx_tuple = [slice(None), slice(None)] + [slice(*b) for b in boundaries]

            idx_tuple = tuple(idx_tuple)
            out_fft[idx_tuple] = X[idx_tuple]
        return torch.fft.ifftn(out_fft, s=new_size, norm="forward", dim=axis)

    if len(axis) == 1:
        return F.interpolate(x, size=new_size[0], mode="linear", align_corners=True)
    if len(axis) == 2:
        return F.interpolate(x, size=new_size, mode="bicubic", align_corners=True)

    X = torch.fft.rfftn(x.float(), norm="forward", dim=axis)

    new_fft_size = list(new_size)
    new_fft_size[-1] = new_fft_size[-1] // 2 + 1  # Redundant last coefficient
    new_fft_size_c = [min(i, j) for (i, j) in zip(new_fft_size, X.shape[-len(axis) :])]
    out_fft = torch.zeros(
        [x.shape[0], x.shape[1], *new_fft_size], device=x.device, dtype=torch.cfloat
    )

    mode_indexing = [((None, (m + 1) // 2), (-(m // 2), None)) for m in new_fft_size_c[:-1]] + [((None, new_fft_size_c[-1]), )]
    for i, boundaries in enumerate(itertools.product(*mode_indexing)):
        idx_tuple = [slice(None), slice(None)] + [slice(*b) for b in boundaries]

        idx_tuple = tuple(idx_tuple)
        out_fft[idx_tuple] = X[idx_tuple]
    y = torch.fft.irfftn(out_fft, s=new_size, norm="forward", dim=axis)

    return y


def iterative_resample(x, res_scale, axis):
    if isinstance(axis, list) and isinstance(res_scale, (float, int)):
        res_scale = [res_scale] * len(axis)
    if not isinstance(axis, list) and isinstance(res_scale,list):
      raise Exception("Axis is not a list but Scale factors are")
    if isinstance(axis, list) and isinstance(res_scale,list) and len(res_scale) != len(axis):
      raise Exception("Axis and Scal factor are in different sizes")

    if isinstance(axis, list):
        for i in range(len(res_scale)):
            rs = res_scale[i]
            a = axis[i]
            x = resample(x, rs, a)
        return x

    old_res = x.shape[axis]
    X = torch.fft.rfft(x, dim=axis, norm="forward")
    newshape = list(x.shape)
    new_res = int(round(res_scale * newshape[axis]))
    newshape[axis] = new_res // 2 + 1

    Y = torch.zeros(newshape, dtype=X.dtype, device=x.device)

    modes = min(new_res, old_res)
    sl = [slice(None)] * x.ndim
    sl[axis] = slice(0, modes // 2 + 1)
    Y[tuple(sl)] = X[tuple(sl)]
    y = torch.fft.irfft(Y, n=new_res, dim=axis, norm="forward")
    return y
