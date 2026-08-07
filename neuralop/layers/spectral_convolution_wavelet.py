from typing import Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import torch
import torch.nn as nn

try: 
    from pytorch_wavelets import DWT1D, IDWT1D
    from pytorch_wavelets import DWT, IDWT
    _HAS_PTW = True
except Exception: 
    _HAS_PTW = False
    
try:
    import pywt
    from ptwt.conv_transform_3 import wavedec3 as ptwt_wavedec3
    from ptwt.conv_transform_3 import waverec3 as ptwt_waverec3
    _HAS_PYWT_PTWT = True
except Exception:
    _HAS_PYWT_PTWT = False
    
def _ensure_tuple_size(size: Union[int, Sequence[int]], n_dim: int) -> Tuple[int, ...]:
    if n_dim == 1:
        if isinstance(size, int):
            return (size,)
        raise ValueError("For n_dim=1, size must be an int (signal length).")
    else:
        if not (isinstance(size, (list, tuple)) and len(size) == n_dim):
            raise ValueError(f"For n_dim={n_dim}, size must be a list/tuple of length {n_dim}.")
        return tuple(int(s) for s in size)

def _einsum_pattern(n_dim: int) -> str:
    """Return einsum pattern to map (B, Cin, *S) x (Cin, Cout, *S) → (B, Cout, *S)."""
    axes = "xyzuvw"
    idx = axes[:n_dim]
    return f"bi{idx},io{idx}->bo{idx}"


class SpectralConvWavelet(nn.Module):
    """Implements Unified Wavelet-based Spectral Convolution for n_dim ∈ {1,2,3}.
    described in [1]_

    Parameters
    ----------
    in_channels, out_channels : int
    level : int
        Decomposition levels (we act on the last level only).
    size : int | Tuple[int, ...]
        Nominal spatial size. For n_dim=1 provide an int; for 2D/3D a tuple/list.
    n_dim : {1,2,3}
        Dimensionality selector.
    wavelet : str
        Wavelet name (e.g., 'db4'). For 3D we pass pywt.Wavelet(wavelet) to ptwt.
    mode : str
        Boundary mode (e.g., 'symmetric' for 1D/2D; 'periodization' or 'symmetric' for 3D).

    .. [1] : Tripura, T., and Chakraborty, S. “Wavelet neural operator: a neural operator for parametric partial differential equations” (2022). arXiv preprint arXiv:2205.02191, https://arxiv.org/pdf/2205.02191.

    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        level: int,
        size: Union[int, Sequence[int]],
        n_dim: int,
        wavelet: str = "db4",
        mode: str = "symmetric",
    ) -> None:
        super().__init__()
        
        if n_dim not in (1, 2, 3):
            raise ValueError("n_dim must be 1, 2, or 3")
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("in_channels/out_channels must be positive")
        if level < 1:
            raise ValueError("level must be >= 1")
        
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.level = int(level)
        self.n_dim = int(n_dim)
        self.wavelet = wavelet
        self.mode = mode
        self.size = _ensure_tuple_size(size, self.n_dim)

        # Dependency checks per-dimension
        if self.n_dim in (1, 2) and not _HAS_PTW:
            raise ImportError("pytorch_wavelets is required for n_dim=1 or 2")
        if self.n_dim == 3 and not _HAS_PYWT_PTWT:
            raise ImportError("pywt + ptwt are required for n_dim=3")
        
        # Dependency checks per-dimension
        if self.n_dim in (1, 2) and not _HAS_PTW:
            raise ImportError("pytorch_wavelets is required for n_dim=1 or 2")
        if self.n_dim == 3 and not _HAS_PYWT_PTWT:
            raise ImportError("pywt + ptwt are required for n_dim=3")
        
        # Initialization scale
        self.scale = 1.0 / (self.in_channels * self.out_channels)
        
        # Infer last-level approx subband shape to allocate weights
        with torch.no_grad():
            if self.n_dim == 1:
                dummy = torch.randn(1, 1, self.size[0])
                dwt = DWT1D(wave=self.wavelet, J=self.level, mode=self.mode)
                yl, _ = dwt(dummy)
                self.modes = (yl.shape[-1],)
                # Weights: approx (A) and detail (D)
                self.weights_A = nn.Parameter(self.scale * torch.rand(self.in_channels, self.out_channels, *self.modes))
                self.weights_D = nn.Parameter(self.scale * torch.rand(self.in_channels, self.out_channels, *self.modes))

            elif self.n_dim == 2:
                dummy = torch.randn(1, 1, *self.size)
                dwt = DWT(wave=self.wavelet, J=self.level, mode=self.mode)
                yl, _ = dwt(dummy)
                self.modes = (yl.shape[-2], yl.shape[-1])
                # Weights: approx (A) and H/V/D details
                self.weights_A = nn.Parameter(self.scale * torch.rand(self.in_channels, self.out_channels, *self.modes))
                self.weights_H = nn.Parameter(self.scale * torch.rand(self.in_channels, self.out_channels, *self.modes))
                self.weights_V = nn.Parameter(self.scale * torch.rand(self.in_channels, self.out_channels, *self.modes))
                self.weights_D = nn.Parameter(self.scale * torch.rand(self.in_channels, self.out_channels, *self.modes))

            else:  # n_dim == 3
                # ptwt expects tensors; we can probe with a spatial-only dummy, shapes match coefficient dims
                dummy = torch.randn(*self.size).unsqueeze(0)  # (1, Z, Y, X) no channel dim
                mode_data = ptwt_wavedec3(dummy, pywt.Wavelet(self.wavelet), level=self.level, mode=self.mode)
                A = mode_data[0]  # approx
                self.modes = (A.shape[-3], A.shape[-2], A.shape[-1])
                # Weights: approx + 7 detail combinations
                self.weights_A   = nn.Parameter(self.scale * torch.rand(self.in_channels, self.out_channels, *self.modes))
                self.weights_aad = nn.Parameter(self.scale * torch.rand(self.in_channels, self.out_channels, *self.modes))
                self.weights_ada = nn.Parameter(self.scale * torch.rand(self.in_channels, self.out_channels, *self.modes))
                self.weights_add = nn.Parameter(self.scale * torch.rand(self.in_channels, self.out_channels, *self.modes))
                self.weights_daa = nn.Parameter(self.scale * torch.rand(self.in_channels, self.out_channels, *self.modes))
                self.weights_dad = nn.Parameter(self.scale * torch.rand(self.in_channels, self.out_channels, *self.modes))
                self.weights_dda = nn.Parameter(self.scale * torch.rand(self.in_channels, self.out_channels, *self.modes))
                self.weights_ddd = nn.Parameter(self.scale * torch.rand(self.in_channels, self.out_channels, *self.modes))
                
        # Precompute einsum pattern for mul
        self._einsum = _einsum_pattern(self.n_dim)

    # ------------------------------- helpers -------------------------------
    @staticmethod
    def _level_adjust(nominal_last: int, actual_last: int, base_level: int) -> int:
        """Mimic reference logic: adjust DWT level only from the last axis length."""
        if actual_last > nominal_last:
            factor = int(np.log2(actual_last // nominal_last))
            return max(1, base_level + factor)
        elif actual_last < nominal_last:
            factor = int(np.log2(nominal_last // actual_last))
            return max(1, base_level - factor)
        else:
            return base_level

    def _mul(self, coeff: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        """Apply per-position linear mixing: (B,Cin,*S) × (Cin,Cout,*S) → (B,Cout,*S)."""
        return torch.einsum(self._einsum, coeff, W)
        
    # ------------------------------- forward -------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dimension check for incoming tensor (Assuming B,C,S... format)
        assert x.ndim == 2 + self.n_dim, f"Expected (B,C,{self.n_dim}D), got {x.shape}"
        B, Cin = x.shape[:2]
        assert Cin == self.in_channels, "Input channels must match in_channels"

        if self.n_dim == 1:
            # Choose level based on last axis only (to match reference behavior)
            J = self._level_adjust(self.size[-1], x.shape[-1], self.level)
            dwt = DWT1D(wave=self.wavelet, J=J, mode=self.mode).to(x.device)
            yl, yh = dwt(x)              # yl: (B,C,LJ), yh: list length J of (B,C,Lj)
            out_yl = self._mul(yl, self.weights_A)
            out_yh = [torch.zeros_like(t, device=x.device) for t in yh]
            out_yh[-1] = self._mul(yh[-1].clone(), self.weights_D)
            idwt = IDWT1D(wave=self.wavelet, mode=self.mode).to(x.device)
            y = idwt((out_yl, out_yh))
            return y
        
        if self.n_dim == 2:
            J = self._level_adjust(self.size[-1], x.shape[-1], self.level)
            dwt = DWT(wave=self.wavelet, J=J, mode=self.mode).to(x.device)
            yl, yh = dwt(x)              # yl: (B,C,HJ,WJ); yh: list length J of (B,C,3,Hj,Wj)
            out_yl = self._mul(yl, self.weights_A)
            out_yh = [torch.zeros_like(t, device=x.device) for t in yh]
            last = yh[-1]
            # H,V,D at index 0,1,2
            out_last = torch.zeros_like(last)
            out_last[:, :, 0] = self._mul(last[:, :, 0].clone(), self.weights_H)
            out_last[:, :, 1] = self._mul(last[:, :, 1].clone(), self.weights_V)
            out_last[:, :, 2] = self._mul(last[:, :, 2].clone(), self.weights_D)
            out_yh[-1] = out_last
            idwt = IDWT(wave=self.wavelet, mode=self.mode).to(x.device)
            y = idwt((out_yl, out_yh))
            return y

        # n_dim == 3
        # ptwt APIs operate per-sample; loop over batch
        out = torch.zeros((B, self.out_channels, *x.shape[-3:]), device=x.device, dtype=x.dtype)
        wav = pywt.Wavelet(self.wavelet)
        for b in range(B):
            # Each slice has shape (C, Z, Y, X)
            xb = x[b]
            # Decompose
            coeffs = ptwt_wavedec3(xb, wav, level=self.level, mode=self.mode)
            # coeffs: [A, D1, D2, ..., DJ]; we use last level only
            A = coeffs[0]           # (C_in, Zj, Yj, Xj)
            Dj = coeffs[1]          # dict with keys among {'aad','ada','add','daa','dad','dda','ddd'}

            # Apply weights
            A_out = torch.einsum("i" + "xyz" + ",io" + "xyz" + "->o" + "xyz", A, self.weights_A)
            D_out = {}
            # map available keys safely
            wmap = {
                'aad': self.weights_aad,
                'ada': self.weights_ada,
                'add': self.weights_add,
                'daa': self.weights_daa,
                'dad': self.weights_dad,
                'dda': self.weights_dda,
                'ddd': self.weights_ddd,
            }
            for k, v in Dj.items():
                Wk = wmap.get(k, None)
                if Wk is None:
                    # unseen detail key; pass zeros
                    D_out[k] = torch.zeros_like(v)
                else:
                    D_out[k] = torch.einsum("i" + "xyz" + ",io" + "xyz" + "->o" + "xyz", v.clone(), Wk)

            # Zero higher levels (>=2) to mirror reference
            coeffs_out: List[Union[torch.Tensor, Dict[str, torch.Tensor]]] = [A_out, D_out]
            for _ in range(2, len(coeffs)):
                zero_dict = {k: torch.zeros_like(next(iter(D_out.values()))) for k in D_out.keys()}
                coeffs_out.append(zero_dict)

            yb = ptwt_waverec3(tuple(coeffs_out), wav)
            out[b] = yb
        return out
