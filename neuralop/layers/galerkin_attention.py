"""Galerkin-style linear attention via spectral kernel methods.

This module provides GalerkinAttention, a linearized attention mechanism
that replaces O(n^2 d) softmax attention with an O(n d^2) spectral kernel
computed in the frequency domain. Inspired by Galerkin methods for PDEs.

Compatible with the neuraloperator layers API (in_channels, out_channels,
n_heads, head_n_channels) and supports optional rotary positional embeddings.
"""
import torch
from torch import nn
import math
from torch.nn.init import xavier_uniform_


class GalerkinAttention(nn.Module):
    """Galerkin-style linear attention via spectral kernel methods.

    Replaces the O(n^2 d) softmax attention with an O(n d^2) linearized
    attention kernel computed in the frequency domain, inspired by the
    Galerkin method for solving PDEs. The kernel is obtained by projecting
    query-key interactions onto a low-rank frequency basis.

    Specifically computes:
        K(x,y) ~ sum_{m=1}^{M} qhat_m(x) * khat_m(y)*
    where M = n_modes limits the number of frequency modes, and the
    resulting kernel is applied to values v(y) via irfft.

    This is complementary to AttentionKernelIntegral which uses quadrature-
    based kernel integration. Here the kernel is implicit (spectral) rather
    than assembled explicitly.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    n_heads : int
        Number of attention heads.
    head_n_channels : int
        Dimension of each attention head (d // n_heads).
    n_modes : int, optional
        Number of low-frequency Fourier modes to retain. Default is 16.
    project_query : bool, optional
        Whether to project query with a linear layer. Default True.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        n_heads,
        head_n_channels,
        n_modes=16,
        project_query=True,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.head_n_channels = head_n_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        if n_modes < 1:
            raise ValueError(f"n_modes must be >= 1, got {n_modes}")
        self.n_modes = n_modes
        self.project_query = project_query

        if project_query:
            self.to_q = nn.Linear(in_channels, head_n_channels * n_heads, bias=False)
        else:
            self.to_q = nn.Identity()

        self.to_k = nn.Linear(in_channels, head_n_channels * n_heads, bias=False)
        self.to_v = nn.Linear(in_channels, head_n_channels * n_heads, bias=False)

        self.to_out = (
            nn.Linear(head_n_channels * n_heads, out_channels)
            if head_n_channels * n_heads != out_channels
            else nn.Identity()
        )

        self.init_gain = 1 / math.sqrt(head_n_channels)
        self.diagonal_weight = self.init_gain
        self.initialize_qkv_weights()

        self.alpha = nn.Parameter(torch.ones(1))

    def init_weight(self, weight, init_fn):
        """Initialize projection matrix with gain and optional diagonal bias.

        Initializes weights for each head with predefined initialization function and gain.
        See Table 8 in https://arxiv.org/pdf/2105.14995.pdf
        """
        for param in weight.parameters():
            if param.ndim > 1:
                for h in range(self.n_heads):
                    init_fn(param[h * self.head_n_channels:(h + 1) * self.head_n_channels, :],
                            gain=self.init_gain)

    def initialize_qkv_weights(self):
        """Initialize QKV projection weights with xavier uniform and diagonal bias."""
        init_fn = xavier_uniform_
        if self.project_query:
            self.init_weight(self.to_q, init_fn)
        self.init_weight(self.to_k, init_fn)
        self.init_weight(self.to_v, init_fn)

    def normalize_wrt_domain(self, u, norm_fn):
        """Normalize input with respect to domain, reshape for multi-head attention."""
        batch_size = u.shape[0]
        num_points = u.shape[2]
        # InstanceNorm1d expects [N, C, L], so transpose from [B, H, N, C] to [BH, C, N]
        u = u.permute(0, 1, 3, 2).contiguous()
        u = u.view(batch_size * self.n_heads, self.head_n_channels, num_points)
        u = norm_fn(u)
        u = u.view(batch_size, self.n_heads, self.head_n_channels, num_points)
        # Transpose back to [B, H, N, C]
        return u.permute(0, 1, 3, 2).contiguous()

    def forward(
        self,
        u_src,
        pos_src=None,
        positional_embedding_module=None,
        u_qry=None,
        pos_qry=None,
        weights=None,
        return_kernel=False,
    ):
        """Forward pass for Galerkin-style linear attention.

        Parameters
        ----------
        u_src : torch.Tensor
            Source input [batch_size, num_points, in_channels].
        pos_src : torch.Tensor, optional
            Source positions [batch_size, num_points, pos_dim].
        positional_embedding_module : nn.Module, optional
            Rotary embedding module (e.g. RotaryEmbedding2D).
        u_qry : torch.Tensor, optional
            Query input. If None, defaults to u_src (self-attention).
        return_kernel : bool, optional
            If True, return (output, kernel).

        Returns
        -------
        torch.Tensor or tuple
        """
        if u_qry is None:
            u_qry = u_src

        batch_size, num_src, _ = u_src.shape
        num_qry = u_qry.shape[1]
        pos_dim = pos_src.shape[-1] if pos_src is not None else 0

        if num_qry != num_src:
            raise ValueError(
                f"GalerkinAttention currently only supports self-attention "
                f"(num_qry == num_src), got num_qry={num_qry} and num_src={num_src}. "
                f"For cross-attention with different lengths, use AttentionKernelIntegral."
            )

        q = self.to_q(u_qry)
        k = self.to_k(u_src)
        v = self.to_v(u_src)

        q = q.view(batch_size, -1, self.n_heads, self.head_n_channels)
        k = k.view(batch_size, -1, self.n_heads, self.head_n_channels)
        v = v.view(batch_size, -1, self.n_heads, self.head_n_channels)

        q = q.permute(0, 2, 1, 3).contiguous()
        k = k.permute(0, 2, 1, 3).contiguous()
        v = v.permute(0, 2, 1, 3).contiguous()

        if positional_embedding_module is not None:
            if pos_dim == 2:
                pos_x = pos_src[..., 0]
                pos_y = pos_src[..., 1]
                k_freqs_x = positional_embedding_module(pos_x)
                k_freqs_y = positional_embedding_module(pos_y)
                k_freqs_x = k_freqs_x.unsqueeze(1).repeat([1, self.n_heads, 1, 1])
                k_freqs_y = k_freqs_y.unsqueeze(1).repeat([1, self.n_heads, 1, 1])

                if u_qry is u_src and pos_qry is None:
                    q_freqs_x, q_freqs_y = k_freqs_x, k_freqs_y
                else:
                    pq_x = pos_qry[..., 0] if pos_qry is not None else pos_x
                    pq_y = pos_qry[..., 1] if pos_qry is not None else pos_y
                    q_freqs_x = positional_embedding_module(pq_x)
                    q_freqs_y = positional_embedding_module(pq_y)
                    q_freqs_x = q_freqs_x.unsqueeze(1).repeat([1, self.n_heads, 1, 1])
                    q_freqs_y = q_freqs_y.unsqueeze(1).repeat([1, self.n_heads, 1, 1])

                q = positional_embedding_module.apply_2d_rotary_pos_emb(q, q_freqs_x, q_freqs_y)
                k = positional_embedding_module.apply_2d_rotary_pos_emb(k, k_freqs_x, k_freqs_y)
            elif pos_dim == 1:
                k_freqs = positional_embedding_module(pos_src[..., 0])
                k_freqs = k_freqs.unsqueeze(1).repeat([1, self.n_heads, 1, 1])
                q_freqs = positional_embedding_module(
                    pos_qry[..., 0] if pos_qry is not None else pos_src[..., 0]
                )
                q_freqs = q_freqs.unsqueeze(1).repeat([1, self.n_heads, 1, 1])
                q = positional_embedding_module.apply_1d_rotary_pos_emb(q, q_freqs)
                k = positional_embedding_module.apply_1d_rotary_pos_emb(k, k_freqs)

        q_fft = torch.fft.rfft(q, dim=2, norm="ortho")
        k_fft = torch.fft.rfft(k, dim=2, norm="ortho")

        n_freqs = q_fft.shape[2]
        M = min(self.n_modes, n_freqs)

        q_m = q_fft[:, :, :M, :]
        k_m = k_fft[:, :, :M, :]

        kernel = torch.einsum("bhnm, bhnm -> bhnm", q_m, k_m.conj())
        kernel = kernel * self.alpha

        kernel_conj_flip = kernel.conj().flip(-2)
        kernel_full = torch.cat([kernel, kernel_conj_flip], dim=2)[:, :, :n_freqs, :]

        attn = torch.fft.irfft(kernel_full, n=num_src, dim=2, norm="ortho")

        if weights is not None:
            if weights.dim() == 1:
                attn = attn * weights.view(1, 1, -1, 1)
            else:
                attn = attn * weights.view(weights.shape[0], 1, -1, 1)

        out = attn * v
        out = out.permute(0, 2, 1, 3).contiguous()
        out = out.view(batch_size, num_src, self.n_heads * self.head_n_channels)
        out = self.to_out(out)

        if return_kernel:
            return out, kernel
        return out
