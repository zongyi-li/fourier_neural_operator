"""Example: Galerkin Attention for PDE Surrogate Modeling.

This example demonstrates GalerkinAttention, a linearized attention mechanism
that replaces O(n^2 d) softmax attention with an O(n d^2) spectral kernel.
Inspired by Galerkin methods for PDEs, it projects query-key interactions
onto a low-rank frequency basis.

This is complementary to AttentionKernelIntegral (quadrature-based kernel
integration) -- here the kernel is implicit (spectral) rather than assembled.

The linearized kernel is especially effective for PDE surrogates where the
attention pattern is expected to be smooth / low-rank in frequency.
"""
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from neuralop.layers.galerkin_attention import GalerkinAttention
from neuralop.layers.attention_kernel_integral import AttentionKernelIntegral
from neuralop.layers.embeddings import RotaryEmbedding2D


def compare_attention_patterns():
    """Compare attention patterns from GalerkinAttention vs AttentionKernelIntegral."""
    d = 64
    n_heads = 4
    head_n_channels = d // n_heads
    n_modes = 16
    num_points = 64

    gal = GalerkinAttention(
        in_channels=d, out_channels=d,
        n_heads=n_heads, head_n_channels=head_n_channels,
        n_modes=n_modes,
    )
    attn_int = AttentionKernelIntegral(
        in_channels=d, out_channels=d,
        n_heads=n_heads, head_n_channels=head_n_channels,
    )

    x = torch.randn(2, num_points, d)
    pos = torch.randn(2, num_points, 2)

    out_gal, kernel_gal = gal(x, pos_src=pos, return_kernel=True)
    out_ki = attn_int(x, pos)

    print(f"Galerkin output: {out_gal.shape}, kernel: {kernel_gal.shape}")
    print(f"KernelIntegral output: {out_ki.shape}")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    kernel_np = kernel_gal[0, 0].detach().cpu().abs().numpy()
    axes[0].imshow(kernel_np, cmap="RdBu_r", aspect="auto")
    axes[0].set_title(f"Galerkin kernel (head 0, M={n_modes})")
    axes[0].set_xlabel("freq mode")
    axes[0].set_ylabel("freq mode")

    diff = (out_gal - out_ki).abs().mean().item()
    axes[1].text(0.5, 0.5, f"Abs diff vs KernelIntegral\nmean={diff:.4f}",
                 ha='center', va='center', transform=axes[1].transAxes)
    axes[1].set_title("Comparison")

    inp = x[0, :, :head_n_channels].detach().cpu()
    axes[2].imshow(inp.T @ inp, cmap="coolwarm")
    axes[2].set_title(f"Input Gram matrix (feat dim {head_n_channels})")

    plt.tight_layout()
    plt.savefig("galerkin_attention_comparison.png", dpi=150)
    print("Saved galerkin_attention_comparison.png")
    plt.close()


def benchmark_complexity():
    """Show theoretical complexity advantage of Galerkin vs standard attention."""
    print("\n=== Complexity Comparison ===")
    print(f"{'n (seq len)':>12} | {'d (emb dim)':>12} | {'Std O(n^2d)':>14} | {'Gal O(nd^2)':>14} | {'Speedup':>8}")
    print("-" * 70)

    for n in [64, 128, 256, 512, 1024]:
        for d in [64, 128, 256]:
            std_ops = n * n * d
            gal_ops = n * d * d
            speedup = std_ops / gal_ops
            if speedup >= 1:
                print(f"{n:>12} | {d:>12} | {std_ops:>14,} | {gal_ops:>14,} | {speedup:>7.1f}x")


def demo_training():
    """Train GalerkinAttention on a simple regression task."""
    print("\n=== Training Demo ===")
    d = 64
    n_heads = 4
    head_n_channels = d // n_heads
    n_modes = 16
    num_points = 128

    model = GalerkinAttention(
        in_channels=d, out_channels=d,
        n_heads=n_heads, head_n_channels=head_n_channels,
        n_modes=n_modes,
    )

    x = torch.randn(32, num_points, d)
    target = torch.randn(32, num_points, d)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(1, 51):
        out = model(x)
        loss = loss_fn(out, target)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d} | loss={loss.item():.4f} | "
                  f"alpha={model.alpha.item():.4f}")

    final_loss = loss.item()
    print(f"\nFinal loss: {final_loss:.4f} after 50 epochs")


if __name__ == "__main__":
    print("Galerkin Attention Example")
    print("=" * 50)
    benchmark_complexity()
    demo_training()
    print("\nDone!")
