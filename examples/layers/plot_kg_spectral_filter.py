"""

Klein-Gordon Spectral Filter
=============================

This example visualizes the Klein-Gordon (KG) spectral convolution filter
and compares it to other common spectral filters used in neural operators.

The KG spectral filter is derived from the dispersion relation of the
Klein-Gordon equation:

.. math::
   H(k) = \\cos\\left(T \\sqrt{c^2 |k|^2 + \\chi^2}\\right)

where :math:`T` is the propagation time, :math:`c` the wave speed,
and :math:`\\chi` the mass parameter. This filter is the exact
single-step solution operator for the Klein-Gordon PDE and is a member
of the Matern kernel family (Whittle 1954).

We compare:

1. **KG filter**: oscillatory bandpass with mass gap
2. **Low-pass (GCN-style)**: 1 - |k|/k_max
3. **Gaussian (diffusion)**: exp(-sigma |k|^2)
4. **FNO truncation**: sharp rectangular cutoff in frequency space

"""

# %%
# .. raw:: html
#
#    <div style="margin-top: 3em;"></div>
#
# Setup
# -----

import torch
import numpy as np
import matplotlib.pyplot as plt

from neuralop.layers.kg_spectral_conv import KGSpectralConv

device = torch.device("cpu")

# %%
# .. raw:: html
#
#    <div style="margin-top: 3em;"></div>
#
# 1D spectral filter comparison
# -----------------------------
# We plot four different spectral filter profiles as a function of
# wavenumber :math:`|k|`.

k = np.linspace(0, 20, 500)

# KG filter for several mass values
T, c = 1.0, 1.0
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Panel 1: KG filter with varying chi (mass parameter)
ax = axes[0]
for chi in [0.0, 1.0, 5.0, 10.0]:
    omega = np.sqrt(c**2 * k**2 + chi**2)
    H = np.cos(T * omega)
    ax.plot(k, H, label=f"$\\chi={chi}$")
ax.set_xlabel("Wavenumber $|k|$")
ax.set_ylabel("Filter response $H(k)$")
ax.set_title("KG filter: varying mass $\\chi$")
ax.legend()
ax.axhline(0, color="gray", linewidth=0.5)
ax.set_ylim(-1.2, 1.2)

# Panel 2: KG filter with varying T (propagation time)
ax = axes[1]
chi = 2.0
for T_val in [0.1, 0.5, 1.0, 2.0]:
    omega = np.sqrt(c**2 * k**2 + chi**2)
    H = np.cos(T_val * omega)
    ax.plot(k, H, label=f"$T={T_val}$")
ax.set_xlabel("Wavenumber $|k|$")
ax.set_ylabel("Filter response $H(k)$")
ax.set_title("KG filter: varying time $T$")
ax.legend()
ax.axhline(0, color="gray", linewidth=0.5)
ax.set_ylim(-1.2, 1.2)

# Panel 3: Comparison with standard filters
ax = axes[2]
chi, T_val = 2.0, 0.8
omega = np.sqrt(c**2 * k**2 + chi**2)
H_kg = np.cos(T_val * omega)
H_lowpass = np.maximum(0, 1 - k / k.max())
H_gauss = np.exp(-0.02 * k**2)
H_fno = np.where(k <= 8, 1.0, 0.0)

ax.plot(k, H_kg, label="KG ($\\chi=2, T=0.8$)", linewidth=2)
ax.plot(k, H_lowpass, "--", label="Low-pass (GCN)", alpha=0.8)
ax.plot(k, H_gauss, "--", label="Gaussian (diffusion)", alpha=0.8)
ax.plot(k, H_fno, ":", label="FNO truncation", alpha=0.8)
ax.set_xlabel("Wavenumber $|k|$")
ax.set_ylabel("Filter response $H(k)$")
ax.set_title("Filter comparison")
ax.legend()
ax.axhline(0, color="gray", linewidth=0.5)
ax.set_ylim(-1.2, 1.2)

fig.suptitle("Klein-Gordon spectral filter vs standard neural operator filters", y=1.02)
plt.tight_layout()
fig.show()

# %%
# .. raw:: html
#
#    <div style="margin-top: 3em;"></div>
#
# 2D spectral filter visualization
# ---------------------------------
# The KG filter is isotropic (depends only on :math:`|k|`), creating
# concentric rings in 2D Fourier space.

kx = np.linspace(-15, 15, 200)
ky = np.linspace(-15, 15, 200)
KX, KY = np.meshgrid(kx, ky)
K2 = KX**2 + KY**2

fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))

configs = [
    (0.5, 1.0, 0.0, "Wave ($\\chi=0$)"),
    (0.5, 1.0, 3.0, "KG ($\\chi=3$)"),
    (0.5, 1.0, 8.0, "KG ($\\chi=8$)"),
    (1.0, 1.0, 3.0, "KG ($T=1.0$)"),
]

for ax, (T_val, c_val, chi_val, title) in zip(axes, configs):
    omega = np.sqrt(c_val**2 * K2 + chi_val**2)
    H = np.cos(T_val * omega)
    im = ax.imshow(
        H,
        extent=[kx[0], kx[-1], ky[0], ky[-1]],
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        origin="lower",
    )
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("$k_x$")
    ax.set_ylabel("$k_y$")

fig.colorbar(im, ax=axes, shrink=0.8, label="$H(k)$")
fig.suptitle("2D Klein-Gordon spectral filter in Fourier space", y=1.02)
fig.show()

# %%
# .. raw:: html
#
#    <div style="margin-top: 3em;"></div>
#
# Parameter efficiency demonstration
# -----------------------------------
# We compare the number of learnable parameters between a standard
# ``SpectralConv`` (unconstrained Fourier weights) and the
# ``KGSpectralConv`` (physics-constrained).

from neuralop.layers.spectral_convolution import SpectralConv

in_ch = 16
out_ch = 16
print(f"{'Configuration':<35} {'SpectralConv':<18} {'KGSpectralConv':<18} {'Ratio':<8}")
print("-" * 85)

for label, modes in [
    ("1D, modes=32", (32,)),
    ("1D, modes=64", (64,)),
    ("2D, modes=16x16", (16, 16)),
    ("2D, modes=32x32", (32, 32)),
    ("3D, modes=8x8x8", (8, 8, 8)),
]:
    fno = SpectralConv(in_ch, out_ch, modes)
    kg = KGSpectralConv(in_ch, out_ch, modes)

    n_fno = sum(p.numel() for p in fno.parameters())
    n_kg = sum(p.numel() for p in kg.parameters())
    ratio = n_fno / n_kg

    print(f"{label:<35} {n_fno:<18,} {n_kg:<18,} {ratio:<8.1f}x")

# %%
# .. raw:: html
#
#    <div style="margin-top: 3em;"></div>
#
# Applying the KG filter to a signal
# ------------------------------------
# We apply the KG spectral conv to a 1D signal containing a Gaussian
# pulse and show how different mass parameters affect the output.

nx = 256
x_grid = torch.linspace(0, 2 * np.pi, nx)
signal = torch.exp(-20 * (x_grid - np.pi) ** 2).unsqueeze(0).unsqueeze(0)

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(x_grid.numpy(), signal[0, 0].numpy(), "k-", linewidth=2, label="Input")

for chi_val in [0.1, 2.0, 5.0]:
    layer = KGSpectralConv(
        1, 1, n_modes=(32,), init_T=1.0, init_c=1.0, init_chi=chi_val, bias=False
    )
    with torch.no_grad():
        layer.channel_weight.fill_(1.0)
        out = layer(signal)
    ax.plot(
        x_grid.numpy(),
        out[0, 0].detach().numpy(),
        label=f"KG output ($\\chi={chi_val}$)",
    )

ax.set_xlabel("$x$")
ax.set_ylabel("Amplitude")
ax.set_title("Effect of KG spectral filter on a Gaussian pulse")
ax.legend()
plt.tight_layout()
fig.show()

# %%
# .. raw:: html
#
#    <div style="margin-top: 3em;"></div>
#
# Summary
# -------
# The ``KGSpectralConv`` layer provides a physics-constrained alternative
# to the unconstrained ``SpectralConv`` used in FNO. Key properties:
#
# - **Oscillatory bandpass**: Unlike diffusion-based (Gaussian) low-pass
#   filters, the KG filter retains and modulates high-frequency modes.
#
# - **Mass gap**: The :math:`\chi` parameter creates a minimum oscillation
#   frequency, providing built-in regularization.
#
# - **Parameter efficient**: 3 scalar parameters (T, c, chi) vs
#   O(C_in * C_out * prod(modes)) for standard FNO.
#
# - **Physically interpretable**: Each parameter has a clear physical
#   meaning tied to the Klein-Gordon dispersion relation.
#
# - **Matern kernel connection**: The Green's function of the KG equation
#   is the Matern kernel family. The standard RBF kernel (used in SVMs
#   and GPs) is the :math:`\nu \to \infty` limit (diffusion).
