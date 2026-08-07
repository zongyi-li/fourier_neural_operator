"""
.. _wave_equation_fd_vis :

A finite difference solver for the wave equation in 2 dimensions
================================================================
A demonstration of the FiniteDiff utility for solving the 2D wave equation,
a fundamental hyperbolic PDE that describes propagation of disturbances at
finite speed.
"""

# %%
# .. raw:: html
#
#    <div style="margin-top: 3em;"></div>
#
# Import the library
# ------------------
# We first import our `neuralop` library and required dependencies.
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from neuralop.losses.differentiation import FiniteDiff

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
# .. raw:: html
#
#    <div style="margin-top: 3em;"></div>
#
# Defining our problem
# --------------------
# We solve the 2D wave equation with periodic boundary conditions:
#
# .. math::
#    \frac{\partial^2 u}{\partial t^2} = c^2 \left(\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2}\right)
#
# Unlike parabolic PDEs (diffusion, Navier-Stokes) which smooth out initial
# conditions, the wave equation preserves sharp features as they propagate.
# This makes it a challenging benchmark for neural operators.
#
# We use leapfrog (Verlet) time integration, which is symplectic and conserves
# energy — essential for hyperbolic PDEs where forward Euler would be unstable.

## Simulation parameters
Lx, Ly = 2.0, 2.0  # Domain lengths
nx, ny = 128, 128  # Grid resolution
c = 1.0  # Wave speed
T = 2.0  # Total simulation time
dt = 0.005  # Time step (CFL: c*dt/dx < 1/sqrt(2))

## Create grid (periodic: endpoint=False, so spacing = L/n)
dx = Lx / nx
dy = Ly / ny
X = torch.linspace(0, Lx - dx, nx, device=device).repeat(ny, 1).T
Y = torch.linspace(0, Ly - dy, ny, device=device).repeat(nx, 1)
nt = int(T / dt)

## CFL stability check
cfl = c * dt / min(dx, dy) * np.sqrt(2)
print(f"CFL number: {cfl:.3f} (must be < 1 for stability)")

## Initialize finite difference operator
fd = FiniteDiff(dim=2, h=(dx, dy))

# %%
# .. raw:: html
#
#    <div style="margin-top: 3em;"></div>
#
# Initial conditions
# ------------------
# We initialize with a superposition of Gaussian wave packets at different
# positions. The initial velocity is zero, so the packets will split into
# outgoing and ingoing waves.

## Initial displacement: two Gaussian bumps
u = (
    torch.exp(-((X - 0.6) ** 2 + (Y - 0.6) ** 2) / 0.02)
    + 0.7 * torch.exp(-((X - 1.4) ** 2 + (Y - 1.4) ** 2) / 0.03)
).to(device)

## Initial velocity: zero — use Taylor expansion for 2nd-order accurate start:
## u^{-1} = u^0 + 0.5 * (c*dt)^2 * laplacian(u^0)
lap_u0 = fd.laplacian(u)
u_prev = u + 0.5 * (c * dt) ** 2 * lap_u0

# %%
# .. raw:: html
#
#    <div style="margin-top: 3em;"></div>
#
# Simulate evolution using leapfrog integration
# ----------------------------------------------
# The leapfrog scheme for the wave equation is:
#
# .. math::
#    u^{n+1} = 2u^n - u^{n-1} + c^2 \Delta t^2 \nabla^2 u^n
#
# This is time-reversible and conserves a discrete energy, making it
# ideal for long-time wave propagation.

u_evolution = [u.clone()]
save_every = max(1, nt // 200)

for step in range(nt):
    lap_u = fd.laplacian(u)
    u_next = 2 * u - u_prev + (c * dt) ** 2 * lap_u
    u_prev = u.clone()
    u = u_next.clone()

    if (step + 1) % save_every == 0:
        u_evolution.append(u.clone())

u_evolution = torch.stack(u_evolution).cpu().numpy()
print(f"Saved {len(u_evolution)} frames from {nt} time steps")

# %%
# .. raw:: html
#
#    <div style="margin-top: 3em;"></div>
#
# Animate our solution
# --------------------
# Unlike diffusion which smooths out, watch how the wave equation preserves
# structure as pulses propagate, wrap around the domain, and interfere.

num_frames = min(100, len(u_evolution))
frame_indices = (
    torch.linspace(0, len(u_evolution) - 1, num_frames, dtype=torch.int).cpu().numpy()
)
u_frames = u_evolution[frame_indices]

vmin, vmax = u_evolution.min(), u_evolution.max()
fig, ax = plt.subplots(figsize=(6, 6))
cmap_u = ax.imshow(
    u_frames[0], extent=[0, Lx, 0, Ly], origin="lower", cmap="RdBu_r",
    vmin=vmin, vmax=vmax,
)
ax.set_title("2D Wave Equation")
plt.colorbar(cmap_u, ax=ax, shrink=0.75)


def update(frame):
    cmap_u.set_data(u_frames[frame])
    t_val = frame_indices[frame] * save_every * dt
    ax.set_title(f"2D Wave Equation (t = {t_val:.3f})")
    ax.set_xticks([])
    ax.set_yticks([])
    return (cmap_u,)


ani = animation.FuncAnimation(
    fig, update, frames=len(u_frames), interval=50, blit=False
)
