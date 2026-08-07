"""
.. _klein_gordon_1d_fd_vis :

A finite difference solver for the 1D Klein-Gordon equation
===============================================================
An intro to our loss module's finite difference utility demonstrating
its use to create a simple numerical solver for the Klein-Gordon equation
in one spatial dimension. The Klein-Gordon equation is a relativistic
wave equation with a mass term, widely used in quantum field theory
and particle physics.
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
from neuralop.losses.differentiation import FiniteDiff

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
# .. raw:: html
#
#    <div style="margin-top: 3em;"></div>
#
# Defining our problem
# --------------------
# The Klein-Gordon equation is a second-order hyperbolic PDE:
#
# .. math::
#    \frac{\partial^2 u}{\partial t^2} = c^2 \frac{\partial^2 u}{\partial x^2} - m^2 u
#
# where :math:`c` is the wave speed and :math:`m` is the mass parameter.
# When :math:`m = 0`, this reduces to the standard wave equation.
# When :math:`m > 0`, the mass term introduces dispersion: different
# frequency components travel at different speeds, causing wave packets
# to spread over time.
#
# We use leapfrog (Verlet) time integration, which is symplectic and
# second-order accurate—well-suited for conservative wave systems.

## Simulation parameters
L = 1.0  # Domain length
nx = 256  # Grid resolution
c = 1.0  # Wave speed
T = 2.0  # Total simulation time
dt = 0.5 * L / (nx * c)  # CFL-stable time step (Courant number = 0.5)
nt = int(T / dt)
dx = L / nx

# We demonstrate three different mass parameter values:
# m=0 (pure wave), m=2 (moderate dispersion), m=10 (strong dispersion)
mass_values = [0.0, 2.0, 10.0]

## Create periodic 1D grid
x = torch.linspace(0, L, nx + 1, device=device)[:-1]  # Exclude endpoint for periodic BC

## Initialize 1D finite difference operator
fd = FiniteDiff(dim=1, h=dx, periodic_in_x=True)

# %%
# .. raw:: html
#
#    <div style="margin-top: 3em;"></div>
#
# Initial conditions
# ------------------
# We use a Gaussian wave packet :math:`u(x, 0) = \exp(-\sigma (x - x_0)^2)`
# centered in the domain with zero initial velocity :math:`\partial_t u(x,0) = 0`.
# The leapfrog scheme is initialized with a Taylor expansion to second order.

x0 = L / 2
sigma = 800.0  # Controls width of the Gaussian
u0 = torch.exp(-sigma * (x - x0) ** 2).to(device)

n_save = 200  # Number of snapshots to store
save_interval = max(1, nt // n_save)

# %%
# .. raw:: html
#
#    <div style="margin-top: 3em;"></div>
#
# Simulate evolution for each mass value
# ----------------------------------------
# We solve the Klein-Gordon equation using leapfrog integration:
#
# .. math::
#    u^{n+1} = 2u^{n} - u^{n-1} + \Delta t^2 \left( c^2 \frac{\partial^2 u}{\partial x^2} - m^2 u \right)
#
# and compare three regimes: :math:`m = 0` (pure wave),
# :math:`m = 2` (moderate), and :math:`m = 10` (strong dispersion).

all_evolutions = {}
all_energies = {}
for m in mass_values:
    # Initialize fields
    u = u0.clone()

    # Taylor expansion for first leapfrog step:
    # u(dt) = u(0) + dt*u_t(0) + 0.5*dt^2*u_tt(0)
    # With u_t(0)=0: u(dt) = u(0) + 0.5*dt^2*(c^2*u_xx - m^2*u)
    u_xx = fd.dx(u, order=2)
    u_prev = u + 0.5 * dt**2 * (c**2 * u_xx - m**2 * u)

    snapshots = [u.cpu().numpy().copy()]
    energies = []

    for step in range(nt):
        u_xx = fd.dx(u, order=2)
        u_next = 2 * u - u_prev + dt**2 * (c**2 * u_xx - m**2 * u)

        # Compute energy using time-centered velocity (accurate for leapfrog)
        if (step + 1) % save_interval == 0:
            du_dt = (u_next - u_prev) / (2 * dt)
            du_dx = fd.dx(u)
            energy = (
                0.5 * torch.sum(du_dt**2 + c**2 * du_dx**2 + m**2 * u**2).item() * dx
            )
            energies.append(energy)
            snapshots.append(u.cpu().numpy().copy())

        u_prev = u.clone()
        u = u_next.clone()

    all_evolutions[m] = np.array(snapshots)
    all_energies[m] = np.array(energies)

# %%
# .. raw:: html
#
#    <div style="margin-top: 3em;"></div>
#
# Comparing wave behavior across mass values
# -------------------------------------------
# The plots below show the evolution of the wave field for each mass parameter.
# As the mass increases, dispersion becomes stronger and the initial
# Gaussian packet spreads into oscillatory tails.

x_np = x.cpu().numpy()

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, m in zip(axes, mass_values):
    evolution = all_evolutions[m]
    n_snapshots = evolution.shape[0]
    # Show a few representative time slices
    indices = np.linspace(0, n_snapshots - 1, 6, dtype=int)
    for idx in indices:
        t_val = idx * save_interval * dt
        ax.plot(x_np, evolution[idx], label=f"t={t_val:.2f}", alpha=0.8)
    ax.set_title(f"m = {m}")
    ax.set_xlabel("x")
    ax.set_ylabel("u")
    ax.legend(fontsize=7, loc="upper right")
    ax.set_ylim(-1.2, 1.2)

fig.suptitle("Klein-Gordon equation: effect of mass parameter", y=1.02)
plt.tight_layout()
fig.show()

# %%
# .. raw:: html
#
#    <div style="margin-top: 3em;"></div>
#
# Spacetime diagram (kymograph)
# -----------------------------
# A spacetime diagram shows the full time evolution at a glance.
# The m=0 case shows clean wave splitting; larger m shows dispersive spreading.

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, m in zip(axes, mass_values):
    evolution = all_evolutions[m]
    extent = [0, L, 0, T]
    im = ax.imshow(
        evolution,
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap="RdBu_r",
        vmin=-0.5,
        vmax=0.5,
    )
    ax.set_title(f"m = {m}")
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    plt.colorbar(im, ax=ax, shrink=0.8)

fig.suptitle("Klein-Gordon spacetime diagrams", y=1.02)
plt.tight_layout()
fig.show()

# %%
# .. raw:: html
#
#    <div style="margin-top: 3em;"></div>
#
# Energy conservation check
# -------------------------
# The Klein-Gordon equation conserves a total energy:
#
# .. math::
#    E = \int \frac{1}{2} \left[ \left(\frac{\partial u}{\partial t}\right)^2 + c^2 \left(\frac{\partial u}{\partial x}\right)^2 + m^2 u^2 \right] dx
#
# We verify energy is approximately conserved for each mass value.
# The velocity :math:`\partial_t u` is computed using the centered difference
# :math:`(u^{n+1} - u^{n-1}) / 2\Delta t`, which is consistent with the
# leapfrog scheme.

print("Energy conservation (relative drift over full simulation):")
for m in mass_values:
    energies = all_energies[m]
    if len(energies) > 1:
        drift = abs(energies[-1] - energies[0]) / (abs(energies[0]) + 1e-12)
        print(
            f"  m={m:5.1f}: E_initial={energies[0]:.4f}, E_final={energies[-1]:.4f}, drift={drift:.2e}"
        )
