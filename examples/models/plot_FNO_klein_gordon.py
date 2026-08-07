"""
Training an FNO on the Klein-Gordon Equation
=============================================

We train a Fourier Neural Operator (FNO) to learn the time evolution of the
Klein-Gordon equation, a relativistic wave equation with mass:

.. math::
   \\frac{\\partial^2 u}{\\partial t^2} = c^2 \\frac{\\partial^2 u}{\\partial x^2} - m^2 u

This tutorial demonstrates:

1. Generating Klein-Gordon training data using a leapfrog finite-difference solver
2. Training an FNO to map initial conditions to later-time solutions
3. Evaluating the trained model on unseen initial conditions
4. Testing generalization across different mass parameters

The Klein-Gordon equation is hyperbolic, unlike the elliptic Darcy-Flow
equation used in most neural operator examples. Hyperbolic PDEs present
additional challenges: solutions are oscillatory and dispersive, and
small errors can grow over time.
"""

# %%
# .. raw:: html
#
#    <div style="margin-top: 3em;"></div>
#
# Import dependencies
# -------------------
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
from neuralop.models import FNO
from neuralop.training import AdamW
from neuralop.utils import count_model_params
from neuralop import H1Loss

device = "cpu"

# %%
# .. raw:: html
#
#    <div style="margin-top: 3em;"></div>
#
# Klein-Gordon data generation
# ----------------------------
# We generate training data by solving the Klein-Gordon equation with a
# leapfrog (Verlet) finite-difference scheme. Each sample uses a random
# initial condition composed of a superposition of Fourier modes.
#
# The solver maps from input :math:`u(x, t=0)` to output :math:`u(x, t=T)`,
# learning the time-evolution operator.


def solve_klein_gordon_1d(u0, nx, dx, dt, nt, c, m):
    """Solve the 1D Klein-Gordon equation using leapfrog integration.

    Parameters
    ----------
    u0 : Tensor of shape (nx,)
        Initial condition u(x, t=0)
    nx : int
        Number of spatial grid points
    dx : float
        Spatial step size
    dt : float
        Time step size
    nt : int
        Number of time steps
    c : float
        Wave speed
    m : float
        Mass parameter

    Returns
    -------
    u : Tensor of shape (nx,)
        Solution at time t = nt * dt
    """
    u = u0.clone()

    # Compute spatial second derivative using periodic finite differences
    def laplacian_1d(v):
        return (torch.roll(v, -1, 0) + torch.roll(v, 1, 0) - 2 * v) / dx**2

    # Initialize first step via Taylor expansion (zero initial velocity)
    u_xx = laplacian_1d(u)
    u_prev = u + 0.5 * dt**2 * (c**2 * u_xx - m**2 * u)

    for _ in range(nt):
        u_xx = laplacian_1d(u)
        u_next = 2 * u - u_prev + dt**2 * (c**2 * u_xx - m**2 * u)
        u_prev = u.clone()
        u = u_next.clone()

    return u


def generate_klein_gordon_dataset(n_samples, nx, mass, seed=0):
    """Generate input-output pairs for the Klein-Gordon equation.

    Each input is a random superposition of low-frequency Fourier modes.
    The output is the solution after evolving for a fixed time T.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate
    nx : int
        Spatial resolution
    mass : float
        Mass parameter m
    seed : int
        Random seed for reproducibility

    Returns
    -------
    inputs : Tensor of shape (n_samples, 1, nx)
        Initial conditions
    outputs : Tensor of shape (n_samples, 1, nx)
        Solutions at time T
    """
    rng = torch.Generator().manual_seed(seed)
    L = 1.0
    c = 1.0
    dx = L / nx
    dt = 0.4 * dx / c  # CFL-stable time step
    T = 0.5  # Evolve for fixed time
    nt = int(T / dt)

    x = torch.linspace(0, L, nx + 1)[:-1]
    inputs = torch.zeros(n_samples, 1, nx)
    outputs = torch.zeros(n_samples, 1, nx)

    n_modes = 5  # Superposition of low-frequency modes

    for i in range(n_samples):
        # Random initial condition: sum of sinusoidal modes
        u0 = torch.zeros(nx)
        for k in range(1, n_modes + 1):
            amp = torch.randn(1, generator=rng).item() * 0.5
            phase = torch.rand(1, generator=rng).item() * 2 * np.pi
            u0 += amp * torch.sin(2 * np.pi * k * x / L + phase)

        u_final = solve_klein_gordon_1d(u0, nx, dx, dt, nt, c, mass)
        inputs[i, 0] = u0
        outputs[i, 0] = u_final

    return inputs, outputs


# %%
# .. raw:: html
#
#    <div style="margin-top: 3em;"></div>
#
# Generate training and test data
# --------------------------------
# We generate data for mass :math:`m = 5`, which produces
# moderate dispersion—enough to make the problem nontrivial.

print("Generating Klein-Gordon training data (m=5) ...")
sys.stdout.flush()

nx = 64
mass_train = 5.0
n_train = 500
n_test = 100

train_x, train_y = generate_klein_gordon_dataset(n_train, nx, mass_train, seed=0)
test_x, test_y = generate_klein_gordon_dataset(n_test, nx, mass_train, seed=42)

print(f"  Training samples: {train_x.shape[0]}")
print(f"  Test samples: {test_x.shape[0]}")
print(f"  Resolution: {nx}")
print(f"  Mass parameter: {mass_train}")
sys.stdout.flush()

# %%
# .. raw:: html
#
#    <div style="margin-top: 3em;"></div>
#
# Setting up data loaders
# -----------------------

train_dataset = torch.utils.data.TensorDataset(train_x, train_y)
test_dataset = torch.utils.data.TensorDataset(test_x, test_y)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32)

# %%
# .. raw:: html
#
#    <div style="margin-top: 3em;"></div>
#
# Creating the FNO model
# ----------------------
# We use a 1D FNO with 16 Fourier modes, suitable for our resolution of 64 points.
# The architecture is the same as in the original FNO paper by Li et al. (2021).

model = FNO(
    n_modes=(16,),
    in_channels=1,
    out_channels=1,
    hidden_channels=32,
    projection_channel_ratio=2,
)
model = model.to(device)

n_params = count_model_params(model)
print(f"\nFNO model has {n_params} parameters.")
sys.stdout.flush()

# %%
# .. raw:: html
#
#    <div style="margin-top: 3em;"></div>
#
# Training the model
# ------------------
# We train with AdamW optimizer and H1 loss, which penalizes errors in
# both the solution values and their spatial gradients.

optimizer = AdamW(model.parameters(), lr=1e-2, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

h1loss = H1Loss(d=1)

print("\n### TRAINING ###")
print(f"Epochs: 30, Optimizer: AdamW, Loss: H1")
sys.stdout.flush()

# Simple training loop (no data_processor needed for synthetic data)
model.train()
for epoch in range(30):
    epoch_loss = 0.0
    n_batches = 0
    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        pred = model(batch_x)
        loss = h1loss(pred, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        n_batches += 1

    scheduler.step()
    if (epoch + 1) % 10 == 0:
        avg_loss = epoch_loss / n_batches
        print(f"  Epoch {epoch + 1:3d}: train H1 loss = {avg_loss:.4f}")
        sys.stdout.flush()

# %%
# .. raw:: html
#
#    <div style="margin-top: 3em;"></div>
#
# Evaluating on held-out test data
# ---------------------------------
# We compute the relative L2 error on unseen initial conditions
# drawn from the same distribution.

model.eval()
test_errors = []
with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        pred = model(batch_x)
        # Per-sample relative L2 error
        for i in range(batch_x.shape[0]):
            err = torch.norm(pred[i] - batch_y[i]) / torch.norm(batch_y[i])
            test_errors.append(err.item())

mean_error = np.mean(test_errors)
print(f"\nTest relative L2 error (m={mass_train}): {mean_error:.4f}")
sys.stdout.flush()

# %%
# .. raw:: html
#
#    <div style="margin-top: 3em;"></div>
#
# Visualizing predictions
# -----------------------
# We show 3 test samples: input (initial condition), ground-truth output,
# and the FNO prediction.

x_grid = np.linspace(0, 1, nx, endpoint=False)

fig, axes = plt.subplots(3, 1, figsize=(10, 8))
for idx in range(3):
    data_x = test_x[idx, 0].numpy()
    data_y = test_y[idx, 0].numpy()
    with torch.no_grad():
        pred_y = model(test_x[idx : idx + 1].to(device)).cpu().numpy()[0, 0]

    ax = axes[idx]
    ax.plot(x_grid, data_x, "k--", label="Initial cond.", alpha=0.5)
    ax.plot(x_grid, data_y, "b-", label="Ground truth", linewidth=2)
    ax.plot(x_grid, pred_y, "r--", label="FNO prediction", linewidth=2)
    ax.set_ylabel("u(x)")
    ax.legend(loc="upper right", fontsize=8)
    if idx == 0:
        ax.set_title(f"FNO predictions on Klein-Gordon (m={mass_train})")
    if idx == 2:
        ax.set_xlabel("x")

plt.tight_layout()
fig.show()

# %%
# .. raw:: html
#
#    <div style="margin-top: 3em;"></div>
#
# Cross-mass generalization
# -------------------------
# A key question for neural operators applied to parametric PDEs: does an
# FNO trained at one mass parameter generalize to other mass values?
#
# We test the model (trained at :math:`m = 5`) on data generated with
# different mass parameters. This reveals the limits of in-distribution
# training for hyperbolic PDEs.

mass_test_values = [0.0, 2.0, 5.0, 10.0, 20.0]
generalization_errors = []

print("\nCross-mass generalization (trained at m=5):")
for m_test in mass_test_values:
    test_x_m, test_y_m = generate_klein_gordon_dataset(50, nx, m_test, seed=99)
    errors = []
    with torch.no_grad():
        for i in range(test_x_m.shape[0]):
            pred = model(test_x_m[i : i + 1].to(device))
            err = torch.norm(pred - test_y_m[i : i + 1]) / torch.norm(
                test_y_m[i : i + 1]
            )
            errors.append(err.item())
    mean_err = np.mean(errors)
    generalization_errors.append(mean_err)
    marker = " (training)" if m_test == mass_train else ""
    print(f"  m={m_test:5.1f}: relative L2 error = {mean_err:.4f}{marker}")

sys.stdout.flush()

# %%
# .. raw:: html
#
#    <div style="margin-top: 3em;"></div>
#
# Generalization error vs mass parameter
# ----------------------------------------
# The plot shows how prediction error increases as the test mass parameter
# moves further from the training mass. This is expected: the mass term
# :math:`m^2 u` fundamentally changes the dispersion relation, so the
# learned operator does not transfer directly.

fig, ax = plt.subplots(figsize=(8, 4))
colors = ["green" if m == mass_train else "steelblue" for m in mass_test_values]
ax.bar(range(len(mass_test_values)), generalization_errors, color=colors)
ax.set_xticks(range(len(mass_test_values)))
ax.set_xticklabels([f"m={m}" for m in mass_test_values])
ax.set_ylabel("Relative L2 error")
ax.set_title("Cross-mass generalization of FNO (trained at m=5)")
ax.axhline(
    y=generalization_errors[mass_test_values.index(mass_train)],
    color="green",
    linestyle="--",
    alpha=0.5,
    label="Training mass error",
)
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
# This example demonstrated:
#
# - The FNO can learn the Klein-Gordon time-evolution operator for a fixed mass.
# - Training on hyperbolic PDEs works with the standard neuralop pipeline.
# - Cross-mass generalization degrades as the test mass moves further from
#   the training mass. This is expected since the mass term changes the
#   underlying dispersion relation, creating qualitatively different dynamics.
#
# For better generalization across parameters, one could include the mass as
# an additional input channel, or train on data sampled from a range of masses.
# This is an open area of research in the neural operator community.
