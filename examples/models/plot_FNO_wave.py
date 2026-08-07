"""
.. _fno_wave_equation :

Training an FNO on the 2D Wave Equation
========================================

We train a Fourier Neural Operator (FNO) to learn the solution operator
for the 2D wave equation: given an initial condition, predict the solution
at a later time.

The wave equation is a fundamental hyperbolic PDE:

.. math::
    \\frac{\\partial^2 u}{\\partial t^2} = c^2 \\nabla^2 u

Unlike elliptic problems (Darcy flow) and parabolic problems (diffusion),
hyperbolic PDEs propagate sharp features without smoothing — making them
a challenging and important benchmark for neural operators.
"""

# %%
# .. raw:: html
#
#    <div style="margin-top: 3em;"></div>
#
# Import dependencies
# -------------------
# We import the necessary modules from ``neuralop`` for training a Fourier Neural Operator.

import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
from neuralop.models import FNO
from neuralop.training import AdamW
from neuralop.utils import count_model_params
from neuralop import LpLoss, H1Loss
from neuralop.losses.differentiation import FiniteDiff

device = "cpu"

# %%
# .. raw:: html
#
#    <div style="margin-top: 3em;"></div>
#
# Generate wave equation dataset
# ------------------------------
# We use the leapfrog scheme to generate ground-truth wave equation data.
# Each sample has a random superposition of Gaussian bumps as the initial
# condition, and we solve forward to a target time.
#
# Our neural operator will learn the mapping:
#
# .. math::
#    \mathcal{G}: u(x, y, t=0) \mapsto u(x, y, t=T)


def generate_wave_samples(n_samples, nx=64, c=1.0, T=0.5, dt=0.005, seed=42):
    """Generate wave equation samples using leapfrog integration.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate.
    nx : int
        Grid resolution (nx x nx).
    c : float
        Wave speed.
    T : float
        Target time to predict.
    dt : float
        Time step.

    Returns
    -------
    inputs : Tensor of shape (n_samples, 1, nx, nx)
        Initial conditions u(x, y, t=0).
    outputs : Tensor of shape (n_samples, 1, nx, nx)
        Solutions u(x, y, t=T).
    """
    Lx = 2.0
    dx = Lx / nx  # periodic grid spacing (no endpoint)
    nt = int(T / dt)
    fd = FiniteDiff(dim=2, h=(dx, dx))

    inputs = []
    outputs = []
    rng = np.random.default_rng(seed)

    for _ in range(n_samples):
        # Random initial condition: superposition of 1-4 Gaussian bumps
        n_bumps = rng.integers(1, 5)
        x_coords = torch.linspace(0, Lx - dx, nx, device=device)
        X = x_coords.repeat(nx, 1).T
        Y = x_coords.repeat(nx, 1)

        u = torch.zeros(nx, nx, device=device)
        for _ in range(n_bumps):
            cx = rng.uniform(0.3, 1.7)
            cy = rng.uniform(0.3, 1.7)
            sigma = rng.uniform(0.02, 0.06)
            amp = rng.uniform(0.5, 1.5)
            u += amp * torch.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / sigma)

        inputs.append(u.clone())

        # Leapfrog integration with 2nd-order accurate start (zero initial velocity):
        # u^{-1} = u^0 + 0.5 * (c*dt)^2 * laplacian(u^0)
        lap0 = fd.laplacian(u)
        u_prev = u + 0.5 * (c * dt) ** 2 * lap0
        for _ in range(nt):
            lap = fd.laplacian(u)
            u_next = 2 * u - u_prev + (c * dt) ** 2 * lap
            u_prev = u.clone()
            u = u_next.clone()

        outputs.append(u.clone())

    inputs = torch.stack(inputs).unsqueeze(1)  # (N, 1, nx, nx)
    outputs = torch.stack(outputs).unsqueeze(1)  # (N, 1, nx, nx)
    return inputs, outputs


# %%
# .. raw:: html
#
#    <div style="margin-top: 3em;"></div>
#
# Create dataset and data loaders
# --------------------------------
# We generate a small dataset with 200 training and 50 test samples
# at 64x64 resolution. In practice, a larger dataset would be used.

print("Generating training data...")
sys.stdout.flush()
train_x, train_y = generate_wave_samples(200, nx=64, c=1.0, T=0.5, seed=42)
print("Generating test data...")
sys.stdout.flush()
test_x, test_y = generate_wave_samples(50, nx=64, c=1.0, T=0.5, seed=12345)

# Wrap in TensorDataset and DataLoader
train_dataset = torch.utils.data.TensorDataset(train_x, train_y)
test_dataset = torch.utils.data.TensorDataset(test_x, test_y)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)

print(f"Training samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
print(f"Input shape: {train_x.shape}, Output shape: {train_y.shape}")
sys.stdout.flush()

# %%
# .. raw:: html
#
#    <div style="margin-top: 3em;"></div>
#
# Creating the FNO model
# ----------------------
# We use higher Fourier modes than for Darcy flow, since wave propagation
# involves multi-scale features that travel across the domain.

model = FNO(
    n_modes=(12, 12),
    in_channels=1,
    out_channels=1,
    hidden_channels=32,
    projection_channel_ratio=2,
)
model = model.to(device)

n_params = count_model_params(model)
print(f"\nOur model has {n_params} parameters.")
sys.stdout.flush()

# %%
# .. raw:: html
#
#    <div style="margin-top: 3em;"></div>
#
# Training setup
# ---------------
# We use the same training workflow as in the Darcy-Flow example:
# AdamW optimizer, cosine scheduler, H1 loss for training.

optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

l2loss = LpLoss(d=2, p=2)
h1loss = H1Loss(d=2)

train_loss = h1loss
eval_losses = {"h1": h1loss, "l2": l2loss}

print("\n### MODEL ###\n", model)
print("\n### OPTIMIZER ###\n", optimizer)
print("\n### LOSSES ###")
print(f"\n * Train: {train_loss}")
print(f"\n * Test: {eval_losses}")
sys.stdout.flush()

# %%
# .. raw:: html
#
#    <div style="margin-top: 3em;"></div>
#
# Training loop
# ---------------
# Since we generate our own data without a ``DataProcessor``, we run a
# manual training loop. The FNO learns to map initial wavefield
# configurations to their evolved states.

n_epochs = 30
for epoch in range(n_epochs):
    model.train()
    epoch_loss = 0.0
    n_batches = 0

    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        pred = model(x_batch)
        loss = train_loss(pred, y_batch)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        n_batches += 1

    scheduler.step()
    avg_loss = epoch_loss / n_batches

    if (epoch + 1) % 5 == 0:
        # Evaluate on test set
        model.eval()
        test_l2 = 0.0
        n_test = 0
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                pred = model(x_batch)
                test_l2 += l2loss(pred, y_batch).item()
                n_test += 1
        print(
            f"Epoch {epoch + 1:3d}/{n_epochs} | "
            f"Train H1: {avg_loss:.4f} | "
            f"Test L2: {test_l2 / n_test:.4f}"
        )
        sys.stdout.flush()

# %%
# .. raw:: html
#
#    <div style="margin-top: 3em;"></div>
#
# Visualizing predictions
# ------------------------
# We visualize the input (initial condition), ground truth (solution at t=T),
# and FNO prediction for several test samples.

model.eval()
fig = plt.figure(figsize=(10, 8))

for index in range(3):
    x = test_x[index : index + 1].to(device)
    y_true = test_y[index].squeeze().cpu().numpy()

    with torch.no_grad():
        y_pred = model(x).squeeze().cpu().numpy()

    x_np = test_x[index].squeeze().cpu().numpy()

    vmin = min(y_true.min(), y_pred.min())
    vmax = max(y_true.max(), y_pred.max())

    # Input (initial condition)
    ax = fig.add_subplot(3, 3, index * 3 + 1)
    ax.imshow(x_np, cmap="RdBu_r", origin="lower")
    if index == 0:
        ax.set_title("Initial condition")
    plt.xticks([], [])
    plt.yticks([], [])

    # Ground truth at t=T
    ax = fig.add_subplot(3, 3, index * 3 + 2)
    ax.imshow(y_true, cmap="RdBu_r", origin="lower", vmin=vmin, vmax=vmax)
    if index == 0:
        ax.set_title("Ground truth (t=0.5)")
    plt.xticks([], [])
    plt.yticks([], [])

    # FNO prediction
    ax = fig.add_subplot(3, 3, index * 3 + 3)
    ax.imshow(y_pred, cmap="RdBu_r", origin="lower", vmin=vmin, vmax=vmax)
    if index == 0:
        ax.set_title("FNO prediction")
    plt.xticks([], [])
    plt.yticks([], [])

fig.suptitle("FNO predictions on 2D Wave Equation", y=0.98)
plt.tight_layout()
fig.show()
