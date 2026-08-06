"""
Benchmark: FNO vs KG-FNO on Klein-Gordon Equation
==================================================

Compares standard FNO (unconstrained SpectralConv) against KG-FNO
(KGSpectralConv with physics-constrained filter) on the Klein-Gordon
equation time-evolution task.

Both models map u(x, t=0) -> u(x, t=T) for the 1D Klein-Gordon PDE:
    d²u/dt² = c² d²u/dx² - m² u

Metrics: relative L2 error, H1 loss, parameter count, training time.
"""

import time
import json
import torch
import numpy as np
from neuralop.models import FNO
from neuralop.layers.kg_spectral_conv import KGSpectralConv

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# ── Data generation ─────────────────────────────────────────────────


def solve_klein_gordon_1d(u0, nx, dx, dt, nt, c, m):
    """Leapfrog solver for 1D Klein-Gordon equation."""
    u = u0.clone()

    def laplacian(v):
        return (torch.roll(v, -1, 0) + torch.roll(v, 1, 0) - 2 * v) / dx**2

    u_xx = laplacian(u)
    u_prev = u + 0.5 * dt**2 * (c**2 * u_xx - m**2 * u)

    for _ in range(nt):
        u_xx = laplacian(u)
        u_next = 2 * u - u_prev + dt**2 * (c**2 * u_xx - m**2 * u)
        u_prev = u.clone()
        u = u_next.clone()

    return u


def generate_dataset(n_samples, nx, mass, c=1.0, T=0.5, seed=0):
    """Generate KG input-output pairs from random Fourier initial conditions."""
    rng = np.random.RandomState(seed)
    dx = 1.0 / nx
    dt = 0.4 * dx / c  # CFL stable
    nt = int(T / dt)

    inputs = []
    outputs = []

    for _ in range(n_samples):
        # Random superposition of Fourier modes
        n_fourier = rng.randint(3, 10)
        x = torch.linspace(0, 1, nx, device="cpu")
        u0 = torch.zeros(nx)
        for _ in range(n_fourier):
            k = rng.randint(1, nx // 4)
            amp = rng.randn() * 0.5
            phase = rng.uniform(0, 2 * np.pi)
            u0 += amp * torch.sin(2 * np.pi * k * x + phase)

        # Normalize amplitude
        u0 = u0 / (u0.abs().max() + 1e-8)

        u_final = solve_klein_gordon_1d(u0, nx, dx, dt, nt, c, mass)
        inputs.append(u0.unsqueeze(0))
        outputs.append(u_final.unsqueeze(0))

    return torch.stack(inputs), torch.stack(outputs)


# ── Training loop ───────────────────────────────────────────────────


def relative_l2(pred, target):
    """Relative L2 error per sample, averaged."""
    diff = (pred - target).flatten(1)
    ref = target.flatten(1)
    return (diff.norm(dim=1) / (ref.norm(dim=1) + 1e-8)).mean().item()


def train_and_evaluate(
    model, train_x, train_y, test_x, test_y, epochs=50, lr=1e-2, batch_size=32, label=""
):
    """Train model and return metrics dictionary."""
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    n_train = train_x.shape[0]
    train_x_d = train_x.to(device)
    train_y_d = train_y.to(device)
    test_x_d = test_x.to(device)
    test_y_d = test_y.to(device)

    # Training
    t0 = time.time()
    model.train()
    loss_history = []

    for epoch in range(epochs):
        perm = torch.randperm(n_train)
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, n_train, batch_size):
            idx = perm[i : i + batch_size]
            bx = train_x_d[idx]
            by = train_y_d[idx]

            pred = model(bx)
            loss = ((pred - by) ** 2).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / n_batches
        loss_history.append(avg_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  [{label}] Epoch {epoch+1:3d}/{epochs}  loss={avg_loss:.6f}")

    train_time = time.time() - t0

    # Evaluation
    model.eval()
    with torch.no_grad():
        pred_test = model(test_x_d)
        pred_train = model(train_x_d)

    test_rl2 = relative_l2(pred_test, test_y_d)
    train_rl2 = relative_l2(pred_train, train_y_d)

    result = {
        "label": label,
        "n_params": n_params,
        "train_time_s": round(train_time, 2),
        "train_rel_l2": round(train_rl2, 6),
        "test_rel_l2": round(test_rl2, 6),
        "final_loss": round(loss_history[-1], 8),
    }
    return result


# ── Main benchmark ──────────────────────────────────────────────────

if __name__ == "__main__":
    # Configuration
    nx = 64
    n_train = 500
    n_test = 100
    hidden = 32
    modes = 16
    epochs = 80
    lr = 1e-2

    # Mass values: 0 (wave), 5 (moderate KG), 15 (heavy KG)
    mass_values = [0.0, 5.0, 15.0]

    all_results = []

    for mass in mass_values:
        print(f"\n{'='*60}")
        print(f"  MASS = {mass}  (nx={nx}, {n_train} train, {n_test} test)")
        print(f"{'='*60}")

        train_x, train_y = generate_dataset(n_train, nx, mass, seed=42)
        test_x, test_y = generate_dataset(n_test, nx, mass, seed=999)

        # Standard FNO
        fno = FNO(
            n_modes=(modes,),
            in_channels=1,
            out_channels=1,
            hidden_channels=hidden,
            n_layers=4,
        )
        res_fno = train_and_evaluate(
            fno,
            train_x,
            train_y,
            test_x,
            test_y,
            epochs=epochs,
            lr=lr,
            label=f"FNO m={mass}",
        )
        res_fno["mass"] = mass
        res_fno["model"] = "FNO"
        all_results.append(res_fno)

        # KG-FNO
        fno_kg = FNO(
            n_modes=(modes,),
            in_channels=1,
            out_channels=1,
            hidden_channels=hidden,
            n_layers=4,
            conv_module=KGSpectralConv,
        )
        res_kg = train_and_evaluate(
            fno_kg,
            train_x,
            train_y,
            test_x,
            test_y,
            epochs=epochs,
            lr=lr,
            label=f"KG-FNO m={mass}",
        )
        res_kg["mass"] = mass
        res_kg["model"] = "KG-FNO"
        all_results.append(res_kg)

        # Print comparison
        print(f"\n  Results for mass={mass}:")
        print(
            f"  {'Model':<12} {'Params':>8} {'Train L2':>10} {'Test L2':>10} {'Time':>8}"
        )
        print(f"  {'-'*50}")
        print(
            f"  {'FNO':<12} {res_fno['n_params']:>8,} {res_fno['train_rel_l2']:>10.4%} {res_fno['test_rel_l2']:>10.4%} {res_fno['train_time_s']:>7.1f}s"
        )
        print(
            f"  {'KG-FNO':<12} {res_kg['n_params']:>8,} {res_kg['train_rel_l2']:>10.4%} {res_kg['test_rel_l2']:>10.4%} {res_kg['train_time_s']:>7.1f}s"
        )
        ratio = res_fno["n_params"] / res_kg["n_params"]
        print(f"  Parameter ratio: {ratio:.1f}x fewer in KG-FNO")

    # Final summary
    print(f"\n\n{'='*60}")
    print(f"  FINAL SUMMARY")
    print(f"{'='*60}")
    print(
        f"\n  {'Model':<12} {'Mass':>6} {'Params':>8} {'Train L2':>10} {'Test L2':>10} {'Time':>8}"
    )
    print(f"  {'-'*58}")
    for r in all_results:
        print(
            f"  {r['model']:<12} {r['mass']:>6.0f} {r['n_params']:>8,} {r['train_rel_l2']:>10.4%} {r['test_rel_l2']:>10.4%} {r['train_time_s']:>7.1f}s"
        )

    # Save results
    with open("benchmark_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nResults saved to benchmark_results.json")
