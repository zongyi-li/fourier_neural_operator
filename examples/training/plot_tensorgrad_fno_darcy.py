"""
Training an FNO with TensorGRaD
==============================

This example compares three optimizers on Darcy-Flow:

- AdamW baseline
- TensorGRaD low-rank gradients with a 25% budget
- TensorGRaD low-rank + sparse residual gradients with a matched 20% + 5%
  budget

The example follows the standard FNO Darcy workflow while using
``DarcyDataset(download=True)`` so it does not rely on repository-local data.
It records train H1, validation H1, and validation L2 losses for all three
methods and visualizes their predictions side by side.
"""

# sphinx_gallery_thumbnail_number = 2

# %%
# .. raw:: html
#
#    <div style="margin-top: 3em;"></div>
#
# Import dependencies
# -------------------

from pathlib import Path
import sys

for path in [Path.cwd(), *Path.cwd().parents]:
    if (path / "neuralop" / "__init__.py").exists():
        repo_root = path
        break
else:
    repo_root = None

if repo_root is not None and str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from neuralop import H1Loss, LpLoss, Trainer
from neuralop.data.datasets.darcy import DarcyDataset
from neuralop.models import FNO
from neuralop.training import (
    AdamW as BaselineAdamW,
    TensorGRaD,
    fno_tensorgrad_param_groups,
)
from neuralop.utils import count_model_params

plt.rcParams["figure.dpi"] = 110


# %%
# .. raw:: html
#
#    <div style="margin-top: 3em;"></div>
#
# Configuration
# -------------
# TensorGRaD is applied to the spectral convolution weights. Other FNO
# parameters, such as lifting/projection layers and skip connections, remain in
# a regular AdamW parameter group.

N_TRAIN = 1000
N_TEST = 100
BATCH_SIZE = 128
N_EPOCHS = 15
LEARNING_RATE = 1e-2
WEIGHT_DECAY = 1e-4
HIDDEN_CHANNELS = 24
N_MODES = (8, 8)
UPDATE_PROJ_GAP = 4
LOW_RANK_SCALE = 10.0
SPARSE_SCALE = 1.0
LAMBDA_SPARSE = 1.0
SPARSE_TYPE = "randk"

LOW_RANK_BUDGET = 0.25
LOW_RANK_SPARSE_LOW_RANK_BUDGET = 0.20
LOW_RANK_SPARSE_SPARSE_BUDGET = 0.05
assert abs(
    LOW_RANK_BUDGET
    - (LOW_RANK_SPARSE_LOW_RANK_BUDGET + LOW_RANK_SPARSE_SPARSE_BUDGET)
) < 1e-12

DATA_ROOT = Path.home() / ".cache" / "neuraloperator" / "darcy"
TEST_RESOLUTIONS = [16, 32]
TEST_BATCH_SIZES = [BATCH_SIZE, BATCH_SIZE]

torch.manual_seed(0)
np.random.seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# %%
# .. raw:: html
#
#    <div style="margin-top: 3em;"></div>
#
# Loading the Darcy-Flow dataset
# ------------------------------
# ``DarcyDataset(download=True)`` downloads the public Darcy files from Zenodo
# if they are not already present in ``DATA_ROOT``.

darcy_dataset = DarcyDataset(
    root_dir=DATA_ROOT,
    n_train=N_TRAIN,
    n_tests=[N_TEST, N_TEST],
    batch_size=BATCH_SIZE,
    test_batch_sizes=TEST_BATCH_SIZES,
    train_resolution=16,
    test_resolutions=TEST_RESOLUTIONS,
    encode_input=False,
    encode_output=True,
    download=True,
)

train_loader = DataLoader(
    darcy_dataset.train_db,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)
test_loaders = {
    resolution: DataLoader(
        darcy_dataset.test_dbs[resolution],
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    for resolution, batch_size in zip(TEST_RESOLUTIONS, TEST_BATCH_SIZES)
}
data_processor = darcy_dataset.data_processor.to(device)
print(f"Loaded Darcy-Flow test resolutions: {list(test_loaders)}")


# %%
# .. raw:: html
#
#    <div style="margin-top: 3em;"></div>
#
# Creating the FNO model
# ----------------------


def make_model():
    torch.manual_seed(0)
    return FNO(
        n_modes=N_MODES,
        in_channels=1,
        out_channels=1,
        hidden_channels=HIDDEN_CHANNELS,
        projection_channel_ratio=2,
    ).to(device)


preview_model = make_model()
print(f"FNO parameters: {count_model_params(preview_model):,}")
del preview_model


# %%
# .. raw:: html
#
#    <div style="margin-top: 3em;"></div>
#
# Creating the optimizers
# -----------------------
# The compressed-gradient budgets are matched: low-rank uses a 25% budget, and
# low-rank+sparse uses a 20% low-rank budget plus a 5% sparse residual budget.


def adamw_optimizer(model):
    return BaselineAdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )


def tensorgrad_low_rank_optimizer(model):
    param_groups = fno_tensorgrad_param_groups(
        model,
        rank=LOW_RANK_BUDGET,
        min_params=1000,
        update_proj_gap=UPDATE_PROJ_GAP,
        scale=LOW_RANK_SCALE,
    )
    return TensorGRaD(param_groups, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)


def tensorgrad_low_rank_sparse_optimizer(model):
    param_groups = fno_tensorgrad_param_groups(
        model,
        rank=LOW_RANK_SPARSE_LOW_RANK_BUDGET,
        sparse_ratio=LOW_RANK_SPARSE_SPARSE_BUDGET,
        min_params=1000,
        update_proj_gap=UPDATE_PROJ_GAP,
        scale=LOW_RANK_SCALE,
        sparse_scale=SPARSE_SCALE,
        sparse_type=SPARSE_TYPE,
        lambda_sparse=LAMBDA_SPARSE,
    )
    return TensorGRaD(param_groups, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)


print(f"Low-rank budget: {LOW_RANK_BUDGET:.0%}")
print(
    "Low-rank + sparse budget: "
    f"{LOW_RANK_SPARSE_LOW_RANK_BUDGET:.0%} + "
    f"{LOW_RANK_SPARSE_SPARSE_BUDGET:.0%} = "
    f"{LOW_RANK_BUDGET:.0%}"
)


# %%
# .. raw:: html
#
#    <div style="margin-top: 3em;"></div>
#
# Setting up losses
# -----------------

l2loss = LpLoss(d=2, p=2)
h1loss = H1Loss(d=2)

train_loss = h1loss
eval_losses = {"h1": h1loss, "l2": l2loss}


# %%
# .. raw:: html
#
#    <div style="margin-top: 3em;"></div>
#
# Training all three models
# -------------------------
# We subclass ``Trainer`` only to store epoch-wise train and validation metrics
# so we can plot all three methods on the same axes.


def to_float(value):
    if torch.is_tensor(value):
        return float(value.detach().cpu())
    if value is None:
        return None
    return float(value)


class HistoryTrainer(Trainer):
    def train_one_epoch(self, epoch, train_loader, training_loss):
        train_err, avg_loss, avg_lasso_loss, epoch_train_time = super().train_one_epoch(
            epoch,
            train_loader,
            training_loss,
        )
        self._latest_history_row = {
            "epoch": epoch,
            "step": (epoch + 1) * len(train_loader),
            "train_err": to_float(train_err),
            "avg_loss": to_float(avg_loss),
            "epoch_train_time": to_float(epoch_train_time),
        }
        return train_err, avg_loss, avg_lasso_loss, epoch_train_time

    def evaluate_all(
        self,
        epoch,
        eval_losses,
        test_loaders,
        eval_modes,
        max_autoregressive_steps=None,
    ):
        eval_metrics = super().evaluate_all(
            epoch=epoch,
            eval_losses=eval_losses,
            test_loaders=test_loaders,
            eval_modes=eval_modes,
            max_autoregressive_steps=max_autoregressive_steps,
        )
        row = dict(getattr(self, "_latest_history_row", {"epoch": epoch}))
        row.update({key: to_float(value) for key, value in eval_metrics.items()})
        self.history.append(row)
        return eval_metrics


def train_with_optimizer(name, optimizer_builder):
    torch.manual_seed(0)
    np.random.seed(0)
    model = make_model()
    optimizer = optimizer_builder(model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)

    print(f"\n### {name} ###")
    print(f"Optimizer: {optimizer.__class__.__name__}")

    trainer = HistoryTrainer(
        model=model,
        n_epochs=N_EPOCHS,
        device=device,
        data_processor=data_processor,
        wandb_log=False,
        eval_interval=1,
        use_distributed=False,
        verbose=True,
    )
    trainer.history = []
    trainer.train(
        train_loader=train_loader,
        test_loaders=test_loaders,
        optimizer=optimizer,
        scheduler=scheduler,
        regularizer=False,
        training_loss=train_loss,
        eval_losses=eval_losses,
    )
    return model, trainer.history


optimizer_builders = {
    "AdamW": adamw_optimizer,
    "Low-rank 25%": tensorgrad_low_rank_optimizer,
    "Low-rank 20% + sparse 5%": tensorgrad_low_rank_sparse_optimizer,
}

models = {}
histories = {}
for name, optimizer_builder in optimizer_builders.items():
    models[name], histories[name] = train_with_optimizer(name, optimizer_builder)


# %%
# .. raw:: html
#
#    <div style="margin-top: 3em;"></div>
#
# Reporting best validation losses
# --------------------------------

plot_resolution = max(test_loaders)
eval_h1_key = f"{plot_resolution}_h1"
eval_l2_key = f"{plot_resolution}_l2"


def best_metric(history, key):
    best_row = min(history, key=lambda row: row[key])
    return best_row[key], best_row["step"]


summary_rows = []
for name, history in histories.items():
    best_h1, best_h1_step = best_metric(history, eval_h1_key)
    best_l2, best_l2_step = best_metric(history, eval_l2_key)
    summary_rows.append(
        {
            "method": name,
            "best_val_h1": best_h1,
            "best_val_h1_step": best_h1_step,
            "best_val_l2": best_l2,
            "best_val_l2_step": best_l2_step,
        }
    )

header = (
    f"{'method':<28} {'best_h1':>12} {'h1_step':>8} "
    f"{'best_l2':>12} {'l2_step':>8}"
)
summary_lines = [header, "-" * len(header)]
for row in summary_rows:
    summary_lines.append(
        f"{row['method']:<28} "
        f"{row['best_val_h1']:>12.5g} "
        f"{row['best_val_h1_step']:>8} "
        f"{row['best_val_l2']:>12.5g} "
        f"{row['best_val_l2_step']:>8}"
    )

summary_text = "\n".join(summary_lines)
print(summary_text)


# %%
# .. raw:: html
#
#    <div style="margin-top: 3em;"></div>
#
# Loss curves
# -----------
# All panels use optimizer steps on the x-axis. Validation H1 and L2 are shown
# at the highest test resolution.

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for name, history in histories.items():
    steps = [row["step"] for row in history]
    axes[0].plot(steps, [row["avg_loss"] for row in history], marker="o", label=name)
    axes[1].plot(steps, [row[eval_h1_key] for row in history], marker="o", label=name)
    axes[2].plot(steps, [row[eval_l2_key] for row in history], marker="o", label=name)

axes[0].set_title("Train H1 loss")
axes[1].set_title(f"Validation H1 loss ({plot_resolution} x {plot_resolution})")
axes[2].set_title(f"Validation L2 loss ({plot_resolution} x {plot_resolution})")

for ax in axes:
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()

plt.tight_layout()
fig.show()


# %%
# .. raw:: html
#
#    <div style="margin-top: 3em;"></div>
#
# Visualizing predictions
# -----------------------
# The columns compare the input, ground truth, AdamW prediction, low-rank
# prediction, and low-rank+sparse prediction.

test_samples = test_loaders[plot_resolution].dataset
prediction_labels = ["AdamW", "Low-rank 25%", "Low-rank 20% + sparse 5%"]

fig = plt.figure(figsize=(12, 7))
for index in range(3):
    data = test_samples[index]
    data = data_processor.preprocess(data, batched=False)

    x = data["x"].to(device)
    y = data["y"].to(device)

    with torch.no_grad():
        predictions = {
            label: models[label](x.unsqueeze(0)).squeeze().detach().cpu()
            for label in prediction_labels
        }

    panels = [
        ("Input x", x[0].detach().cpu(), "gray"),
        ("Ground truth y", y.squeeze().detach().cpu(), None),
        *[(label, predictions[label], None) for label in prediction_labels],
    ]
    for column, (title, image, cmap) in enumerate(panels):
        ax = fig.add_subplot(3, 5, index * 5 + column + 1)
        ax.imshow(image, cmap=cmap)
        if index == 0:
            ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])

fig.suptitle(
    f"Darcy-Flow FNO predictions at {plot_resolution} x {plot_resolution}",
    y=0.98,
)
plt.tight_layout()
fig.show()
