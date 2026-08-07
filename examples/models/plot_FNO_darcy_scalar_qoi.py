"""
FNO scalar QoI prediction with resolution-invariant readout
===========================================================

This example trains an FNO to predict a scalar quantity of interest (QoI)
from Darcy-Flow fields. We define the target as:

``y_scalar = mean(y_field)``

and use a resolution-invariant readout to map ``(B, C, H, W) -> (B, 1)``.
The model is trained at 16×16 and evaluated at both the training resolution
and zero-shot at 32×32.

By default the example uses a small in-memory synthetic dataset so it runs
without downloading anything.  To use the real Darcy-Flow dataset from Zenodo,
set ``USE_DARCY_DATA = True`` below and note that output-field normalization is
disabled (``encode_output=False``) because the target is a derived scalar QoI
rather than the full output field.
"""

# %%
# .. raw:: html
#
#    <div style="margin-top: 3em;"></div>
#
# Configuration
# -------------
# Set ``USE_DARCY_DATA = True`` to download and use the real Darcy-Flow
# dataset from Zenodo instead of the synthetic stand-in.

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from neuralop import Trainer
from neuralop.data.datasets import load_darcy_flow_small
from neuralop.data.transforms.data_processors import DefaultDataProcessor
from neuralop.models import FNO, ResolutionInvariantReadout
from neuralop.training import AdamW

USE_DARCY_DATA = False  # set True to use the real Zenodo dataset
N_EPOCHS = 3
BATCH_SIZE = 32
device = "cuda" if torch.cuda.is_available() else "cpu"


# %%
# .. raw:: html
#
#    <div style="margin-top: 3em;"></div>
#
# Dataset
# -------
# We use a tiny synthetic Darcy-like dataset when ``USE_DARCY_DATA`` is False,
# or load the real small Darcy-Flow dataset otherwise.


class ScalarQOILoss(nn.Module):
    """Sum of squared errors against scalar QoI computed from y, ignoring extra kwargs."""

    def forward(self, out, y, **kwargs):
        y_scalar = y.reshape(y.shape[0], -1).mean(dim=1, keepdim=True)
        return torch.sum((out - y_scalar) ** 2)


class SyntheticDarcyLikeDataset(Dataset):
    """Small deterministic dataset for smoke testing the QoI example."""

    def __init__(self, n_samples: int, resolution: int):
        self.n_samples = n_samples
        coords = torch.linspace(0.0, 1.0, resolution)
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        self.base_x = (xx + yy).unsqueeze(0)
        self.base_y = (torch.sin(torch.pi * xx) * torch.cos(torch.pi * yy)).unsqueeze(0)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        scale = (idx + 1) / self.n_samples
        x = self.base_x * scale
        y = self.base_y * (1.0 + 0.5 * scale)
        return {"x": x, "y": y}


if USE_DARCY_DATA:
    # Disable output normalization: target is a derived scalar, not the field.
    train_loader, test_loaders, data_processor = load_darcy_flow_small(
        n_train=256,
        batch_size=BATCH_SIZE,
        n_tests=[64, 64],
        test_resolutions=[16, 32],
        test_batch_sizes=[BATCH_SIZE, BATCH_SIZE],
        encode_output=False,
    )
else:
    train_loader = DataLoader(
        SyntheticDarcyLikeDataset(n_samples=32, resolution=16),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    test_loaders = {
        16: DataLoader(
            SyntheticDarcyLikeDataset(n_samples=8, resolution=16), batch_size=BATCH_SIZE
        ),
        32: DataLoader(
            SyntheticDarcyLikeDataset(n_samples=8, resolution=32), batch_size=BATCH_SIZE
        ),
    }
    data_processor = DefaultDataProcessor()

data_processor = data_processor.to(device)


# %%
# .. raw:: html
#
#    <div style="margin-top: 3em;"></div>
#
# Model
# -----
# The FNO maps input fields to a latent field, then the
# :class:`~neuralop.models.ResolutionInvariantReadout` pools the field to a
# single scalar per sample.

model = FNO(
    n_modes=(8, 8),
    in_channels=1,
    out_channels=16,
    hidden_channels=24,
    projection_channel_ratio=2,
    readout=ResolutionInvariantReadout(
        in_channels=16,
        out_dim=1,
        reduce="mean",
        head="mlp",
        mlp_hidden_dim=32,
    ),
).to(device)


# %%
# .. raw:: html
#
#    <div style="margin-top: 3em;"></div>
#
# Training
# --------

optimizer = AdamW(model.parameters(), lr=5e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)
qoi_loss = ScalarQOILoss()

trainer = Trainer(
    model=model,
    n_epochs=N_EPOCHS,
    device=device,
    data_processor=data_processor,
    wandb_log=False,
    eval_interval=1,
    use_distributed=False,
    verbose=True,
)

trainer.train(
    train_loader=train_loader,
    test_loaders=test_loaders,
    optimizer=optimizer,
    scheduler=scheduler,
    regularizer=False,
    training_loss=qoi_loss,
    eval_losses={"qoi_mse": qoi_loss},
)


# %%
# .. raw:: html
#
#    <div style="margin-top: 3em;"></div>
#
# Zero-shot evaluation at multiple resolutions
# --------------------------------------------
# The readout pools the field regardless of spatial size, so the same trained
# model can be evaluated at resolutions it was never trained on.

model.eval()
with torch.no_grad():
    for resolution in sorted(test_loaders):
        sample = next(iter(test_loaders[resolution]))
        sample = data_processor.preprocess(sample)
        out = model(sample["x"])
        y_scalar = (
            sample["y"].reshape(sample["y"].shape[0], -1).mean(dim=1, keepdim=True)
        )
        mse = torch.mean((out - y_scalar) ** 2).item()
        print(f"Resolution {resolution}x{resolution} scalar QoI MSE: {mse:.6f}")
