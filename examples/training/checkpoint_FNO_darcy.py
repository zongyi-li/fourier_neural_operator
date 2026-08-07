"""
Checkpointing and loading training states
=========================================

Demonstrating the ``Trainer``'s saving and loading functionality, 
which makes it easy to checkpoint and resume training states.

"""

# %%
# .. raw:: html
#
#    <div style="margin-top: 3em;"></div>
#
# Import dependencies
# -------------------

import torch
import matplotlib.pyplot as plt
import sys
from neuralop.models import FNO
from neuralop import Trainer
from neuralop.training import AdamW
from neuralop.data.datasets import load_darcy_flow_small
from neuralop.utils import count_model_params
from neuralop import LpLoss, H1Loss

device = "cpu"


# %%
# .. raw:: html
#
#    <div style="margin-top: 3em;"></div>
#
# Loading the Darcy-Flow dataset
# ------------------------------
train_loader, test_loaders, data_processor = load_darcy_flow_small(
    n_train=1000,
    batch_size=32,
    test_resolutions=[16, 32],
    n_tests=[100, 50],
    test_batch_sizes=[32, 32],
)


# %%
# .. raw:: html
#
#    <div style="margin-top: 3em;"></div>
#
# Creating the FNO model
# ----------------------

model = FNO(
    n_modes=(16, 16),
    in_channels=1,
    out_channels=1,
    hidden_channels=32,
    projection_channel_ratio=2,
    factorization="tucker",
    rank=0.42,
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
# Creating the optimizer and scheduler
# ------------------------------------
optimizer = AdamW(model.parameters(), lr=8e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)


# %%
# .. raw:: html
#
#    <div style="margin-top: 3em;"></div>
#
# Creating the losses
# ------------------
l2loss = LpLoss(d=2, p=2)
h1loss = H1Loss(d=2)

train_loss = h1loss
eval_losses = {"h1": h1loss, "l2": l2loss}


# %%
# .. raw:: html
#
#    <div style="margin-top: 3em;"></div>
#
# Displaying configuration
# ------------------------

print("\n### MODEL ###\n", model)
print("\n### OPTIMIZER ###\n", optimizer)
print("\n### SCHEDULER ###\n", scheduler)
print("\n### LOSSES ###")
print(f"\n * Train: {train_loss}")
print(f"\n * Test: {eval_losses}")
sys.stdout.flush()


# %%
# .. raw:: html
#
#    <div style="margin-top: 3em;"></div>
#
# Creating the trainer
# --------------------
trainer = Trainer(
    model=model,
    n_epochs=20,
    device=device,
    data_processor=data_processor,
    wandb_log=False,
    eval_interval=3,
    use_distributed=False,
    verbose=True,
)


# %%
# .. raw:: html
#
#    <div style="margin-top: 3em;"></div>
#
# Training the model
# ------------------
# We train and save checkpoints

trainer.train(
    train_loader=train_loader,
    test_loaders={},
    optimizer=optimizer,
    scheduler=scheduler,
    regularizer=False,
    training_loss=train_loss,
    eval_losses=eval_losses,
    save_every=1,
    save_dir="./checkpoints",
)


# .. resume_from_dir:
# resume training from saved checkpoint at epoch 10

trainer = Trainer(
    model=model,
    n_epochs=20,
    device=device,
    data_processor=data_processor,
    wandb_log=False,
    eval_interval=3,
    use_distributed=False,
    verbose=True,
)

trainer.train(
    train_loader=train_loader,
    test_loaders={},
    optimizer=optimizer,
    scheduler=scheduler,
    regularizer=False,
    training_loss=train_loss,
    eval_losses=eval_losses,
    resume_from_dir="./checkpoints",
)
