"""
Training an FNO on Darcy-Flow using scale-consistent training scheme (https://arxiv.org/abs/2507.18813)
==============================================================================

This example demonstrates how to train a Fourier Neural Operator (FNO) on the Darcy-Flow problem using a scale-consistent training scheme. 
The dataset used is a boundary-valued Darcy-Flow dataset, and the model is trained using both standard and scale-consistent training methods.
"""

# %%
# 
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import sys
from neuralop.models import FNO
from neuralop.training import AdamW
from neuralop.data.datasets.darcy import load_darcy_pt
from neuralop.data.datasets.pt_dataset import PTDataset
from neuralop.utils import count_model_params
from neuralop import LpLoss, H1Loss
from neuralop.losses.scale_consistency_losses import LossSelfconsistency
from neuralop import Trainer
from neuralop.training.scale_consistency import SelfConsistencyTrainer

# %%
# Choose device
device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")

# %%
# Loading the boundary-valued Darcy-Flow dataset

dataset = PTDataset(root_dir = "neuralop/data/datasets/data",
                    dataset_name = "darcy",
                    n_train=1024,
                    n_tests=[128,],
                    batch_size=32,
                    test_batch_sizes=[32,],
                    train_resolution=128,
                    test_resolutions=[128,],
                    channel_dim=1,
                    channels_squeezed=False)

train_loader = DataLoader(dataset.train_db, batch_size=32,)
test_loader = DataLoader(dataset.test_dbs[128], batch_size=32,)
test_loaders = {128: test_loader}

data_processor = dataset.data_processor.to(device)

# %%
# Create a simple FNO model

model = FNO(n_modes=(32, 32),
             in_channels=5, 
             out_channels=1,
             hidden_channels=32, 
             projection_channel_ratio=2)
model = model.to(device)

n_params = count_model_params(model)
print(f'\nOur model has {n_params} parameters.')
sys.stdout.flush()

# %%
# Training setup


# %%
# Then create the losses
l2loss = LpLoss(d=2, p=2)
h1loss = H1Loss(d=2)

train_loss = l2loss
eval_losses={'h1': h1loss, 'l2': l2loss}

# %%
# Training the modelpy
# ---------------------

print('\n### MODEL ###\n', model)
print('\n### LOSSES ###')
print(f'\n * Train: {train_loss}')
print(f'\n * Test: {eval_losses}')
sys.stdout.flush()

# %% 
# Use a standard the trainer:
optimizer = AdamW(model.parameters(), lr=1e-3,  weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

trainer = Trainer(model=model, n_epochs=20,
                  device=device,
                  data_processor=data_processor,
                  wandb_log=False,
                  eval_interval=3,
                  use_distributed=False,
                  verbose=True)

trainer.train(train_loader=train_loader,
            test_loaders=test_loaders,
            optimizer=optimizer,
            scheduler=scheduler, 
            regularizer=False, 
            training_loss=train_loss,
            eval_losses=eval_losses)


# %% 
# Now use scale-consistent trainer:
optimizer = AdamW(model.parameters(), lr=1e-3,  weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

trainer = SelfConsistencyTrainer(model=model, n_epochs=20,
            selfconsistency_loss = LossSelfconsistency,
            device=device,
            data_processor=data_processor,
            wandb_log=False,
            eval_interval=3,
            use_distributed=False,
            verbose=True)

trainer.train(train_loader=train_loader,
            test_loaders=test_loaders,
            optimizer=optimizer,
            scheduler=scheduler, 
            regularizer=False, 
            training_loss=train_loss,
            eval_losses=eval_losses)