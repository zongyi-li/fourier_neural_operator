"""
Training an TFNO with navier-stokes
===============================================
Work in-progress...

A demo of the TFNO on Navier Stokes that also considerate the number of flops 
required for forward and backward process. Current version only consider training
on a single gpu or cpu. Future work will add arguments to the script that allow users 
to pick training mode on multiple gpus (if available). This allows testing speeds 
(flops) when training accross multiple gpus. 

Will continue to update in the next few days...

"""

import torch
import matplotlib.pyplot as plt
from neuralop.models import TFNO
from neuralop.data.datasets import load_navier_stokes_pt
from neuralop.utils import count_model_params
from neuralop.training import AdamW
from neuralop import LpLoss, H1Loss
from torchtnt.utils.flops import FlopTensorDispatchMode
from copy import deepcopy
from collections import defaultdict

from torch.profiler import profile, record_function, ProfilerActivity

# Load dataset 
train_loader, test_loaders, output_encoder = load_navier_stokes_pt(
    n_train=20,
    batch_size=2,
    test_resolutions=[128],
    n_tests=[3],
    test_batch_sizes=[2],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model 
model = TFNO(
    max_n_modes=(16, 16),
    n_modes=[128, 128],
    hidden_channels=32,
    in_channels=1,
    out_channels=1,
).to(device)

n_params = count_model_params(model)
print(f"# Parameters: {n_params}")

# Prepare input for FLOP count 
batch = next(iter(train_loader))
x = batch["x"].to(device)
x.requires_grad_(True)

# Count FLOPs 
with FlopTensorDispatchMode(model) as ftdm:
    out = model(x)
    out.mean().backward()
    forward_flops = deepcopy(ftdm.flop_counts)
    ftdm.reset()
    backward_flops = deepcopy(ftdm.flop_counts)

def get_max_flops(flop_count_dict, max_value=0):
    for _, value in flop_count_dict.items():
        if isinstance(value, int):
            max_value = max(max_value, value)
        elif isinstance(value, defaultdict):
            max_value = max(max_value, get_max_flops(value, max_value))
    return max_value

print(f"Max FLOPs (forward): {get_max_flops(forward_flops):,}")
print(f"Max FLOPs (backward): {get_max_flops(backward_flops):,}")

# Losses, optimizer, scheduler 
l2loss = LpLoss(d=2, p=2)
h1loss = H1Loss(d=2)
train_loss = h1loss
eval_losses = {"h1": h1loss, "l2": l2loss}

optimizer = AdamW(model.parameters(), lr=8e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

# Training loop 
epochs = 3
for epoch in range(epochs):
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=True, profile_memory=True) as prof:
        for batch in train_loader:
            # forward
            with record_function(f"epoch_{epoch+1}_forward"):
                optimizer.zero_grad()
                out = model(batch["x"].to(device))

            with record_function(f"epoch_{epoch+1}_backward"):
                loss = train_loss(out, batch["y"].to(device))
                loss.backward()
                optimizer.step()
            
            prof.step() 
    
    print(f"Epoch {epoch+1} profiling:")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=5))
    prof.export_chrome_trace(f"epoch_{epoch+1}_trace.json")
    scheduler.step()


# %%
# Plot the prediction, and compare with the ground-truth
# For debugging we trained on a small resolution for small number of epochs
#
# In practice we would train a Neural Operator on one or multiple GPUs

model.eval()
test_samples = test_loaders[128].dataset

fig = plt.figure(figsize=(7, 7))
for index in range(3):
    data = test_samples[index]
    x = data["x"].to(device)
    y = data["y"].to(device)
    out = model(x.unsqueeze(0))

    ax = fig.add_subplot(3, 3, index * 3 + 1)
    ax.imshow(x.cpu().squeeze().detach().numpy(), cmap="gray")
    if index == 0: ax.set_title("Input x")
    plt.xticks([]); plt.yticks([])

    ax = fig.add_subplot(3, 3, index * 3 + 2)
    ax.imshow(y.cpu().squeeze().detach().numpy())
    if index == 0: ax.set_title("Ground-truth y")
    plt.xticks([]); plt.yticks([])

    ax = fig.add_subplot(3, 3, index * 3 + 3)
    ax.imshow(out.cpu().squeeze().detach().numpy())
    if index == 0: ax.set_title("Model prediction")
    plt.xticks([]); plt.yticks([])

fig.suptitle("Inputs, Ground Truth, and Predictions", y=0.98)
plt.tight_layout()
plt.savefig("tfno_predictions.png")
# plt.show()
