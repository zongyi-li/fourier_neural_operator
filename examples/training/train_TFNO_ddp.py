import os
import argparse
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from collections import defaultdict

from neuralop.models import TFNO
from neuralop.data.datasets import load_navier_stokes_pt
from neuralop.utils import count_model_params
from neuralop.training import AdamW
from neuralop import LpLoss, H1Loss

from torchtnt.utils.flops import FlopTensorDispatchMode
from torch.profiler import profile, record_function, ProfilerActivity


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def summarize_flops(flops_dict):
    summary = defaultdict(int)
    for scope in flops_dict:
        for op, count in flops_dict[scope].items():
            summary[op] += count
    return dict(summary)


def train(rank, world_size, args):
    setup(rank, world_size)
    try: 
        device = torch.device(f"cuda:{rank}")

        train_loader, test_loaders, output_encoder = load_navier_stokes_pt(
            n_train=args.n_train,
            batch_size=args.batch_size,
            test_resolutions=[128],
            n_tests=[3],
            test_batch_sizes=[2],
        )
        dataset = train_loader.dataset  # from the original loader
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
        train_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            pin_memory=True,
            num_workers=2,
        )

        # sampler = DistributedSampler(train_loader.dataset, num_replicas=world_size, rank=rank)
        # train_loader = DataLoader(train_loader.dataset, batch_size=args.batch_size, sampler=sampler, pin_memory=True, num_workers=2)

        model = TFNO(
            max_n_modes=(16, 16),
            n_modes=[128, 128],
            hidden_channels=32,
            in_channels=1,
            out_channels=1,
        ).to(device)
        model = DDP(model, device_ids=[rank])

        if rank == 0:
            print(f"# Parameters: {count_model_params(model)}")

        l2loss = LpLoss(d=2, p=2)
        h1loss = H1Loss(d=2)
        train_loss = h1loss

        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        if rank == 0:
            batch = next(iter(train_loader))
            x = batch["x"].to(device)
            x.requires_grad_(True)

            with FlopTensorDispatchMode(model) as ftdm:
                with torch.no_grad():
                    _ = model(x)
                torch.cuda.synchronize()

                # Forward
                out = model(x)
                forward_flops = summarize_flops(ftdm.flop_counts)
                ftdm.reset()

                # Backward
                loss = train_loss(out, batch["y"].to(device))
                loss.backward()
                torch.cuda.synchronize()
                backward_flops = summarize_flops(ftdm.flop_counts)

                print("Forward FLOPs Summary:", forward_flops)
                print("Backward FLOPs Summary:", backward_flops)

        print(f"[Rank {rank}] Starting training on {torch.cuda.get_device_name(rank)}")
        print(f"[Rank {rank}] Dataset size: {len(train_loader.dataset)} | Batch size: {args.batch_size}")

        for epoch in range(args.epochs):
            sampler.set_epoch(epoch)
            model.train()

            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                        record_shapes=True, profile_memory=True) as prof:
                for batch in train_loader:
                    optimizer.zero_grad()
                    x = batch["x"].to(device, non_blocking=True)
                    y = batch["y"].to(device, non_blocking=True)

                    with record_function("forward"):
                        out = model(x)

                    with record_function("backward"):
                        loss = train_loss(out, y)
                        loss.backward()
                        optimizer.step()

                    prof.step()

            if rank == 0:
                print(f"Epoch {epoch+1}: Loss = {loss.item():.6f}")
                print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=5))
                prof.export_chrome_trace(f"trace_rank{rank}_epoch{epoch+1}.json")
            scheduler.step()
    finally: 
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-train", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=8e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    args = parser.parse_args()

    world_size = 1  # torch.cuda.device_count() for multi-GPU
    print('No of GPUs:', world_size)
    mp.spawn(train, args=(world_size, args), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
