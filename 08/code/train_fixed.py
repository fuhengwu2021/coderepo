#!/usr/bin/env python3
"""
Sample distributed training script for SLURM.

This script demonstrates how to run distributed training with PyTorch DDP
using SLURM's environment variables.

Usage examples:

1. Using srun with explicit task configuration:
    srun -N 1 --gres=gpu:2 --ntasks-per-node=2 python train.py
    srun -N 2 --gres=gpu:1 --ntasks-per-node=1 python train.py

2. Using torchrun (recommended):
    srun -N 1 --gres=gpu:2 torchrun --nproc_per_node=2 train.py
    srun -N 2 --gres=gpu:1 torchrun --nproc_per_node=1 --nnodes=2 train.py

3. Single GPU (no distributed):
    srun -N 1 --gres=gpu:1 python train.py

4. Using sbatch with a batch script (see chapter8.md for examples)
"""

import os
import argparse
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler


class SimpleDataset(Dataset):
    """Simple synthetic dataset for demonstration."""
    def __init__(self, size=1000, input_dim=10):
        self.size = size
        self.input_dim = input_dim
        self.data = torch.randn(size, input_dim)
        self.targets = torch.randint(0, 2, (size,))
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


class SimpleModel(nn.Module):
    """Simple neural network for demonstration."""
    def __init__(self, input_dim=10, hidden_dim=64, num_classes=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


def setup_distributed():
    """
    Initialize distributed training using SLURM environment variables.
    
    Returns None if running in single-process mode.
    """
    # Disable IPv6 warnings
    os.environ['NCCL_SOCKET_IFNAME'] = '^docker,lo'
    
    # Check if we're in a SLURM environment
    if 'SLURM_PROCID' in os.environ:
        # SLURM mode: use SLURM environment variables
        rank = int(os.environ['SLURM_PROCID'])
        world_size = int(os.environ['SLURM_NTASKS'])
        local_rank = int(os.environ['SLURM_LOCALID'])
        
        # Get master address from SLURM
        if 'SLURM_JOB_NODELIST' in os.environ:
            master_addr = os.popen(
                f"scontrol show hostnames {os.environ['SLURM_JOB_NODELIST']} | head -n 1"
            ).read().strip()
        else:
            master_addr = 'localhost'
        
        master_port = os.environ.get('MASTER_PORT', '29500')
    else:
        # Non-SLURM mode: use torchrun environment variables
        rank = int(os.environ.get('RANK', 0))
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        master_addr = os.environ.get('MASTER_ADDR', 'localhost')
        master_port = os.environ.get('MASTER_PORT', '29500')
    
    # If world_size is 1, we're running single process - skip distributed setup
    if world_size == 1:
        print("Running in single-process mode (no distributed training)")
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        return 0, 1, 0, device, False
    
    # Set up the process group for distributed training
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = str(master_port)
    
    dist.init_process_group(
        backend='nccl',
        rank=rank,
        world_size=world_size
    )
    
    # Set device
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    
    return rank, world_size, local_rank, device, True


def cleanup_distributed(is_distributed):
    """Clean up distributed training."""
    if is_distributed:
        dist.destroy_process_group()


def train_one_epoch(model, dataloader, optimizer, criterion, device, rank, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if rank == 0 and batch_idx % 10 == 0:
            print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
    
    avg_loss = total_loss / num_batches
    return avg_loss


def main():
    parser = argparse.ArgumentParser(description='Distributed Training with SLURM')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size per GPU')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--dataset-size', type=int, default=1000, help='Dataset size')
    parser.add_argument('--input-dim', type=int, default=10, help='Input dimension')
    parser.add_argument('--num-workers', type=int, default=0, help='DataLoader workers (0=main process only)')
    args = parser.parse_args()
    
    # Setup distributed training (or single process)
    rank, world_size, local_rank, device, is_distributed = setup_distributed()
    
    if rank == 0:
        print(f'Training configuration:')
        print(f'  World size: {world_size}')
        print(f'  Rank: {rank}, Local rank: {local_rank}')
        print(f'  Device: {device}')
        print(f'  Distributed: {is_distributed}')
        print(f'  Epochs: {args.epochs}')
        print(f'  Batch size per GPU: {args.batch_size}')
        print(f'  Total batch size: {args.batch_size * world_size}')
        print(f'  DataLoader workers: {args.num_workers}')
    
    # Create model and move to device
    model = SimpleModel(input_dim=args.input_dim).to(device)
    
    # Wrap with DDP only if distributed
    if is_distributed:
        model = DDP(model, device_ids=[local_rank])
    
    # Create dataset and dataloader
    dataset = SimpleDataset(size=args.dataset_size, input_dim=args.input_dim)
    
    if is_distributed:
        # Use DistributedSampler for multi-process training
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=True
        )
    else:
        # Use regular DataLoader for single process
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True
        )
    
    # Setup optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    if rank == 0:
        print('\nStarting training...\n')
    
    for epoch in range(args.epochs):
        # Set epoch for distributed sampler (important for shuffling)
        if is_distributed:
            sampler.set_epoch(epoch)
        
        avg_loss = train_one_epoch(
            model, dataloader, optimizer, criterion, device, rank, epoch
        )
        
        if rank == 0:
            print(f'Epoch {epoch} completed, Average Loss: {avg_loss:.4f}\n')
    
    if rank == 0:
        print('Training completed!')
    
    # Cleanup
    cleanup_distributed(is_distributed)


if __name__ == '__main__':
    main()
