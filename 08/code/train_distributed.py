#!/usr/bin/env python3
"""
Complete distributed training script with checkpointing support.

This script demonstrates a full training workflow with:
- Distributed training (DDP)
- Checkpoint saving and resumption
- Proper error handling
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


class MyModel(nn.Module):
    """Simple model for demonstration."""
    def __init__(self, input_dim=10, hidden_dim=128, num_classes=2):
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
    """Initialize distributed training using SLURM or torchrun environment variables."""
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


def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch):
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
    
    avg_loss = total_loss / num_batches
    return avg_loss


def save_checkpoint(model, optimizer, epoch, checkpoint_dir, rank):
    """Save checkpoint (only on rank 0)."""
    if rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save latest checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        torch.save(checkpoint, f'{checkpoint_dir}/latest.pt')
        
        # Save periodic checkpoint
        if epoch % 10 == 0:
            torch.save(checkpoint, f'{checkpoint_dir}/checkpoint_epoch_{epoch}.pt')
            print(f"Saved checkpoint at epoch {epoch}")


def load_checkpoint(model, optimizer, checkpoint_dir, rank):
    """Load checkpoint if resuming."""
    latest_checkpoint = f'{checkpoint_dir}/latest.pt'
    
    if os.path.exists(latest_checkpoint):
        if rank == 0:
            print(f"Loading checkpoint from {latest_checkpoint}")
        
        checkpoint = torch.load(latest_checkpoint, map_location=f'cuda:{rank % torch.cuda.device_count()}')
        
        if hasattr(model, 'module'):
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        
        if rank == 0:
            print(f"Resuming from epoch {start_epoch}")
        
        return start_epoch
    else:
        if rank == 0:
            print("No checkpoint found, starting from scratch")
        return 0


def main():
    parser = argparse.ArgumentParser(description='Distributed Training with Checkpointing')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Checkpoint directory')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size per GPU')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    args = parser.parse_args()
    
    # Setup distributed training
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
        print(f'  Learning rate: {args.lr}')
        print(f'  Checkpoint dir: {args.checkpoint_dir}')
    
    # Create model and move to device
    model = MyModel().to(device)
    
    # Wrap with DDP only if distributed
    if is_distributed:
        model = DDP(model, device_ids=[local_rank])
    
    # Create dataset and dataloader
    dataset = SimpleDataset(size=1000, input_dim=10)
    
    if is_distributed:
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
            num_workers=2,
            pin_memory=True
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
    
    # Setup optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    # Load checkpoint if resuming
    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(model, optimizer, args.checkpoint_dir, rank)
    
    # Training loop
    if rank == 0:
        print('\nStarting training...\n')
    
    for epoch in range(start_epoch, args.epochs):
        # Set epoch for distributed sampler (important for shuffling)
        if is_distributed:
            sampler.set_epoch(epoch)
        
        avg_loss = train_one_epoch(model, dataloader, optimizer, criterion, device, epoch)
        
        if rank == 0:
            print(f'Epoch {epoch} completed, Average Loss: {avg_loss:.4f}')
        
        # Save checkpoint
        save_checkpoint(model, optimizer, epoch, args.checkpoint_dir, rank)
    
    if rank == 0:
        print('\nTraining completed!')
    
    # Cleanup
    cleanup_distributed(is_distributed)


if __name__ == '__main__':
    main()
