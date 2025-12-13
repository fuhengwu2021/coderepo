"""
First multi-GPU distributed training using PyTorch DDP.

Trains ResNet18 from torchvision on FashionMNIST dataset using distributed training.
Uses the same model as single_gpu_baseline.py for fair comparison.

Usage:
    # Set OMP_NUM_THREADS before running to avoid warning
    OMP_NUM_THREADS=4 torchrun --nproc_per_node=2 code/multi_gpu_ddp.py
    
    # Or use the launch script (sets OMP_NUM_THREADS automatically)
    bash code/launch_torchrun.sh
"""
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms
import time
from mdaisy import get_resnet18_fashionmnist

def setup():
    """Initialize the process group using torchrun environment variables"""
    # torchrun sets these environment variables automatically
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    return rank, dist.get_world_size(), local_rank

def cleanup():
    """Clean up the process group"""
    dist.destroy_process_group()

def train_ddp():
    """Run distributed training using PyTorch DDP"""
    rank, world_size, local_rank = setup()
    
    # Data loading with distributed sampler
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    train_dataset = datasets.FashionMNIST(
        root='./data', train=True, download=(rank == 0), transform=transform
    )
    sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(
        train_dataset, batch_size=128, sampler=sampler, num_workers=2
    )
    
    # Create model (same as single_gpu_baseline.py)
    model = get_resnet18_fashionmnist(num_classes=10).cuda()
    
    # Wrap with DDP
    model = DDP(model, device_ids=[local_rank])
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    model.train()
    start_time = time.time()
    
    num_epochs = 3  # Enough to see training progress, completes in < 30s
    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)  # Important for shuffling
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.cuda(), target.cuda()
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        if rank == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%", flush=True)
    
    total_time = time.time() - start_time
    if rank == 0:
        print(f"\nTotal training time: {total_time:.2f}s")
    
    cleanup()

if __name__ == "__main__":
    train_ddp()
