"""
Check which distributed backends are available in PyTorch.

Usage:
    python check_backends.py
"""

import torch
import torch.distributed as dist

def check_backends():
    """Check and display available PyTorch distributed backends."""
    print("=" * 60)
    print("PyTorch Distributed Backend Availability")
    print("=" * 60)
    print(f"PyTorch version: {torch.__version__}")
    print()
    
    backends = {
        'NCCL': dist.is_nccl_available(),
        'Gloo': dist.is_gloo_available(),
        'MPI': dist.is_mpi_available(),
    }
    
    # UCC might not be available in all PyTorch versions
    if hasattr(dist, 'is_ucc_available'):
        backends['UCC'] = dist.is_ucc_available()
    
    print("Backend Status:")
    print("-" * 60)
    for backend, available in backends.items():
        status = "✓ Available" if available else "✗ Not available"
        print(f"{backend:10s}: {status}")
    
    print()
    print("=" * 60)
    print("Backend Details:")
    print("=" * 60)
    
    # NCCL details
    if backends['NCCL']:
        print("\nNCCL (NVIDIA Collective Communications Library):")
        print("  - Optimized for GPU communication")
        print("  - Requires CUDA and multiple GPUs")
        print("  - Supports: all_reduce, all_gather, broadcast, etc.")
        print("  - Best for: Multi-GPU training on single/multi-node")
    
    # Gloo details
    if backends['Gloo']:
        print("\nGloo:")
        print("  - Works on CPU and GPU")
        print("  - Good for CPU-based distributed training")
        print("  - Supports: all_reduce, all_gather, broadcast, etc.")
        print("  - Note: Does NOT support all_to_all on CPU")
        print("  - Best for: CPU training or when NCCL unavailable")
    
    # MPI details
    if backends['MPI']:
        print("\nMPI (Message Passing Interface):")
        print("  - Requires MPI installation (OpenMPI, MPICH)")
        print("  - Works on CPU and GPU")
        print("  - Supports: all collective operations including all_to_all")
        print("  - Best for: HPC clusters with MPI infrastructure")
    else:
        print("\nMPI:")
        print("  - Not available (requires MPI installation)")
        print("  - Install with: conda install mpi4py or pip install mpi4py")
    
    # UCC details
    if 'UCC' in backends and backends['UCC']:
        print("\nUCC (Unified Communication Collective):")
        print("  - Unified interface for multiple communication libraries")
        print("  - Can use NCCL, MPI, or other backends under the hood")
    elif 'UCC' in backends:
        print("\nUCC:")
        print("  - Not available in this PyTorch version")

if __name__ == '__main__':
    check_backends()
