"""
Plot training time vs number of GPUs for CIFAR-10 extended training.
"""
import matplotlib.pyplot as plt
import numpy as np

# Data from CIFAR-10 experiments (20 epochs)
gpus = np.array([1, 2, 4, 6, 8])
times = np.array([73.00, 46.47, 27.72, 21.24, 18.20])  # in seconds

# Calculate speedup
speedup = times[0] / times

# Create figure
fig, ax1 = plt.subplots(1, 1, figsize=(6, 4.5))

# Plot: Training time vs number of GPUs
ax1.plot(gpus, times, 'o-', linewidth=2, markersize=8, color='#2E86AB')
ax1.set_xlabel('Number of GPUs', fontsize=12)
ax1.set_ylabel('Training Time (seconds)', fontsize=12)
ax1.set_title('CIFAR-10 Training Time vs Number of GPUs\n(20 epochs, ResNet18)', fontsize=14, fontweight='bold', pad=15)
ax1.grid(True, alpha=0.3)
ax1.set_xticks(gpus)

# Annotations removed to avoid overlap with title

plt.tight_layout(pad=2.0)
plt.savefig('cifar10_scaling_performance.png', dpi=300, bbox_inches='tight', pad_inches=0.2)
print("Plot saved as cifar10_scaling_performance.png")
