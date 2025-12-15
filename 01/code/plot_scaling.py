"""
Plot training time vs number of GPUs to show scaling efficiency.
"""
import matplotlib.pyplot as plt
import numpy as np

# Data from experiments
gpus = np.array([1, 2, 4, 6, 8])
times = np.array([8.78, 5.91, 3.69, 2.92, 2.44])

# Calculate speedup
speedup = times[0] / times

# Create figure with single plot
fig, ax1 = plt.subplots(1, 1, figsize=(6, 3.25))

# Plot: Training time vs number of GPUs
ax1.plot(gpus, times, 'o-', linewidth=2, markersize=8, color='#2E86AB')
ax1.set_xlabel('Number of GPUs', fontsize=10)
ax1.set_ylabel('Training Time (seconds)', fontsize=10)
ax1.set_title('FashionMNIST Training Time vs Number of GPUs', fontsize=12, pad=15)
ax1.grid(True, alpha=0.3)
ax1.set_xticks(gpus)

# Annotations removed to avoid overlap with title

plt.tight_layout(pad=2.0)
plt.savefig('fashionmnist_scaling_performance.png', dpi=300, bbox_inches='tight', pad_inches=0.2)
print("Plot saved as fashionmnist_scaling_performance.png")
