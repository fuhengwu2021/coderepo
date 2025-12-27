"""
Plot training time vs number of GPUs to show scaling efficiency.
"""
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

# Data from experiments
gpus = np.array([1, 2, 4, 6, 8])
times = np.array([8.78, 5.91, 3.69, 2.92, 2.44])

# Calculate speedup
speedup = times[0] / times

# Create figure with single plot
fig, ax1 = plt.subplots(1, 1, figsize=(6, 3.25))

# Plot: Training time vs number of GPUs
ax1.plot(gpus, times, 'o-', linewidth=1, markersize=4, color='#2E86AB')
ax1.set_xlabel('Number of GPUs', fontsize=8)
ax1.set_ylabel('Training Time (seconds)', fontsize=8)
ax1.set_title('FashionMNIST Training Time vs Number of GPUs', fontsize=9, pad=5)
ax1.grid(True, alpha=0.3)
ax1.set_xticks(gpus)

# Annotations removed to avoid overlap with title

# Determine output directory: if ../img exists, use it; otherwise use current directory
# Get script file path and check its parent directory for img folder
# Handle symlinks by checking both the resolved path and the original path
script_file = Path(__file__)
script_dir_resolved = script_file.resolve().parent
script_dir_original = script_file.parent

# Check both resolved and original paths for ../img
for script_dir in [script_dir_resolved, script_dir_original]:
    parent_dir = script_dir.parent
    img_dir = parent_dir / 'img'
    if img_dir.exists() and img_dir.is_dir():
        output_dir = img_dir
        break
else:
    # If no img folder found, use the script's directory (resolved)
    output_dir = script_dir_resolved

plt.tight_layout(pad=2.0)
output_path = output_dir / 'fashionmnist_scaling_performance.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.2)
print(f"Plot saved as {output_path}")
