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
fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))

# Plot: Training time vs number of GPUs
ax1.plot(gpus, times, 'o-', linewidth=2, markersize=8, color='#2E86AB')
ax1.set_xlabel('Number of GPUs', fontsize=12)
ax1.set_ylabel('Training Time (seconds)', fontsize=12)
ax1.set_title('Training Time vs Number of GPUs', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_xticks(gpus)

# Add annotations for each point
for i, (gpu, time) in enumerate(zip(gpus, times)):
    speedup_val = speedup[i]
    ax1.annotate(f'{time:.2f}s\n({speedup_val:.2f}×)', 
                xy=(gpu, time), 
                xytext=(5, 10), 
                textcoords='offset points',
                fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

plt.tight_layout()
plt.savefig('scaling_performance.png', dpi=300, bbox_inches='tight')
print("Plot saved as scaling_performance.png")
