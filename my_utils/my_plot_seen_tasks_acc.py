import matplotlib.pyplot as plt
import numpy as np

# Data from acc_across_CL_stream
acc_across_CL_stream = [
    [0.9535, 0.4970, 0.6045, 0.5540, 0.5720],  # Training Split 0
    [0.8495, 0.8995, 0.6220, 0.5575, 0.5575],  # Training Split 1
    [0.8120, 0.7530, 0.9020, 0.5300, 0.5475],  # Training Split 2
    [0.7535, 0.7310, 0.8410, 0.8900, 0.5390],  # Training Split 3
    [0.7015, 0.6635, 0.8025, 0.7990, 0.9365],  # Training Split 4
]

# Increase default font size
plt.rcParams.update({'font.size': 14})

# Plot setup
plt.figure(figsize=(10, 6))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
markers = ['o', 's', '^', 'D', 'v']
stream_points = list(range(5))  # 0, 1, 2, 3, 4

# Plot each task's accuracy, but only after it's been seen
for task_id in range(5):
    # Only plot from the point where the task was seen
    task_acc = [acc_across_CL_stream[i][task_id] for i in range(task_id, 5)]
    plt.plot(range(task_id, 5), task_acc, label=f'Task {task_id+1}', 
             marker=markers[task_id], color=colors[task_id], linewidth=2)

# Add labels and title
plt.xlabel('Continual Learning Stream (After Training Task N)', fontsize=16)
plt.ylabel('Accuracy on Seen Tasks', fontsize=16)
plt.xticks(stream_points, [f'Task {i+1}' for i in stream_points], fontsize=14)
plt.yticks(fontsize=14)
plt.ylim(0.6, 1.0)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(loc='lower left', fontsize=14)

# Save and show the plot
plt.tight_layout()
plt.savefig('seen_tasks_accuracy.png', dpi=300)
plt.show()