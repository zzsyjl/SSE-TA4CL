import matplotlib.pyplot as plt
import numpy as np

# Data for CIFAR100 with 5 splits
ewc_cifar100 = [
    [0.9535, 0.4970, 0.6045, 0.5540, 0.5720],  # Training Split 0
    [0.8495, 0.8995, 0.6220, 0.5575, 0.5575],  # Training Split 1
    [0.8120, 0.7530, 0.9020, 0.5300, 0.5475],  # Training Split 2
    [0.7535, 0.7310, 0.8410, 0.8900, 0.5390],  # Training Split 3
    [0.7015, 0.6635, 0.8025, 0.7990, 0.9365],  # Training Split 4
]
magmax_cifar100 = [[0.906, 0.65, 0.726, 0.7045, 0.6815], [0.8905, 0.826, 0.741, 0.705, 0.686], [0.886, 0.8305, 0.8315, 0.6925, 0.693], [0.8635, 0.8285, 0.823, 0.795, 0.689], [0.8585, 0.817, 0.8215, 0.8125, 0.799]]
SSE_cifar100 = [[0.9024, 0.7099, 0.7634, 0.7536, 0.7233], [0.9055, 0.8352, 0.7568, 0.7485, 0.7147], [0.8962, 0.8221, 0.8562, 0.7578, 0.7147], [0.8924, 0.8273, 0.8488, 0.8459, 0.7284], [0.8786, 0.8203, 0.8398, 0.8357, 0.8129]]

# Data for CIFAR10 with 10 splits
ewc_cifar100_10splits = [
[0.9580, 0.4940, 0.4060, 0.5340, 0.4520, 0.6800, 0.5000, 0.5370, 0.5220, 0.4400],  # Training Split 0
[0.8970, 0.8730, 0.5160, 0.5070, 0.5770, 0.7140, 0.5160, 0.5790, 0.5640, 0.5350],  # Training Split 1
[0.7100, 0.6180, 0.8500, 0.4470, 0.4670, 0.6770, 0.4290, 0.4900, 0.4780, 0.4130],  # Training Split 2
[0.7820, 0.5990, 0.8200, 0.9090, 0.5240, 0.6690, 0.4770, 0.5590, 0.4940, 0.5010],  # Training Split 3
[0.7430, 0.6140, 0.7080, 0.8280, 0.8650, 0.6820, 0.5330, 0.5660, 0.5020, 0.5140],  # Training Split 4
[0.7570, 0.6900, 0.7080, 0.8010, 0.7880, 0.8660, 0.4850, 0.5960, 0.5320, 0.5990],  # Training Split 5
[0.7220, 0.6530, 0.6800, 0.7080, 0.6990, 0.8040, 0.8160, 0.5480, 0.4690, 0.5400],  # Training Split 6
[0.6690, 0.6670, 0.6530, 0.6940, 0.7230, 0.8140, 0.7330, 0.9220, 0.5620, 0.5300],  # Training Split 7
[0.6710, 0.6480, 0.6270, 0.6680, 0.6830, 0.7740, 0.6750, 0.8330, 0.8960, 0.4790],  # Training Split 8
[0.6600, 0.6890, 0.5860, 0.6300, 0.6890, 0.7560, 0.6790, 0.7870, 0.8140, 0.9110],  # Training Split 9
]

magmax_cifar100_10splits = [[0.941, 0.612, 0.556, 0.63, 0.592, 0.766, 0.601, 0.68, 0.601, 0.582], [0.91, 0.801, 0.611, 0.665, 0.654, 0.777, 0.675, 0.736, 0.631, 0.649], [0.885, 0.773, 0.808, 0.673, 0.683, 0.785, 0.677, 0.753, 0.641, 0.656], [0.878, 0.744, 0.803, 0.821, 0.704, 0.787, 0.675, 0.743, 0.632, 0.672], [0.845, 0.739, 0.764, 0.808, 0.812, 0.795, 0.661, 0.75, 0.648, 0.664], [0.84, 0.74, 0.739, 0.807, 0.809, 0.854, 0.671, 0.76, 0.645, 0.669], [0.839, 0.732, 0.739, 0.794, 0.806, 0.85, 0.769, 0.735, 0.627, 0.656], [0.813, 0.747, 0.729, 0.798, 0.815, 0.848, 0.753, 0.826, 0.642, 0.65], [0.795, 0.737, 0.734, 0.785, 0.8, 0.846, 0.754, 0.815, 0.734, 0.64], [0.796, 0.727, 0.706, 0.775, 0.784, 0.84, 0.761, 0.817, 0.733, 0.758]]
SSE_cifar100_10splits = [[0.908, 0.7265, 0.7508, 0.7299, 0.7945, 0.7539, 0.747, 0.7787, 0.7702, 0.7841], 
                        [0.8965, 0.8268, 0.7635, 0.7495, 0.8117, 0.7727, 0.7606, 0.7973, 0.7735, 0.7841], 
                        [0.8993, 0.8104, 0.8279, 0.7476, 0.8152, 0.7806, 0.7691, 0.8041, 0.7557, 0.7786], 
                        [0.8965, 0.8221, 0.8206, 0.8069, 0.8135, 0.7743, 0.7793, 0.7956, 0.7816, 0.7694], 
                        [0.9092, 0.8255, 0.8095, 0.7886, 0.8556, 0.7806, 0.7708, 0.8041, 0.7735, 0.7934], 
                        [0.8965, 0.8238, 0.8381, 0.7847, 0.8584, 0.8009, 0.7912, 0.7922, 0.7896, 0.7841], 
                        [0.905, 0.8339, 0.8333, 0.7965, 0.8601, 0.8166, 0.8014, 0.7956, 0.7816, 0.7841], 
                        [0.8993, 0.8255, 0.8349, 0.7886, 0.8428, 0.8135, 0.8166, 0.8091, 0.7816, 0.7897], 
                        [0.9035, 0.8255, 0.846, 0.8102, 0.848, 0.8166, 0.8031, 0.8311, 0.7977, 0.797], 
                        [0.8908, 0.8087, 0.8238, 0.8043, 0.8384, 0.8119, 0.8217, 0.826, 0.8155, 0.8118]]

methods = ["EWC", "MagMax", "SSE(ours)"]

# CIFAR100 datasets with 5 splits
cifar100_datasets = [ewc_cifar100, magmax_cifar100, SSE_cifar100]
# CIFAR10 datasets with 10 splits
cifar10_datasets = [ewc_cifar100_10splits, magmax_cifar100_10splits, SSE_cifar100_10splits]

# Increase default font size
plt.rcParams.update({'font.size': 18})

# Function to create a plot for each method
def create_plot(acc_across_CL_stream, method_name, dataset_name, num_splits):
    plt.figure(figsize=(10, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    stream_points = list(range(num_splits))  
    
    # Plot each task's accuracy, but only after it's been seen
    for task_id in range(num_splits):
        # Only plot from the point where the task was seen
        task_acc = [acc_across_CL_stream[i][task_id] for i in range(task_id, num_splits)]
        plt.plot(range(task_id, num_splits), task_acc, label=f'Task {task_id+1}', 
                 marker=markers[task_id % len(markers)], color=colors[task_id % len(colors)], linewidth=2)
    
    # Add labels and title
    # plt.xlabel('Continual Learning Stream (After Training Task N)', fontsize=24)
    plt.xticks(stream_points, [f'Task {i+1}' for i in stream_points], fontsize=20)
    plt.yticks(fontsize=22)
    plt.ylim(0.6, 1.0)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='lower left', fontsize=22)
    plt.title(f'{method_name} on {dataset_name} {num_splits}-splits', fontsize=24)
    
    # Save and show the plot
    plt.tight_layout()
    plt.savefig(f'seen_tasks_accuracy_{method_name}_{dataset_name}_{num_splits}splits.png', dpi=300)
    
    # Return the figure for the combined plot
    return plt.gcf()

# Create individual plots for all datasets
individual_plots = []

# CIFAR100 plots
for i, method in enumerate(methods):
    fig = create_plot(cifar100_datasets[i], method, "CIFAR100", 5)
    individual_plots.append(fig)

# CIFAR10 plots
for i, method in enumerate(methods):
    fig = create_plot(cifar10_datasets[i], method, "CIFAR10", 10)
    individual_plots.append(fig)

# Close all individual figures 
plt.close('all')

# Create combined figure with all 6 plots
fig, axes = plt.subplots(2, 3, figsize=(24, 16), dpi=300)
axes = axes.flatten()

# Function to create a plot directly on an axis
def plot_on_axis(ax, acc_across_CL_stream, method_name, dataset_name, num_splits):
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    stream_points = list(range(num_splits))
    
    # Plot each task's accuracy, but only after it's been seen
    for task_id in range(num_splits):
        # Only plot from the point where the task was seen
        task_acc = [acc_across_CL_stream[i][task_id] for i in range(task_id, num_splits)]
        ax.plot(range(task_id, num_splits), task_acc, label=f'Task {task_id+1}', 
                marker=markers[task_id % len(markers)], color=colors[task_id % len(colors)], linewidth=2)
    
    # Add labels and title
    ax.set_xticks(stream_points)
    ax.set_xticklabels([f'Task {i+1}' for i in stream_points], fontsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ax.set_ylim(0.6, 1.0)
    ax.grid(True, linestyle='--', alpha=0.7)
    if num_splits <= 6:  # Only show legend for plots with fewer tasks to avoid clutter
        ax.legend(loc='lower left', fontsize=14)
    ax.set_title(f'{method_name} on {dataset_name} {num_splits}-splits', fontsize=18)

# Plot CIFAR100 on the first row
for i, method in enumerate(methods):
    plot_on_axis(axes[i], cifar100_datasets[i], method, "CIFAR100", 5)

# Plot CIFAR10 on the second row
for i, method in enumerate(methods):
    plot_on_axis(axes[i+3], cifar10_datasets[i], method, "CIFAR10", 10)

# Add a common x-label and y-label
fig.text(0.5, 0.08, 'Continual Learning Stream (After Training Task N)', ha='center', fontsize=20)
fig.text(0.08, 0.5, 'Accuracy on Seen Tasks', va='center', rotation='vertical', fontsize=20)

# Adjust layout and save
plt.tight_layout(rect=[0.1, 0.1, 1, 0.95])
plt.savefig('all_methods_combined_plot.png', dpi=300)
plt.close() 