import matplotlib.pyplot as plt
import numpy as np

# Data for CIFAR10 with 10 splits
ewc_cifar10_10splits = [
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

magmax_cifar10_10splits = [[0.941, 0.612, 0.556, 0.63, 0.592, 0.766, 0.601, 0.68, 0.601, 0.582], [0.91, 0.801, 0.611, 0.665, 0.654, 0.777, 0.675, 0.736, 0.631, 0.649], [0.885, 0.773, 0.808, 0.673, 0.683, 0.785, 0.677, 0.753, 0.641, 0.656], [0.878, 0.744, 0.803, 0.821, 0.704, 0.787, 0.675, 0.743, 0.632, 0.672], [0.845, 0.739, 0.764, 0.808, 0.812, 0.795, 0.661, 0.75, 0.648, 0.664], [0.84, 0.74, 0.739, 0.807, 0.809, 0.854, 0.671, 0.76, 0.645, 0.669], [0.839, 0.732, 0.739, 0.794, 0.806, 0.85, 0.769, 0.735, 0.627, 0.656], [0.813, 0.747, 0.729, 0.798, 0.815, 0.848, 0.753, 0.826, 0.642, 0.65], [0.795, 0.737, 0.734, 0.785, 0.8, 0.846, 0.754, 0.815, 0.734, 0.64], [0.796, 0.727, 0.706, 0.775, 0.784, 0.84, 0.761, 0.817, 0.733, 0.758]]
SSE_cifar10_10splits = [[0.908, 0.7265, 0.7508, 0.7299, 0.7945, 0.7539, 0.747, 0.7787, 0.7702, 0.7841], 
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
cifar10_datasets = [ewc_cifar10_10splits, magmax_cifar10_10splits, SSE_cifar10_10splits]

# Set smaller default font size
plt.rcParams.update({'font.size': 14})

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
    
    # Add labels and title with smaller font sizes
    # plt.xlabel('Continual Learning Stream (After Training Task N)', fontsize=16)
    plt.xticks(stream_points, [f'Task {i+1}' for i in stream_points], fontsize=16)
    plt.yticks(fontsize=22)
    plt.ylim(0.4, 1.0)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Create a compact legend with small font size
    leg = plt.legend(loc='lower left', fontsize=16, ncol=2)
    leg.get_frame().set_linewidth(0.5)
    
    plt.title(f'{method_name} on {dataset_name} {num_splits}-splits', fontsize=24)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(f'seen_tasks_accuracy_{method_name}_{dataset_name}_{num_splits}splits.png', dpi=300)

# Create plots for each method (CIFAR10 only)
for i, method in enumerate(methods):
    create_plot(cifar10_datasets[i], method, "CIFAR10", 10)

print("CIFAR10 10-splits plots created successfully!") 