import numpy as np
import torch
import time
from src.merging.task_vectors import TaskVector, merge_max_abs, merge_rnd_mix
from src.merging.ties import merge_methods, state_dict_to_vector, vector_to_state_dict
from src.eval import eval_single_dataset, eval_task_aware, eval_task_agnostic
from src.args import parse_arguments
import matplotlib.pyplot as plt
import os

import pdb
    

def calculate_similarity_all(vector1, vector2, threshold=1e-8):
    """Calculate the percentage of all parameters that are the same (within threshold)"""
    diff = np.abs(vector1 - vector2)
    similar_params = np.sum(diff < threshold)
    total_params = len(vector1)
    return (similar_params / total_params) * 100

def calculate_similarity_nonzero(tv, merved_tv, threshold=1e-8):
    """
    Calculate the percentage of non-zero parameters that are the same (within threshold)
    Only considers parameters of the merged task vector that has a non-zero value
    """
    mask = np.abs(merved_tv) > threshold
    
    # Only compare parameters where at least one vector has non-zero value
    diff = np.abs(tv[mask] - merved_tv[mask])
    similar_params = np.sum(diff < threshold)
    total_params = np.sum(mask)
    
    if total_params == 0:  # Handle case where all parameters are zero
        return 100.0
    
    return (similar_params / total_params) * 100

def calculate_histogram_stats(params, bins=20, range=(-1e-2, 1e-2)):
    """Calculate histogram statistics"""
    hist, bin_edges = np.histogram(params, bins=bins, range=range, density=True)
    total_count = len(params)
    counts, _ = np.histogram(params, bins=bins, range=range)
    return hist, bin_edges, counts, total_count

def write_histogram_data(f, hist_data):
    """Write histogram data to file in formatted way"""
    hist, bin_edges, counts, total_count = hist_data
    
    f.write("\nHistogram Data (20 bins):\n")
    f.write("Bin Range | Count | Percentage | Density\n")
    f.write("-" * 50 + "\n")
    
    for i in range(len(hist)):
        bin_start = bin_edges[i]
        bin_end = bin_edges[i+1]
        percentage = (counts[i] / total_count) * 100
        f.write(f"[{bin_start:7.4f}, {bin_end:7.4f}] | {counts[i]:7d} | {percentage:8.2f}% | {hist[i]:.4f}\n")
    f.write("\n")

if __name__ == '__main__':
    args = parse_arguments()
    print('Program start time:', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    suffix = ""    
    pretrained_checkpoint = f'checkpoints/{args.model}/zeroshot.pt'
    
    if args.sequential_finetuning:
        if args.pre_merged_mode:
            method = "seq-ft-pre-merged-dist"
            args.save = f'checkpoints/{args.model}/pre_merged_distance/sequential_finetuning/{args.split_strategy}_incremental'
        else:
            method = "seq-ft"
            args.save = f'checkpoints/{args.model}/sequential_finetuning/{args.split_strategy}_incremental'
    elif args.merged_finetuning_mode:
        method = "merged-ft"
        args.save = f'checkpoints/{args.model}/merged_finetuning/{args.split_strategy}_incremental'
    else:
        if args.pre_merged_mode:
            method = "ind-ft-pre-merged-dist"
            args.save = f'checkpoints/{args.model}/pre_merged_distance/{args.split_strategy}_incremental'
        else:
            method = "ind-ft"
            args.save = f'checkpoints/{args.model}/{args.split_strategy}_incremental'

    name = f"merging-{args.dataset}-{args.n_splits}-{method}"
    
    # preload task vectors
    pre_merged_mode_label = f'{args.pre_merged_mode}' if args.pre_merged_mode is not None else ''
    model_dist_type = f'_{args.model_dist_type}_' if args.model_dist_type == 'L2' else ''
    kd_weight = f'_kd{args.kd_weight}_' if args.kd_weight > 0 else ''
    reg_weight_label = f'{args.reg_weight}' if args.reg_weight > 0 else ''
    temp_label = f'_temp{args.temperature}_' if args.temperature > 0 else ''

    task_vectors = [
        TaskVector(pretrained_checkpoint, f'{args.save}/{args.dataset}-{args.n_splits}/ft-epochs-{args.epochs}-seed:{args.seed}{suffix}/finetuned_{pre_merged_mode_label}{reg_weight_label}{model_dist_type}{kd_weight}{temp_label}{_idx}.pt')
        for _idx in range(args.n_splits)
    ]

    merged_tv = merge_max_abs(task_vectors)

    # Write all statistics to a file in current working directory
    stats_file = f'task_vectors_stats_{args.dataset}_splits{args.n_splits}_seed{args.seed}{pre_merged_mode_label}{reg_weight_label}{model_dist_type}{kd_weight}{temp_label}.txt'
    
    with open(stats_file, 'w') as f:
        f.write(f"Analysis Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\n")
        f.write(f"Dataset: {args.dataset}, Splits: {args.n_splits}, Seed: {args.seed}\n")
        f.write(f"Method: {method}\n\n")
        
        # First get merged params for similarity comparison
        merged_params = []
        for param_name, param in merged_tv.vector.items():
            merged_params.extend(param.cpu().numpy().flatten())
        merged_params = np.array(merged_params)
        
        # Analyze each task vector
        print("\nCalculating statistics for each task vector...")
        for idx, task_vector in enumerate(task_vectors):
            tv_params = []
            for param_name, param in task_vector.vector.items():
                tv_params.extend(param.cpu().numpy().flatten())
            tv_params = np.array(tv_params)
            
            # Calculate basic statistics
            mean_val = np.mean(tv_params)
            std_val = np.std(tv_params)
            sparsity = np.mean(np.abs(tv_params) < 1e-8)
            similarity_all = calculate_similarity_all(tv_params, merged_params)
            similarity_nonzero = calculate_similarity_nonzero(tv_params, merged_params)
            
            
            # Write statistics
            f.write(f"============ Task Vector {idx} Statistics ============\n")
            f.write(f"Mean: {mean_val:.6f}\n")
            f.write(f"Std: {std_val:.6f}\n")
            f.write(f"Sparsity (% of near-zero values): {sparsity*100:.2f}%\n")
            f.write(f"Similarity with Merged Task Vector (all params): {similarity_all:.2f}%\n")
            f.write(f"Similarity with Merged Task Vector (non-zero params): {similarity_nonzero:.2f}%\n")
            f.write(f"Min: {np.min(tv_params):.6f}\n")
            f.write(f"Max: {np.max(tv_params):.6f}\n")
            f.write(f"Median: {np.median(tv_params):.6f}\n")
            
            # Calculate and write histogram data
            hist_data = calculate_histogram_stats(tv_params)
            write_histogram_data(f, hist_data)
            
            f.write("\n")
            
            # Print basic stats to console
            print(f"\nTask Vector {idx} Statistics:")
            print(f"Mean: {mean_val:.6f}")
            print(f"Std: {std_val:.6f}")
            print(f"Sparsity: {sparsity*100:.2f}%")
            print(f"Similarity with Merged Task Vector (all params): {similarity_all:.2f}%")
            print(f"Similarity with Merged Task Vector (non-zero params): {similarity_nonzero:.2f}%")

        
        # Analyze merged task vector
        merged_params = []
        for param_name, param in merged_tv.vector.items():
            merged_params.extend(param.cpu().numpy().flatten())
        merged_params = np.array(merged_params)
        
        # Write merged vector statistics
        f.write("============ Merged Task Vector Statistics ============\n")
        f.write(f"Mean: {np.mean(merged_params):.6f}\n")
        f.write(f"Std: {np.std(merged_params):.6f}\n")
        f.write(f"Sparsity: {np.mean(np.abs(merged_params) < 1e-8)*100:.2f}%\n")
        f.write(f"Min: {np.min(merged_params):.6f}\n")
        f.write(f"Max: {np.max(merged_params):.6f}\n")
        f.write(f"Median: {np.median(merged_params):.6f}\n")
        
        # Calculate and write histogram data for merged vector
        hist_data = calculate_histogram_stats(merged_params)
        write_histogram_data(f, hist_data)

    print(f"\nAll statistics have been written to: {stats_file}")
    print('Program end time:', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
