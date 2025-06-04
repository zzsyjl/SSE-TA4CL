import numpy as np
import torch
import wandb
import time
from src.merging.task_vectors import TaskVector, merge_max_abs, merge_rnd_mix
from src.merging.ties import merge_methods, state_dict_to_vector, vector_to_state_dict
from src.eval import eval_single_dataset, eval_task_aware, eval_task_agnostic
from src.args import parse_arguments

import pdb
# Config

def exchange_params_between_tvs(merged_tv: TaskVector, tv1: TaskVector, tv2: TaskVector, only_change_tv1_params=False, threshold=1e-8) -> TaskVector:
    """
    Exchange parameters in merged_tv based on their similarity to tv1 and tv2.
    For each parameter, choose the value from tv1 if it's closer to merged_tv,
    otherwise choose from tv2.
    
    Args:
        merged_tv: The merged task vector
        tv1: First task vector
        tv2: Second task vector
        threshold: Minimum absolute value to consider a parameter non-zero
    
    Returns:
        TaskVector: A new task vector with exchanged parameters
    """
    with torch.no_grad():
        new_vector = {}
        
        for key in merged_tv.vector:
            merged_tensor = merged_tv.vector[key]
            tv1_tensor = tv1.vector[key]
            tv2_tensor = tv2.vector[key]
        
            
            # Calculate differences
            diff1 = torch.abs(tv1_tensor - merged_tensor)
            diff2 = torch.abs(tv2_tensor - merged_tensor)
            
            # Create masks for selecting parameters
            select_tv2_mask = (diff1 < threshold)
            select_tv1_mask = (diff2 < threshold)
            
            if only_change_tv1_params:
                merged_tensor[select_tv2_mask] = tv2_tensor[select_tv2_mask]
            else:
                merged_tensor[select_tv1_mask] = tv1_tensor[select_tv1_mask]
                merged_tensor[select_tv2_mask] = tv2_tensor[select_tv2_mask]
            
            new_vector[key] = merged_tensor

    return TaskVector(vector=new_vector)
    

args = parse_arguments()
pretrained_checkpoint = f'checkpoints/ViT-B-16/zeroshot.pt'
args.save = f'checkpoints/ViT-B-16/sequential_finetuning/class_incremental'
args.dataset = 'ImageNetR'
args.model = 'ViT-B-16'
args.n_splits = 5
args.epochs = 10
args.seed = 5
args.split_strategy = "class"
ONLY_CHANGE_TV1_PARAMS = True

task_vectors = [
        TaskVector(pretrained_checkpoint, f'{args.save}/{args.dataset}-{args.n_splits}/ft-epochs-{args.epochs}-seed:{args.seed}/finetuned_{_idx}.pt')
        for _idx in range(args.n_splits)
    ]

merged_tv = merge_max_abs(task_vectors)
merged_tv = exchange_params_between_tvs(merged_tv, task_vectors[0], task_vectors[-1], only_change_tv1_params=ONLY_CHANGE_TV1_PARAMS)


image_encoder = merged_tv.apply_to(pretrained_checkpoint, scaling_coef=0.5)
# Evaluate
res = eval_task_agnostic(image_encoder, args)
print(f"task-agnostic eval on {args.dataset}. Accuracies:\n{res}")

