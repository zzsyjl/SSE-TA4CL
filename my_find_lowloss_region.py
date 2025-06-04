import numpy as np
import torch
import wandb

from src.merging.task_vectors import TaskVector, merge_max_abs, merge_rnd_mix
from src.merging.ties import merge_methods, state_dict_to_vector, vector_to_state_dict
from src.eval import eval_single_dataset, eval_task_aware, eval_task_agnostic, do_eval
from src.args import parse_arguments

from src import utils
from src.datasets.common import get_dataloader, maybe_dictionarize
from src.heads import get_classification_head
from src.modeling import ImageClassifier, ImageEncoder
from src.datasets.registry import get_dataset
from src.cl_utils import get_dataset_and_classifier_for_split

import pdb
import torch
import torch.nn as nn
import copy
import math

def rotate_vector_in_plane(a: torch.Tensor, b: torch.Tensor, angle: float) -> torch.Tensor:
    """
    Rotate vector c = b - a within the plane defined by vectors a and b by the specified angle.

    Parameters:
    ----------
    a : torch.Tensor
        The first high-dimensional vector defining the plane.
    b : torch.Tensor
        The second high-dimensional vector defining the plane.
    angle : float
        The rotation angle in radians.

    Returns:
    -------
    torch.Tensor
        The rotated vector c.
    """
    if a.shape != b.shape:
        raise ValueError("Vectors 'a' and 'b' must have the same shape.")

    # Compute vector c
    c = b - a

    # Compute norms
    norm_a = torch.norm(a)
    if norm_a.item() == 0:
        raise ValueError("Vector 'a' must be non-zero.")

    # Create orthonormal basis using Gram-Schmidt
    u = a / norm_a
    proj_b_on_u = torch.dot(b.flatten(), u.flatten()) * u
    b_perp = b - proj_b_on_u
    norm_b_perp = torch.norm(b_perp)
    
    if norm_b_perp.item() == 0:
        raise ValueError("Vectors 'a' and 'b' are colinear; plane is undefined.")
    
    v = b_perp / norm_b_perp

    # Express c in the orthonormal basis (u, v)
    alpha = torch.dot(c.flatten(), u.flatten())
    beta = torch.dot(c.flatten(), v.flatten())

    # Compute rotation
    cos_theta = math.cos(angle)
    sin_theta = math.sin(angle)

    alpha_rot = alpha * cos_theta - beta * sin_theta
    beta_rot = alpha * sin_theta + beta * cos_theta

    # Reconstruct the rotated vector
    rotated_c = alpha_rot * u + beta_rot * v

    return rotated_c

def rotate_models(model_a: nn.Module, model_b: nn.Module, angle: float) -> nn.Module:
    """
    Create a new model by rotating the difference between model_b and model_a's parameters
    by a specified angle within the plane defined by each corresponding pair of parameters.

    Parameters:
    ----------
    model_a : nn.Module
        The first PyTorch model.
    model_b : nn.Module
        The second PyTorch model.
    angle : float
        The rotation angle in radians.

    Returns:
    -------
    nn.Module
        The new rotated model.
    """
    # Ensure both models have the same architecture
    if type(model_a) != type(model_b):
        raise ValueError("Models must be of the same architecture.")
    
    # Create a deep copy of model_a to hold the new parameters
    new_model = copy.deepcopy(model_a)

    # Get the state dictionaries
    state_dict_a = model_a.state_dict()
    state_dict_b = model_b.state_dict()
    state_dict_new = new_model.state_dict()

    # Iterate through all parameters
    for key in state_dict_a:
        param_a = state_dict_a[key]
        param_b = state_dict_b[key]

        if param_a.shape != param_b.shape:
            raise ValueError(f"Shape mismatch for parameter '{key}': {param_a.shape} vs {param_b.shape}")

        # Flatten the parameters to 1D tensors
        param_a_flat = param_a.view(-1)
        param_b_flat = param_b.view(-1)

        # Rotate the difference vector
        try:
            rotated_c_flat = rotate_vector_in_plane(param_a_flat, param_b_flat, angle)
        except ValueError as e:
            print(f"Skipping parameter '{key}': {e}")
            rotated_c_flat = param_b_flat - param_a_flat  # If rotation is undefined, keep the difference as is

        # Reshape back to original shape
        rotated_c = rotated_c_flat.view_as(param_a)

        # Update the new parameter: a + rotated_c
        state_dict_new[key] = param_a + rotated_c

    # Load the updated state dictionary into the new model
    new_model.load_state_dict(state_dict_new)

    return new_model

if __name__ == '__main__':

    args = parse_arguments()
    split_idx = 0
    search_coef_flag = False # 决定是搜索scaling_coef还是旋转角度

    pretrained_checkpoint = f'checkpoints/{args.model}/zeroshot.pt'
    task_ckpt1 = f'checkpoints/{args.model}/{args.split_strategy}_incremental/{args.dataset}-{args.n_splits}/ft-epochs-{args.epochs}-seed:{args.seed}/finetuned_{split_idx}.pt'
    scaling_coef_ls = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
    rotate_angle_ls = [math.radians(i * 10) for i in range(-18, 19)] # +-180 degrees

    if search_coef_flag:
        candidate_ls = scaling_coef_ls
        task_vector1 = TaskVector(pretrained_checkpoint, task_ckpt1)
    else:
        candidate_ls = rotate_angle_ls
        pretrained_model = torch.load(pretrained_checkpoint)
        task_model = torch.load(task_ckpt1)


    

    accs = []

    for candidate in candidate_ls:
        if search_coef_flag:
            image_encoder = task_vector1.apply_to(pretrained_checkpoint, scaling_coef=candidate)
        else:
            # pdb.set_trace()
            image_encoder = rotate_models(pretrained_model, task_model, candidate)

        classification_head = get_classification_head(args, args.dataset)
        model = ImageClassifier(image_encoder, classification_head)

        full_dataset = get_dataset(
            args.dataset,
            image_encoder.train_preprocess,
            location=args.data_location,
            batch_size=args.batch_size
        )




        dataset = get_dataset_and_classifier_for_split(
            full_dataset, split_idx, None, args, remap_labels=False, return_classifier=False
        )
        metrics = do_eval(model, dataset.test_loader, args.device)
        accs.append(metrics['top1'])
        print(f"Find low-loss region for coef/angle {candidate}. Accuracy: {accs[-1]}")
    print(f"Accs of each coef/angle: {accs}")
