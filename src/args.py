import os
import random
import numpy as np
import argparse

import torch


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    

def parse_arguments():
    parser = argparse.ArgumentParser()
    
    # My added arguments
    parser.add_argument(
        "--merged-finetuning-mode",
        type=str,
        default=None,
        choices=["max", "rnd", "avg", "learned", "finetune_model"],
        help="Mode to merge finetuned adapters before next split finetuning.",
    )
    parser.add_argument('--reg_weight', type=float, default=0,
        help='Weight for model distance between current and pretrained models')
    # Change the argument name from merged_finetuning_mode to pre_merged_mode
    parser.add_argument('--pre_merged_mode', type=str, choices=['max', 'avg', 'no_merge', 'pretrained', 'None'], default=None,
        help='Mode for merging previous task vectors, to provide a constraint between the current and merged models')    
    parser.add_argument('--model_dist_type', type=str, choices=['L1', 'L2'], default='L1',
        help='Type of distance metric for model distance')
    # Add to argument parser
    parser.add_argument('--kd_weight', type=float, default=0,
                        help='Weight for knowledge distillation loss')
    parser.add_argument('--temperature', type=float, default=0,
                        help='Temperature for knowledge distillation')
    parser.add_argument('--validset_ratio', type=float, default=0.1,
                        help='Ratio of training data to use as validation set')
    # add seed for validation set
    parser.add_argument('--seed_for_validset', type=int, default=42,
                        help='Random seed for creating validation set')
    # Add parameter for merge_learned epochs
    parser.add_argument('--merge_learned_epochs', type=int, default=100,
                        help='Number of epochs when using learned merge method or finetuning merged model directly')
    parser.add_argument(
        "--finetune_merged_lr",
        type=float,
        default=1e-3,
        help="Learning rate for directly finetuning merged model",
    )
    parser.add_argument(
        "--finetune_merged_init",
        type=str,
        default="avg",
        choices=["avg", "max"],
        help="Initial merging method before finetuning merged model",
    )
    # End of my added arguments
    
    # DATASETS
    parser.add_argument(
        "--data-location",
        type=str,
        default=os.path.expanduser('~/data'),
        help="The root directory for the datasets.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--eval-datasets",
        default=None,
        type=lambda x: x.split(","),
        help="Which datasets to use for evaluation. Split by comma, e.g. CIFAR101,CIFAR102.",
    )
    parser.add_argument(
        "--train-dataset",
        default=None,
        type=lambda x: x.split(","),
        help="Which dataset(s) to patch on.",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default=None,
        help="Name of the experiment, for organization purposes only."
    )

    # MODEL/TRAINING
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="The type of model (e.g. RN50, ViT-B-32).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="Learning rate."
    )
    parser.add_argument(
        "--wd",
        type=float,
        default=0.1,
        help="Weight decay"
    )
    parser.add_argument(
        "--ls",
        type=float,
        default=0.0,
        help="Label smoothing."
    )
    parser.add_argument(
        "--warmup_length",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
    )
    parser.add_argument('--skip-eval', action='store_true')
    
    # LOAD/SAVE PATHS
    parser.add_argument(
        "--load",
        type=lambda x: x.split(","),
        default=None,
        help="Optionally load _classifiers_, e.g. a zero shot classifier or probe or ensemble both.",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Optionally save a _classifier_, e.g. a zero shot classifier or probe.",
    )
    parser.add_argument(
        "--results-db",
        type=str,
        default=None,
        help="Where to store the results, else does not store",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory for caching features and encoder",
    )
    parser.add_argument(
        "--openclip-cachedir",
        type=str,
        default='checkpoints/ViT-B-16/cachedir/open_clip',
        help='Directory for caching models from OpenCLIP'
    )
    
    # CL SPLITS
    parser.add_argument(
        "--n_splits",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--split_strategy",
        type=str,
        default=None,
        choices=[None, 'data', 'class']
    )
    parser.add_argument(
        "--sequential_finetuning",
        action='store_true'
    )
    
    # CL METHODS
    parser.add_argument(
        "--lwf_lamb",
        type=float,
        default=0.0,
        help="LWF lambda"
    )
    parser.add_argument(
        "--ewc_lamb",
        type=float,
        default=0.0,
        help="EWC lambda"
    )
    
    # OTHER    
    parser.add_argument(
        '--seed',
        default=5,
        type=int
    )
    parser.add_argument(
        "--wandb_entity_name",
        type=str,
        default="YOUR-WANDB-ACCOUNT"
    )
    
    parsed_args = parser.parse_args()
    parsed_args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    seed_everything(parsed_args.seed)
    
    assert parsed_args.lwf_lamb == 0.0 or parsed_args.ewc_lamb == 0.0, \
        "Lambda for LWF and EWC are mutually exclusive"
    
    if parsed_args.load is not None and len(parsed_args.load) == 1:
        parsed_args.load = parsed_args.load[0]

    return parsed_args
