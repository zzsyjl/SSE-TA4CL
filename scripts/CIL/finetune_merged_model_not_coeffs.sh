#!/bin/bash
"""这个脚本对merged model本身进行Finetuning"""

set -e

eval "$(conda shell.bash hook)"
conda activate magmax

model=$1
dataset=$2
epochs=$3
n_splits=$4
seed=$5
finetune_merged_init=${6:-avg}  # Initial merge mode (avg or max)
finetune_merged_lr=${7:-1e-3}   # Learning rate for finetuning
pre_merged_mode=${8:-none}      # Regularization mode
reg_weight=${9:-0}              # Regularization weight
validset_ratio=${10:-0.1}       # Validation set ratio (default: 0.1)
merge_learned_epochs=${11:-100} # Number of epochs for finetuning

# TRAIN
out_dir=outs/${model}/finetune_merged_model/class_incremental/ft/${dataset}
mkdir -p ${out_dir}

python finetune_splitted.py \
    --model ${model} \
    --dataset ${dataset} \
    --epochs ${epochs} \
    --n_splits ${n_splits} \
    --split_strategy class \
    --merged-finetuning-mode finetune_model \
    --finetune_merged_init ${finetune_merged_init} \
    --finetune_merged_lr ${finetune_merged_lr} \
    --seed ${seed} \
    --pre_merged_mode ${pre_merged_mode} \
    --reg_weight ${reg_weight} \
    --merge_learned_epochs ${merge_learned_epochs} \
    --validset_ratio ${validset_ratio} \
        |& tee -a ${out_dir}/splits:${n_splits}-init:${finetune_merged_init}-lr:${finetune_merged_lr}-ep:${epochs}-seed:${seed}-vs:${validset_ratio}.out


# # MERGE
# out_dir=outs/${model}/merged_finetuning/class_incremental/merging/${dataset}
# mkdir -p ${out_dir}

# python merge_splitted.py \
#     --model ${model} \
#     --dataset ${dataset} \
#     --epochs ${epochs} \
#     --n_splits ${n_splits} \
#     --split_strategy class \
#     --merged-finetuning-mode ${merged_finetuning_mode} \
#     --seed ${seed} \
#         |& tee -a ${out_dir}/merge-${n_splits}-mode:${merged_finetuning_mode}-ep:${epochs}-seed:${seed}.out