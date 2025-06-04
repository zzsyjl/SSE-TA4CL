#!/bin/env/bash
"""这个脚本实现了我所提出的mergedStart方法"""

set -e

eval "$(conda shell.bash hook)"
conda activate magmax

model=$1
dataset=$2
epochs=$3
n_splits=$4
seed=$5
merged_finetuning_mode=$6
pre_merged_mode=$7
reg_weight=$8
validset_ratio=${9:-0.1}  # 添加验证集比例参数，默认为0.1
merge_learned_epochs=${10}

# TRAIN
out_dir=outs/${model}/merged_finetuning/class_incremental/ft/${dataset}
mkdir -p ${out_dir}

python finetune_splitted.py \
    --model ${model} \
    --dataset ${dataset} \
    --epochs ${epochs} \
    --n_splits ${n_splits} \
    --split_strategy class \
    --merged-finetuning-mode ${merged_finetuning_mode} \
    --seed ${seed} \
    --pre_merged_mode ${pre_merged_mode} \
    --reg_weight ${reg_weight} \
    --merge_learned_epochs ${merge_learned_epochs} \
    --validset_ratio ${validset_ratio} \
        |& tee -a ${out_dir}/splits:${n_splits}-mode:${merged_finetuning_mode}-ep:${epochs}-seed:${seed}-vs:${validset_ratio}.out


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