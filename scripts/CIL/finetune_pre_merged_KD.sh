#!/bin/env/bash

# 这个脚本实现了yichen所提出的pre_merged_distance方法, 把之前任务的task vector merge, 以提供一个约束, 使训练的模型更接近之前任务的模型


set -e

eval "$(conda shell.bash hook)"
conda activate magmax

model=$1
dataset=$2
epochs=$3
n_splits=$4
seed=$5
pre_merged_mode=$6
kd_weight=$7
temperature=$8

# TRAIN
out_dir=outs/${model}/pre_merged_distance/class_incremental/ft/${dataset}
mkdir -p ${out_dir}

python my_finetune_splitted_pre_merged_dist.py \
    --model ${model} \
    --dataset ${dataset} \
    --epochs ${epochs} \
    --n_splits ${n_splits} \
    --split_strategy class \
    --pre_merged_mode ${pre_merged_mode} \
    --sequential_finetuning \
    --seed ${seed} \
    --kd_weight ${kd_weight} \
    --temperature ${temperature} \
        |& tee ${out_dir}/splits:${n_splits}-mode:${pre_merged_mode}-ep:${epochs}-seed:${seed}.out

# MERGE
out_dir=outs/${model}/pre_merged_distance/class_incremental/merging/${dataset}
mkdir -p ${out_dir}

python merge_splitted.py \
    --model ${model} \
    --dataset ${dataset} \
    --epochs ${epochs} \
    --n_splits ${n_splits} \
    --split_strategy class \
    --pre_merged_mode ${pre_merged_mode} \
    --kd_weight ${kd_weight} \
    --temperature ${temperature} \
    --sequential_finetuning \
    --seed ${seed} \
        |& tee -a ${out_dir}/merge-${n_splits}-mode:${pre_merged_mode}-ep:${epochs}-seed:${seed}.out