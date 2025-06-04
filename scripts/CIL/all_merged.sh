#!/bin/env/bash

set -e

model=ViT-B-16
seed=5
lwf_lambda=0.3
ewc_lambda=1e6

epochs=10
dataset=CIFAR100
# for n_splits in 5 10 20 50 ; do
for n_splits in 5 10; do
    for merged_finetuning_mode in rnd max avg; do
        bash scripts/CIL/finetune_merged.sh ${model} ${dataset} ${epochs} ${n_splits} ${seed} ${merged_finetuning_mode}
    done
done

epochs=10
dataset=ImageNetR
# for n_splits in 5 10 20 50 ; do
for n_splits in 5 10; do
    for merged_finetuning_mode in rnd max avg; do
        bash scripts/CIL/finetune_merged.sh ${model} ${dataset} ${epochs} ${n_splits} ${seed} ${merged_finetuning_mode}
    done
done