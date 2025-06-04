#!/bin/bash

datasets=("CIFAR100" "ImageNetR")
splits=(5 10)
sparsities=(0.01 0.1 0.3 0.5 0.7)


for dataset in "${datasets[@]}"; do
    for split in "${splits[@]}"; do
        for sparsity in "${sparsities[@]}"; do
            python save_LS_ckpt_from_finetuned_ckpts.py \
                --DARE "True" \
                --dataset-name "$dataset" \
                --splits "$split" \
                --sparsity "$sparsity"
        done
    done
done