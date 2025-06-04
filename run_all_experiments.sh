#!/bin/bash

# Setup
eval "$(conda shell.bash hook)"
conda activate magmax

# Parameters that remain constant across runs
model="vit_b_16"  # Set your model here
dataset="cifar100"  # Set your dataset here
epochs=10  # Set number of epochs
seed=42  # Set seed
validset_ratio=0.1  # Validation set ratio
merge_learned_epochs=5  # Set merge learned epochs
reg_weight=0.1  # Set regularization weight

# Array of configurations
sparse_ft_options=("false" "true")  # w/o and w/ Sparse FT
strategies=("randmix" "avg" "max")  # RandMix, Avg, Max
splits=(5 10)  # 5 splits, 10 splits
init_types=("ind" "seq" "equi")  # ind-init, seq-init, equi-init

# Run all combinations
for sparse_ft in "${sparse_ft_options[@]}"; do
    for strategy in "${strategies[@]}"; do
        for n_splits in "${splits[@]}"; do
            for init_type in "${init_types[@]}"; do
                # Set sparse FT flag if needed
                sparse_ft_flag=""
                if [ "$sparse_ft" == "true" ]; then
                    sparse_ft_flag="--use_sparse_ft"
                fi
                
                # Create descriptive names for logging
                sparse_ft_name=$([ "$sparse_ft" == "true" ] && echo "with_sparse_ft" || echo "without_sparse_ft")
                
                echo "Running experiment: $init_type-init, $strategy, $n_splits splits, $sparse_ft_name"
                
                # Create a custom output directory for this configuration
                custom_out_dir="outs/${model}/merged_finetuning/class_incremental/ft/${dataset}/${sparse_ft_name}/${init_type}-init"
                mkdir -p $custom_out_dir
                
                # Set the output file name
                out_file="${custom_out_dir}/splits:${n_splits}-strategy:${strategy}-ep:${epochs}-seed:${seed}.out"
                
                # Call finetune script but redirect output to our custom file
                python finetune_splitted.py \
                    --model ${model} \
                    --dataset ${dataset} \
                    --epochs ${epochs} \
                    --n_splits ${n_splits} \
                    --split_strategy class \
                    --merged-finetuning-mode ${strategy} \
                    --seed ${seed} \
                    --pre_merged_mode ${init_type} \
                    --reg_weight ${reg_weight} \
                    --merge_learned_epochs ${merge_learned_epochs} \
                    --validset_ratio ${validset_ratio} \
                    ${sparse_ft_flag} \
                    |& tee ${out_file}
                
                echo "Experiment completed: $init_type-init, $strategy, $n_splits splits, $sparse_ft_name"
                echo "Output saved to: ${out_file}"
                echo "========================================================"
            done
        done
    done
done

echo "All experiments completed successfully!" 