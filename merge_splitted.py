import numpy as np
import torch
import wandb
import time
from src.merging.task_vectors import TaskVector, merge_max_abs, merge_rnd_mix, merge_learned
from src.merging.ties import merge_methods, state_dict_to_vector, vector_to_state_dict
from src.eval import eval_single_dataset, eval_task_aware, eval_task_agnostic
from src.args import parse_arguments

import pdb
# Config
args = parse_arguments()
pretrained_checkpoint = f'checkpoints/{args.model}/zeroshot.pt'


def evaluate_individial_fts(task_vectors, args, task_agnostic=True):
    _eval_f = eval_task_agnostic if task_agnostic else eval_task_aware
    _eval_name = "task-agnostic" if task_agnostic else "task-aware"
    
    print('#' * 100 + f"\nPerforming {_eval_name} evaluation of individual finetunings.")

    results = []
    for idx in range(args.n_splits):
        print(f"\nEVAL: {args.dataset}-{args.n_splits} ({args.split_strategy} incremental) - split idx: {idx}")

        # Create the task vectors
        image_encoder = task_vectors[idx].apply_to(pretrained_checkpoint, scaling_coef=1.0)

        # Evaluate
        res = _eval_f(image_encoder, args)
        results.append(res)
        print(f"{_eval_name} eval on {args.dataset} after task {idx}. Accuracies:\n{res}")

    print(f"{_eval_name} evaluation of individual finetunings: final results:\n{results}\n" + '#' * 100 + '\n')


def evaluate_merged_fts(task_vectors, args, merging_f, scaling_coef, task_agnostic=True, only_final=False, use_validset=False):
    """
    这个函数循环每一步都对seen task的ckpts进行merge, 然后进行eval
    only_final: 控制跳过前面的merge, 仅在最后一个循环对所有ckpts进行merge然后进行eval, (其实就是我下面写得evaluate_merging_for_one_mode)
    """
    _eval_f = eval_task_agnostic if task_agnostic else eval_task_aware
    _eval_name = "task-agnostic" if task_agnostic else "task-aware"
    
    print('#' * 100 + f"\nPerforming {_eval_name} evaluation of merged finetunings.")

    results = []
    best_results = []  # 存储使用best_coeffs的结果
    
    for idx in range(args.n_splits):
        if idx == 0 and use_validset:
            continue
        if only_final and idx != args.n_splits - 1:
            continue
        
        print(f"\nEVAL: {args.dataset}-{args.n_splits} ({args.split_strategy} incremental) - split idx: {idx}")

        # _tvs = task_vectors[:idx+1]
        
        if use_validset:
            if idx == 1:
                _tvs = [task_vectors[0], task_vectors[1]]
            else:
                _tvs = [merged_tv, task_vectors[idx]]  # 选择使用的是last_coeffs还是best_coeffs的merged_tv
            
            # 在使用validset时总是返回两个task vectors
            best_merged_tv, merged_tv = merging_f(_tvs, pretrained_checkpoint, args, validation_splits=range(idx+1), validation_proportion=args.validset_ratio)
            
            # 评估best_coeffs生成的task vector
            print(f"Evaluating with BEST coefficients task vector:")
            best_image_encoder = best_merged_tv.apply_to(pretrained_checkpoint, scaling_coef=scaling_coef)
            best_res = _eval_f(best_image_encoder, args)
            best_results.append(best_res)
            print(f"{_eval_name} eval with BEST coefficients on {args.dataset} after task {idx}. Accuracies:\n{best_res}")
            
            # 评估last_coeffs生成的task vector
            print(f"Evaluating with LAST coefficients task vector:")
            image_encoder = merged_tv.apply_to(pretrained_checkpoint, scaling_coef=scaling_coef)
            res = _eval_f(image_encoder, args)
            results.append(res)
            print(f"{_eval_name} eval with LAST coefficients on {args.dataset} after task {idx}. Accuracies:\n{res}")
        else:
            _tvs = task_vectors[:idx+1]
            merged_tv = merging_f(_tvs)
            image_encoder = merged_tv.apply_to(pretrained_checkpoint, scaling_coef=scaling_coef)
            
            # 评估
            res = _eval_f(image_encoder, args)
            results.append(res)
            print(f"{_eval_name} eval on {args.dataset} after task {idx}. Accuracies:\n{res}")
    
    print(f"{_eval_name} evaluation of merged finetunings: final results:\n{results}\n" + '#' * 100 + '\n')
    
    # 如果有best_results，打印它们
    if best_results:
        print(f"{_eval_name} evaluation with BEST coefficients: final results:\n{best_results}\n" + '#' * 100 + '\n')

def evaluate_merging_for_one_mode(task_vectors, merge_mode, dataset, n_splits, split_strategy, n_coeffs=20):
    """
    把search_evaluate_merging中测试多种merging方法, 改为只测试一种方法.
    """
    if merge_mode in ['max', 'no_merge', 'pretrained']:
        merging_f = merge_max_abs
        scaling_coef = 0.5
    elif merge_mode == 'rnd':
        merging_f = merge_rnd_mix
        scaling_coef = 1.0
    elif merge_mode == 'avg':
        merging_f = sum
        scaling_coef = 1.0/n_splits
    print(f"\nMerging with function: {merging_f.__name__}")
    merged_tv = merging_f(task_vectors)
    
    # Apply the resulting task vector
    results = {}


    image_encoder = merged_tv.apply_to(pretrained_checkpoint, scaling_coef=scaling_coef)
    # Evaluate
    _r = eval_single_dataset(image_encoder, dataset, args)['top1']
    results[scaling_coef] = _r

    print(f"Results with function {merging_f.__name__}:\n{results}")

def search_evaluate_merging(task_vectors, dataset, n_splits, split_strategy, n_coeffs=20):
    print(f"\nEVAL: {dataset}-{n_splits} ({split_strategy} incremental)")
    
    funcs_and_coeffs = [
        # (merge_rnd_mix, np.linspace(0.5, 1.5, num=n_coeffs+1)[1:]),
        # (merge_max_abs, np.linspace(0.0, 1.0, num=n_coeffs+1)[1:]),
        # (sum, np.linspace(0.0, 2.0/n_splits, num=n_coeffs+1)[1:]),
        (merge_rnd_mix, [1.0]),
        (merge_max_abs, [0.5]),
        (sum, [1.0/n_splits]),
    ]

    for f, coeffs in funcs_and_coeffs:
        print(f"\nMerging with function: {f.__name__}")
        merged_tv = f(task_vectors)
        
        # Apply the resulting task vector
        results = {}
        for scaling_coef in coeffs:
            print(f"Scaling coeff: {scaling_coef}")
            image_encoder = merged_tv.apply_to(pretrained_checkpoint, scaling_coef=scaling_coef)
            # Evaluate
            _r = eval_single_dataset(image_encoder, dataset, args)['top1']
            wandb.log({
                f"merging/{f.__name__}": _r * 100.0,
                "helpers/merging/alpha": scaling_coef,
            })
            results[scaling_coef] = _r

        print(f"Results with function {f.__name__}:\n{results}")
        
    # TIES merging
    reset_type = 'topk'
    reset_thresh = 20
    resolve = 'mass'
    merge = 'dis-mean'
    tv_flat_checks = torch.vstack([state_dict_to_vector(tv.vector) for tv in task_vectors])
    
    print(f"\nMerging with TIES merging: pruning {reset_type}-{reset_thresh}, resolve sign by {resolve}, merge by {merge}")
    
    merged_flat_tv = merge_methods(
        reset_type,
        tv_flat_checks,
        reset_thresh=reset_thresh,
        resolve_method=resolve,
        merge_func=merge,
    )
    merged_tv = vector_to_state_dict(
        merged_flat_tv, task_vectors[0].vector, remove_keys=[]
    )
    merged_tv = TaskVector(vector=merged_tv)

    # Apply the resulting task vector
    results = {}
    # for scaling_coef in np.linspace(0.5, 1.5, num=n_coeffs+1)[1:]:
    for scaling_coef in [0.55]:
        print(f"Scaling coeff: {scaling_coef}")
        image_encoder = merged_tv.apply_to(pretrained_checkpoint, scaling_coef=scaling_coef)
        # Evaluate
        _r = eval_single_dataset(image_encoder, dataset, args)['top1']
        wandb.log({
            f"merging/TIES": _r * 100.0,
            "helpers/merging/alpha": scaling_coef,
        })
        results[scaling_coef] = _r
            
    print(f"Results with function TIES:\n{results}")


if __name__ == '__main__':
    print('Program start time:', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    suffix = ""
    if args.lwf_lamb > 0.0:
        method = "lwf"
        args.save = f'checkpoints/{args.model}/lwf'
        suffix = f"-lamb:{args.lwf_lamb}"
    elif args.ewc_lamb > 0.0:
        method = "ewc"
        args.save = f'checkpoints/{args.model}/ewc'
        suffix = f"-lamb:{args.ewc_lamb}"
    
    # args.save = f'checkpoints/{args.model}/{merged_ft_dir}{sequential_ft_dir}{args.split_strategy}_incremental'
    
    elif args.sequential_finetuning:
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

    wandb.init(
        project="magmax",
        group="merging-CIL",
        entity="zzsyjl",
        mode='disabled',
        name=name,
        config=args,
        tags=["merging", "CIL", f"{args.dataset}", f"{method}"],
    )
    
    # preload task vectors
    if args.model_dist_type is None or args.model_dist_type == 'L1':
        model_dist_type = ''
    else:
        model_dist_type = f'_{args.model_dist_type}_'
    kd_weight = f'_kd{args.kd_weight}_' if args.kd_weight > 0 else ''
    reg_weight_label = f'{args.reg_weight}' if args.reg_weight > 0 else ''
    temp_label = f'_temp{args.temperature}_' if args.temperature > 0 else ''
    pre_merged_mode_label = f'{args.pre_merged_mode}' if args.pre_merged_mode is not None else ''

    task_vectors = [
        TaskVector(pretrained_checkpoint, f'{args.save}/{args.dataset}-{args.n_splits}/ft-epochs-{args.epochs}-seed:{args.seed}{suffix}/finetuned_{pre_merged_mode_label}{reg_weight_label}{model_dist_type}{kd_weight}{temp_label}{_idx}.pt')
        for _idx in range(args.n_splits)
    ]

    # evaluate_individial_fts(task_vectors, args, task_agnostic=True)

    if args.pre_merged_mode in ['max', 'no_merge', 'pretrained']:
        merging_f = merge_max_abs
        scaling_coef = 0.5
    elif args.pre_merged_mode == 'avg':
        merging_f = sum
        scaling_coef = 1.0/args.n_splits
    else:
        merging_f = merge_max_abs
        scaling_coef = 0.5

    # # 使用validset. 注意和下边注释的代码不能同时存在
    # merging_f = merge_learned
    # scaling_coef = 1.0
    # args.validset_ratio = 0.006
    # args.seed_for_validset = 5
    # args.batch_size = 64
    # evaluate_merged_fts(task_vectors, args, merging_f, scaling_coef, task_agnostic=True, only_final=False, use_validset=True)

    # 原有代码, 不用validset
    # merging_f = merge_rnd_mix
    # scaling_coef = 1.0
    merging_f = merge_max_abs
    scaling_coef = 0.5
    # merging_f = sum
    # scaling_coef = 1.0/args.n_splits
    evaluate_merged_fts(task_vectors, args, merging_f, scaling_coef, task_agnostic=True, only_final=False, use_validset=False)
    
    # evaluate_merging_for_one_mode(task_vectors, args.pre_merged_mode, args.dataset, args.n_splits, args.split_strategy)
    # evaluate_merged_fts(task_vectors, args, merge_max_abs, 0.5, task_agnostic=False)

    # if args.merged_finetuning_mode: # max, avg, rnd, 是表示我所提出的mergedStart 用何种方式来merge得到每次finetune的起点
    #     evaluate_merging_for_one_mode(task_vectors, args.merged_finetuning_mode, args.dataset, args.n_splits, args.split_strategy)
    # else:
    #     search_evaluate_merging(task_vectors, args.dataset, args.n_splits, args.split_strategy)

    # print('Program end time:', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))