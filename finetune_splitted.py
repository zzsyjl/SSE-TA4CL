import os
import time
import torch
import gc
import numpy as np
from copy import deepcopy
from torch.utils.data import ConcatDataset, Subset, DataLoader


from src.args import parse_arguments
from src.datasets.common import get_dataloader, maybe_dictionarize
from src.datasets.registry import get_dataset
from src.modeling import ImageEncoder, ImageClassifier
from src.utils import cosine_lr, LabelSmoothing
from src.cl_utils import get_dataset_and_classifier_for_split
from src.merging.task_vectors import TaskVector, merge_max_abs, merge_rnd_mix, merge_learned
from src.eval import evaluate, eval_task_agnostic
from src.heads import get_classification_head

import pdb

def model_dist(current_model, pretrained_model, dist_type='L1'):
    """Calculate distance between parameters of current and pretrained models"""
    loss = 0
    for (name1, param1), (name2, param2) in zip(
        current_model.module.image_encoder.named_parameters(),
        pretrained_model.module.image_encoder.named_parameters()
    ):
        if param1.requires_grad:  # Only consider trainable parameters
            if dist_type == 'L1':
                loss += torch.abs(param1 - param2).sum()
            elif dist_type == 'L2':
                loss += torch.norm(param1 - param2, 2)
    return loss

def finetune_merged_model(args, task_vectors, split_idx, pretrained_checkpoint, 
                          validation_splits=None, num_epochs=10, lr=1e-3, 
                          merge_mode='avg', validation_proportion=0.1):
    """
    Directly finetune a merged model (instead of learning coefficients) on validation data.
    
    Args:
        args: Runtime arguments with dataset information
        task_vectors: List of TaskVector objects to merge
        split_idx: Current split index
        pretrained_checkpoint: Path to the pretrained model checkpoint
        validation_splits: List of validation split indices to use (defaults to all splits up to split_idx)
        num_epochs: Number of training epochs
        lr: Learning rate for model optimization
        merge_mode: Strategy for initial merging ('avg' or 'max')
        validation_proportion: Proportion of training data to use for validation
        
    Returns:
        Trained image encoder model
    """
    if len(task_vectors) == 0:
        # If no task vectors, return the pretrained model
        return torch.load(pretrained_checkpoint)
        
    # Adjust epochs based on validation proportion
    base_proportion = 0.1  # Base proportion that num_epochs is calibrated for
    if validation_proportion < base_proportion:
        adjusted_epochs = int(num_epochs * (base_proportion / validation_proportion))
        print(f"Based on validation proportion {validation_proportion} (base: {base_proportion}), "
              f"adjusting training epochs from {num_epochs} to {adjusted_epochs}")
        num_epochs = adjusted_epochs
    
    # =========== 1. Initialize model by merging task vectors ===========
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create initially merged model based on merge_mode
    if merge_mode == 'avg':
        print(f"Merging {len(task_vectors)} task vectors using average method")
        merged_tv = sum(task_vectors)
        scaling_coef = 1.0 / len(task_vectors)
        image_encoder = merged_tv.apply_to(pretrained_checkpoint, scaling_coef=scaling_coef)
    elif merge_mode == 'max':
        print(f"Merging {len(task_vectors)} task vectors using max_abs method")
        merged_tv = merge_max_abs(task_vectors)
        image_encoder = merged_tv.apply_to(pretrained_checkpoint, scaling_coef=0.5)
    else:
        raise ValueError(f"Unsupported merge mode: {merge_mode}")
    
    image_encoder = image_encoder.to(device)
    
    # =========== 2. Prepare split-specific data loaders ===========
    if validation_splits is None:
        validation_splits = list(range(split_idx + 1))
    
    print("Preparing validation datasets for each split...")
    preprocess_fn = image_encoder.train_preprocess
    
    # 创建每个split的加载器
    split_loaders = []
    split_classification_heads = []
    
    full_dataset = get_dataset(
        args.dataset,
        preprocess_fn,
        location=args.data_location,
        batch_size=args.batch_size
    )
    
    for split_i in validation_splits:
        print(f"Creating validation loader for split {split_i}")
        dataset = deepcopy(full_dataset)
        # 获取特定split的数据集和分类头
        split_dataset, split_head = get_dataset_and_classifier_for_split(
            dataset, split_i, image_encoder, args
        )
        split_head = split_head.to(device)
        split_classification_heads.append(split_head)
        
        # 创建验证子集和加载器
        train_dataset = split_dataset.train_dataset
        dataset_size = len(train_dataset)
        val_size = int(dataset_size * validation_proportion)
        
        if val_size > 0:
            torch.manual_seed(args.seed_for_validset)
            indices = torch.randperm(dataset_size)[:val_size].tolist()
            val_subset = Subset(train_dataset, indices)
            
            # 为每个split创建单独的加载器
            split_loader = DataLoader(
                val_subset,
                batch_size=max(args.batch_size // len(validation_splits), 16),  # 至少16个样本
                shuffle=True,
                num_workers=getattr(args, 'num_workers', 4)
            )
            split_loaders.append(split_loader)
            print(f"  Split {split_i}: total samples {dataset_size}, using {val_size} for validation ({validation_proportion:.1%})")
    
    # =========== 3. Prepare optimizer ===========
    # 直接优化image_encoder而不是整个模型
    params = [p for p in image_encoder.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=lr)
    
    # 准备损失函数
    criterion = torch.nn.CrossEntropyLoss()
    
    # =========== 4. Training loop ===========
    best_accuracy = 0.0
    best_state_dict = None
    
    print(f"Starting model finetuning for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        # 每个epoch开始前设置为训练模式
        image_encoder.train()
        running_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        
        # 为每个split单独训练
        for split_i, (loader, head) in enumerate(zip(split_loaders, split_classification_heads)):
            split_correct = 0
            split_total = 0
            split_loss = 0.0
            
            for batch_idx, batch in enumerate(loader):
                # 清零梯度
                optimizer.zero_grad()
                
                # 准备数据
                batch = maybe_dictionarize(batch)
                inputs = batch['images'].to(device)
                targets = batch['labels'].to(device)
                
                # 创建临时模型用于此split
                temp_model = ImageClassifier(image_encoder, head)
                temp_model.to(device)
                
                # 前向传播
                outputs = temp_model(inputs)
                
                # 计算损失
                loss = criterion(outputs, targets)
                
                # 反向传播和参数更新
                loss.backward()
                optimizer.step()
                
                # 计算准确率
                _, predicted = outputs.max(1)
                batch_correct = predicted.eq(targets).sum().item()
                batch_size = targets.size(0)
                
                split_correct += batch_correct
                split_total += batch_size
                split_loss += loss.item()
            
            # 累计所有split的统计信息
            epoch_correct += split_correct
            epoch_total += split_total
            running_loss += split_loss
            
            # 输出每个split的准确率
            if len(loader) > 0:  # 避免除以零
                print(f"  Split {split_i} accuracy: {100.0 * split_correct / split_total:.2f}%, "
                      f"average loss: {split_loss / len(loader):.4f}")
        
        # 计算整体准确率
        if epoch_total > 0:
            epoch_accuracy = 100.0 * epoch_correct / epoch_total
            
            if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == num_epochs - 1:
                print(f"Epoch {epoch+1}/{num_epochs} completed, "
                      f"overall accuracy: {epoch_accuracy:.2f}%, "
                      f"average loss: {running_loss/sum(len(l) for l in split_loaders):.4f}")
            
            # 更新最佳模型
            if epoch_accuracy > best_accuracy:
                best_accuracy = epoch_accuracy
                best_state_dict = deepcopy(image_encoder.state_dict())
    
    # =========== 5. 应用最佳模型并返回结果 ===========
    print(f"Finetuning complete. Best accuracy: {best_accuracy:.2f}%")
    
    # 加载最佳状态如果有
    if best_state_dict is not None:
        image_encoder.load_state_dict(best_state_dict)
    
    # 创建task vector
    pretrained_model = torch.load(pretrained_checkpoint)
    
    # 创建最终的task vector
    merged_tv = TaskVector(pretrained_checkpoint=pretrained_checkpoint, 
                          finetuned_state_dict=image_encoder.state_dict())
    
    # 返回兼容merge_learned的格式
    return merged_tv

def finetune(args):
    train_dataset = args.dataset
    reg_weight_label = f'_reg{args.reg_weight}_' if args.reg_weight > 0 else ''
    model_dist_type = f'_{args.model_dist_type}_' if args.model_dist_type is not None else ''
    
    ckpdir = os.path.join(args.save,
                          f"{train_dataset}-{args.n_splits}",
                          f"ft-epochs-{args.epochs}-seed:{args.seed}"
                          )

    # 创建列表来存储所有split的评估结果
    all_evaluation_results = []
    # 创建列表来存储所有split的last_coeffs
    all_last_coeffs = []

    # finetune for each split separately
    for split_idx in range(args.n_splits):
        print(f"\n##### SPLIT {split_idx} #####")
        ft_path = os.path.join(ckpdir, f'finetuned_{args.merged_finetuning_mode}{reg_weight_label}{model_dist_type}{args.validset_ratio}_{split_idx}.pt')
        if os.path.exists(ft_path):
            print(f"Skipping finetuning on split {split_idx}, "
                  f"ckpt already exists under {ft_path}")
            continue

        assert train_dataset is not None, "Please provide a training dataset."
        # pdb.set_trace()
        if args.load is not None and args.load.endswith('pt'):
            image_encoder = ImageEncoder.load(args.load, keep_lang=True)
        elif args.sequential_finetuning and split_idx != 0:
            prev_ckpt = os.path.join(ckpdir, f'finetuned_{split_idx-1}.pt')
            print(f'Loading image encoder from prev task {prev_ckpt=}')
            image_encoder = torch.load(prev_ckpt, weights_only=True)
        elif args.merged_finetuning_mode and split_idx != 0:
            # init from merged task vector
            task_vectors = [
                TaskVector(f'checkpoints/{args.model}/zeroshot.pt', 
                          f'{args.save}/{args.dataset}-{args.n_splits}/ft-epochs-{args.epochs}-seed:{args.seed}/finetuned_{args.merged_finetuning_mode}{reg_weight_label}{model_dist_type}{args.validset_ratio}_{_idx}.pt')
                for _idx in range(split_idx)
            ]
            print(f'Successfully merged image encoder for finetuning on split {split_idx}, from task vectors up to split {split_idx-1}')


            if args.merged_finetuning_mode == 'max':
                merged_tv = merge_max_abs(task_vectors)
                image_encoder = merged_tv.apply_to(f'checkpoints/{args.model}/zeroshot.pt', scaling_coef=0.5)
            elif args.merged_finetuning_mode == 'rnd':
                merged_tv = merge_rnd_mix(task_vectors)
                image_encoder = merged_tv.apply_to(f'checkpoints/{args.model}/zeroshot.pt', scaling_coef=1.0)
            elif args.merged_finetuning_mode == 'avg':
                merged_tv = sum(task_vectors)
                image_encoder = merged_tv.apply_to(f'checkpoints/{args.model}/zeroshot.pt', scaling_coef=1.0/args.n_splits)
            elif args.merged_finetuning_mode == 'learned' or args.merged_finetuning_mode == 'finetune_model':
                # 对于"learned"模式，直接使用前一个split在evaluate时得到的image_encoder
                image_encoder = prev_learned_image_encoder
                print(f'Reusing previous learned image encoder for finetuning on split {split_idx}')

            
        else:
            print('Building image encoder.')
            image_encoder = ImageEncoder(args, keep_lang=True)

        # Load pretrained model if using regularization with pre_merged_mode
        if args.pre_merged_mode == 'pretrained' and args.reg_weight > 0:
            print('Loading pretrained model for regularization.')
            pretrained_model = torch.load(f'checkpoints/{args.model}/zeroshot.pt')
            pretrained_model.cuda()
            pretrained_model.eval()

        if split_idx==0 and not os.path.exists(f'checkpoints/{args.model}/zeroshot.pt'):
            image_encoder.save(f'checkpoints/{args.model}/zeroshot.pt')

        preprocess_fn = image_encoder.train_preprocess
        print_every = 100

        dataset = get_dataset(
            train_dataset,
            preprocess_fn,
            location=args.data_location,
            batch_size=args.batch_size
        )
        dataset, classification_head = get_dataset_and_classifier_for_split(
            dataset, split_idx, image_encoder, args
        )
        
        devices = list(range(torch.cuda.device_count()))
        print('Using devices', devices)
        
        model = ImageClassifier(image_encoder, classification_head)
        model.freeze_head()
        model.freeze_lang()
        model = torch.nn.DataParallel(model, device_ids=devices)

        # If using regularization with pre_merged_mode, prepare pretrained model with same classification head
        if args.pre_merged_mode == 'pretrained' and args.reg_weight > 0:
            pretrained_model = ImageClassifier(pretrained_model, classification_head)
            pretrained_model.freeze_head()
            pretrained_model.freeze_lang()
            pretrained_model = torch.nn.DataParallel(pretrained_model, device_ids=devices)

        if args.ls > 0:
            loss_fn = LabelSmoothing(args.ls)
        else:
            loss_fn = torch.nn.CrossEntropyLoss()

        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
        num_batches = len(dataset.train_loader)
        scheduler = cosine_lr(optimizer, args.lr, args.warmup_length, args.epochs * num_batches)
        data_loader = get_dataloader(dataset, is_train=True, args=args, image_encoder=None)
        n_batches = len(data_loader)

        if args.save is not None:
            os.makedirs(ckpdir, exist_ok=True)

        for epoch in range(args.epochs):
            model = model.cuda()
            model.train()

            for i, batch in enumerate(data_loader):
                start_time = time.time()

                step = i + epoch * num_batches
                scheduler(step)
                optimizer.zero_grad()

                batch = maybe_dictionarize(batch)
                inputs = batch['images'].to('cuda:0')
                labels = batch['labels'].to('cuda:0')
                data_time = time.time() - start_time

                logits = model(inputs)

                loss_cls = loss_fn(logits, labels)
                
                # Add regularization term if using pretrained mode
                if args.pre_merged_mode == 'pretrained' and args.reg_weight > 0:
                    loss_dist = model_dist(model, pretrained_model, args.model_dist_type)
                    loss = loss_cls + args.reg_weight * loss_dist
                    
                    if step % print_every == 0 or i + 1 == n_batches:
                        print(f'loss_cls: {loss_cls.item():.6f}, loss_dist: {loss_dist.item():.6f}')
                else:
                    loss = loss_cls

                loss.backward()

                torch.nn.utils.clip_grad_norm_(params, 1.0)

                optimizer.step()
                batch_time = time.time() - start_time

                if step % print_every == 0 or i + 1 == n_batches:
                    percent_complete = 100 * i / len(data_loader)
                    print(
                        f"Train Epoch: {epoch} [{percent_complete:.0f}% {i}/{len(dataset.train_loader)}]\t"
                        f"Loss: {loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}", flush=True
                    )

        # Evaluate after each split
        image_encoder = model.module.image_encoder

        if args.save is not None:
            image_encoder.save(ft_path)

        # 评估 merged task vector
        if args.merged_finetuning_mode == 'learned' or args.merged_finetuning_mode == 'finetune_model':
            # 创建当前split的task vector
            current_task_vector = TaskVector(f'checkpoints/{args.model}/zeroshot.pt', 
                                           f'{args.save}/{args.dataset}-{args.n_splits}/ft-epochs-{args.epochs}-seed:{args.seed}/finetuned_{args.merged_finetuning_mode}{reg_weight_label}{model_dist_type}{args.validset_ratio}_{split_idx}.pt')
            
            if split_idx == 0:
                # 第一个split不需要合并，直接保存task vector供下次使用
                prev_merged_tv = current_task_vector
                task_vector_to_apply = current_task_vector
                print(f'First split, saving task vector directly for next merge')
            else:
                # 从第二个split开始，每次只合并两个task vector：
                # 1. 前一次得到的merged task vector
                # 2. 当前split的新task vector


                # # 试验代码开始, 增加ind task vector来merge
                # args_save_ind = f'checkpoints/{args.model}/{args.split_strategy}_incremental'
                # current_task_vector_ind = TaskVector(f'checkpoints/{args.model}/zeroshot.pt', 
                #                            f'{args_save_ind}/{args.dataset}-{args.n_splits}/ft-epochs-{args.epochs}-seed:5/finetuned_{split_idx}.pt')
                # print(f'Merging previous merged task vector with current split {split_idx} task vector')
                # task_vectors = [prev_merged_tv, current_task_vector, current_task_vector_ind]
                # 试验代码结束

                task_vectors = [prev_merged_tv, current_task_vector]
                if args.merged_finetuning_mode == 'finetune_model':
                    merged_tv = finetune_merged_model(args, task_vectors, split_idx, f'checkpoints/{args.model}/zeroshot.pt', 
                                                    validation_splits=range(split_idx+1), 
                                                    validation_proportion=args.validset_ratio,
                                                    num_epochs=args.merge_learned_epochs)
                    # 为了兼容性，为all_last_coeffs添加一个占位符
                    dummy_coeffs = torch.ones(2)  # 仅用于保持数据结构一致
                    all_last_coeffs.append([dummy_coeffs, dummy_coeffs])
                else:
                    best_merged_tv, merged_tv, best_coeffs, last_coeffs = merge_learned(task_vectors, split_idx, f'checkpoints/{args.model}/zeroshot.pt', args, 
                                                            validation_splits=range(split_idx+1), 
                                                            validation_proportion=args.validset_ratio,
                                                            num_epochs=args.merge_learned_epochs)
                    
                    # 保存last_coeffs到列表中
                    all_last_coeffs.append(last_coeffs)
                
                # 保存新的merged task vector供下次使用
                prev_merged_tv = merged_tv
                task_vector_to_apply = merged_tv
            
            # 应用最新的task vector
            image_encoder = task_vector_to_apply.apply_to(f'checkpoints/{args.model}/zeroshot.pt', scaling_coef=1.0)
            
            # 评估并收集结果
            eval_results = eval_task_agnostic(image_encoder, args)
            all_evaluation_results.append(eval_results)
            
            # 保存image_encoder用于下一个split
            globals()['prev_learned_image_encoder'] = image_encoder
            print(f'Saved learned image encoder for potential reuse in next split')
            
            # 在模型评估后释放GPU内存
            del image_encoder
            gc.collect()
            torch.cuda.empty_cache()
        
        else:
            eval_results = evaluate(image_encoder, args)
            all_evaluation_results.append(eval_results)

    # 在所有split训练完成后打印汇总结果
    print("\n" + "="*50)
    print(f"Summary of all evaluation results for {args.dataset} with {args.n_splits} splits:")
    print("="*50)
    
    for i, result in enumerate(all_evaluation_results):
        print(f"Split {i} evaluation results:")
        print(result[:i+1])
        print('mean accuracy:', np.mean(result[:i+1]))
        if i > 0:
            print(f'prev_coeffs: {all_last_coeffs[i-1][0]}')
            print(f'current_coeffs: {all_last_coeffs[i-1][1]}')
        print("-"*30)


            





if __name__ == '__main__':
    args = parse_arguments()
    
    args.lr = 1e-5
    args.batch_size = 128
    sequential_ft_dir = 'sequential_finetuning/' if args.sequential_finetuning else ''
    merged_ft_dir = 'merged_finetuning/' if args.merged_finetuning_mode else ''
    reg_dir = 'pretrained_reg/' if args.pre_merged_mode == 'pretrained' and args.reg_weight > 0 else ''
    args.save = f'checkpoints/{args.model}/{merged_ft_dir}{sequential_ft_dir}{reg_dir}{args.split_strategy}_incremental'

    print('='*100)
    print(f'Finetuning {args.model} on {args.dataset} ({args.n_splits} splits)')
    print('='*100)

    finetune(args)
