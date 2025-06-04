import os
import time
import torch
import torch.nn.functional as F

from src.args import parse_arguments
from src.datasets.common import get_dataloader, maybe_dictionarize
from src.datasets.registry import get_dataset
from src.modeling import ImageEncoder, ImageClassifier
from src.utils import cosine_lr, LabelSmoothing
from src.cl_utils import get_dataset_and_classifier_for_split
from src.merging.task_vectors import TaskVector, merge_max_abs, merge_rnd_mix
from src.eval import evaluate

import pdb

def model_dist(current_model, merged_model, dist_type='L1'):
    """Calculate L1 distance between parameters of current and merged models"""
    l1_loss = 0
    for (name1, param1), (name2, param2) in zip(
        current_model.module.image_encoder.named_parameters(),
        merged_model.module.image_encoder.named_parameters()
    ):
        if param1.requires_grad:  # Only consider trainable parameters
            if dist_type == 'L1':
                l1_loss += torch.abs(param1 - param2).sum()
            elif dist_type == 'L2':
                l1_loss += torch.norm(param1 - param2, 2)
    return l1_loss

def distillation_loss(student_logits, teacher_logits, temperature=2.0):
    """
    Compute knowledge distillation loss between student (current) and teacher (pre-merged) models
    Args:
        student_logits: logits from current model
        teacher_logits: logits from pre-merged model
        temperature: softmax temperature for distillation (higher temperature = softer probabilities)
    """
    soft_targets = F.softmax(teacher_logits / temperature, dim=-1)
    student_log_softmax = F.log_softmax(student_logits / temperature, dim=-1)
    kd_loss = -(soft_targets * student_log_softmax).sum(dim=-1).mean()
    return kd_loss * (temperature ** 2)  # Scale loss with temperature

def finetune(args):
    train_dataset = args.dataset
    ckpdir = os.path.join(args.save,
                          f"{train_dataset}-{args.n_splits}",
                          f"ft-epochs-{args.epochs}-seed:{args.seed}"
                          )
    model_dist_type = f'_{args.model_dist_type}_' if args.model_dist_type is not None else ''
    kd_weight = f'_kd{args.kd_weight}_' if args.kd_weight > 0 else ''
    reg_weight_label = f'_reg{args.reg_weight}_' if args.reg_weight > 0 else ''
    temp_label = f'_temp{args.temperature}_' if args.temperature > 0 else ''


    # finetune for each split separately
    for split_idx in range(args.n_splits):
        print(f"\n##### SPLIT {split_idx} #####")
        ft_path = os.path.join(ckpdir, f'finetuned_{args.pre_merged_mode}{reg_weight_label}{model_dist_type}{kd_weight}{temp_label}{split_idx}.pt')
        if os.path.exists(ft_path):
            print(f"Skipping finetuning on split {split_idx}, "
                  f"ckpt already exists under {ft_path}")
            continue

        assert train_dataset is not None, "Please provide a training dataset."
        # pdb.set_trace()
        if args.load is not None and args.load.endswith('pt'):
            image_encoder = ImageEncoder.load(args.load, keep_lang=True)
        elif args.sequential_finetuning and split_idx != 0:
            prev_ckpt = os.path.join(ckpdir, f'finetuned_{args.pre_merged_mode}{reg_weight_label}{model_dist_type}{kd_weight}{temp_label}{split_idx-1}.pt')
            print(f'Loading image encoder from prev task {prev_ckpt=}')
            image_encoder = torch.load(prev_ckpt)
        else: # 实测, 在没有seq-finetuning的时候, 这里的if-else是执行这里的, 这里也是会加载预训练的权重的, 不是随机初始化
            print('Building image encoder.') 
            image_encoder = ImageEncoder(args, keep_lang=True)
        
        


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
        
        model = ImageClassifier(image_encoder, classification_head)
        model.freeze_head()
        model.freeze_lang()

        if args.pre_merged_mode and split_idx != 0:
            # init from merged task vector
            task_vectors = [
                TaskVector(f'checkpoints/{args.model}/zeroshot.pt', 
                            f'{args.save}/{args.dataset}-{args.n_splits}/ft-epochs-{args.epochs}-seed:{args.seed}/finetuned_{args.pre_merged_mode}{reg_weight_label}{model_dist_type}{kd_weight}{temp_label}{_idx}.pt')
                for _idx in range(split_idx)
            ]
            print(f'Successfully merged image encoder for finetuning on split {split_idx}, from task vectors up to split {split_idx-1}')
            if args.pre_merged_mode == 'max':
                merged_tv = merge_max_abs(task_vectors)
                pre_merged_model = merged_tv.apply_to(f'checkpoints/{args.model}/zeroshot.pt', scaling_coef=0.5)

            elif args.pre_merged_mode == 'avg':
                merged_tv = sum(task_vectors)
                pre_merged_model = merged_tv.apply_to(f'checkpoints/{args.model}/zeroshot.pt', scaling_coef=1.0/args.n_splits)
            elif args.pre_merged_mode == 'no_merge':
                pre_merged_model = task_vectors[-1].apply_to(f'checkpoints/{args.model}/zeroshot.pt', scaling_coef=1.0)
            elif args.pre_merged_mode == 'pretrained':
                pre_merged_model = torch.load(f'checkpoints/{args.model}/zeroshot.pt')
                
            pre_merged_model.cuda()
            pre_merged_model.eval()  # Set pre-merged model to eval mode

            pre_merged_model = ImageClassifier(pre_merged_model, classification_head)
            pre_merged_model.freeze_head()
            pre_merged_model.freeze_lang()

        devices = list(range(torch.cuda.device_count()))
        print('Using devices', devices)
        model = torch.nn.DataParallel(model, device_ids=devices)

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

                # Add either KD loss or L1 regularization when using merged finetuning (except first split)
                if args.pre_merged_mode and split_idx != 0:
                    if args.kd_weight > 0:
                        # Knowledge distillation loss
                        with torch.no_grad():
                            teacher_logits = pre_merged_model(inputs)
                        loss_kd = distillation_loss(logits, teacher_logits, temperature=args.temperature)
                        loss = loss_cls + args.kd_weight * loss_kd
                        loss_dist = torch.tensor(0.0).to(loss.device)
                    else:
                        # L1 distance loss
                        loss_dist = model_dist(model, pre_merged_model, args.model_dist_type)
                        loss = loss_cls + args.reg_weight * loss_dist
                        loss_kd = torch.tensor(0.0).to(loss.device)  
                else:
                    loss = loss_cls
                    loss_dist = torch.tensor(0.0).to(loss.device)  
                    loss_kd = torch.tensor(0.0).to(loss.device)  

                loss.backward()

                torch.nn.utils.clip_grad_norm_(params, 1.0)

                optimizer.step()
                batch_time = time.time() - start_time

                if step % print_every == 0 or i + 1 == n_batches:
                    percent_complete = 100 * i / len(data_loader)
                    print(
                        f"Train Epoch: {epoch} [{percent_complete:.0f}% {i}/{len(dataset.train_loader)}]\t"
                        f"Loss: {loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}",flush=True
                    )
                    if split_idx >0:
                        print('loss_cls:', loss_cls.item(), 'loss_dist:', loss_dist.item(), 'loss_kd:', loss_kd.item())

        # Evaluate
        image_encoder = model.module.image_encoder
        evaluate(image_encoder, args)

        if args.save is not None:
            image_encoder.save(ft_path)



if __name__ == '__main__':
    args = parse_arguments()
    
    args.lr = 1e-5
    args.batch_size = 128
    sequential_ft_dir = 'sequential_finetuning/' if args.sequential_finetuning else ''
    pre_merged_dist_dir = 'pre_merged_distance/' if args.pre_merged_mode else ''
    args.save = f'checkpoints/{args.model}/{pre_merged_dist_dir}{sequential_ft_dir}{args.split_strategy}_incremental'

    print('='*100)
    print(f'Finetuning {args.model} on {args.dataset} ({args.n_splits} splits)')
    print('Program start time:', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    print('='*100)

    finetune(args)

    print('Program end time:', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))