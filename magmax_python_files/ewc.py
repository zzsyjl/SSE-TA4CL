import os
import time
import torch
from copy import deepcopy
import itertools

from src.args import parse_arguments
from src.datasets.common import get_dataloader, maybe_dictionarize
from src.datasets.registry import get_dataset
from src.modeling import ImageEncoder, ImageClassifier
from src.utils import cosine_lr, LabelSmoothing
from src.cl_utils import get_dataset_and_classifier_for_split
from src.eval import evaluate, eval_task_agnostic



def compute_fisher_matrix_diag(model, trn_loader):
    print("Starting computing diagonal of Fisher Information Matrix")
    
    # Store Fisher Information
    fisher = {n: torch.zeros(p.shape).to("cuda") for n, p in model.named_parameters()
                if p.requires_grad}
    # Compute fisher information for specified number of samples -- rounded to the batch size
    num_samples = len(trn_loader.dataset)
    n_samples_batches = (num_samples // trn_loader.batch_size + 1) if num_samples > 0 \
        else (len(trn_loader.dataset) // trn_loader.batch_size)
    # Do forward and backward pass to compute the fisher information
    model.train()
    model = model.cuda()
    for images, targets in itertools.islice(trn_loader, n_samples_batches):
        images = images.to("cuda")
        outputs = model(images)
        preds = outputs.argmax(1)
        loss = torch.nn.functional.cross_entropy(outputs, preds)
        # self.optimizer.zero_grad()
        loss.backward()
        # Accumulate all gradients from loss with regularization
        for n, p in model.named_parameters():
            if p.grad is not None:
                fisher[n] += p.grad.pow(2) * len(targets)
    
    # Apply mean across all samples
    n_samples = n_samples_batches * trn_loader.batch_size
    fisher = {n: (p / n_samples) for n, p in fisher.items()}
    
    print("Finished computing diagonal of Fisher Information Matrix")

    return fisher


def calc_ewc_loss(backbone, fisher, older_params):
    """Returns the distillation loss value"""
    loss_reg = 0
    # Eq. 3: elastic weight consolidation quadratic penalty
    for n, p in backbone.named_parameters():
        if n in fisher.keys():
            loss_reg += torch.sum(fisher[n] * (p - older_params[n]).pow(2)) / 2
    return loss_reg


def finetune(args):
    train_dataset = args.dataset
    eval_results_history = []
    
    # finetune for each split separately
    for split_idx in range(args.n_splits):
        print(f"\n##### SPLIT {split_idx} #####")
        ckpdir = os.path.join(args.save, f"{train_dataset}-{args.n_splits}", f"ft-epochs-{args.epochs}-seed:{args.seed}-lamb:{args.ewc_lamb}")
        ft_path = os.path.join(ckpdir, f'finetuned_{split_idx}.pt')

        if split_idx == 0:
            print('Building image encoder.')
            image_encoder = ImageEncoder(args, keep_lang=True)

        preprocess_fn = image_encoder.train_preprocess
        print_every = 10

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
        
        if split_idx == 0:
            fisher = {n: torch.zeros(p.shape, device="cuda") for n, p in model.named_parameters() if p.requires_grad}

        if args.save is not None:
            os.makedirs(ckpdir, exist_ok=True)
        
        model = model.cuda()
        model.train()

        for epoch in range(args.epochs):
            for i, batch in enumerate(data_loader):
                start_time = time.time()

                step = i + epoch * num_batches
                scheduler(step)
                optimizer.zero_grad()

                batch = maybe_dictionarize(batch)
                inputs = batch['images'].to('cuda:0')
                labels = batch['labels'].to('cuda:0')
                data_time = time.time() - start_time
                
                if split_idx > 0:
                    # ewc_lamb
                    logits = model(inputs)
                    clsf_loss = loss_fn(logits, labels)
                    
                    ewc_loss = calc_ewc_loss(model, fisher, old_params)
                    
                    loss = clsf_loss + args.ewc_lamb * ewc_loss
                else:
                    logits = model(inputs)
                    loss = loss_fn(logits, labels)

                loss.backward()

                torch.nn.utils.clip_grad_norm_(params, 1.0)

                optimizer.step()
                batch_time = time.time() - start_time

                if step % print_every == 0 or i + 1 == n_batches:
                    percent_complete = 100 * i / len(data_loader)
                    if split_idx == 0:
                        print(
                            f"Train Epoch: {epoch} [{percent_complete:.0f}% {i}/{len(dataset.train_loader)}]\t"
                            f"Loss: {loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}", flush=True
                        )
                    else:
                        print(
                            f"Train Epoch: {epoch} [{percent_complete:.0f}% {i}/{len(dataset.train_loader)}]\t"
                            f"Loss: {loss.item():.6f}\t Loss clsf: {clsf_loss.item():.6f}\tLoss EWC: {ewc_loss:.6f}\t"
                            f"Data (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}", flush=True
                        )
        # Evaluate
        image_encoder = model.module.image_encoder
        
        print(f"\nEvaluating after training split {split_idx}...")
        # 首先使用evaluate评估整体性能
        eval_results = evaluate(image_encoder, args)
        overall_acc = eval_results.get(f"{args.dataset}:top1", 0)
        
        # 直接使用eval_task_agnostic评估每个split的性能
        print(f"\nEvaluating performance on each split...")
        split_accs = eval_task_agnostic(image_encoder, args)
        
        # 计算所有split的平均准确率
        avg_split_acc = sum(split_accs) / len(split_accs) if split_accs else 0
        
        # 记录结果
        split_results = {
            'split': split_idx,
            'overall_acc': overall_acc,
            'split_accuracies': {i: acc for i, acc in enumerate(split_accs)},
            'avg_split_acc': avg_split_acc
        }
        eval_results_history.append(split_results)

        if args.save is not None:
            image_encoder.save(ft_path)

        # Store current parameters for the next task
        old_params = {n: p.clone().detach().to("cuda") for n, p in model.named_parameters() if p.requires_grad}

        # calculate Fisher information
        curr_fisher = compute_fisher_matrix_diag(model, data_loader)
        # merge fisher information, we do not want to keep fisher information for each task in memory
        alpha = 0.5
        for n in fisher.keys():
            fisher[n] = (alpha * fisher[n] + (1 - alpha) * curr_fisher[n])

    # Print all historical evaluation results
    print("\n" + "="*50)
    print("EVALUATION HISTORY AFTER EACH SPLIT")
    print("="*50)
    for result in eval_results_history:
        split_idx = result['split']
        print(f"\nResults after training split {split_idx}:")
        print("-" * 30)
        print(f"Overall accuracy: {result['overall_acc']:.4f}")
        
        print("Split accuracies:")
        for s_idx, acc in result['split_accuracies'].items():
            print(f"  Split {s_idx}: {acc:.4f}")
        print(f"Average split accuracy: {result['avg_split_acc']:.4f}")


if __name__ == '__main__':
    args = parse_arguments()
    
    # args.model = 'ViT-B-16'
    args.lr = 1e-5
    args.batch_size = 128
    args.sequential_finetuning = True
    args.split_strategy = 'class'
    args.save = f'checkpoints/{args.model}/ewc'
    args.eval_datasets = [args.dataset]

    print('='*100)
    print(f'Finetuning {args.model} on {args.dataset} ({args.n_splits} splits)')
    print('='*100)

    finetune(args)
