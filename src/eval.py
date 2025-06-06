import os
import json
import tqdm
from copy import deepcopy

import torch

from src import utils
from src.datasets.common import get_dataloader, maybe_dictionarize
from src.heads import get_classification_head
from src.modeling import ImageClassifier, ImageEncoder
from src.datasets.registry import get_dataset
from src.cl_utils import get_dataset_and_classifier_for_split



def eval_given_dataset(image_encoder, dataset, dataset_name, args):
    classification_head = get_classification_head(args, dataset_name, classnames=dataset.classnames)
    model = ImageClassifier(image_encoder, classification_head)
    dataloader = dataset.test_loader
    metrics = do_eval(model, dataloader, args.device)
    
    print(f"Done evaluating. Accuracy: {metrics['top1']:.4f}")
    
    return metrics


def eval_single_dataset(image_encoder, dataset_name, args):
    classification_head = get_classification_head(args, dataset_name)
    model = ImageClassifier(image_encoder, classification_head)

    dataset = get_dataset(
        dataset_name,
        model.val_preprocess,
        location=args.data_location,
        batch_size=args.batch_size
    )
    dataloader = get_dataloader(
        dataset, is_train=False, args=args, image_encoder=None)

    metrics = do_eval(model, dataloader, args.device)
    
    print(f"Done evaluating on {dataset_name}. Accuracy: {metrics['top1']:.4f}")
    
    return metrics


def eval_task_aware(image_encoder, args):
    text_encoder = ImageEncoder(args, keep_lang=True)
    full_dataset = get_dataset(
        args.dataset,
        image_encoder.train_preprocess,
        location=args.data_location,
        batch_size=args.batch_size
    )

    accs = []
    for split_idx in range(args.n_splits):
        dataset = deepcopy(full_dataset)
        dataset, classification_head = get_dataset_and_classifier_for_split(
            dataset, split_idx, text_encoder, args 
        )
        model = ImageClassifier(image_encoder, classification_head)
        metrics = do_eval(model, dataset.test_loader, args.device)
        accs.append(metrics['top1'])
        print(f"Task-aware eval on split {split_idx} of dataset {args.dataset}. Accuracy: {accs[-1]:.4f}")
        
    return accs


def eval_task_agnostic(image_encoder, args):
    classification_head = get_classification_head(args, args.dataset)
    model = ImageClassifier(image_encoder, classification_head)

    full_dataset = get_dataset(
        args.dataset,
        image_encoder.train_preprocess,
        location=args.data_location,
        batch_size=args.batch_size
    )

    accs = []
    for split_idx in range(args.n_splits):
        dataset = deepcopy(full_dataset)
        dataset = get_dataset_and_classifier_for_split(
            dataset, split_idx, None, args, remap_labels=False, return_classifier=False
        )
        metrics = do_eval(model, dataset.test_loader, args.device)
        accs.append(round(metrics['top1'], 4))
        print(f"Task-agnostic eval on split {split_idx} of dataset {args.dataset}. Accuracy: {accs[-1]:.4f}")
        
    return accs


@torch.no_grad()
def do_eval(model, dl, device):    
    correct, n = 0., 0.
    model.eval()
    for data in tqdm.tqdm(dl):
        data = maybe_dictionarize(data)
        x = data['images'].to(device)
        y = data['labels'].to(device)

        logits = utils.get_logits(x, model)
        pred = logits.argmax(dim=1, keepdim=True).to(device)
        correct += pred.eq(y.view_as(pred)).sum().item()
        n += y.size(0)

    metrics = {'top1': correct / n}
    
    return metrics


def evaluate(image_encoder, args):
    if args.eval_datasets is None:
        return
    info = vars(args)
    for dataset_name in args.eval_datasets:
        print('Evaluating on', dataset_name)

        results = eval_single_dataset(image_encoder, dataset_name, args)

        if 'top1' in results:
            print(f"{dataset_name} Top-1 accuracy: {results['top1']:.4f}")
        for key, val in results.items():
            if 'worst' in key or 'f1' in key.lower() or 'pm0' in key:
                print(f"{dataset_name} {key}: {val:.4f}")
            info[dataset_name + ':' + key] = val

    if args.results_db is not None:
        dirname = os.path.dirname(args.results_db)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        with open(args.results_db, 'a+') as f:
            f.write(json.dumps(info) + '\n')
        print(f'Results saved to {args.results_db}.')
    else:
        print('Results not saved (to do so, use --results_db to specify a path).')

    return info
