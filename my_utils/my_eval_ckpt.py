from src.eval import eval_single_dataset, eval_task_aware, eval_task_agnostic
from src.args import parse_arguments
import torch
args = parse_arguments()
args.model = 'ViT-B-16'

# # eval my reproduced DARE method on CIFAR100 and ImageNetR
# for dataset_name in ['CIFAR100', 'ImageNetR']:
#     for splits in [5, 10]:
#         # for finetune_mode in ['ind', 'seq']: # 
#         for sparsity in [0.01, 0.1, 0.3, 0.5, 0.7]:
#             torch.cuda.empty_cache()
#             model_path = f'Localize-and-Stitch/LS_{dataset_name}-{splits}_DAREratio{sparsity}.pt'
#             print(f"Loading model from {model_path}")
#             image_encoder = torch.load(model_path)
#             eval_single_dataset(image_encoder, dataset_name, args)

# # eval the zeroshot model on CIFAR100 and ImageNetR
# for dataset_name in ['CIFAR100', 'ImageNetR']:
#     image_encoder = torch.load(f'checkpoints/ViT-B-16/zeroshot.pt')
#     eval_single_dataset(image_encoder, dataset_name, args)

# eval the fisher merged model
image_encoder = torch.load(f'./fisher_merged.pt')
eval_single_dataset(image_encoder, 'CIFAR100', args)