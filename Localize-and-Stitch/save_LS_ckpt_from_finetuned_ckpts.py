import os
import torch
import pdb 
import numpy as np
from pathlib import Path

import sys
import os
sys.path.append(os.path.abspath('../../'))
from localize_utils import *

import time
import sys
# TODO: change to your checkpoint folders
# root = '/data/common/task-arithmetic'
root = '/data/hdc/jinglong/magmax'
sys.path.append(root)

from eval import eval_single_dataset
from args import parse_arguments
import pickle

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

def create_log_dir(path, filename='log.txt'):
    import logging
    if not os.path.exists(path):
        os.makedirs(path)
    logger = logging.getLogger(path)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(path+'/'+filename)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

# exam_datasets = ['SUN397', 'Cars', 'RESISC45', 'EuroSAT', 'SVHN', 'GTSRB', 'MNIST', 'DTD'] 

model_type = 'ViT-B-16' 
args = parse_arguments()
# args.data_location = root + '/data'
args.model = model_type
args.save = root + '/checkpoints/' + model_type
# args.save = root + '/task_vectors_checkpoints/' + model
args.log = True
pretrained_checkpoint = root+'/checkpoints/'+model_type+'/zeroshot.pt'
# pretrained_checkpoint = root+'/task_vectors_checkpoints/'+model+'/zeroshot.pt'

str_time_ = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))

graft_args = parse_arguments()
graft_args.checkpoint_location = root+'/ckpt'
graft_args.sparsity = args.sparsity
graft_args.sigmoid_bias = 5
args.logs_path = '../logs/'+model_type+'/'

if args.log:
    log = create_log_dir(args.logs_path, 'log_dataless_localize_stitch_{}.txt'.format(str_time_))

# start training masks
final_model = torch.load(pretrained_checkpoint)
pretrained_model = torch.load(pretrained_checkpoint)
model = torch.load(pretrained_checkpoint)
pretrained_model_dic = pretrained_model.state_dict()

trainable_params = {}
frozen = ["model.positional_embedding", "model.text_projection", "model.logit_scale", "model.token_embedding.weight", "model.ln_final.weight", "model.ln_final.bias"]
for k, v in pretrained_model_dic.items():
    if k not in frozen:
        trainable_params[k] = v

start_time = time.time()

# TODO: change to your finetuned checkpoints path, make dataset_name is None



print(f'Start Localize and Stitch for {args.dataset_name}-{args.splits}')
# Clear lists and CUDA cache
masks, finetuned_models, proportions = [], [], []
torch.cuda.empty_cache()

for i in range(args.splits):
    # Clear CUDA cache at start of each iteration
    torch.cuda.empty_cache()
    
    # Move models to CPU when no longer needed
    if i > 0:
        del localizer
        torch.cuda.empty_cache()
    
    finetuned_checkpoint = root+'/checkpoints/'+model_type+f'/class_incremental/{args.dataset_name}-{args.splits}/ft-epochs-10-seed:5/finetuned_{i}.pt'
    try:
        finetuned_model = torch.load(finetuned_checkpoint)
    except:
        finetuned_model = pickle.load(open(finetuned_checkpoint, 'rb'))

    # MY: change model_type so that won't construct classifier head
    localizer = Localizer(trainable_params, final_model, pretrained_model, finetuned_model, args.dataset_name, args, graft_args,
                           model_type='vit_but_no_classfier_head', DARE=args.DARE) 
    mask, proportion = localizer.interpolate_model(round_=True, return_mask=True)
    # mask_cat = torch.cat(mask, dim=0)
    # print('mask_cat.mean():', mask_cat.mean())
    # test = eval_single_dataset(localizer.model, dataset_name, args)["top1"]
            
    masks.append(mask)
    finetuned_models.append(finetuned_model.cpu())
    proportions.append(proportion.cpu().item())
    # tests.append(test)

localize_time = time.time() - start_time
print('localize_time:', localize_time)

stitcher = Stitcher(trainable_params, model, pretrained_model, finetuned_models, masks, DARE_ratio=graft_args.sparsity if args.DARE else 1.0) # YJL
image_encoder = stitcher.interpolate_models()
stitch_time = time.time() - start_time - localize_time

print('stitch_time:', stitch_time)
DARE_label = f'_DAREratio{graft_args.sparsity}' if args.DARE else ''
image_encoder.save(f'LS_{args.dataset_name}-{args.splits}{DARE_label}.pt')
print(f'Success in saving L&S_{args.dataset_name}-{args.splits}{DARE_label}.pt')
del masks, finetuned_models, proportions
torch.cuda.empty_cache()