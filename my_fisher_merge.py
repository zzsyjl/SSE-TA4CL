import logging
import os
import re
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import defaultdict

# Set up logging
logger = logging.getLogger(__name__)

def setup_logging(debug=False):
    """Configure logging level and format"""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

# --------------------------------------------------------------------
# Import the same utility functions used in your codebase.
# --------------------------------------------------------------------
from src.datasets.registry import get_dataset
from src.cl_utils import get_dataset_and_classifier_for_split
from src.modeling import ImageEncoder
from src.args import parse_arguments

#################################################################################
# Small wrapper to produce an object with a .loss attribute for Fisher calculation.
#################################################################################
class FisherOutputWrapper(nn.Module):
    """
    Wraps your finetuned model so that forward(...) returns an object with a '.loss' attribute
    for the Fisher merger code to access.
    """
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, images, labels):
        """
        The fisher computation loop expects: 
          outputs = model(**inputs)
          loss = outputs.loss
        So we return an object which has a .loss attribute.
        """
        logits = self.base_model(images)
        loss = self.loss_fn(logits, labels)
        # Build a small object with a .loss attribute
        outputs = type('FisherOutputs', (object,), {})()
        outputs.loss = loss
        return outputs

#################################################################################
# Fisher merging function (trimmed version), originally from fisher_merge.py
#################################################################################
def fisher_merging(models_to_merge: list,
                   trainers: list,
                   exclude_param_names_regex: list,
                   nums_fisher_examples: list,
                   fisher_scaling_coefficients: list = None,
                   normalize_fisher_weight: bool = True,
                   minimal_fisher_weight: float = 1e-6):
    """
    Merge multiple models using the Fisher weights.
    """
    def get_param_names_to_merge(input_param_names, exclude_param_names_regex):
        excluded = set()
        for regex_str in exclude_param_names_regex:
            pattern = re.compile(regex_str)
            for name in input_param_names:
                if pattern.search(name):
                    excluded.add(name)
        return [n for n in input_param_names if n not in excluded]

    def get_param_squared_gradients(model: nn.Module, param_names_to_merge: list):
        """
        Collect param.grad^2 for a model. Returns dict[param_name] = grad^2
        """
        param_squared_gradients = {}
        for param_name, param_value in model.named_parameters():
            if param_name in param_names_to_merge and param_value.grad is not None:
                param_squared_gradients[param_name] = param_value.grad.detach() ** 2
        return param_squared_gradients

    def get_models_fisher_norm(models_to_merge_param_dict, models_to_merge_fisher_weights_list):
        """
        Compute the L2 norm across all parameters for each model’s Fisher weights.
        """
        models_fisher_norm_dict = {}
        # Get the device from the first parameter (they should all be on the same device)
        device = next(iter(models_to_merge_param_dict.values()))[0].device
        
        for param_name, _ in models_to_merge_param_dict.items():
            models_fisher = torch.stack(
                [model_to_merge_fisher_weights[param_name].to(device)
                 for model_to_merge_fisher_weights in models_to_merge_fisher_weights_list],
                dim=0
            )
            # Instead of using all dims, flatten the tensor after the first dimension
            # and then take norm across the flattened dimensions
            batch_size = models_fisher.size(0)
            models_fisher_flat = models_fisher.reshape(batch_size, -1)
            models_fisher_norm = torch.norm(models_fisher_flat, dim=1)
            models_fisher_norm_dict[param_name] = models_fisher_norm

        # shape [num_models_to_merge, num_params], then reduce across parameters
        models_fisher_norm_tensor = torch.stack(list(models_fisher_norm_dict.values()), dim=1)
        models_fisher_norm = torch.norm(models_fisher_norm_tensor, dim=1)
        return models_fisher_norm.to(device)

    def merging_with_fisher_weights(models_to_merge_param_dict,
                                    models_to_merge_fisher_weights_list,
                                    fisher_scaling_coefficients,
                                    normalize_fisher_weight=True,
                                    minimal_fisher_weight=1e-6):
        merged_params = {}
        # Get the device from the first parameter
        device = next(iter(models_to_merge_param_dict.values()))[0].device

        # Optionally normalize each model's Fisher stats by its L2 norm
        if normalize_fisher_weight:
            # shape [num_models_to_merge]
            models_fisher_norm = get_models_fisher_norm(
                models_to_merge_param_dict=models_to_merge_param_dict,
                models_to_merge_fisher_weights_list=models_to_merge_fisher_weights_list
            )

        for param_name, param_value_list in models_to_merge_param_dict.items():
            param_values = torch.stack(param_value_list, dim=0)
            logger.debug(f"{param_name} param_values shape: {param_values.shape}")

            # Check if this parameter has fisher weights
            try:
                models_to_merge_fisher_weights = torch.stack(
                    [mw[param_name].to(device) for mw in models_to_merge_fisher_weights_list],
                    dim=0
                )
            except KeyError:
                logger.debug(f"No fisher weights for {param_name}, using simple averaging")
                # If no fisher weights, just average the parameters
                merged_params[param_name] = param_values.mean(dim=0)
                continue

            if models_to_merge_fisher_weights.numel() == 0:
                logger.debug(f"Empty fisher weights for {param_name}, using simple averaging")
                merged_params[param_name] = param_values.mean(dim=0)
                continue

            models_to_merge_fisher_weights = models_to_merge_fisher_weights + minimal_fisher_weight
            logger.debug(f"{param_name} fisher_weights shape: {models_to_merge_fisher_weights.shape}")

            # shape [num_models_to_merge, 1, 1, ...]
            scaling_coeffs = fisher_scaling_coefficients.to(device)
            logger.debug(f"Before unsqueeze, scaling_coeffs shape: {scaling_coeffs.shape}")
            
            # Create a list of dimensions to match param_values
            expand_shape = [-1] + [1] * (param_values.dim() - 1)
            reshaped_scaling_coeffs = scaling_coeffs.view(*expand_shape)
            logger.debug(f"After reshape, scaling_coeffs shape: {reshaped_scaling_coeffs.shape}")

            if normalize_fisher_weight:
                _models_fisher_norm = 1.0 / (models_fisher_norm + minimal_fisher_weight)
                normalized_fisher_norm = _models_fisher_norm / _models_fisher_norm.sum()
                
                # Reshape normalized_fisher_norm to match param dimensions
                normalized_fisher_norm = normalized_fisher_norm.to(device)
                
                # Match the shape of param_values
                normalized_fisher_norm = normalized_fisher_norm.view(*expand_shape)
                logger.debug(f"normalized_fisher_norm shape: {normalized_fisher_norm.shape}")
                
                reshaped_scaling_coeffs = reshaped_scaling_coeffs * normalized_fisher_norm

            logger.debug("Final shapes before multiplication:")
            logger.debug(f"  - reshaped_scaling_coeffs: {reshaped_scaling_coeffs.shape}")
            logger.debug(f"  - models_to_merge_fisher_weights: {models_to_merge_fisher_weights.shape}")
            logger.debug(f"  - param_values: {param_values.shape}")

            numerator = (reshaped_scaling_coeffs * models_to_merge_fisher_weights * param_values).sum(dim=0)
            denominator = (reshaped_scaling_coeffs * models_to_merge_fisher_weights).sum(dim=0)

            merged_param = numerator / denominator
            merged_params[param_name] = merged_param

        return merged_params

    from collections import defaultdict
    models_to_merge_param_dict = defaultdict(list)
    models_to_merge_fisher_weights_list = []

    # Sanity checks
    assert len(models_to_merge) == len(trainers) == len(nums_fisher_examples), \
        "Number of models, trainers, and nums_fisher_examples must match."

    # Step 2: For each model, compute an average Fisher across num_fisher_examples
    for model_idx, (model, trainer, n_fisher_ex) in enumerate(zip(models_to_merge, trainers, nums_fisher_examples)):
        param_dict = dict(model.named_parameters())  # param_name -> tensor

        # Decide which params to merge
        param_names_to_merge = get_param_names_to_merge(
            list(param_dict.keys()),
            exclude_param_names_regex
        )

        for pn in param_names_to_merge:
            models_to_merge_param_dict[pn].append(param_dict[pn])

        # Walk through the dataloader, compute squared grads for fisher
        dataloader = trainer.get_train_dataloader()
        batch_size = trainer._train_batch_size
        batches_fisher_weights_list = []
        num_computed_examples = 0

        for step, data_batch in tqdm(enumerate(dataloader), 
                                     desc=f"[Model {model_idx}] Fisher Computing"):
            if num_computed_examples >= n_fisher_ex:
                break

            inputs = trainer._prepare_inputs(data_batch)
            outputs = model(**inputs)  # This should produce an object with .loss
            loss = outputs.loss
            model.zero_grad()
            loss.backward()

            batch_fisher_weights = get_param_squared_gradients(model, param_names_to_merge)
            batches_fisher_weights_list.append(batch_fisher_weights)

            num_computed_examples += batch_size

        # Aggregate across all batches for this model
        model_fisher_weights = defaultdict(torch.Tensor)
        for bf in batches_fisher_weights_list:
            for k, v in bf.items():
                if k not in model_fisher_weights:
                    model_fisher_weights[k] = v.clone()
                else:
                    model_fisher_weights[k] += v

        # Average across total examples used
        for k in model_fisher_weights:
            model_fisher_weights[k] /= float(num_computed_examples)
        models_to_merge_fisher_weights_list.append(model_fisher_weights)

    # Step 3: Merge with Fisher weights
    if fisher_scaling_coefficients is None:
        fisher_scaling_coefficients = torch.ones(len(models_to_merge)) / len(models_to_merge)
    else:
        fisher_scaling_coefficients = torch.tensor(fisher_scaling_coefficients, dtype=torch.float32)

    merged_param_dict = merging_with_fisher_weights(
        models_to_merge_param_dict,
        models_to_merge_fisher_weights_list,
        fisher_scaling_coefficients,
        normalize_fisher_weight,
        minimal_fisher_weight
    )
    return merged_param_dict


#################################################################################
# Trainer that uses the actual split dataset (train_loader) from "get_dataset_and_classifier_for_split."
#################################################################################
class SplitTrainer:
    """
    This trainer holds a reference to the splitted dataset's train_loader,
    and prepares a dict with "images" and "labels" for each batch to feed 
    into the model(...) call during the Fisher calculation.
    """
    def __init__(self, splitted_dataset):
        self.train_loader = splitted_dataset.train_loader
        # Keep track of the batch size for Fisher computation
        self._train_batch_size = self.train_loader.batch_size

    def get_train_dataloader(self):
        return self.train_loader

    def _prepare_inputs(self, batch):
        """
        The batch from DataLoader is typically (images, labels).
        We must return a dictionary: {"images": <tensor>, "labels": <tensor>}
        so that model(**inputs) sees images=..., labels=...
        """
        images, labels = batch
        # Move to GPU if desired; adapt if you have multiple GPUs
        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        return {"images": images, "labels": labels}


#################################################################################
# Main script that:
# (1) Loads each finetuned model
# (2) Builds the splitted dataset (per-split) by re-using the logic from cl_utils
# (3) Wraps each model to produce .loss
# (4) Runs Fisher merging
# (5) Saves the final merged model
#################################################################################
def main():
    # Set up logging with debug mode if needed
    setup_logging(debug=False)  # Set to False for production use
    logger.info("Starting Fisher merging process...")
    # ---------------------------------------------------------------------------------
    # Here, define the relevant arguments to replicate the splitting and dataset loading.
    # Adjust these paths and arguments as needed for your setup.
    # ---------------------------------------------------------------------------------
    args = parse_arguments()
    
    args.dataset = "CIFAR100"
    args.split_strategy = "class"   # or "data"
    args.data_location = "~/data"   # where your CIFAR100 data is stored
    args.batch_size = 32
    args.num_workers = 4
    args.n_splits = 5
    args.epochs = 10
    args.seed = 5
    args.model = "ViT-B-16"  # example model name
    args.device = "cuda"  # or "cpu" if no GPU
        

    args.save = f'checkpoints/{args.model}/{args.split_strategy}_incremental'

    # ---------------------------------------------------------------------------------
    # Step 1: Create a "template" ImageEncoder so we can obtain the correct preprocess
    #         function. We'll use it for dataset loading, as get_dataset requires "preprocess".
    # ---------------------------------------------------------------------------------
    print("Building a temporary ImageEncoder to retrieve the preprocess transforms...")
    temp_encoder = ImageEncoder(args, keep_lang=True)  # loads an open_clip model
    preprocess_fn = temp_encoder.train_preprocess
    
    # ---------------------------------------------------------------------------------
    # Step 2: For each split index, load the splitted dataset the same way finetune_splitted.py does.
    #         We'll store each splitted dataset in a list, plus we’ll load each corresponding model.
    # ---------------------------------------------------------------------------------
    splitted_datasets = []
    models_to_merge = []
    for split_idx in range(args.n_splits):
        # Load the base dataset
        dataset = get_dataset(
            args.dataset,
            preprocess_fn,
            location=args.data_location,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
        # We'll produce the splitted dataset for the given split index
        # Passing the 'temp_encoder' as the text_encoder param. 
        # If your code has synergy with lang encoders, adapt as needed.
        splitted_dataset, _ = get_dataset_and_classifier_for_split(
            dataset, 
            split_idx,
            temp_encoder,  # just for building classification head if needed, or ignoring
            args,
            remap_labels=True,
            return_classifier=True
        )
        splitted_datasets.append(splitted_dataset)

        # Load the finetuned model for this split
        
        ckpt_path = os.path.join(args.save,
            f"{args.dataset}-{args.n_splits}",
            f"ft-epochs-{args.epochs}-seed:{args.seed}",
            f'finetuned_{split_idx}.pt'
            )

        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
        print(f"Loading model from {ckpt_path}")
        model = torch.load(ckpt_path)
        model.eval().cuda()
        
        # Wrap it so its forward(...) produces .loss
        wrapped_model = FisherOutputWrapper(model)
        models_to_merge.append(wrapped_model)

        # print the current GPU memory usage on device 0
        print(f"Current GPU memory usage on device 0: {torch.cuda.memory_summary(device=0, abbreviated=False)}")

    # ---------------------------------------------------------------------------------
    # Step 3: Build a trainer for each splitted dataset
    # ---------------------------------------------------------------------------------
    trainers = []
    for ds in splitted_datasets:
        trainer = SplitTrainer(ds)
        trainers.append(trainer)

    # ---------------------------------------------------------------------------------
    # Step 4: Fisher merging
    # By default, let's do 1024 examples per split for the Fisher approximation.
    # Adjust "nums_fisher_examples" as needed.
    # ---------------------------------------------------------------------------------
    nums_fisher_examples = [1024] * args.n_splits

    # You may exclude some parameter names from Fisher merging, e.g. BN layers:
    exclude_param_names_regex = [r'logit_scale', r'bn', r'norm']

    print("\nPerforming Fisher merging of all splits ...")
    merged_params_dict = fisher_merging(
        models_to_merge=models_to_merge,
        trainers=trainers,
        exclude_param_names_regex=exclude_param_names_regex,
        nums_fisher_examples=nums_fisher_examples,
        fisher_scaling_coefficients=None,  # or [1, 1, 1, 1, 1] if you want equal weighting
        normalize_fisher_weight=True,
        minimal_fisher_weight=1e-6
    )
    print("Fisher merging done.\n")

    # ---------------------------------------------------------------------------------
    # Step 5: Load these merged params into a fresh copy of one of the models
    #         Note: we pick the 0-th model checkpoint as a "template" for architecture.
    # ---------------------------------------------------------------------------------
    print("Loading final merged parameters into a fresh model template...")
    template_ckpt_path = os.path.join(args.save,
            f"{args.dataset}-{args.n_splits}",
            f"ft-epochs-{args.epochs}-seed:{args.seed}",
            f'finetuned_0.pt'
            )
    merged_model = torch.load(template_ckpt_path)  # Using the 0th as the template
    merged_model.eval().cuda()

    with torch.no_grad():
        for name, param in merged_model.named_parameters():
            if name in merged_params_dict:
                param.copy_(merged_params_dict[name].to(param.device))

    # ---------------------------------------------------------------------------------
    # Step 6: Save the merged model
    # ---------------------------------------------------------------------------------
    out_path = "fisher_merged.pt"
    print(f"Saving final merged model to {out_path}")
    torch.save(merged_model, out_path)
    print("Done!")
    logger.info("Fisher merging completed successfully!")


if __name__ == '__main__':
    main()