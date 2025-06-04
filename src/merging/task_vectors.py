import torch
import pdb
from src.eval import do_eval

class TaskVector():
    def __init__(self, pretrained_checkpoint=None, finetuned_checkpoint=None, vector=None, finetuned_state_dict=None):
        """Initializes the task vector from a pretrained and a finetuned checkpoints.
        
        This can either be done by passing two state dicts (one corresponding to the
        pretrained model, and another to the finetuned model), or by directly passying in
        the task vector state dict.
        """
        if vector is not None:
            self.vector = vector
        else:
            with torch.no_grad():
                assert pretrained_checkpoint
                pretrained_state_dict = torch.load(pretrained_checkpoint).state_dict()

                if finetuned_state_dict:
                    print(f"Creating task vector from finetuned_state_dict based on {pretrained_checkpoint=}")
                elif finetuned_checkpoint:
                    print(f"Creating task vector from {finetuned_checkpoint=} based on {pretrained_checkpoint=}")
                    finetuned_state_dict = torch.load(finetuned_checkpoint).state_dict()

                self.vector = {}
                # print(pretrained_state_dict.keys())
                # print(finetuned_state_dict.keys())
                for key in pretrained_state_dict:
                    # print(pretrained_state_dict[key].dtype)
                    if pretrained_state_dict[key].dtype in [torch.int64, torch.uint8]:
                        print(f"Key {key} has dtype {pretrained_state_dict[key].dtype} -- skipping!")
                        continue
                    self.vector[key] = finetuned_state_dict[key] - pretrained_state_dict[key]
    
    def __add__(self, other):
        """Add two task vectors together."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                if key not in other.vector:
                    print(f'Warning, key {key} is not present in both task vectors.')
                    continue
                new_vector[key] = self.vector[key] + other.vector[key]
        return TaskVector(vector=new_vector)

    def __radd__(self, other):
        if other is None or isinstance(other, int):
            return self
        return self.__add__(other)

    def __neg__(self):
        """Negate a task vector."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                new_vector[key] = - self.vector[key]
        return TaskVector(vector=new_vector)
    
    def __truediv__(self, other):
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                new_vector[key] = self.vector[key] / other
        return TaskVector(vector=new_vector)

    def __mul__(self, other):
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                new_vector[key] = self.vector[key] * other
        return TaskVector(vector=new_vector)

    def apply_to(self, pretrained_checkpoint, scaling_coef=1.0):
        """Apply a task vector to a pretrained model."""
        with torch.no_grad():
            pretrained_model = torch.load(pretrained_checkpoint)
            new_state_dict = {}
            pretrained_state_dict = pretrained_model.state_dict()
            for key in pretrained_state_dict:
                if key not in self.vector:
                    print(f'Warning: key {key} is present in the pretrained state dict but not in the task vector')
                    continue
                new_state_dict[key] = pretrained_state_dict[key] + scaling_coef * self.vector[key]
        pretrained_model.load_state_dict(new_state_dict, strict=False)
        return pretrained_model


def merge_rnd_mix(task_vectors):
    """Randomly mix multiple task vectors together."""
    if len(task_vectors) == 0:
        return task_vectors[0]
    
    with torch.no_grad():
        new_vector = {}
        for key in task_vectors[0].vector:
            _rand_indices = torch.randint(0, len(task_vectors), task_vectors[0].vector[key].shape)
            new_vector[key] = task_vectors[0].vector[key] * (_rand_indices == 0)
            for i in range(1, len(task_vectors)):
                new_vector[key] += task_vectors[i].vector[key] * (_rand_indices == i)
    
    return TaskVector(vector=new_vector)


def merge_max_abs(task_vectors):
    """Mix multiple task vectors together by highest parameter value."""
    if len(task_vectors) == 0:
        return task_vectors[0]
        
    with torch.no_grad():
        new_vector = {}
        
        # Iterate over keys in the first task vector
        for key in task_vectors[0].vector:
            # Get the initial tensor for the current key
            max_abs_tensor = task_vectors[0].vector[key]
            
            # Iterate over the remaining task vectors
            for task_vector in task_vectors[1:]:
                current_tensor = task_vector.vector[key]
                
                # Update max_abs_tensor to keep the element-wise maximum absolute values
                max_abs_tensor = torch.where(current_tensor.abs() >= max_abs_tensor.abs(), current_tensor, max_abs_tensor)
            
            # Assign the final tensor to the new_vector dictionary
            new_vector[key] = max_abs_tensor

    return TaskVector(vector=new_vector)


def merge_learned(task_vectors, split_idx,pretrained_checkpoint, args, 
                  validation_splits=None, num_epochs=10, lr=1e-3, 
                  init_strategy='proportional', regularization=0.0, validation_proportion=0.1):
    """
    Merge task vectors by learning coefficients through gradient descent on validation data.
    
    Args:
        task_vectors: List of TaskVector objects to merge
        pretrained_checkpoint: Path to the pretrained model checkpoint
        args: Runtime arguments with dataset information
        validation_splits: List of validation split indices to use (if None, use all splits)
        num_epochs: Number of training epochs for proportion=0.1 (will be adjusted based on actual proportion)
        lr: Learning rate for coefficient optimization
        init_strategy: Strategy for initializing coefficients ('uniform', 'random', or 'proportional')
        regularization: L1 regularization strength to encourage sparse coefficients
        validation_proportion: Proportion of training data to use for validation (between 0 and 1)
        
    Returns:
        Tuple of TaskVector objects: The merged task vectors with learned coefficients
    """
    import torch
    import torch.nn.functional as F
    from torch.func import functional_call
    from src.heads import get_classification_head
    from src.datasets.common import get_dataloader, maybe_dictionarize
    from src.datasets.registry import get_dataset
    from src.cl_utils import get_dataset_and_classifier_for_split
    from src.modeling import ImageClassifier
    from copy import deepcopy
    from torch.utils.data import ConcatDataset, Subset, DataLoader

    if len(task_vectors) == 1:
        return task_vectors[0], task_vectors[0]
    
    # 调整epoch数量，基于validation_proportion和数据集
    base_proportion = 0.1  # 基础比例，与默认的num_epochs匹配
    
    # 在赋予validset比例时, 已经给考虑IN-R乘以4倍来得到 same number sample per class, epoch就不变了.
    # # ImageNetR数据集需要更多训练轮数
    # if args.dataset == 'ImageNetR':
    #     print(f"数据集是ImageNetR，将训练轮次乘以4倍: {num_epochs} -> {num_epochs * 4}")
    #     num_epochs *= 4
    
    if validation_proportion < base_proportion:
        # 如果比例减少，增加epoch数量；反之亦然
        adjusted_epochs = int(num_epochs * (base_proportion / validation_proportion))
        print(f"基于验证比例 {validation_proportion} (基础比例 {base_proportion})，将训练轮次从 {num_epochs} 调整为 {adjusted_epochs}")
        num_epochs = adjusted_epochs
    
    # =========== 1. 初始化参数和模型 ===========
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载预训练模型
    pretrained_model = torch.load(pretrained_checkpoint)
    
    # 将任务向量移至指定设备并确保requires_grad=False
    for tv in task_vectors:
        for key in tv.vector:
            tv.vector[key] = tv.vector[key].to(device).detach()
    
    # 基于选定策略初始化系数
    n_vectors = len(task_vectors)
    if init_strategy == 'proportional':
        # prev_vector占split_idx/split_idx+1, 当前vector占1/split_idx+1
        coeffs = torch.ones(n_vectors, device=device)
        coeffs[0] = split_idx / (split_idx + 1)
        coeffs[1] = 1 / (split_idx + 1)

    elif init_strategy == 'uniform':
        coeffs = torch.ones(n_vectors, device=device) / n_vectors
    elif init_strategy == 'random':
        coeffs = torch.rand(n_vectors, device=device)
    # elif init_strategy == 'proportional':
    #     # 按比例初始化（这里简单地使用递增权重）
    #     coeffs = torch.ones(n_vectors, device=device)
    #     for i in range(n_vectors):
    #         coeffs[i] = (i + 1)  # 简单的递增权重示例
    else:
        raise ValueError(f"Unknown initialization strategy: {init_strategy}")
    
    # 将系数转换为需要梯度的参数
    coeffs = torch.nn.Parameter(coeffs, requires_grad=True)
    
    # 准备优化器
    optimizer = torch.optim.Adam([coeffs], lr=lr)
    
    # 使用交叉熵损失
    criterion = torch.nn.CrossEntropyLoss()
    
    # =========== 2. 准备验证数据 ===========
    if validation_splits is None:
        validation_splits = list(range(n_vectors))
    
    # 创建合并的验证数据集
    print("准备验证数据集...")
    preprocess_fn = pretrained_model.train_preprocess
    
    # 获取数据集的分类头部
    classification_head = get_classification_head(args, args.dataset)
    classification_head = classification_head.to(device)
    
    # 准备所有验证拆分的组合数据集
    if len(validation_splits) > 0:
        full_dataset = get_dataset(
            args.dataset,
            preprocess_fn,
            location=args.data_location,
            batch_size=args.batch_size
        )
        
        combined_datasets = []
        for split_idx in validation_splits:
            print(f"添加拆分 {split_idx} 到验证集")
            dataset = deepcopy(full_dataset)
            split_dataset = get_dataset_and_classifier_for_split(
                dataset, split_idx, pretrained_model, args, remap_labels=False, return_classifier=False
            )
            
            # 根据validation_proportion参数只使用训练集的一部分作为验证
            train_dataset = split_dataset.train_dataset # 使用测试集作为验证集, 只是为了验证训练和eval的代码正确性
            dataset_size = len(train_dataset)
            val_size = int(dataset_size * validation_proportion)
            
            if val_size > 0:
                # 创建验证子集
                torch.manual_seed(args.seed_for_validset) # 固定随机种子
                indices = torch.randperm(dataset_size)[:val_size].tolist()
                val_subset = Subset(train_dataset, indices)
                combined_datasets.append(val_subset)
                print(f"  拆分 {split_idx}: 总样本 {dataset_size}, 使用 {val_size} 样本进行验证 ({validation_proportion:.1%})")
        
        if len(combined_datasets) > 0:
            # 合并所有验证子集
            combined_val_dataset = ConcatDataset(combined_datasets)
            
            # Use a default value for num_workers if not in args
            num_workers = getattr(args, 'num_workers', 4)  # Default to 4 if not specified
            
            val_loader = DataLoader(
                combined_val_dataset, 
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=num_workers
            )
            print(f"合并验证集大小: {len(combined_val_dataset)} 样本")
        else:
            raise ValueError("未能创建任何验证子集，请检查validation_proportion是否过小")
    else:
        raise ValueError("至少需要一个验证集拆分来进行系数优化")
    
    # =========== 3. 为ImageClassifier创建正确的参数字典 ===========
    def get_merged_params_for_classifier(base_model, task_vectors, coeffs):
        """计算合并后的参数字典，为ImageClassifier格式化正确的键"""
        merged_params = {}
        
        # 原始状态字典中的键映射到ImageClassifier中的位置
        for key, base_param in base_model.state_dict().items():
            # 将基础模型的参数复制到merged_params中，保持设备一致
            base_param_tensor = base_param.clone().detach().to(device)
            
            # 映射键：确保键的路径格式正确
            # 例如：将"model.xxx"转换为"image_encoder.model.xxx"
            target_key = key
            if not key.startswith("classification_head"):
                # 如果不是分类头部分的键，则加上image_encoder前缀
                target_key = f"image_encoder.{key}"
            
            merged_params[target_key] = base_param_tensor
            
            # 对task vector中的参数应用相同的映射
            for i, task_vector in enumerate(task_vectors):
                # 检查原始键是否在任务向量中
                if key in task_vector.vector:
                    tv_param = task_vector.vector[key].to(device)
                    # 累加加权任务向量
                    merged_params[target_key] = merged_params[target_key] + coeffs[i] * tv_param
                    
        return merged_params
    
    # =========== 4. 系数优化训练循环 ===========
    best_accuracy = 0.0
    best_coeffs = coeffs.clone().detach()
    
    print(f"开始系数优化，共{num_epochs}轮...")
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, batch in enumerate(val_loader):
            # 每个批次开始时清零梯度
            optimizer.zero_grad()
            
            # 创建模型副本进行本次迭代
            model = deepcopy(pretrained_model)
            model = model.to(device)
            classifier = ImageClassifier(model, classification_head)
            classifier.to(device)
            
            # 获取调整后的参数字典（格式化为ImageClassifier的结构）
            merged_params = get_merged_params_for_classifier(model, task_vectors, coeffs)
            
            # 前向传播
            batch = maybe_dictionarize(batch)
            inputs, targets = batch['images'].to(device), batch['labels'].to(device)
            
            # 确保分类头部的参数也包含在内
            for name, param in classification_head.named_parameters():
                merged_params[f"classification_head.{name}"] = param.to(device)
            
            # 使用functional_call进行前向计算
            outputs = functional_call(classifier, merged_params, (inputs,))
            
            # 计算损失
            loss = criterion(outputs, targets)
            
            # 添加L1正则化（可选）
            if regularization > 0:
                reg_loss = regularization * torch.norm(coeffs, 1)
                loss = loss + reg_loss
            
            loss.backward()
            optimizer.step()
            
            # 计算准确率统计
            _, predicted = outputs.max(1)
            batch_correct = predicted.eq(targets).sum().item()
            batch_size = targets.size(0)
            
            correct += batch_correct
            total += batch_size
            running_loss += loss.item()
            
            # 打印进度
            # if (batch_idx + 1) % 5 == 0:
            # print(f"轮次 {epoch+1}/{num_epochs}, 批次 {batch_idx+1}/{len(val_loader)}, "
            #         f"批次准确率: {100.0 * batch_correct / batch_size:.2f}%, 损失: {loss.item():.4f}, "
            #         f"系数: {coeffs.tolist()}")
        
        # 计算整个验证集上的准确率
        epoch_accuracy = 100.0 * correct / total
        
        if (epoch + 1) % 10 == 0:
            print(f"轮次 {epoch+1}/{num_epochs} 完成, "
                  f"验证准确率: {epoch_accuracy:.2f}%, "
                  f"平均损失: {running_loss/len(val_loader):.4f}, "
                  f"系数: {coeffs.tolist()}")
        
        # 更新最佳模型
        if epoch_accuracy > best_accuracy:
            best_accuracy = epoch_accuracy
            best_coeffs = coeffs.clone().detach()
    
    # =========== 5. 生成最终合并向量 ===========
    print(f"优化完成。最佳准确率: {best_accuracy:.2f}%")
    print(f"最佳系数: {best_coeffs.tolist()}")
    print(f"最终系数: {coeffs.tolist()}")
    
    # 创建两个最终向量：一个使用最佳系数，一个使用最后的系数
    best_vector = {}
    last_vector = {}
    
    for key in task_vectors[0].vector:
        if any(key not in tv.vector for tv in task_vectors):
            continue
        
        # 创建CPU上的零张量
        best_vector[key] = torch.zeros_like(task_vectors[0].vector[key].cpu())
        last_vector[key] = torch.zeros_like(task_vectors[0].vector[key].cpu())
        
        for i, task_vector in enumerate(task_vectors):
            if key in task_vector.vector:
                # 将贡献移至CPU并添加到相应向量
                best_contribution = (best_coeffs[i].cpu() * task_vector.vector[key].cpu())
                last_contribution = (coeffs[i].cpu() * task_vector.vector[key].cpu())
                
                best_vector[key] += best_contribution
                last_vector[key] += last_contribution
    
    return TaskVector(vector=best_vector), TaskVector(vector=last_vector), best_coeffs, coeffs


