acc_lists = [
    # 从评估结果中提取的准确率数据
    [[0.8716, 0.773, 0.7839, 0.7909, 0.7724], [0.8786, 0.8212, 0.7888, 0.7892, 0.7759], [0.8793, 0.8335, 0.834, 0.7849, 0.7534], [0.8678, 0.8405, 0.8431, 0.8298, 0.7543], [0.8701, 0.8361, 0.8447, 0.8281, 0.7966]]
]

import numpy as np

def calculate_cl_metrics(acc_list):
    """
    计算连续学习的三个关键度量指标
    参数:
        acc_list: 二维列表，包含每个split在所有已见任务上的精度
    """
    num_splits = len(acc_list)
    
    # 1. Last-acc: 最后一个任务训练结束后，在所有任务上的acc的平均值
    last_acc = float(np.mean(acc_list[-1]))
    print(f"Last-acc: {last_acc:.4f}")
    
    # 2. Forgetting: 对每个任务，找到其历史最高acc，减去最后一个任务训练完时其acc
    forgetting_values = []
    final_accs = acc_list[-1]
    
    # 计算每个任务的遗忘值
    for task_idx in range(len(final_accs)):
        # 找出每个任务的历史最高精度
        max_acc = -float('inf')
        for split_idx in range(num_splits):
            # 只考虑当前split包含此任务的情况
            if task_idx < len(acc_list[split_idx]):
                max_acc = max(max_acc, acc_list[split_idx][task_idx])
        
        # 计算遗忘值
        forgetting = max_acc - final_accs[task_idx]
        forgetting_values.append(forgetting)
    
    avg_forgetting = float(np.mean(forgetting_values))
    print(f"Forgetting: {avg_forgetting:.4f}")
    
    # 3. Avg-acc: 每个任务训练完之后，把seen tasks的acc算出平均值
    avg_acc_per_step = []
    for split_idx in range(num_splits):
        # 每个split只考虑已见任务
        seen_tasks_acc = acc_list[split_idx][:split_idx+1]
        avg_acc_per_step.append(np.mean(seen_tasks_acc))
    
    avg_acc = float(np.mean(avg_acc_per_step))
    print(f"Avg-acc: {avg_acc:.4f}")
    
    return {
        "last_acc": round(last_acc, 4),
        "forgetting": round(avg_forgetting, 4),
        "avg_acc": round(avg_acc, 4)
    }

# Calculate and display the metrics
for i, acc_list in enumerate(acc_lists):
    print(f"\nMetrics for acc_list {i+1}:")
    metrics = calculate_cl_metrics(acc_list)
    print(metrics)

