import numpy as np

def compute_metrics(fold_results):
    """
    计算每个指标（WT, TC, ET, Mean）的 mean ± std。
    
    参数:
        fold_results: list of list，每个子列表格式为 [WT, TC, ET, Mean]，共5个fold

    返回:
        dict: 包含 WT, TC, ET, Mean 各自的 mean 和 std
    """
    fold_array = np.array(fold_results)  # 转换为 (5, 4) 的数组
    means = np.mean(fold_array, axis=0)
    stds = np.std(fold_array, axis=0)

    metric_names = ['WT', 'TC', 'ET', 'Mean']
    result = {}
    for i, name in enumerate(metric_names):
        mean_percent = means[i] * 100
        std_percent = stds[i] * 100
        result[name] = {
            'mean': mean_percent,
            'std': std_percent,
            'formatted': f"{mean_percent:.2f}±{std_percent:.2f}"
        }
    return result


# 每行为一个fold的WT, TC, ET, Mean
fold_results = [
[0.9257, 0.8861, 0.8465, 0.8861],
[0.9280, 0.9004, 0.8573, 0.8952],
[0.9193, 0.8814, 0.8655, 0.8888],
[0.9105, 0.9034, 0.8510, 0.8883],
[0.9270, 0.9135, 0.8554, 0.8986],

]


metrics = compute_metrics(fold_results)
for name, values in metrics.items():
    print(f"{name}: {values['formatted']}")


