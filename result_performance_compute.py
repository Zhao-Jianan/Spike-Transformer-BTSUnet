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
    [0.8713, 0.8129, 0.7272, 0.8038],
    [0.8848, 0.8354, 0.6802, 0.8001],
    [0.8489, 0.7774, 0.6718, 0.7661],
    [0.8615, 0.8002, 0.7079, 0.7899],
    [0.8560, 0.8350, 0.6727, 0.7879],



]


metrics = compute_metrics(fold_results)
for name, values in metrics.items():
    print(f"{name}: {values['formatted']}")


