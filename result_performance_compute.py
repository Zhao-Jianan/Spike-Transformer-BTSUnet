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
    [0.8290, 0.7997, 0.7233, 0.7840],
    [0.8182, 0.8011, 0.7115, 0.7769],
    [0.8205, 0.7819, 0.7413, 0.7812],
    [0.7705, 0.8198, 0.7783, 0.7895],
    [0.8105, 0.8075, 0.7692, 0.7957],


]



metrics = compute_metrics(fold_results)
for name, values in metrics.items():
    print(f"{name}: {values['formatted']}")


