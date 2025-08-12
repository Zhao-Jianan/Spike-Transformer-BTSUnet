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
[0.8531, 0.8121, 0.7171, 0.7941],
[0.8857, 0.8344, 0.6839, 0.8013],
[0.8724, 0.7988, 0.7174, 0.7962],
[0.8689, 0.8395, 0.7492, 0.8192],
[0.8526, 0.8380, 0.6998, 0.7968],

]




metrics = compute_metrics(fold_results)
for name, values in metrics.items():
    print(f"{name}: {values['formatted']}")


