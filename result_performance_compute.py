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
[0.9466, 0.9395, 0.9019, 0.9293],
[0.9467, 0.9409, 0.8984, 0.9287],
[0.9443, 0.9491, 0.9034, 0.9323],
[0.9403, 0.9381, 0.9023, 0.9269],
[0.9401, 0.9422, 0.9031, 0.9285],


]







metrics = compute_metrics(fold_results)
for name, values in metrics.items():
    print(f"{name}: {values['formatted']}")


