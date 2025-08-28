import numpy as np
from utilities.logger import logger

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
[0.8779, 	0.8222, 	0.7663, 	0.8221],
[0.8802, 	0.8406, 	0.6912, 	0.8040],
[0.8827, 	0.8060, 	0.7222, 	0.8037],
[0.8655, 	0.8275, 	0.7595, 	0.8175],
[0.8552, 	0.8265, 	0.7156, 	0.7991],














]




metrics = compute_metrics(fold_results)
for name, values in metrics.items():
    logger.info(f"{name}: {values['formatted']}")


