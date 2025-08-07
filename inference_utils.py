import os
os.chdir(os.path.dirname(__file__))
import torch
import nibabel as nib
import numpy as np

        
def convert_prediction_to_label_backup(mean_prob: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    BraTS标签转换，输入 mean_prob 顺序：TC, WT, ET
    返回标签：0=BG, 1=TC(NCR/NET), 2=ED, 4=ET
    """
    assert mean_prob.shape[0] == 3, "Expected 3 channels: TC, WT, ET"
    tc_prob, wt_prob, et_prob = mean_prob[0], mean_prob[1], mean_prob[2]

    # 阈值化各通道
    et = (et_prob > threshold).astype(np.uint8)
    tc = (tc_prob > threshold).astype(np.uint8)
    wt = (wt_prob > threshold).astype(np.uint8)

    label = np.zeros_like(tc, dtype=np.uint8)

    label[wt == 1] = 2
    label[tc == 1] = 1
    label[et == 1] = 4  # ET优先级最高

    return label


def convert_prediction_to_label(mean_prob: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    BraTS标签转换，输入 mean_prob 顺序：TC, WT, ET
    返回标签：0=BG, 1=TC(NCR/NET), 2=ED, 4=ET
    """
    assert mean_prob.shape[0] == 3, "Expected 3 channels: TC, WT, ET"
    tc_prob, wt_prob, et_prob = mean_prob[0], mean_prob[1], mean_prob[2]

    # 阈值化
    et = (et_prob > threshold)
    tc = (tc_prob > threshold) & (~et)  # 排除ET区域
    wt = (wt_prob > threshold) & (~et) & (~tc)  # 排除ET和TC区域

    label = np.zeros_like(tc_prob, dtype=np.uint8)
    label[wt] = 2
    label[tc] = 1
    label[et] = 4

    return label


def convert_prediction_to_label_suppress_fp(mean_prob: np.ndarray, threshold: float = 0.5, bg_margin: float = 0.1) -> np.ndarray:
    """
    BraTS 标签转换，加入背景保护机制。
    输入 mean_prob 顺序：TC, WT, ET
    返回标签图：每个 voxel 值为 {0, 1, 2, 4}
    """
    assert mean_prob.shape[0] == 3, "Expected 3 channels: TC, WT, ET"
    tc_prob, wt_prob, et_prob = mean_prob[0], mean_prob[1], mean_prob[2]

    # 阈值生成掩码   
    et_mask = (et_prob >= threshold)
    tc_mask = (tc_prob >= threshold) & (~et_mask)
    wt_mask = (wt_prob >= threshold) & (~et_mask) & (~tc_mask)

    # 背景保护：如果所有类别的最大值都很小，就强制为背景
    overall_max_prob = np.max(mean_prob, axis=0)
    suppress_mask = overall_max_prob < (threshold + bg_margin)
    
    # 独立三通道标签图
    label = np.zeros_like(tc_prob, dtype=np.uint8)

    # 按照 ET > TC > WT 的优先级赋值（互斥标签）
    label[wt_mask] = 2         # 先赋 WT
    label[tc_mask] = 1         # TC 会覆盖 WT 的值为 1
    label[et_mask] = 4         # ET 会覆盖 TC 的值为 4

    label[suppress_mask] = 0   # 背景保护

    return label



def check_all_folds_ckpt_exist(ckpt_dir):
    """
    检查 fold1~fold5 的 checkpoint 是否都存在。
    若缺少任意一个，则报错退出。
    """
    missing_folds = []
    for fold in range(1, 6):
        ckpt_path = os.path.join(ckpt_dir, f"best_model_fold{fold}.pth")
        if not os.path.isfile(ckpt_path):
            missing_folds.append(fold)

    if missing_folds:
        raise FileNotFoundError(f"[Warning] Missing checkpoint(s) for fold(s): {missing_folds} in {ckpt_dir}")
    else:
        print("All 5 fold checkpoints found.")

def check_test_txt_exist(txt_path):
    """
    参数应是 test_cases.txt 的完整路径。
    """
    if not os.path.isfile(txt_path):
        raise FileNotFoundError(f"[Warning] Missing test_cases.txt file in {txt_path}")
    else:
        print("test_cases.txt file found.")
        
        
        
def check_all_folds_val_txt_exist(val_cases_dir):
    """
    检查 val_cases_dir 中是否存在 val_cases_fold1.txt 到 val_cases_fold5.txt。
    若缺少任意一个，则报错退出。
    """
    missing_txts = []
    for fold in range(1, 6):
        txt_path = os.path.join(val_cases_dir, f"val_cases_fold{fold}.txt")
        if not os.path.isfile(txt_path):
            missing_txts.append(fold)

    if missing_txts:
        raise FileNotFoundError(f"[Warning] Missing val_cases_fold txt file(s) for fold(s): {missing_txts} in {val_cases_dir}")
    else:
        print("All 5 fold val_cases txt files found.")
        

        
def restore_to_original_shape(cropped_label, original_shape, crop_start):
    """
    将中心裁剪过的预测结果还原回原图大小。
    """
    restored = np.zeros(original_shape, dtype=cropped_label.dtype)
    z, y, x = crop_start
    dz, dy, dx = cropped_label.shape
    restored[z:z+dz, y:y+dy, x:x+dx] = cropped_label
    return restored   


def read_case_list(txt_path):
    with open(txt_path, "r") as f:
        return sorted([line.strip() for line in f.readlines() if line.strip()]) 
    
    
def convert_label_to_onehot(label):
    """
    label: Tensor [D, H, W], value in {0, 1, 2, 4}
    return: one-hot [3, D, H, W], corresponding to TC, WT, ET
    """
    tc = ((label == 1) | (label == 4)).float()  # TC = label 1 or 4
    wt = ((label == 1) | (label == 2) | (label == 4)).float()  # WT = label 1 or 2 or 4
    et = (label == 4).float()  # ET = label 4

    return torch.stack([tc, wt, et], dim=0)  # [3, D, H, W]