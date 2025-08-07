import os
os.chdir(os.path.dirname(__file__))
import torch
import nibabel as nib
import numpy as np
from scipy.ndimage import binary_dilation, binary_opening, label, generate_binary_structure


def postprocess_brats_label(pred_mask: np.ndarray) -> np.ndarray:
    """
    BraTS预测标签后处理：
    - ET (4): 向外扩张一圈，只吸收外部的NCR和ED，不吞噬ET内部的NCR
    - NCR/NET (1): 外部NCR做开运算，ET内部NCR保持原样
    - ED (2): 保持原状
    """

    structure = generate_binary_structure(3, 1)

    # 原始标签
    et_mask = (pred_mask == 4)
    ncr_mask = (pred_mask == 1)
    edema_mask = (pred_mask == 2)

    print("Before Postprocessing:")
    print("Sum ET:", np.sum(et_mask))
    print("Sum ED:", np.sum(edema_mask))
    print("Sum NCR:", np.sum(ncr_mask))

    # Step 1: 分离ET内部的NCR（要保护的）与ET外部的NCR（可处理的）
    ncr_inside_et = ncr_mask & et_mask
    ncr_outside_et = ncr_mask & (~et_mask)

    # Step 2: 对外部NCR做开运算
    ncr_outside_processed = binary_opening(ncr_outside_et, structure=structure, iterations=1)
    ncr_processed = ncr_outside_processed | ncr_inside_et

    # 被剥掉的外部NCR边缘
    ncr_removed = ncr_outside_et & (~ncr_outside_processed)

    # Step 3: 构造ET的“外壳”：从ET外面包一圈，不含ET原始区域
    et_outer_shell = binary_dilation(et_mask, structure=structure, iterations=1) & (~et_mask)

    # Step 4: 只允许ET扩张到其“外壳”中满足条件的区域（NCR外部边缘 or ED）
    et_expand_target = et_outer_shell & (ncr_removed | edema_mask)

    # Step 5: 最终ET = 原始ET + 允许扩张区域（外壳目标）
    et_final = et_mask | et_expand_target

    # Step 6: 构建最终mask
    new_mask = np.zeros_like(pred_mask, dtype=np.uint8)
    new_mask[et_final] = 4
    new_mask[(edema_mask) & (new_mask == 0)] = 2
    new_mask[(ncr_processed) & (new_mask == 0)] = 1

    # Step 7: 剩余被删掉的NCR边缘如果未被覆盖，强制转ET（避免留背景）
    ncr_remaining = ncr_removed & (new_mask == 0)
    new_mask[ncr_remaining] = 4

    print("Postprocessing results:")
    print("Sum ET:", np.sum(new_mask == 4))
    print("Sum ED:", np.sum(new_mask == 2))
    print("Sum NCR:", np.sum(new_mask == 1))

    return new_mask


def postprocess_brats_label_nnstyle(pred_mask: np.ndarray) -> np.ndarray:
    """
    nnU-Net风格的BraTS预测标签后处理：
    对每一类标签（ET=4, NCR=1, ED=2）分别：
    - 保留最大连通区域
    - 其余区域设为背景（label=0）
    """

    def keep_largest_connected_component(mask: np.ndarray) -> np.ndarray:
        """
        仅保留mask中最大的连通区域，其余设为False
        """
        structure = generate_binary_structure(3, 1)
        labeled, num_features = label(mask, structure)
        if num_features == 0:
            return np.zeros_like(mask, dtype=bool)
        largest_component = np.argmax(np.bincount(labeled.flat)[1:]) + 1  # +1 因为0是背景
        return labeled == largest_component

    # 输出初始化为背景
    print("Postprocessing BraTS label...")
    new_mask = np.zeros_like(pred_mask, dtype=np.uint8)

    for label_id in [4, 2, 1]:  # ET, ED, NCR 按优先级顺序处理
        binary_mask = (pred_mask == label_id)
        largest_component_mask = keep_largest_connected_component(binary_mask)
        new_mask[largest_component_mask] = label_id

    print("Postprocessing done.")

    return new_mask

