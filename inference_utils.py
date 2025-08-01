import os
import torch
import nibabel as nib
import numpy as np
import torch.nn.functional as F
from config import config as cfg
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, NormalizeIntensityd,
    Orientationd, Spacingd, ToTensord
    )

from scipy.ndimage import binary_dilation, binary_opening, label, generate_binary_structure



def preprocess_for_inference(image_paths, center_crop=False):
    """
    image_paths: list of 4 modality paths [t1, t1ce, t2, flair]
    
    Returns:
        x_seq: torch.Tensor, shape (B=1, C, D, H, W)
    """
    data_dict = {"image": image_paths}
    
    # Step 1: Load + Channel First
    load_transform = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
    ])
    data = load_transform(data_dict)
    data["image"] = data["image"].permute(0, 3, 1, 2).contiguous()
    print("Loaded image shape:", data["image"].shape)  # (C, D, H, W)
    
    img_meta = data["image"].meta
    img_spacing = img_meta.get("pixdim", None)

    # Step 2: Spatial Normalization (Orientation + Spacing)
    need_orientation_or_spacing = False
    if img_meta.get("spatial_shape") is None:
        need_orientation_or_spacing = True
    else:
        if not torch.allclose(torch.tensor(img_spacing[:3]), torch.tensor([1.0, 1.0, 1.0])):
            need_orientation_or_spacing = True
        if not (img_meta.get("original_channel_dim", None) == 0 and img_meta.get("original_affine", None) is not None):
            need_orientation_or_spacing = True
    
    if need_orientation_or_spacing:
        print("DO PREPROCESS!!!")
        preprocess = Compose([
            Orientationd(keys=["image"], axcodes="RAS"),
            Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
        ])
        data = preprocess(data)
    
    # Step 3: Center Crop
    def _center_crop_fn(img: torch.Tensor, crop_size=(144,144,144)):
        _, D, H, W = img.shape
        cd, ch, cw = crop_size
        sd = (D - cd) // 2
        sh = (H - ch) // 2
        sw = (W - cw) // 2
        cropped = img[:, sd:sd+cd, sh:sh+ch, sw:sw+cw]
        crop_start = (sd, sh, sw)
        return cropped, (D, H, W), crop_start
    
    if center_crop:
        print("Applying center crop...")
        data["image"], original_shape, crop_start = _center_crop_fn(data["image"])
    else:
        print("No center crop applied.")
        original_shape = data["image"].shape[1:]  # (D, H, W)
        crop_start = (0, 0, 0)
    
    
    # Step 4: Intensity Normalization
    normalize = NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True)
    data = normalize(data)
    
    # Step 5: ToTensor
    to_tensor = ToTensord(keys=["image"])
    data = to_tensor(data)
    
    # Step 6: Add batch dimension
    img = data["image"]  # shape: (C, D, H, W)
    img = img.unsqueeze(0) # (B=1, C, D, H, W)
    
    print("Preprocessed image shape:", img.shape)  # (B=1, C, D, H, W)
    
    return img, {
        "original_shape": original_shape,  # (D, H, W)
        "crop_start": crop_start           # (sd, sh, sw)
    }
    
    
    
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
    tc_mask = (tc_prob >= threshold)
    wt_mask = (wt_prob >= threshold)
    et_mask = (et_prob >= threshold)

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


def check_test_txt_exist(test_cases_txt):
    """
    检查 test_cases_txt 中是否存在 test_cases.txt。
    若缺少，则报错退出。
    """
    missing_txts = False

    txt_path = os.path.join(test_cases_txt, f"test_cases.txt")
    if not os.path.isfile(txt_path):
        missing_txts = True

    if missing_txts:
        raise FileNotFoundError(f"[Warning] Missing test_cases.txt file in {test_cases_txt}")
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