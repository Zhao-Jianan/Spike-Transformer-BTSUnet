import os
os.chdir(os.path.dirname(__file__))
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import torch
import nibabel as nib
import numpy as np
from einops import rearrange
from spike_former_unet_model import spike_former_unet3D_8_384
import torch.nn.functional as F
from config import config as cfg
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, NormalizeIntensityd,
    Orientationd, Spacingd, ToTensord
    )
from copy import deepcopy
import time
from scipy.ndimage import binary_dilation, binary_opening, label, generate_binary_structure
from inference_helper import TemporalSlidingWindowInference
from tqdm import tqdm


def preprocess_for_inference(image_paths):
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
    data["image"] = rearrange(data["image"], 'c h w d -> c d h w')
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
    
    # Step 3: Intensity Normalization
    normalize = NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True)
    data = normalize(data)
    
    # Step 4: ToTensor
    to_tensor = ToTensord(keys=["image"])
    data = to_tensor(data)
    
    # Step 5: Add batch dimension
    img = data["image"]  # shape: (C, D, H, W)
    img = img.unsqueeze(0) # (B=1, C, D, H, W)
    
    return img



def convert_prediction_to_label(mean_prob: np.ndarray, threshold: float = 0.5) -> np.ndarray:
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



def pred_single_case_soft(case_dir, prob_save_dir, model, inference_engine, device):
    case_name = os.path.basename(case_dir)
    print(f"Processing case: {case_name}")
    image_paths = [os.path.join(case_dir, f"{mod}.nii.gz") for mod in cfg.modalities]

    x_batch = preprocess_for_inference(image_paths).to(device)

    with torch.no_grad():
        output = inference_engine(x_batch, model)

    output_prob = torch.sigmoid(output).squeeze(0).cpu().numpy()  # [C, D, H, W]
    prob_path = os.path.join(prob_save_dir, f"{case_name}_prob.npy")
    np.save(prob_path, output_prob)
    print(f"Saved probability map: {prob_path}")



def run_inference_soft(case_dir, save_dir, model, inference_engine, device):
    os.makedirs(save_dir, exist_ok=True)
    pred_single_case_soft(case_dir, save_dir, model, inference_engine, device)

def run_inference_folder_soft(case_root, save_dir, model, inference_engine, device):
    os.makedirs(save_dir, exist_ok=True)
    case_dirs = sorted([
        os.path.join(case_root, name) for name in os.listdir(case_root)
        if os.path.isdir(os.path.join(case_root, name))
    ])
    print(f"Found {len(case_dirs)} cases to infer.")

    for case_dir in tqdm(case_dirs, desc="Soft Voting Inference"):
        run_inference_soft(case_dir, save_dir, model, inference_engine, device)


def soft_ensemble(prob_base_dir, case_dir, ckpt_dir):
    case_dir = case_dir
    prob_base_dir = prob_base_dir

    for fold in range(1, 6):
        print(f"Running inference for fold {fold}")
        model_ckpt = os.path.join(ckpt_dir, f"best_model_fold{fold}.pth")
        model = spike_former_unet3D_8_384(
            num_classes=cfg.num_classes,
            T=cfg.T,
            norm_type=cfg.norm_type,
            step_mode=cfg.step_mode).to(cfg.device)
        model.load_state_dict(torch.load(model_ckpt, map_location=cfg.device))
        model.eval()

        inference_engine = TemporalSlidingWindowInference(
            patch_size=cfg.inference_patch_size,
            overlap=cfg.overlap,
            sw_batch_size=16,
            mode="constant", # "gaussian", "constant"
            encode_method=cfg.encode_method,
            T=cfg.T,
            num_classes=cfg.num_classes
        )

        fold_prob_dir = os.path.join(prob_base_dir, f"fold{fold}")
        run_inference_folder_soft(case_dir, fold_prob_dir, model, inference_engine, cfg.device)


def ensemble_soft_voting(prob_root, case_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    case_names = sorted(list(set([f.split('_prob.npy')[0] for f in os.listdir(os.path.join(prob_root, 'fold1'))])))

    for case in tqdm(case_names, desc="Soft Voting Ensemble"):
        prob_list = []
        for fold in range(1, 6):
            prob_path = os.path.join(prob_root, f"fold{fold}", f"{case}_prob.npy")
            prob = np.load(prob_path)
            prob_list.append(prob)

        mean_prob = np.mean(np.stack(prob_list, axis=0), axis=0)  # [C, D, H, W]
        print(f"Mean probability shape for case {case}: {mean_prob.shape}")

        label_np = convert_prediction_to_label(mean_prob)

        print("Label shape before transposing:", label_np.shape)  # (D, H, W)
        label_np = np.transpose(label_np, (1, 2, 0))
        print("Label shape before postprocessing:", label_np.shape)  # (H, W, D)
        # 后处理
        # label_np = postprocess_brats_label(label_np)
        print(f"Processed case {case}, label shape: {label_np.shape}")
        
        ref_nii_path = os.path.join(case_dir, case, f"{cfg.modalities[cfg.modalities.index('t1ce')]}.nii.gz")
        ref_nii = nib.load(ref_nii_path)
        pred_nii = nib.Nifti1Image(label_np, affine=ref_nii.affine, header=ref_nii.header)

        save_path = os.path.join(output_dir, f"{case}_pred_mask.nii.gz")
        nib.save(pred_nii, save_path)


def main():
    # BraTS2018 inference
    prob_base_dir = "/hpc/ajhz839/validation/test_prob_folds/"
    ensemble_output_dir = "/hpc/ajhz839/validation/test_pred_soft_ensemble/"
    case_dir = "/hpc/ajhz839/validation/val/"
    ckpt_dir = "./checkpoint/experiment_41/"

    soft_ensemble(prob_base_dir, case_dir, ckpt_dir)
    ensemble_soft_voting(prob_base_dir, case_dir, ensemble_output_dir)

    
    # # Clinical data inference
    # prob_base_dir = "/hpc/ajhz839/inference/pred/test_prob_folds/"
    # ensemble_output_dir = "/hpc/ajhz839/inference/pred/test_pred_soft_ensemble/"
    # case_dir = "/hpc/ajhz839/inference/clinical_data/"
    # ckpt_dir = "./checkpoint/"

    # soft_ensemble(prob_base_dir, case_dir, ckpt_dir)
    # ensemble_soft_voting(prob_base_dir, case_dir, ensemble_output_dir)
    
    
    print("Inference completed.")

if __name__ == "__main__":
    main()
