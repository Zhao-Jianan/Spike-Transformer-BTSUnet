import os
os.chdir(os.path.dirname(__file__))
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import nibabel as nib
import numpy as np
from spike_former_unet_model import spike_former_unet3D_8_384
# from simple_unet_model import spike_former_unet3D_8_384
import torch.nn.functional as F
from config import config as cfg
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, NormalizeIntensityd,
    Orientationd, Spacingd, ToTensord
    )
import time
from scipy.ndimage import binary_dilation, binary_opening, label, generate_binary_structure
from inference_helper import TemporalSlidingWindowInference, TemporalSlidingWindowInferenceWithROI
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



def pred_single_case(case_dir, model, inference_engine, device):
    case_name = os.path.basename(case_dir)
    print(f"Processing case: {case_name}")
    image_paths = [os.path.join(case_dir, f"{case_name}_{mod}.nii") for mod in cfg.modalities]

    x_batch = preprocess_for_inference(image_paths).to(device)
    B, C, D, H, W = x_batch.shape
    brain_width = np.array([[0, 0, 0], [D-1, H-1, W-1]])
    
    with torch.no_grad():
        # output = inference_engine(x_batch, brain_width, model)
        output = inference_engine(x_batch, model)

    output_prob = torch.sigmoid(output).squeeze(0).cpu().numpy()  # [C, D, H, W]

    return output_prob




def run_inference_soft_single(case_dir, save_dir, model, inference_engine, device):
    os.makedirs(save_dir, exist_ok=True)
    
    prob = pred_single_case(case_dir, model, inference_engine, device)

    label_np = convert_prediction_to_label_suppress_fp(prob)  # shape: (D, H, W)
    label_np = np.transpose(label_np, (1, 2, 0))  # to (H, W, D)

    case_name = os.path.basename(case_dir)
    ref_nii_path = os.path.join(case_dir, f"{case_name}_{cfg.modalities[cfg.modalities.index('t1ce')]}.nii")
    ref_nii = nib.load(ref_nii_path)
    pred_nii = nib.Nifti1Image(label_np, affine=ref_nii.affine, header=ref_nii.header)

    save_path = os.path.join(save_dir, f"{case_name}_pred_mask.nii.gz")
    nib.save(pred_nii, save_path)

def run_inference_folder_single(case_root, save_dir, model, inference_engine, device, whitelist=None):
    os.makedirs(save_dir, exist_ok=True)
    
    # # For other dataset
    # all_case_dirs = sorted([
    #     os.path.join(case_root, name) for name in os.listdir(case_root)
    #     if os.path.isdir(os.path.join(case_root, name))
    # ])
         
    # For BraTS2018 dataset
    all_case_dirs = []
    for subfolder in ['HGG', 'LGG']:
        sub_dir = os.path.join(case_root, subfolder)
        if not os.path.isdir(sub_dir):
            continue
        case_dirs = sorted([
            os.path.join(sub_dir, name) for name in os.listdir(sub_dir)
            if os.path.isdir(os.path.join(sub_dir, name))
        ])
        all_case_dirs.extend(case_dirs)

    if whitelist is not None:
        whitelist_set = set(whitelist)
        case_dirs = [d for d in all_case_dirs if os.path.basename(d) in whitelist_set]
    else:
        case_dirs = all_case_dirs

    print(f"Found {len(case_dirs)} cases to run.")

    for case_dir in tqdm(case_dirs, desc="Single Model Inference"):
        run_inference_soft_single(case_dir, save_dir, model, inference_engine, device)


def build_model(ckpt_path):
    model = spike_former_unet3D_8_384(
        num_classes=cfg.num_classes,
        T=cfg.T,
        # norm_type=cfg.norm_type,
        step_mode=cfg.step_mode
    ).to(cfg.device)
    model.load_state_dict(torch.load(ckpt_path, map_location=cfg.device))
    model.eval()
    return model

def build_inference_engine():
    return TemporalSlidingWindowInference( # TemporalSlidingWindowInference, TemporalSlidingWindowInferenceWithROI
        patch_size=cfg.inference_patch_size,
        overlap=cfg.overlap,
        sw_batch_size=8,
        mode="constant",
        encode_method=cfg.encode_method,
        T=cfg.T,
        num_classes=cfg.num_classes
    )            


def run_inference_all_folds(
    build_model_func,
    build_inference_engine_func,
    run_inference_func,
    val_cases_dir,
    ckpt_dir,
    case_dir,
    output_base_dir,
    device='cuda',
    num_folds=5,
    fold_to_run=None  # None表示跑所有fold，否则跑指定fold
):
    inference_engine = build_inference_engine_func()

    # folds从1开始到num_folds
    folds = [fold_to_run] if fold_to_run is not None else list(range(1, num_folds + 1))

    for fold in folds:
        ckpt_path = os.path.join(ckpt_dir, f"best_model_fold{fold}.pth")
        model = build_model_func(ckpt_path)
        model.to(device)
        model.eval()

        val_list_path = os.path.join(val_cases_dir, f"val_cases_fold{fold}.txt")
        if not os.path.exists(val_list_path):
            print(f"Validation case list not found for fold {fold}: {val_list_path}")
            continue

        with open(val_list_path, 'r') as f:
            whitelist = [line.strip() for line in f if line.strip()]

        print(f"Running inference for fold {fold} with {len(whitelist)} cases")

        output_dir = os.path.join(output_base_dir, f"val_fold{fold}_pred")
        os.makedirs(output_dir, exist_ok=True)

        run_inference_func(case_dir, output_dir, model, inference_engine, device, whitelist)

        print(f"Fold {fold} inference done. Results saved in {output_dir}")





def main():
    val_cases_dir = './val_cases/'  # 存放验证集case名单txt的文件夹
    ckpt_dir = "/hpc/ajhz839/checkpoint/experiment_56/"  # 模型ckpt所在目录
    case_dir = "/hpc/ajhz839/data/BraTS2018/train/"
    output_base_dir = "/hpc/ajhz839/validation/"    
        
    run_inference_all_folds(
        build_model_func=build_model,
        build_inference_engine_func=build_inference_engine,
        run_inference_func=run_inference_folder_single,
        val_cases_dir=val_cases_dir,
        ckpt_dir=ckpt_dir,
        case_dir=case_dir,
        output_base_dir=output_base_dir,
        device=cfg.device,
        num_folds=5,
        fold_to_run=None  # 跑全部fold 1~5
    )

    
    # # Clinical data inference
    # prob_base_dir = "/hpc/ajhz839/inference/pred/test_prob_folds/"
    # ensemble_output_dir = "/hpc/ajhz839/inference/pred/test_pred_soft_ensemble/"
    # case_dir = "/hpc/ajhz839/inference/clinical_data/"
    # ckpt_dir = "./checkpoint/"

    # soft_ensemble(prob_base_dir, case_dir, ckpt_dir)
    # ensemble_soft_voting(prob_base_dir, case_dir, ensemble_output_dir)
    
    
    print("Single-model inference completed.")

if __name__ == "__main__":
    main()




