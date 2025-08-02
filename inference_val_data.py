import os
os.chdir(os.path.dirname(__file__))
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import torch
import nibabel as nib
import numpy as np
from spike_former_unet_model import spike_former_unet3D_8_384
# from simple_unet_model import spike_former_unet3D_8_384
import torch.nn.functional as F
from config import config as cfg
import time
from inference_helper import TemporalSlidingWindowInference, TemporalSlidingWindowInferenceWithROI
from tqdm import tqdm
from inference_utils import (
    preprocess_for_inference, convert_prediction_to_label_suppress_fp, postprocess_brats_label,
    check_all_folds_ckpt_exist, check_all_folds_val_txt_exist, restore_to_original_shape
    )


def pred_single_case(case_dir, model, inference_engine, device, center_crop=True):
    case_name = os.path.basename(case_dir)
    print(f"Processing case: {case_name}")
    image_paths = [os.path.join(case_dir, f"{case_name}_{mod}.nii") for mod in cfg.modalities]

    # 获取预处理输出和原图信息（包括原始shape和crop位置）
    x_batch, metadata = preprocess_for_inference(image_paths, center_crop=center_crop)
    x_batch = x_batch.to(device)
    B, C, D, H, W = x_batch.shape
    brain_width = np.array([[0, 0, 0], [D-1, H-1, W-1]])
    
    with torch.no_grad():
        # output = inference_engine(x_batch, brain_width, model)
        output = inference_engine(x_batch, model)

    output_prob = torch.sigmoid(output).squeeze(0).cpu().numpy()  # [C, D, H, W]

    return output_prob, metadata


def run_inference_soft_single(case_dir, save_dir, model, inference_engine, device, center_crop=True):
    os.makedirs(save_dir, exist_ok=True)
    prob, metadata = pred_single_case(case_dir, model, inference_engine, device, center_crop=center_crop)

    label_np = convert_prediction_to_label_suppress_fp(prob)  # shape: (D, H, W)


    if center_crop:
        # 还原原始空间
        print("Restoring label to original shape using metadata...")
        restored_label = restore_to_original_shape(
            label_np,
            metadata["original_shape"],
            metadata["crop_start"]
        )
    else:
        # 如果没有中心裁剪，直接使用原始shape
        restored_label = label_np
        
    # 将标签从 (D, H, W) 转换为 (H, W, D)
    final_label = np.transpose(restored_label, (1, 2, 0))  # to (H, W, D)
    print("Label shape before postprocessing:", restored_label.shape)  # (H, W, D)
    # 后处理
    # final_label = postprocess_brats_label(final_label)


    case_name = os.path.basename(case_dir)
    print(f"Processed case {case_name}, label shape: {final_label.shape}")   
    
    ref_nii_path = os.path.join(case_dir, f"{case_name}_{cfg.modalities[cfg.modalities.index('t1ce')]}.nii")
    ref_nii = nib.load(ref_nii_path)
    pred_nii = nib.Nifti1Image(final_label, affine=ref_nii.affine, header=ref_nii.header)

    save_path = os.path.join(save_dir, f"{case_name}_pred_mask.nii.gz")
    nib.save(pred_nii, save_path)
    

def run_inference_folder_single(case_root, save_dir, model, inference_engine, device, whitelist=None, center_crop=True):
    os.makedirs(save_dir, exist_ok=True)
    
    # For other dataset
    all_case_dirs = sorted([
        os.path.join(case_root, name) for name in os.listdir(case_root)
        if os.path.isdir(os.path.join(case_root, name))
    ])
         
    # # For BraTS2018 dataset
    # all_case_dirs = []
    # for subfolder in ['HGG', 'LGG']:
    #     sub_dir = os.path.join(case_root, subfolder)
    #     if not os.path.isdir(sub_dir):
    #         continue
    #     case_dirs = sorted([
    #         os.path.join(sub_dir, name) for name in os.listdir(sub_dir)
    #         if os.path.isdir(os.path.join(sub_dir, name))
    #     ])
    #     all_case_dirs.extend(case_dirs)

    if whitelist is not None:
        whitelist_set = set(whitelist)
        case_dirs = [d for d in all_case_dirs if os.path.basename(d) in whitelist_set]
    else:
        case_dirs = all_case_dirs

    print(f"Found {len(case_dirs)} cases to run.")

    for case_dir in tqdm(case_dirs, desc="Single Model Inference"):
        run_inference_soft_single(case_dir, save_dir, model, inference_engine, device, center_crop=center_crop)


def build_model(ckpt_path):
    model = spike_former_unet3D_8_384(
        num_classes=cfg.num_classes,
        T=cfg.T,
        norm_type=cfg.norm_type,
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
    center_crop=True,  # 是否进行中心裁剪
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

        run_inference_func(case_dir, output_dir, model, inference_engine, device, whitelist, center_crop=center_crop)

        print(f"Fold {fold} inference done. Results saved in {output_dir}")



def main():
    val_cases_dir = './val_cases/'  # 存放验证集case名单txt的文件夹
    ckpt_dir = "/hpc/ajhz839/checkpoint/experiment_69/"  # 模型ckpt所在目录
    case_dir = "/hpc/ajhz839/data/BraTS2020/MICCAI_BraTS2020_TrainingData/"
    output_base_dir = "/hpc/ajhz839/validation/BraTS2020_val_pred_exp69/"

    check_all_folds_ckpt_exist(ckpt_dir)
    check_all_folds_val_txt_exist(val_cases_dir)

    print("5 fold validation data inference started.")

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
        center_crop=True,  # 是否进行中心裁剪
        fold_to_run=None  # 跑全部fold 1~5
    )

    
    # # Clinical data inference
    # prob_base_dir = "/hpc/ajhz839/inference/pred/test_prob_folds/"
    # ensemble_output_dir = "/hpc/ajhz839/inference/pred/test_pred_soft_ensemble/"
    # case_dir = "/hpc/ajhz839/inference/clinical_data/"
    # ckpt_dir = "./checkpoint/"

    # soft_ensemble(prob_base_dir, case_dir, ckpt_dir)
    # ensemble_soft_voting(prob_base_dir, case_dir, ensemble_output_dir)
    
    
    print("5 fold validation data inference completed.")

if __name__ == "__main__":
    main()




