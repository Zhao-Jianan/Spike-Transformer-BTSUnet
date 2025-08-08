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
import time
from inference_helper import TemporalSlidingWindowInference, TemporalSlidingWindowInferenceWithROI
from tqdm import tqdm
from inference_utils import preprocess_for_inference, convert_prediction_to_label_suppress_fp, postprocess_brats_label


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
        run_inference_soft_single(case_dir, save_dir, model, inference_engine, device)


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


def main():
    print("Starting single-model inference...")
    ckpt_path = "/hpc/ajhz839/checkpoint/experiment_61/best_model_fold1.pth"  # 模型ckpt
    case_dir = "/hpc/ajhz839/data/BraTS2020/MICCAI_BraTS2020_TrainingData/"
    output_dir = "/hpc/ajhz839/validation/BraTS2020_val_fold1_pred_exp61/"
    
    with open('val_cases_fold1.txt', 'r') as f:
        brats2020_case_whitelist = [line.strip() for line in f if line.strip()]
        
    model = build_model(ckpt_path)
    inference_engine = build_inference_engine()
    run_inference_folder_single(case_dir, output_dir, model, inference_engine, cfg.device, brats2020_case_whitelist)

    
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




