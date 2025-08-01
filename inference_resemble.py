import os
os.chdir(os.path.dirname(__file__))
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import torch
import nibabel as nib
import numpy as np
from spike_former_unet_model import spike_former_unet3D_8_384
#from simple_unet_model import spike_former_unet3D_8_384
import torch.nn.functional as F
from config import config as cfg
import time
from inference_helper import TemporalSlidingWindowInference, TemporalSlidingWindowInferenceWithROI
from tqdm import tqdm
import json
from inference_utils import (
    preprocess_for_inference, convert_prediction_to_label_suppress_fp, postprocess_brats_label,
    check_all_folds_ckpt_exist, check_test_txt_exist, restore_to_original_shape, read_case_list
    )



def pred_single_case_soft(case_dir, prob_save_dir, model, inference_engine, device, center_crop=True):
    case_name = os.path.basename(case_dir)
    print(f"Processing case: {case_name}")
    image_paths = [os.path.join(case_dir, f"{case_name}_{mod}.nii") for mod in cfg.modalities]

    x_batch, metadata = preprocess_for_inference(image_paths, center_crop=center_crop)
    if not center_crop:
        metadata = None
        
    # 将数据移动到指定设备
    x_batch = x_batch.to(device)
    
    B, C, D, H, W = x_batch.shape
    brain_width = np.array([[0, 0, 0], [D-1, H-1, W-1]])
    
    with torch.no_grad():
        output = inference_engine(x_batch, model)
        # output = inference_engine(x_batch, brain_width, model)

    output_prob = torch.sigmoid(output).squeeze(0).cpu().numpy()  # [C, D, H, W]
    
    os.makedirs(prob_save_dir, exist_ok=True)
    prob_path = os.path.join(prob_save_dir, f"{case_name}_prob.npy")
    np.save(prob_path, output_prob)
    print(f"Saved probability map: {prob_path}")

    return case_name, metadata

    

def run_inference_folder_soft(case_root, save_dir, model, inference_engine, device, case_list=None,center_crop=True):
    os.makedirs(save_dir, exist_ok=True)

    # 仅包含在 test_case_list 中的目录
    if case_list is not None:
        case_dirs = [
            os.path.join(case_root, name) for name in case_list
            if os.path.isdir(os.path.join(case_root, name))
        ]
    else:
        case_dirs = sorted([
            os.path.join(case_root, name) for name in os.listdir(case_root)
            if os.path.isdir(os.path.join(case_root, name))
        ])

    print(f"Found {len(case_dirs)} cases to infer.")

    metadata_dict = {}

    for case_dir in tqdm(case_dirs, desc="Soft Voting Inference"):
        if center_crop:
            case_name, metadata = pred_single_case_soft(
                case_dir, save_dir, model, inference_engine, device, center_crop=center_crop)
            metadata_dict[case_name] = metadata
        else:
            pred_single_case_soft(case_dir, save_dir, model, inference_engine, device, center_crop=center_crop)
    
    if center_crop:
        return metadata_dict


   
def soft_ensemble(prob_base_dir, case_dir, ckpt_dir, test_case_list, center_crop=True):
    metadata_dir = os.path.join(prob_base_dir, "metadata")
    os.makedirs(metadata_dir, exist_ok=True)
    
    metadata_dict = None
    
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
            sw_batch_size=4,
            mode="constant", # "gaussian", "constant"
            encode_method=cfg.encode_method,
            T=cfg.T,
            num_classes=cfg.num_classes
        )
        
        # inference_engine = TemporalSlidingWindowInferenceWithROI(
        #     patch_size=cfg.inference_patch_size,
        #     overlap=cfg.overlap,
        #     sw_batch_size=4,
        #     mode="gaussian", # "gaussian", "constant"
        #     encode_method=cfg.encode_method,
        #     T=cfg.T,
        #     num_classes=cfg.num_classes
        # )

        fold_prob_dir = os.path.join(prob_base_dir, f"fold{fold}")
        
        if center_crop:
            if fold == 1:
                metadata_dict = run_inference_folder_soft(case_dir, fold_prob_dir, model, inference_engine, cfg.device, test_case_list, center_crop=center_crop)
                # 保存metadata_dict为json文件
                metadata_json_path = os.path.join(metadata_dir, "case_metadata.json")
                with open(metadata_json_path, "w") as f:
                    json.dump(metadata_dict, f)
                print(f"Saved metadata JSON to {metadata_json_path}")
            else:
                run_inference_folder_soft(case_dir, fold_prob_dir, model, inference_engine, cfg.device, test_case_list, center_crop=center_crop)
        else:
            run_inference_folder_soft(case_dir, fold_prob_dir, model, inference_engine, cfg.device, test_case_list, center_crop=center_crop)

            
    return metadata_json_path if center_crop else None


def ensemble_soft_voting(prob_root, case_dir, output_dir, center_crop=True, metadata_json_path=None):
    os.makedirs(output_dir, exist_ok=True)
    
    if metadata_json_path and center_crop:
        with open(metadata_json_path, "r") as f:
            case_metadata = json.load(f)
        
    case_names = sorted(list(set([f.split('_prob.npy')[0] for f in os.listdir(os.path.join(prob_root, 'fold1'))])))

    for case in tqdm(case_names, desc="Soft Voting Ensemble"):
        prob_list = []
        for fold in range(1, 6):
            prob_path = os.path.join(prob_root, f"fold{fold}", f"{case}_prob.npy")
            prob = np.load(prob_path)
            prob_list.append(prob)

        mean_prob = np.mean(np.stack(prob_list, axis=0), axis=0)  # [C, D, H, W]
        print(f"Mean probability shape for case {case}: {mean_prob.shape}")

        label_np = convert_prediction_to_label_suppress_fp(mean_prob)

        print("Label shape before transposing:", label_np.shape)  # (D, H, W)
        label_np = np.transpose(label_np, (1, 2, 0))
        print("Label shape before postprocessing:", label_np.shape)  # (H, W, D)
        # 后处理
        # label_np = postprocess_brats_label(label_np)
        print(f"Processed case {case}, label shape: {label_np.shape}")
        
        if metadata_json_path and center_crop:
            metadata = case_metadata[case]
            original_shape = metadata["original_shape"]  # (D, H, W)
            crop_start = metadata["crop_start"]          # (sd, sh, sw)
            restored_label = restore_to_original_shape(label_np, tuple(original_shape), tuple(crop_start))
        else:
            restored_label = label_np
                
        ref_nii_path = os.path.join(case_dir, case, f"{cfg.modalities[cfg.modalities.index('t1ce')]}.nii.gz")
        ref_nii = nib.load(ref_nii_path)
        pred_nii = nib.Nifti1Image(restored_label, affine=ref_nii.affine, header=ref_nii.header)

        save_path = os.path.join(output_dir, f"{case}_pred_mask.nii.gz")
        nib.save(pred_nii, save_path)


def main():
    # # BraTS2018 inference
    # prob_base_dir = "/hpc/ajhz839/validation/test_prob_folds/"
    # ensemble_output_dir = "/hpc/ajhz839/validation/test_pred_soft_ensemble/"
    # case_dir = "/hpc/ajhz839/validation/val/"
    # ckpt_dir = "/hpc/ajhz839/checkpoint/experiment_41/"

    # soft_ensemble(prob_base_dir, case_dir, ckpt_dir)
    # ensemble_soft_voting(prob_base_dir, case_dir, ensemble_output_dir)
    
    
    # BraTS2020 test data inference
    prob_base_dir = "/hpc/ajhz839/validation/test_prob_folds_exp65/"
    ensemble_output_dir = "/hpc/ajhz839/validation/test_pred_soft_ensemble_exp65/"
    case_dir = "/hpc/ajhz839/data/BraTS2020/MICCAI_BraTS2020_TrainingData/"
    test_cases_txt =  './val_cases/test_cases.txt'
    ckpt_dir = "/hpc/ajhz839/checkpoint/experiment_65/"
    
    center_crop=False

    check_all_folds_ckpt_exist(ckpt_dir)
    check_test_txt_exist(test_cases_txt)

    test_case_list = read_case_list(test_cases_txt)
    metadata_json_path=soft_ensemble(prob_base_dir, case_dir, ckpt_dir, test_case_list, center_crop=center_crop) 
    ensemble_soft_voting(prob_base_dir, case_dir, ensemble_output_dir, center_crop=center_crop, metadata_json_path=metadata_json_path)

    
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
