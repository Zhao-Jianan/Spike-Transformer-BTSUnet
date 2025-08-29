import os
import pickle
os.chdir(os.path.dirname(__file__))
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import nibabel as nib
import numpy as np
from model.spike_former_unet_model import spike_former_unet3D_8_384
# from model.simple_unet_model import spike_former_unet3D_8_384
import torch.nn.functional as F
from config import config as cfg
import time
from tqdm import tqdm
import json
from utilities.logger import logger

from inference.inference_helper import TemporalSlidingWindowInference, TemporalSlidingWindowInferenceWithROI
from inference.inference_preprocess import preprocess_for_inference_test
from inference.inference_postprocess import (
    postprocess_brats_label_nnstyle, postprocess_brats_label, 
    postprocess_brats_label_etminsize, apply_postprocessing_brats, postprocess_brats_label_nnstyle_v2
    )
from inference.inference_utils import (
    convert_prediction_to_label_suppress_fp, check_all_folds_ckpt_exist, check_test_txt_exist, 
    restore_to_original_shape, read_case_list
    )



def pred_single_case_soft(case_dir, prob_save_dir, model, inference_engine, device, center_crop=True, dataset_flag=None):
    case_name = os.path.basename(case_dir)
    logger.info(f"Processing case: {case_name}")
    if dataset_flag == 'BraTS23':
        image_paths = [os.path.join(case_dir, f"{case_name}-{mod}.nii.gz") for mod in cfg.modalities]
    else:
        image_paths = [os.path.join(case_dir, f"{case_name}_{mod}.nii") for mod in cfg.modalities]

    x_batch, metadata = preprocess_for_inference_test(image_paths, center_crop=center_crop)
    if not center_crop:
        metadata = None
        
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
    logger.info(f"Saved probability map: {prob_path}")

    return case_name, metadata

    

def run_inference_folder_soft(case_root, save_dir, model, inference_engine, device, case_list=None,center_crop=True, dataset_flag=None):
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

    logger.info(f"Found {len(case_dirs)} cases to infer.")

    metadata_dict = {}

    for case_dir in tqdm(case_dirs, desc="Soft Voting Inference"):
        if center_crop:
            case_name, metadata = pred_single_case_soft(
                case_dir, save_dir, model, inference_engine, device, center_crop=center_crop, dataset_flag=dataset_flag)
            metadata_dict[case_name] = metadata
        else:
            pred_single_case_soft(case_dir, save_dir, model, inference_engine, device, center_crop=center_crop, dataset_flag=dataset_flag)
    
    if center_crop:
        return metadata_dict



def soft_ensemble(prob_base_dir, case_dir, ckpt_dir, test_case_list, dice_style=1, center_crop=True, prefix=None, dataset_flag=None):
    metadata_dir = os.path.join(prob_base_dir, "metadata")
    os.makedirs(metadata_dir, exist_ok=True)
    
    metadata_dict = None
    
    dice_style_str = "" if dice_style == 1 else f"_dice_style{dice_style}"
    
    for fold in range(1, 6):
        logger.info(f"Running inference for fold {fold}")        
        if prefix == 'slidingwindow':
            model_ckpt = os.path.join(ckpt_dir, f"entire_best_model_fold{fold}{dice_style_str}.pth")
        else:
            model_ckpt = os.path.join(ckpt_dir, f"best_model_fold{fold}{dice_style_str}.pth")

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
                metadata_dict = run_inference_folder_soft(
                    case_dir, fold_prob_dir, model, inference_engine, cfg.device, 
                    test_case_list, center_crop=center_crop, dataset_flag=dataset_flag)
                # 保存metadata_dict为json文件
                metadata_json_path = os.path.join(metadata_dir, "case_metadata.json")
                with open(metadata_json_path, "w") as f:
                    json.dump(metadata_dict, f)
                logger.info(f"Saved metadata JSON to {metadata_json_path}")
            else:
                run_inference_folder_soft(case_dir, fold_prob_dir, model, inference_engine, 
                                          cfg.device, test_case_list, center_crop=center_crop, dataset_flag=dataset_flag)
        else:
            run_inference_folder_soft(case_dir, fold_prob_dir, model, inference_engine, cfg.device, 
                                      test_case_list, center_crop=center_crop, dataset_flag=dataset_flag)

            
    return metadata_json_path if center_crop else None


def ensemble_soft_voting(prob_root, case_dir, output_dir, center_crop=True, 
                         metadata_json_path=None, dataset_flag=None, postprocess_method='solid'):
    os.makedirs(output_dir, exist_ok=True)
    
    if metadata_json_path and center_crop:
        with open(metadata_json_path, "r") as f:
            logger.info("Loading metadata from JSON file...")
            case_metadata = json.load(f)
        
    case_names = sorted(list(set([f.split('_prob.npy')[0] for f in os.listdir(os.path.join(prob_root, 'fold1'))])))

    for case in tqdm(case_names, desc="Soft Voting Ensemble"):
        prob_list = []
        for fold in range(1, 6):
            prob_path = os.path.join(prob_root, f"fold{fold}", f"{case}_prob.npy")
            prob = np.load(prob_path)
            # 读取该 fold 的后处理策略
            if postprocess_method == 'strategy':
                pkl_path = os.path.join(prob_root, f"fold{fold}", "postprocessing_strategy.pkl")
                if os.path.exists(pkl_path):
                    with open(pkl_path, "rb") as f:
                        strategy = pickle.load(f)
                    label_fold = convert_prediction_to_label_suppress_fp(prob, dataset_flag=dataset_flag)
                    label_fold = apply_postprocessing_brats(label_fold, strategy)  # 应用独立策略
                    prob = np.eye(cfg.num_classes)[label_fold]  # 重新转为 one-hot 概率图
                    prob = np.transpose(prob, (3, 0, 1, 2))  # (C, D, H, W)
                else:
                    logger.warning(f"[Warning] No postprocessing_strategy.pkl found for fold {fold}. Skipping postprocessing.")

            prob_list.append(prob)

        mean_prob = np.mean(np.stack(prob_list, axis=0), axis=0)  # [C, D, H, W]
        logger.info(f"Mean probability shape for case {case}: {mean_prob.shape}")

        label_np = convert_prediction_to_label_suppress_fp(mean_prob, dataset_flag=dataset_flag)

        logger.info(f"Label shape before restoring to original shape: {label_np.shape}")  # (D, H, W)

        if metadata_json_path and center_crop:
            logger.info("Restoring label to original shape using metadata...")
            # 使用case_metadata中的信息恢复标签到原始形状
            metadata = case_metadata[case]
            original_shape = metadata["original_shape"]  # (D, H, W)
            crop_start = metadata["crop_start"]          # (sd, sh, sw)
            restored_label = restore_to_original_shape(label_np, tuple(original_shape), tuple(crop_start))
        else:
            restored_label = label_np

        logger.info(f"Label shape before transposing: {restored_label.shape}")  # (D, H, W)
        
        # Apply postprocessing if no specific strategy is provided
        if postprocess_method == 'solid':
            # Apply default postprocessing
            final_label = postprocess_brats_label_nnstyle_v2(final_label)
            
        final_label = np.transpose(restored_label, (1, 2, 0))
                   
        if dataset_flag== 'BraTS23':
            ref_nii_path = os.path.join(case_dir, case, f"{case}-{cfg.modalities[cfg.modalities.index('t1c')]}.nii.gz")
        else:
            ref_nii_path = os.path.join(case_dir, case, f"{case}_{cfg.modalities[cfg.modalities.index('t1ce')]}.nii")
        ref_nii = nib.load(ref_nii_path)
        pred_nii = nib.Nifti1Image(final_label, affine=ref_nii.affine, header=ref_nii.header)

        save_path = os.path.join(output_dir, f"{case}_pred_mask.nii.gz")
        nib.save(pred_nii, save_path)



def inference_BraTS2020_test_data(experiment_id, dice_style, center_crop=True, prefix=None, postprocess_method='solid'):
    # BraTS2020 test data inference
    logger.info(f"Starting inference for BraTS2020 test data with experiment ID {experiment_id} and dice style {dice_style}...")
 
    dice_style_str = "" if dice_style == 1 else f"_dice_style{dice_style}"
    prefix_str = f"_{prefix}" if prefix else ""
        
    prob_base_dir = f"/hpc/ajhz839/inference/BraTS2020/test_prob_folds_exp{experiment_id}{dice_style_str}{prefix_str}/"
    ensemble_output_dir = f"/hpc/ajhz839/inference/BraTS2020/test_pred_soft_ensemble_exp{experiment_id}{dice_style_str}{prefix_str}/"
        
    case_dir = "/hpc/ajhz839/data/BraTS2020/MICCAI_BraTS2020_TrainingData/"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    test_cases_txt = os.path.join(script_dir, 'val_cases/test_cases.txt') 
    ckpt_dir = f"/hpc/ajhz839/checkpoint/experiment_{experiment_id}/"

    check_all_folds_ckpt_exist(ckpt_dir, dice_style, prefix)
    check_test_txt_exist(test_cases_txt)

    test_case_list = read_case_list(test_cases_txt)
    metadata_json_path=soft_ensemble(prob_base_dir, case_dir, ckpt_dir, test_case_list, dice_style=dice_style, center_crop=center_crop, prefix=prefix)

    ensemble_soft_voting(prob_base_dir, case_dir, ensemble_output_dir, 
                         center_crop=center_crop, metadata_json_path=metadata_json_path, postprocess_method=postprocess_method)

    logger.info("Inference completed.")
    

def inference_BraTS2023_test_data(experiment_id, dice_style, center_crop=True, prefix=None, postprocess_method='solid'):
    # BraTS2020 test data inference
    logger.info(f"Starting inference for BraTS2023 test data with experiment ID {experiment_id} and dice style {dice_style}...")
 
    dice_style_str = "" if dice_style == 1 else f"_dice_style{dice_style}"
    prefix_str = f"_{prefix}" if prefix else ""
        
    prob_base_dir = f"/hpc/ajhz839/inference/BraTS2023/test_prob_folds_exp{experiment_id}{dice_style_str}{prefix_str}/"
    ensemble_output_dir = f"/hpc/ajhz839/inference/BraTS2023/test_pred_soft_ensemble_exp{experiment_id}{dice_style_str}{prefix_str}/"
        
    case_dir = "/hpc/ajhz839/data/BraTS2023/val/"
    ckpt_dir = f"/hpc/ajhz839/checkpoint/experiment_{experiment_id}/"

    check_all_folds_ckpt_exist(ckpt_dir, dice_style, prefix)

    test_case_list = sorted([
        d for d in os.listdir(case_dir)
        if os.path.isdir(os.path.join(case_dir, d))
    ])
    
    metadata_json_path=soft_ensemble(
        prob_base_dir, case_dir, ckpt_dir, test_case_list, 
        dice_style=dice_style, center_crop=center_crop, prefix=prefix, dataset_flag='BraTS23'
        )

    ensemble_soft_voting(
        prob_base_dir, case_dir, ensemble_output_dir, 
        center_crop=center_crop, metadata_json_path=metadata_json_path, dataset_flag='BraTS23',
        postprocess_method=postprocess_method
        )

    logger.info("Inference completed.")



def main():
    # # BraTS2018 inference
    # prob_base_dir = "/hpc/ajhz839/validation/test_prob_folds/"
    # ensemble_output_dir = "/hpc/ajhz839/validation/test_pred_soft_ensemble/"
    # case_dir = "/hpc/ajhz839/validation/val/"
    # ckpt_dir = "/hpc/ajhz839/checkpoint/experiment_41/"

    # soft_ensemble(prob_base_dir, case_dir, ckpt_dir)
    # ensemble_soft_voting(prob_base_dir, case_dir, ensemble_output_dir)
    
    
    # BraTS2020 test data inference
    experiment_id = 92
    dice_style = 1
    prefix = None  # "slidingwindow"
    postprocess_method='solid' # 'solid', 'strategy', or 'none'
    inference_BraTS2020_test_data(experiment_id, dice_style, center_crop=True, prefix=prefix, 
                                  postprocess_method=postprocess_method)


    # # BraTS2023 test data inference
    # experiment_id = 75
    # dice_style = 2
    # prefix = None  # "slidingwindow"
    # inference_BraTS2023_test_data(experiment_id, dice_style, center_crop=True, prefix=prefix)
    
   
    # # Clinical data inference
    # prob_base_dir = "/hpc/ajhz839/inference/pred/test_prob_folds/"
    # ensemble_output_dir = "/hpc/ajhz839/inference/pred/test_pred_soft_ensemble/"
    # case_dir = "/hpc/ajhz839/inference/clinical_data/"
    # ckpt_dir = "./checkpoint/"

    # soft_ensemble(prob_base_dir, case_dir, ckpt_dir)
    # ensemble_soft_voting(prob_base_dir, case_dir, ensemble_output_dir)
    
    

if __name__ == "__main__":
    main()
