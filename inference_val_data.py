import os
os.chdir(os.path.dirname(__file__))
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
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
import json
from inference_utils import (
    preprocess_for_inference_valid, convert_prediction_to_label_suppress_fp,
    check_all_folds_ckpt_exist, check_all_folds_val_txt_exist, restore_to_original_shape,
    postprocess_brats_label_nnstyle, postprocess_brats_label
    )

from metrics import dice_score_braTS_overall, dice_score_braTS_per_sample_avg
from inference_dice_compute import dice_score_braTS_style

def pred_single_case(case_dir, model, inference_engine, device, center_crop=True):
    case_name = os.path.basename(case_dir)
    print(f"Processing case: {case_name}")
    image_paths = [os.path.join(case_dir, f"{case_name}_{mod}.nii") for mod in cfg.modalities]
    seg_path = os.path.join(case_dir, f"{case_name}_seg.nii")
    print(f"Image paths: {image_paths}")
    print(f"Ground truth path: {seg_path}")
    
    # 获取预处理输出和原图信息（包括原始shape和crop位置）
    img, gt_label, metadata = preprocess_for_inference_valid(image_paths, seg_path, center_crop=center_crop)
    img = img.to(device)
    gt_label = gt_label.to(device)
    B, C, D, H, W = img.shape
    brain_width = np.array([[0, 0, 0], [D-1, H-1, W-1]])
    
    with torch.no_grad():
        # output = inference_engine(x_batch, brain_width, model)
        output = inference_engine(img, model)


    output_prob = torch.sigmoid(output).squeeze(0).cpu().numpy()  # [C, D, H, W]
    
    return output_prob, output, gt_label, metadata


def run_inference_soft_single(case_dir, save_dir, prob_save_dir, model, inference_engine, device, center_crop=True):
    os.makedirs(save_dir, exist_ok=True)
    if prob_save_dir is not None:
        os.makedirs(prob_save_dir, exist_ok=True)
    
    # 1. 推理获得概率图和 metadata    
    prob, pred_tensor, label_tensor, metadata = pred_single_case(case_dir, model, inference_engine, device, center_crop=center_crop)
    case_name = os.path.basename(case_dir)

    if prob_save_dir:
        prob_save_path = os.path.join(prob_save_dir, f"{case_name}_prob.npy")
        np.save(prob_save_path, prob)
        print(f"Saved probability map: {prob_save_path}")

        # 保存metadata信息        
        # 统一保存所有case的metadata.json路径
        prob_base_dir = os.path.dirname(prob_save_dir.rstrip("/"))
        metadata_json_path = os.path.join(prob_base_dir, "metadata.json")

        # 读取已有metadata，如果文件不存在则创建空字典
        if os.path.exists(metadata_json_path):
            with open(metadata_json_path, "r") as f:
                all_metadata = json.load(f)
        else:
            all_metadata = {}

        # 更新当前case的metadata
        all_metadata[case_name] = {
            "original_shape": metadata["original_shape"],
            "crop_start": metadata["crop_start"]
        }

        # 保存回metadata.json
        with open(metadata_json_path, "w") as f:
            json.dump(all_metadata, f, indent=2)
        print(f"Updated metadata file: {metadata_json_path}")

               
    # # 计算 Dice（logits 阶段）
    # print(f"label_tensor shape: {label_tensor.shape}")
    # print(f"pred_tensor shape: {pred_tensor.shape}")
    
    # print("Unique values in label tensor:", torch.unique(label_tensor))
    # print(f"[label] shape: {label_tensor.shape} | min: {label_tensor.min().item()} | max: {label_tensor.max().item()}")
    # print(f"[logits] shape: {pred_tensor.shape} | min: {pred_tensor.min().item()} | max: {pred_tensor.max().item()}")
    
    pred_prob = torch.sigmoid(pred_tensor)
    pred_bin = (pred_prob > 0.5).float()

    # for i, name in enumerate(['TC', 'WT', 'ET']):
    #     print(f"{name} label voxels: {(label_tensor[0, i] > 0).sum().item()}")
    #     print(f"{name} pred voxels: {(pred_bin[0, i] > 0).sum().item()}")
    #     print(f"[label > 0.5] sum: {(label_tensor[0, i] > 0.5).sum().item()}")       # 目标中正样本体素数
    #     print(f"[pred_bin > 0.5] sum: {(pred_bin[0] > 0.5).sum().item()}")  # 预测中正样本体素数
        

    for i, key in enumerate(['TC', 'WT', 'ET']):
        p = pred_bin[0, i]
        t = label_tensor[0, i]
        inter = (p * t).sum().item()
        p_sum = p.sum().item()
        t_sum = t.sum().item()
        dice_val = (2 * inter + 1e-5) / (p_sum + t_sum + 1e-5)
        print(f"Class {key}: pred sum={p_sum}, target sum={t_sum}, intersection={inter}, dice={dice_val:.6f}")

            

    # 计算 Dice
    dice_dict = dice_score_braTS_per_sample_avg(pred_tensor, label_tensor)
    print(f"Dice - Case {case_name} | TC: {dice_dict['TC']:.4f}, WT: {dice_dict['WT']:.4f}, ET: {dice_dict['ET']:.4f}")
    
    pred_tensor_style = pred_tensor.squeeze(0)  # 从 [1, 3, D, H, W] -> [3, D, H, W]
    label_tensor_style = label_tensor.squeeze(0)
    dice_dict_test2 = dice_score_braTS_style(pred_tensor_style, label_tensor_style)
    tc, wt, et = dice_dict_test2
    print(f"Dice (style) - Case {case_name} | TC: {tc:.4f}, WT: {wt:.4f}, ET: {et:.4f}")
        
    # 将预测结果转换为标签
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
    final_label = postprocess_brats_label_nnstyle(final_label)


    case_name = os.path.basename(case_dir)
    print(f"Processed case {case_name}, label shape: {final_label.shape}")   
    
    ref_nii_path = os.path.join(case_dir, f"{case_name}_{cfg.modalities[cfg.modalities.index('t1ce')]}.nii")
    ref_nii = nib.load(ref_nii_path)
    pred_nii = nib.Nifti1Image(final_label, affine=ref_nii.affine, header=ref_nii.header)

    save_path = os.path.join(save_dir, f"{case_name}_pred_mask.nii.gz")
    nib.save(pred_nii, save_path)
    
    return dice_dict
    

def run_inference_folder_single(case_root, save_dir, prob_save_dir, model, inference_engine, device, whitelist=None, center_crop=True):
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

    dice_dicts = []
    for case_dir in tqdm(case_dirs, desc="Single Model Inference"):
        dice_dict = run_inference_soft_single(case_dir, save_dir, prob_save_dir, model, inference_engine, device, center_crop=center_crop)
        dice_dicts.append(dice_dict)

    # 计算平均 Dice
    if dice_dicts:
        avg_dice = {key: sum(d[key] for d in dice_dicts) / len(dice_dicts) for key in dice_dicts[0]}
        print(f"Average Dice - TC: {avg_dice['TC']:.4f}, WT: {avg_dice['WT']:.4f}, ET: {avg_dice['ET']:.4f}")
    
    return avg_dice if dice_dicts else None


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
    prob_base_dir,
    output_base_dir,
    device='cuda',
    num_folds=5,
    dice_style=1,    
    center_crop=True,  # 是否进行中心裁剪
    fold_to_run=None  # None表示跑所有fold，否则跑指定fold
):
    inference_engine = build_inference_engine_func()

    # folds从1开始到num_folds
    folds = [fold_to_run] if fold_to_run is not None else list(range(1, num_folds + 1))
    
    avg_dice_list = []

    for fold in folds:
        if dice_style == 1:
            ckpt_path = os.path.join(ckpt_dir, f"best_model_fold{fold}.pth")
        elif dice_style == 2:
            ckpt_path = os.path.join(ckpt_dir, f"best_model_fold{fold}_dice_style2.pth")
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
        
        # 创建输出目录和概率保存目录
        output_dir = os.path.join(output_base_dir, f"val_fold{fold}_pred")
        prob_save_dir = os.path.join(prob_base_dir, f"fold{fold}")
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(prob_save_dir, exist_ok=True)

        avg_dice = run_inference_func(case_dir, output_dir, prob_save_dir, model, inference_engine, device, whitelist, 
                           center_crop=center_crop)
        avg_dice_list.append(avg_dice)

        print(f"Fold {fold} inference done. Results saved in {output_dir}")
        print(f"Probabilities saved in {prob_save_dir}")
        
    for i, fold in enumerate(folds):
        if avg_dice_list[i] is not None:
            print(f"Fold {fold} - Average Dice: TC: {avg_dice_list[i]['TC']:.4f}, WT: {avg_dice_list[i]['WT']:.4f}, ET: {avg_dice_list[i]['ET']:.4f}")
        else:
            print(f"Fold {fold} - No valid cases processed.")



def inference_BraTS2020_val_data(experiment_id, dice_style, center_crop=True):
    # BraTS 2020 validation data inference
    val_cases_dir = './val_cases/'  # 存放验证集case名单txt的文件夹    
    ckpt_dir = f"/hpc/ajhz839/checkpoint/experiment_{experiment_id}/"  # 模型ckpt所在目录
    case_dir = "/hpc/ajhz839/data/BraTS2020/MICCAI_BraTS2020_TrainingData/"
    if dice_style == 1:
        output_base_dir = f"/hpc/ajhz839/validation/BraTS2020_val_pred_exp{experiment_id}/"
        prob_base_dir = f"/hpc/ajhz839/validation/BraTS2020_val_prob_folds_exp{experiment_id}/"
    elif dice_style == 2:
        output_base_dir = f"/hpc/ajhz839/validation/BraTS2020_val_pred_exp{experiment_id}_dice_style2/"
        prob_base_dir = f"/hpc/ajhz839/validation/BraTS2020_val_prob_folds_exp{experiment_id}_dice_style2/"

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
        prob_base_dir=prob_base_dir,
        output_base_dir=output_base_dir,
        device=cfg.device,
        num_folds=5,
        center_crop=center_crop,  # 是否进行中心裁剪
        dice_style=dice_style,  # 1表示原始Dice计算，2表示新的Dice计算方式
        fold_to_run=None,  # 跑全部fold 1~5
    )
    
    print("5 fold validation data inference completed.")


def main():
    # BraTS 2020 validation data inference
    experiment_id = 76
    dice_style = 2
    inference_BraTS2020_val_data(experiment_id, dice_style, center_crop=True)

    
    # # Clinical data inference
    # prob_base_dir = "/hpc/ajhz839/inference/pred/test_prob_folds/"
    # ensemble_output_dir = "/hpc/ajhz839/inference/pred/test_pred_soft_ensemble/"
    # case_dir = "/hpc/ajhz839/inference/clinical_data/"
    # ckpt_dir = "./checkpoint/"

    # soft_ensemble(prob_base_dir, case_dir, ckpt_dir)
    # ensemble_soft_voting(prob_base_dir, case_dir, ensemble_output_dir)
    
    
if __name__ == "__main__":
    main()




