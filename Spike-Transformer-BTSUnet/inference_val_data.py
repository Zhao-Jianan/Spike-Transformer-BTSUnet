import os
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
from inference.inference_preprocess import preprocess_for_inference_valid
from inference.inference_postprocess import postprocess_brats_label_nnstyle, postprocess_brats_label, determine_postprocessing_brats
from inference.inference_utils import (
    convert_prediction_to_label_suppress_fp, check_all_folds_ckpt_exist,
    check_all_folds_val_txt_exist, restore_to_original_shape, convert_label_to_onehot
    )
from training.metrics import  dice_score_braTS_per_sample_avg


def compute_avg_dice(dice_list, index_map):
    """
    计算平均 Dice
    支持字典格式({'TC':..., 'WT':..., 'ET':...}) 和 tuple/tensor 格式
    """
    if not dice_list:
        return None, None

    # 检查第一个元素类型，决定解析方式
    first = dice_list[0]
    if isinstance(first, dict):
        avg_dice = {key: sum(d[key] for d in dice_list) / len(dice_list) for key in first}
    else:  # tuple / list / tensor
        def get_value(d, key):
            val = d[index_map[key]]
            return val.item() if torch.is_tensor(val) else float(val)
        avg_dice = {key: sum(get_value(d, key) for d in dice_list) / len(dice_list) for key in index_map}

    mean_dice = sum(avg_dice.values()) / len(avg_dice)
    return avg_dice, mean_dice


def compute_avg_dice_style2(dice_dicts_style2_post):
    """
    计算 style2 格式下的平均 Dice。
    dice_dicts_style2_post: list of tuple / list / tensor
    tuple 或 list -> (TC, WT, ET)
    0-dim tensor -> 单个值（一般错误存储，不推荐）
    """
    index_map = {'TC': 0, 'WT': 1, 'ET': 2}

    # 确保统一格式
    processed = []
    for d in dice_dicts_style2_post:
        if isinstance(d, tuple) or isinstance(d, list):
            processed.append(tuple(float(x) for x in d))
        elif torch.is_tensor(d):
            if d.ndim == 0:
                # 0-dim tensor 不能直接索引，转成 float
                processed.append((float(d.item()), 0.0, 0.0))  
            else:
                processed.append(tuple(float(x) for x in d))
        else:
            raise ValueError(f"Unsupported type in dice_dicts_style2_post: {type(d)}")

    # 求平均
    avg_dice = {key: sum(p[index_map[key]] for p in processed) / len(processed) for key in index_map}
    mean_dice = sum(avg_dice.values()) / len(avg_dice)

    logger.info(f"Average Dice - TC: {avg_dice['TC']:.4f} | WT: {avg_dice['WT']:.4f} | ET: {avg_dice['ET']:.4f} | Mean: {mean_dice:.4f}")
    return avg_dice, mean_dice



def pred_single_case(case_dir, model, inference_engine, device, center_crop=True, dataset_flag=None):
    case_name = os.path.basename(case_dir)
    logger.info(f"Processing case: {case_name}")
    if dataset_flag == 'BraTS23':
        image_paths = [os.path.join(case_dir, f"{case_name}-{mod}.nii.gz") for mod in cfg.modalities]
        seg_path = os.path.join(case_dir, f"{case_name}-seg.nii.gz")
    else:
        image_paths = [os.path.join(case_dir, f"{case_name}_{mod}.nii") for mod in cfg.modalities]
        seg_path = os.path.join(case_dir, f"{case_name}_seg.nii")
    logger.info(f"Image paths: {image_paths}")
    logger.info(f"Ground truth path: {seg_path}")
    
    # 获取预处理输出和原图信息（包括原始shape和crop位置）
    img, gt_label, gt_tensor_hwd, metadata = preprocess_for_inference_valid(image_paths, seg_path, center_crop=center_crop)
    img = img.to(device)
    gt_label = gt_label.to(device)
    gt_tensor_hwd = gt_tensor_hwd.to("cpu")
    B, C, D, H, W = img.shape
    brain_width = np.array([[0, 0, 0], [D-1, H-1, W-1]])

    with torch.no_grad():
        # output = inference_engine(x_batch, brain_width, model)
        output = inference_engine(img, model)


    output_prob = torch.sigmoid(output).squeeze(0).cpu().numpy()  # [C, D, H, W]
    
    return output_prob, output, gt_label, gt_tensor_hwd, metadata


def run_inference_soft_single(case_dir, save_dir, prob_save_dir, model, inference_engine, device, center_crop=True, dataset_flag=None):
    os.makedirs(save_dir, exist_ok=True)
    if prob_save_dir:
        os.makedirs(prob_save_dir, exist_ok=True)
    
    # 1. 推理获得概率图和 metadata    
    prob, pred_tensor, gt_tensor, raw_gt_tensor_hwd, metadata = pred_single_case(
        case_dir, model, inference_engine, device, center_crop=center_crop, dataset_flag=dataset_flag)

    case_name = os.path.basename(case_dir)

    if prob_save_dir:
        prob_save_path = os.path.join(prob_save_dir, f"{case_name}_prob.npy")
        np.save(prob_save_path, prob)

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

        
    # 将预测结果转换为标签（不做 PP）
    label_np = convert_prediction_to_label_suppress_fp(prob, dataset_flag=dataset_flag)  # shape: (D, H, W)
    label_tensor = torch.from_numpy(label_np)

    if center_crop:
        # 还原原始空间
        logger.info("Restoring label to original shape using metadata...")
        restored_label = restore_to_original_shape(
            label_tensor,
            metadata["original_shape"],
            metadata["crop_start"]
        )

    else:
        # 如果没有中心裁剪，直接使用原始shape
        restored_label = label_tensor

    # 将标签从 (D, H, W) 转换为 (H, W, D)
    final_label = restored_label.permute(1, 2, 0)  # to (H, W, D)
    final_label_np = final_label.cpu().numpy()
    
   
    # 保存预测 nii
    if dataset_flag == 'BraTS23':
        ref_nii_path = os.path.join(case_dir, f"{case_name}-{cfg.modalities[cfg.modalities.index('t1c')]}.nii.gz")
    else:    
        ref_nii_path = os.path.join(case_dir, f"{case_name}_{cfg.modalities[cfg.modalities.index('t1ce')]}.nii")
    ref_nii = nib.load(ref_nii_path)
    nib.save(nib.Nifti1Image(final_label_np, affine=ref_nii.affine, header=ref_nii.header),
             os.path.join(save_dir, f"{case_name}_pred_mask.nii.gz"))

    # 保存 GT（(D,H,W) → (H,W,D)）
    raw_gt_np = raw_gt_tensor_hwd.cpu().numpy()

    return final_label_np, raw_gt_np
    

def run_inference_folder_single(case_root, save_dir, prob_save_dir, model, inference_engine, device, 
                                whitelist=None, center_crop=True, dataset_flag=None):
    os.makedirs(save_dir, exist_ok=True)
    # 存储预测和GT数组
    pred_list = []
    gt_list = []
    
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

    logger.info(f"Found {len(case_dirs)} cases to run.")

    for case_dir in tqdm(case_dirs, desc="Single Model Inference"):
        pred_np, gt_np = run_inference_soft_single(
            case_dir, save_dir, prob_save_dir, model, inference_engine, device, center_crop=center_crop, dataset_flag=dataset_flag
        )
        pred_list.append(pred_np)
        gt_list.append(gt_np)

    # 全部完成后，生成后处理策略
    pkl_path = os.path.join(save_dir, "postprocessing_strategy.pkl")
    determine_postprocessing_brats(pred_list, gt_list, labels=[1, 2, 4], save_pkl_path=pkl_path)


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
    prefix=None,  # None表示不使用前缀，'slidingwindow'表示使用滑窗前缀    
    center_crop=True,  # 是否进行中心裁剪
    fold_to_run=None,  # None表示跑所有fold，否则跑指定fold
    dataset_flag=None
):
    inference_engine = build_inference_engine_func()

    # folds从1开始到num_folds
    folds = [fold_to_run] if fold_to_run is not None else list(range(1, num_folds + 1))
        
    dice_style_str = "" if dice_style == 1 else f"_dice_style{dice_style}"

    avg_dice_list = []
    avg_dice_style2_list = []
    avg_dice_post_list = []
    avg_dice_style2_post_list = []

    for fold in folds:
        if prefix == 'slidingwindow':
            ckpt_path = os.path.join(ckpt_dir, f"entire_best_model_fold{fold}{dice_style_str}.pth")
        else:
            ckpt_path = os.path.join(ckpt_dir, f"best_model_fold{fold}{dice_style_str}.pth")
        model = build_model_func(ckpt_path)
        model.to(device)
        model.eval()

        val_list_path = os.path.join(val_cases_dir, f"val_cases_fold{fold}.txt")
        if not os.path.exists(val_list_path):
            logger.warning(f"Validation case list not found for fold {fold}: {val_list_path}")
            continue

        with open(val_list_path, 'r') as f:
            whitelist = [line.strip() for line in f if line.strip()]

        logger.info(f"Running inference for fold {fold} with {len(whitelist)} cases")
        
        # 创建输出目录和概率保存目录
        output_dir = os.path.join(output_base_dir, f"val_fold{fold}_pred")
        os.makedirs(output_dir, exist_ok=True)
        if prob_base_dir:
            prob_save_dir = os.path.join(prob_base_dir, f"fold{fold}")
            os.makedirs(prob_save_dir, exist_ok=True)
        else:
            prob_save_dir = None

        run_inference_func(
            case_dir, output_dir, prob_save_dir, model, inference_engine, device, 
            whitelist, center_crop=center_crop, dataset_flag=dataset_flag
        )

        logger.info(f"Fold {fold} inference done. Results saved in {output_dir}")
        logger.info(f"Probabilities saved in {prob_save_dir}")
        
    # for i, fold in enumerate(folds):
    #     if avg_dice_list[i] is not None:
    #         mean_dice = sum(avg_dice_list[i].values()) / len(avg_dice_list[i])
    #         logger.info(f"Fold {fold} - Average Dice: TC: {avg_dice_list[i]['TC']:.4f}, WT: {avg_dice_list[i]['WT']:.4f}, ET: {avg_dice_list[i]['ET']:.4f} | Mean: {mean_dice:.4f}")            
    #     else:
    #         logger.info(f"Fold {fold} - No valid cases processed.")
            
    # for i, fold in enumerate(folds):
    #     if avg_dice_style2_list[i] is not None:
    #         mean_dice = sum(avg_dice_style2_list[i].values()) / len(avg_dice_style2_list[i])
    #         logger.info(f"Style 2 Fold {fold} - Average Dice: TC: {avg_dice_style2_list[i]['TC']:.4f}, WT: {avg_dice_style2_list[i]['WT']:.4f}, ET: {avg_dice_style2_list[i]['ET']:.4f} | Mean: {mean_dice:.4f}")

    # for i, fold in enumerate(folds):
    #     if avg_dice_post_list[i] is not None:
    #         mean_dice = sum(avg_dice_post_list[i].values()) / len(avg_dice_post_list[i])
    #         logger.info(f"Style 1 with post-processing Fold {fold} - Average Dice: TC: {avg_dice_post_list[i]['TC']:.4f}, WT: {avg_dice_post_list[i]['WT']:.4f}, ET: {avg_dice_post_list[i]['ET']:.4f} | Mean: {mean_dice:.4f}")
            
    # for i, fold in enumerate(folds):
    #     if avg_dice_style2_post_list[i] is not None:
    #         mean_dice = sum(avg_dice_style2_post_list[i].values()) / len(avg_dice_style2_post_list[i])
    #         logger.info(f"Style 2 with post-processing Fold {fold} - Average Dice: TC: {avg_dice_style2_post_list[i]['TC']:.4f}, WT: {avg_dice_style2_post_list[i]['WT']:.4f}, ET: {avg_dice_style2_post_list[i]['ET']:.4f} | Mean: {mean_dice:.4f}")


def inference_BraTS2020_val_data(experiment_id, dice_style, center_crop=True, prefix=None, fold_to_run=None):
    # BraTS 2020 validation data inference
    script_dir = os.path.dirname(os.path.abspath(__file__))
    val_cases_dir = os.path.join(script_dir, 'val_cases')   # 存放验证集case名单txt的文件夹 
    # val_cases_dir = 'val_cases/'   
    ckpt_dir = f"/hpc/ajhz839/checkpoint/experiment_{experiment_id}/"  # 模型ckpt所在目录
    case_dir = "/hpc/ajhz839/data/BraTS2020/MICCAI_BraTS2020_TrainingData/"
    
    dice_style_str = "" if dice_style == 1 else f"_dice_style{dice_style}"
    prefix_str = f"_{prefix}" if prefix else ""
    
    output_base_dir = f"/hpc/ajhz839/validation/BraTS2020_val_pred_exp{experiment_id}{dice_style_str}{prefix_str}/"
    prob_base_dir = None # f"/hpc/ajhz839/validation/BraTS2020_val_prob_folds_exp{experiment_id}{dice_style_str}{prefix_str}/"

    if fold_to_run is None:
        check_all_folds_ckpt_exist(ckpt_dir, dice_style, prefix)
        check_all_folds_val_txt_exist(val_cases_dir)

    logger.info("5 fold validation data inference started.")

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
        dice_style=dice_style,
        prefix=prefix,
        center_crop=center_crop,  # 是否进行中心裁剪
        fold_to_run=fold_to_run,  # 跑全部fold 1~5
    )
    
    logger.info("5 fold validation data inference completed.")
    
    
def inference_BraTS2023_val_data(experiment_id, dice_style, center_crop=True, prefix=None):
    # BraTS 2020 validation data inference
    script_dir = os.path.dirname(os.path.abspath(__file__))
    val_cases_dir = os.path.join(script_dir, 'val_cases')   # 存放验证集case名单txt的文件夹 
    # val_cases_dir = 'val_cases/'   
    ckpt_dir = f"/hpc/ajhz839/checkpoint/experiment_{experiment_id}/"  # 模型ckpt所在目录
    case_dir = "/hpc/ajhz839/data/BraTS2023/train/"

    dice_style_str = "" if dice_style == 1 else f"_dice_style{dice_style}"
    prefix_str = f"_{prefix}" if prefix else ""

    output_base_dir = f"/hpc/ajhz839/validation/BraTS2023_val_pred_exp{experiment_id}{dice_style_str}{prefix_str}/"
    prob_base_dir = None # f"/hpc/ajhz839/validation/BraTS2023_val_prob_folds_exp{experiment_id}{dice_style_str}{prefix_str}/"

    check_all_folds_ckpt_exist(ckpt_dir, dice_style, prefix)
    check_all_folds_val_txt_exist(val_cases_dir)

    logger.info("5 fold validation data inference started.")

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
        dice_style=dice_style,
        prefix=prefix,
        center_crop=center_crop,  # 是否进行中心裁剪
        fold_to_run=None,  # 跑全部fold 1~5
        dataset_flag="BraTS23"  # 指定数据集标志
    )
    
    logger.info("5 fold validation data inference completed.")


def main():
    # BraTS 2020 validation data inference
    experiment_id = 112
    dice_style = 1
    prefix = None
    fold_to_run=1
    inference_BraTS2020_val_data(experiment_id, dice_style, center_crop=True, prefix=prefix, fold_to_run=fold_to_run)
    
    # # BraTS 2023 validation data inference
    # experiment_id = 75
    # dice_style = 2
    # prefix = None
    # inference_BraTS2023_val_data(experiment_id, dice_style, center_crop=True, prefix=prefix)

    
    # # Clinical data inference
    # prob_base_dir = "/hpc/ajhz839/inference/pred/test_prob_folds/"
    # ensemble_output_dir = "/hpc/ajhz839/inference/pred/test_pred_soft_ensemble/"
    # case_dir = "/hpc/ajhz839/inference/clinical_data/"
    # ckpt_dir = "./checkpoint/"

    # soft_ensemble(prob_base_dir, case_dir, ckpt_dir)
    # ensemble_soft_voting(prob_base_dir, case_dir, ensemble_output_dir)
    
    
if __name__ == "__main__":
    main()




