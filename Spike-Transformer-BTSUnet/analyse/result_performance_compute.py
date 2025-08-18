import nibabel as nib
import numpy as np
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import json
from inference.inference_utils import restore_to_original_shape, convert_label_to_onehot
from inference.inference_metrics import (
    compute_dice_from_nifti, load_nifti_as_tensor, compute_soft_dice,
    compute_hd95_from_nifti, compute_sensitivity_specificity_from_nifti,
)


def find_gt_path(gt_root, case_name):
    """
    自动判断是否存在 HGG/LGG 子文件夹，兼容 BraTS2018 和 BraTS2020 目录结构。
    """
    # BraTS2020 结构（无 HGG/LGG）
    candidate_path = os.path.join(gt_root, case_name, f"{case_name}_seg.nii")
    if os.path.exists(candidate_path):
        return candidate_path

    # BraTS2018 结构（有 HGG/LGG）
    for grade in ["HGG", "LGG"]:
        candidate_path = os.path.join(gt_root, grade, case_name, f"{case_name}_seg.nii")
        if os.path.exists(candidate_path):
            return candidate_path

    print(f"GT not found for {case_name} in {gt_root}")
    return None


def print_avg_metrics(metrics_list, prefix="", keys=None, safe=True):
    """
    通用平均指标打印函数
    
    Args:
        metrics_list: List[Dict[str, float or None]]
        prefix: 打印标题前缀
        keys: 指定要处理的key列表（默认为从metrics_list[0].keys()自动获取）
        safe: 若为True，会跳过None；False时直接抛错
    """
    if not metrics_list:
        print(f"\n=== {prefix} Scores ===")
        print("No metrics to display.")
        return

    if keys is None:
        keys = metrics_list[0].keys()

    print(f"\n=== Average {prefix} Scores ===")
    for k in keys:
        try:
            vals = [m[k] for m in metrics_list if not safe or m[k] is not None]
            if len(vals) == 0:
                print(f"{k}: N/A")
            else:
                print(f"{k}: {np.mean(vals):.4f}")
        except Exception as e:
            print(f"{k}: Error - {e}")



def batch_compute_metrics(
    pred_dir, 
    gt_root,
    prob_dir=None,
    is_resemble=False,
    metric_obj=None,
    compute_hd95=False,
    compute_sensitivity_specificity=False,
    metadata_json_path=None
):
    pred_files = sorted([f for f in os.listdir(pred_dir) if f.endswith(".nii.gz")])
    # pred_files = sorted([f for f in os.listdir(pred_dir) if f.endswith(".nii")])    
    all_dice_scores = []
    compare_mode = "crop"  # 默认比较模式为裁剪
    all_soft_dice_scores = []
    all_hd95_scores = []
    all_sensitivity_specificity_scores = []

    if metadata_json_path:
        with open(metadata_json_path, "r") as f:
            case_metadata = json.load(f)
    else:
        case_metadata = None

    for pred_file in pred_files:
        case_name = pred_file.replace("_pred_mask.nii.gz", "")
        # case_name = pred_file.replace(".nii", "")
        print(f"Processing case: {case_name}")
        pred_path = os.path.join(pred_dir, pred_file)
        gt_path = find_gt_path(gt_root, case_name)

        if gt_path is None:
            print(f"[Warning] GT not found for {case_name}")
            continue


        # 计算 Hard Dice
        dice = compute_dice_from_nifti(pred_path, gt_path)
        all_dice_scores.append(dice)
        
        
        # Soft Dice (if probability maps provided)
        if prob_dir is not None and metric_obj is not None:
            if is_resemble:
                prob_path = None
                for fold in range(5):
                    candidate = os.path.join(prob_dir, f"fold{fold}", case_name + "_prob.npy")
                    if os.path.exists(candidate):
                        prob_path = candidate
                        break
                if prob_path is None:
                    print(f"[Warning] Probability file not found for {case_name} in any fold.")
                    continue
            else:
                prob_path = os.path.join(prob_dir, case_name + "_prob.npy")
                if not os.path.exists(prob_path):
                    print(f"[Warning] Probability file not found for {case_name}.")
                    continue

            prob_np = np.load(prob_path)  # (3, D, H, W) 裁剪区域概率图

            gt_tensor_full = load_nifti_as_tensor(gt_path).long()  # full shape GT tensor
            gt_tensor_full = gt_tensor_full.permute(2, 0, 1).contiguous()  # (D,H,W)

            if compare_mode == "full":
                # 恢复预测概率到全图尺寸
                if case_metadata and case_name in case_metadata:
                    original_shape = tuple(case_metadata[case_name]["original_shape"])
                    crop_start = tuple(case_metadata[case_name]["crop_start"])
                    restored_prob = []
                    for c in range(prob_np.shape[0]):
                        restored_c = restore_to_original_shape(prob_np[c], original_shape, crop_start)
                        restored_prob.append(restored_c)
                    prob_np_full = np.stack(restored_prob, axis=0)
                    prob_tensor_for_compare = torch.from_numpy(prob_np_full).float()
                    gt_for_compare = gt_tensor_full
                else:
                    print(f"[Warning] Metadata missing for {case_name}, cannot restore prob to full size, skipping.")
                    continue

            elif compare_mode == "crop":
                # 不恢复预测概率，用原裁剪区概率；裁剪GT到概率图尺寸
                if case_metadata and case_name in case_metadata:
                    crop_start = case_metadata[case_name]["crop_start"]
                    crop_size = prob_np.shape[1:]  # (D,H,W)
                    crop_end = tuple(crop_start[i] + crop_size[i] for i in range(3))
                    gt_for_compare = gt_tensor_full[crop_start[0]:crop_end[0],
                                                   crop_start[1]:crop_end[1],
                                                   crop_start[2]:crop_end[2]]
                    prob_tensor_for_compare = torch.from_numpy(prob_np).float()
                else:
                    print(f"[Warning] Metadata missing for {case_name}, cannot crop GT, skipping.")
                    continue
            else:
                raise ValueError(f"Unsupported compare_mode: {compare_mode}")

            soft_dice = compute_soft_dice(prob_tensor_for_compare, gt_for_compare, metric_obj)
            all_soft_dice_scores.append(soft_dice)
            print(f"{case_name}: Soft Dice: {soft_dice}")

            
        # 计算 HD95
        if compute_hd95:
            hd95 = compute_hd95_from_nifti(pred_path, gt_path)
            all_hd95_scores.append(hd95)
            
        # 计算敏感性和特异性
        if compute_sensitivity_specificity:
            scores = compute_sensitivity_specificity_from_nifti(pred_path, gt_path)
            all_sensitivity_specificity_scores.append(scores)

        print(f"{case_name}: dice: {dice} | HD95: {hd95 if compute_hd95 else 'N/A'}")
        if compute_sensitivity_specificity:
            print(f"{case_name}: sensitivity & specificity: {scores}")

    # 打印平均值
    print_avg_metrics(all_dice_scores, prefix="Hard Dice")
    if all_soft_dice_scores:
        print_avg_metrics(all_soft_dice_scores, prefix="Soft Dice")

    if compute_hd95:
        print_avg_metrics(all_hd95_scores, prefix="HD95", keys=["HD95_WT", "HD95_TC", "HD95_ET", "Mean_HD95"])

    if compute_sensitivity_specificity:
        print_avg_metrics(
            all_sensitivity_specificity_scores,
            prefix="Sensitivity & Specificity",
            keys=[
                "Sensitivity_WT", "Sensitivity_TC", "Sensitivity_ET", "Mean_Sensitivity",
                "Specificity_WT", "Specificity_TC", "Specificity_ET", "Mean_Specificity"
            ]
        )
    
    return all_dice_scores, all_soft_dice_scores, all_hd95_scores, all_sensitivity_specificity_scores


def inference_dice_compute_for_brats20_val_data(experiment_index, dice_score_style, prefix=None, metric_obj=None, metadata_json_path = None):
    """
    计算验证集数据的 Dice 分数
    """

    all_fold_dice_scores = []
    all_fold_soft_dice_scores = []

    gt_root = '../../data/BraTS2020/MICCAI_BraTS2020_TrainingData'
    dice_score_style_str = "" if dice_score_style == 1 else f"_dice_style{dice_score_style}"
    prefix_str = f"_{prefix}" if prefix else ""
    for fold_index in range(1, 6):
        print(f"Processing fold {fold_index} for experiment {experiment_index} with style {dice_score_style}")
        pred_dir = f'../../Project/Pred/BraTS2020/validation_dataset/BraTS2020_val_pred_exp{experiment_index}{dice_score_style_str}{prefix_str}/val_fold{fold_index}_pred'
        prob_dir = f'../../Project/Pred/BraTS2020/validation_dataset/BraTS2020_val_prob_folds_exp{experiment_index}{dice_score_style_str}{prefix_str}/fold{fold_index}'

        print(f"Pred dir: {pred_dir}")
        all_dice_scores, all_soft_dice_scores, _, _ = batch_compute_metrics(
            pred_dir=pred_dir,
            gt_root=gt_root,
            prob_dir=prob_dir, # 概率图
            is_resemble=False, # 是否是5折重叠的概率图
            metric_obj=metric_obj, # BratsDiceMetric 实例
            compute_hd95=False,
            compute_sensitivity_specificity=False,
            metadata_json_path=metadata_json_path
            )
        
        all_fold_dice_scores.append(all_dice_scores)
        all_fold_soft_dice_scores.append(all_soft_dice_scores)

    # 打印所有折的平均值
    print("\n============================================")
    print("\n============================================")
    print("\n=== Average Dice Scores across all folds ===")
    print("\n============================================")
    print("\n============================================")
    for i in range(5):
        print(f"\nFold {i+1}:")
        print_avg_metrics(all_fold_dice_scores[i], prefix="Hard Dice")
        if all_soft_dice_scores:
            print_avg_metrics(all_fold_soft_dice_scores[i], prefix="Soft Dice")


def inference_dice_compute_for_brats20_test_data(experiment_index, dice_score_style, prefix=None, metric_obj=None, metadata_json_path = None):   
        gt_root = '../../data/BraTS2020/MICCAI_BraTS2020_TrainingData'
        dice_score_style_str = "" if dice_score_style == 1 else f"_dice_style{dice_score_style}"
        prefix_str = f"_{prefix}" if prefix else ""
        pred_dir = f'../../Project/Pred/BraTS2020/test_dataset/test_pred_soft_ensemble_exp{experiment_index}{dice_score_style_str}{prefix_str}'
        prob_dir = None # f'./Pred/BraTS2020/test_dataset/test_prob_soft_ensemble_exp{experiment_index}'
        metric_obj = None # BratsDiceMetric()
        
        print(f"Pred dir: {pred_dir}")

        batch_compute_metrics(
            pred_dir=pred_dir,
            gt_root=gt_root,
            prob_dir=prob_dir, # 概率图
            is_resemble=True, # 是否是5折重叠的概率图
            metric_obj=metric_obj, # BratsDiceMetric 实例
            compute_hd95=True,
            compute_sensitivity_specificity=True
            )
        
        
def inference_dice_compute_for_brats23_val_data(experiment_index, dice_score_style, prefix=None, metric_obj=None, metadata_json_path = None):
    """
    计算验证集数据的 Dice 分数
    """

    all_fold_dice_scores = []
    all_fold_soft_dice_scores = []

    gt_root = '../../data/BraTS2020/MICCAI_BraTS2020_TrainingData'
    dice_score_style_str = "" if dice_score_style == 1 else f"_dice_style{dice_score_style}"
    prefix_str = f"_{prefix}" if prefix else ""
    for fold_index in range(1, 6):
        print(f"Processing fold {fold_index} for experiment {experiment_index} with style {dice_score_style}")
        pred_dir = f'../../Project/Pred/BraTS2020/validation_dataset/BraTS2020_val_pred_exp{experiment_index}{dice_score_style_str}{prefix_str}/val_fold{fold_index}_pred'
        prob_dir = f'../../Project/Pred/BraTS2020/validation_dataset/BraTS2020_val_prob_folds_exp{experiment_index}{dice_score_style_str}{prefix_str}/fold{fold_index}'

        print(f"Pred dir: {pred_dir}")
        all_dice_scores, all_soft_dice_scores, _, _ = batch_compute_metrics(
            pred_dir=pred_dir,
            gt_root=gt_root,
            prob_dir=prob_dir, # 概率图
            is_resemble=False, # 是否是5折重叠的概率图
            metric_obj=metric_obj, # BratsDiceMetric 实例
            compute_hd95=False,
            compute_sensitivity_specificity=False,
            metadata_json_path=metadata_json_path
            )
        
        all_fold_dice_scores.append(all_dice_scores)
        all_fold_soft_dice_scores.append(all_soft_dice_scores)

    # 打印所有折的平均值
    print("\n============================================")
    print("\n============================================")
    print("\n=== Average Dice Scores across all folds ===")
    print("\n============================================")
    print("\n============================================")
    for i in range(5):
        print(f"\nFold {i+1}:")
        print_avg_metrics(all_fold_dice_scores[i], prefix="Hard Dice")
        if all_soft_dice_scores:
            print_avg_metrics(all_fold_soft_dice_scores[i], prefix="Soft Dice")
        
        
        
        
        
def inference_dice_compute_nnunet_val_data():
    """
    计算验证集数据的 Dice 分数
    """

    all_fold_dice_scores = []
    
    gt_root = '../../data/BraTS2020/MICCAI_BraTS2020_TrainingData'
    for fold_index in range(0, 5):
        print(f"Processing fold {fold_index}")
        pred_dir = f'../../Compared_exp/Result/nnUnet/BraTS_2020/fold_{fold_index}/validation'

        print(f"Pred dir: {pred_dir}")
        all_dice_scores, all_soft_dice_scores, _, _, _ = batch_compute_metrics(
            pred_dir=pred_dir,
            gt_root=gt_root,
            prob_dir=None, # 概率图
            is_resemble=False, # 是否是5折重叠的概率图
            metric_obj=None, # BratsDiceMetric 实例
            compute_hd95=False,
            compute_sensitivity_specificity=False,
            metadata_json_path=None
            )
        
        all_fold_dice_scores.append(all_dice_scores)

    # 打印所有折的平均值
    print("\n============================================")
    print("\n============================================")
    print("\n=== Average Dice Scores across all folds ===")
    print("\n============================================")
    print("\n============================================")
    for i in range(5):
        print(f"\nFold {i+1}:")
        print_avg_metrics(all_fold_dice_scores[i], prefix="Hard Dice")    

def main():
    batch_compute = True
    
    if batch_compute:
        # 批量计算 Dice
        # # BraTS 2018 val dataset
        # gt_dir = './Pred/nnUNetTrainer'
        # pred_dir = './Pred/test_pred_experiment56'
        # batch_compute_dice(gt_dir, pred_dir)
        
        # # BraTS 2018 Training set
        # gt_root = './data/BraTS2018/MICCAI_BraTS_2018_Data_Training'
        # pred_dir = './Pred/val_fold2_pred_experiment56'
        # batch_compute_dice_trainingset(gt_root, pred_dir)
        

        # BraTS 2020 Validation or Test
        mode = 'test'  # 'val' or 'test'
        experiment_index = 91
        dice_score_style = 1
        prefix = None
        if mode == 'val':
            inference_dice_compute_for_brats20_val_data(experiment_index, dice_score_style, prefix, metric_obj=None, metadata_json_path = None)
        elif mode == 'test':
            inference_dice_compute_for_brats20_test_data(experiment_index, dice_score_style, prefix,metric_obj=None, metadata_json_path = None)


        # # BraTS 2023 Validation or Test
        # mode = 'val'  # 'val' or 'test'
        # experiment_index = 86
        # dice_score_style = 2
        # prefix = None
        # if mode == 'val':
        #     inference_dice_compute_for_brats23_val_data(experiment_index, dice_score_style, prefix, metric_obj=None, metadata_json_path = None)
        # elif mode == 'test':
        #     inference_dice_compute_for_brats20_test_data(experiment_index, dice_score_style, prefix,metric_obj=None, metadata_json_path = None)



        
        # inference_dice_compute_nnunet_val_data()

    else:
        # compute single case Dice
        data_dir = './data/MICCAI_BraTS_2018_Data_Training/HGG/Brats18_2013_27_1'
        pred_dir = './pred'
        case_name = os.path.basename(data_dir)

        gt_mask_path = os.path.join(data_dir, case_name + '_seg.nii')     # ground truth
        pred_mask_path = os.path.join(pred_dir, case_name + '_pred_mask.nii.gz') # model prediction _pred_mask_constant_05.nii
        

        dice_results = compute_dice_from_nifti(pred_mask_path, gt_mask_path)
        print(dice_results)

if __name__ == "__main__":
    main()