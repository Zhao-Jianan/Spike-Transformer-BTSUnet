import nibabel as nib
import numpy as np
import torch
import os
from surface_distance import compute_surface_distances, compute_robust_hausdorff
from metrics import BratsDiceMetric
import json
from inference_utils import restore_to_original_shape, convert_gt_to_structural_onehot

def load_nifti_as_tensor(path):
    """Load NIfTI and return tensor of shape (D, H, W)"""
    nii = nib.load(path)
    data = nii.get_fdata().astype(np.uint8)
    if data.ndim == 4:
        data = data.squeeze()  # in case it's (1, D, H, W)
    return torch.from_numpy(data)

def dice_score(pred_mask, gt_mask):
    intersection = (pred_mask & gt_mask).sum().float()
    denominator = pred_mask.sum().float() + gt_mask.sum().float()
    if denominator == 0:
        return torch.tensor(1.0)
    return 2.0 * intersection / denominator

def compute_dice_from_nifti(pred_path, gt_path):
    pred = load_nifti_as_tensor(pred_path)
    gt = load_nifti_as_tensor(gt_path)
    
    # Convert to label space if needed
    pred = pred.cpu()
    gt = gt.cpu()

    # TC: label 1 or 4
    pred_tc = (pred == 1) | (pred == 4)
    gt_tc   = (gt == 1)   | (gt == 4)

    # WT: label 1 or 2 or 4
    pred_wt = (pred == 1) | (pred == 2) | (pred == 4)
    gt_wt   = (gt == 1)   | (gt == 2)   | (gt == 4)

    # ET: label 4
    pred_et = (pred == 4)
    gt_et   = (gt == 4)
    

    print("Sum TC:", pred_tc.sum().item(), "GT TC:", gt_tc.sum().item())
    print("Sum WT:", pred_wt.sum().item(), "GT WT:", gt_wt.sum().item())
    print("Sum ET:", pred_et.sum().item(), "GT ET:", gt_et.sum().item())
    print('Sum NCR/NET:', (pred == 1).sum().item(), "GT NCR/NET:", (gt == 1).sum().item())
    print('Sum ED:', (pred == 2).sum().item(), "GT ED:", (gt == 2).sum().item())
    print('Sum BG:', (pred == 0).sum().item(), "GT BG:", (gt == 0).sum().item())

    dice_tc = dice_score(pred_tc, gt_tc)
    dice_wt = dice_score(pred_wt, gt_wt)
    dice_et = dice_score(pred_et, gt_et)
    mean_dice = (dice_tc + dice_wt + dice_et) / 3

    return {
        "Dice_TC": round(dice_tc.item(), 4),
        "Dice_WT": round(dice_wt.item(), 4),
        "Dice_ET": round(dice_et.item(), 4),
        "Mean_Dice": round(mean_dice.item(), 4),
    }




def dice_score_braTS_style(pred, target, eps=1e-5):
    def compute_dice(p, t):
        inter = (p * t).sum()
        union = p.sum() + t.sum()
        return (2. * inter + eps) / (union + eps)

    pred_prob = torch.sigmoid(pred)
    pred_bin = (pred_prob > 0.5).float()
    target_bin = target.float()

    dice_tc = compute_dice(pred_bin[0], target_bin[0])
    dice_wt = compute_dice(pred_bin[1], target_bin[1])
    dice_et = compute_dice(pred_bin[2], target_bin[2])

    return dice_tc, dice_wt, dice_et


def convert_label_to_onehot(label):
    """
    label: Tensor [D, H, W], value in {0, 1, 2, 4}
    return: one-hot [3, D, H, W], corresponding to TC, WT, ET
    """
    tc = ((label == 1) | (label == 4)).float()  # TC = label 1 or 4
    wt = ((label == 1) | (label == 2) | (label == 4)).float()  # WT = label 1 or 2 or 4
    et = (label == 4).float()  # ET = label 4

    return torch.stack([tc, wt, et], dim=0)  # [3, D, H, W]


def compute_dice_from_nifti_braTS_style(pred_path, gt_path):
    pred = load_nifti_as_tensor(pred_path).cpu()  # [D, H, W]
    gt = load_nifti_as_tensor(gt_path).cpu()      # [D, H, W]

    pred_onehot = convert_label_to_onehot(pred)   # [3, D, H, W]
    gt_onehot = convert_label_to_onehot(gt)       # [3, D, H, W]

    # Add dummy channel to mimic network output shape: [1, 3, D, H, W]
    pred_input = pred_onehot.unsqueeze(0) * 10.0  # Logit-style input to mimic sigmoid -> 0.5 threshold
    gt_input = gt_onehot.unsqueeze(0)             # Ground truth one-hot

    # Compute hard Dice as in dice_score_braTS
    dice_tc, dice_wt, dice_et = dice_score_braTS_style(pred_input[0], gt_input[0])

    mean_dice = (dice_tc + dice_wt + dice_et) / 3

    return {
        "Dice_TC": round(dice_tc.item(), 4),
        "Dice_WT": round(dice_wt.item(), 4),
        "Dice_ET": round(dice_et.item(), 4),
        "Mean_Dice": round(mean_dice.item(), 4),
    }



def compute_soft_dice(prob_tensor, gt_tensor, metric_obj=None):
    """
    prob_tensor: torch.Tensor, shape [3, D, H, W], soft prediction (float)
    gt_tensor:   torch.Tensor, shape [D, H, W], GT label (int, {0,1,2,4})
    return: dict: {'WT': dice1, 'TC': dice2, 'ET': dice3}
    """
    eps = 1e-5
    soft_dice = {}

    gt_tc = ((gt_tensor == 1) | (gt_tensor == 4)).float()  # TC: label 1 or 4
    gt_wt = ((gt_tensor == 1) | (gt_tensor == 2) | (gt_tensor == 4)).float()  # WT: 1,2,4
    gt_et = (gt_tensor == 4).float()

    pred_tc = prob_tensor[0]  # shape: (D, H, W)
    pred_wt = prob_tensor[1]
    pred_et = prob_tensor[2]

    def dice(pred, target):
        inter = (pred * target).sum()
        union = pred.sum() + target.sum()
        return (2.0 * inter + eps) / (union + eps)

    soft_dice["WT"] = dice(pred_wt, gt_wt)
    soft_dice["TC"] = dice(pred_tc, gt_tc)
    soft_dice["ET"] = dice(pred_et, gt_et)

    return soft_dice


def compute_hd95_from_nifti(pred_path, gt_path, spacing=(1.0, 1.0, 1.0)):
    pred = nib.load(pred_path).get_fdata().astype(np.uint8)
    gt = nib.load(gt_path).get_fdata().astype(np.uint8)

    # TC: label 1 or 4
    pred_tc = np.isin(pred, [1, 4])
    gt_tc   = np.isin(gt, [1, 4])

    # WT: label 1 or 2 or 4
    pred_wt = np.isin(pred, [1, 2, 4])
    gt_wt   = np.isin(gt, [1, 2, 4])

    # ET: label 4
    pred_et = (pred == 4)
    gt_et   = (gt == 4)

    def _compute_hd(pred_mask, gt_mask):
        if not np.any(pred_mask) or not np.any(gt_mask):
            return np.nan
        distances = compute_surface_distances(gt_mask, pred_mask, spacing)
        return compute_robust_hausdorff(distances, percent=95)

    hd95_tc = _compute_hd(pred_tc, gt_tc)
    hd95_wt = _compute_hd(pred_wt, gt_wt)
    hd95_et = _compute_hd(pred_et, gt_et)
    hd95_mean = np.nanmean([hd95_tc, hd95_wt, hd95_et])

    return {
        "HD95_TC": round(hd95_tc, 4) if not np.isnan(hd95_tc) else None,
        "HD95_WT": round(hd95_wt, 4) if not np.isnan(hd95_wt) else None,
        "HD95_ET": round(hd95_et, 4) if not np.isnan(hd95_et) else None,
        "Mean_HD95": round(hd95_mean, 4) if not np.isnan(hd95_mean) else None,
    }


def compute_sensitivity_specificity_from_nifti(pred_path, gt_path):
    pred = nib.load(pred_path).get_fdata().astype(np.uint8)
    gt = nib.load(gt_path).get_fdata().astype(np.uint8)

    # TC: label 1 or 4
    pred_tc = np.isin(pred, [1, 4])
    gt_tc   = np.isin(gt, [1, 4])

    # WT: label 1 or 2 or 4
    pred_wt = np.isin(pred, [1, 2, 4])
    gt_wt   = np.isin(gt, [1, 2, 4])

    # ET: label 4
    pred_et = (pred == 4)
    gt_et   = (gt == 4)

    def _compute_metrics(pred_mask, gt_mask):
        TP = np.logical_and(pred_mask, gt_mask).sum()
        FN = np.logical_and(np.logical_not(pred_mask), gt_mask).sum()
        TN = np.logical_and(np.logical_not(pred_mask), np.logical_not(gt_mask)).sum()
        FP = np.logical_and(pred_mask, np.logical_not(gt_mask)).sum()

        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else np.nan
        specificity = TN / (TN + FP) if (TN + FP) > 0 else np.nan

        return sensitivity, specificity

    sens_tc, spec_tc = _compute_metrics(pred_tc, gt_tc)
    sens_wt, spec_wt = _compute_metrics(pred_wt, gt_wt)
    sens_et, spec_et = _compute_metrics(pred_et, gt_et)

    mean_sens = np.nanmean([sens_tc, sens_wt, sens_et])
    mean_spec = np.nanmean([spec_tc, spec_wt, spec_et])

    return {
        "Sensitivity_TC": round(sens_tc, 4) if not np.isnan(sens_tc) else None,
        "Specificity_TC": round(spec_tc, 4) if not np.isnan(spec_tc) else None,
        "Sensitivity_WT": round(sens_wt, 4) if not np.isnan(sens_wt) else None,
        "Specificity_WT": round(spec_wt, 4) if not np.isnan(spec_wt) else None,
        "Sensitivity_ET": round(sens_et, 4) if not np.isnan(sens_et) else None,
        "Specificity_ET": round(spec_et, 4) if not np.isnan(spec_et) else None,
        "Mean_Sensitivity": round(mean_sens, 4) if not np.isnan(mean_sens) else None,
        "Mean_Specificity": round(mean_spec, 4) if not np.isnan(mean_spec) else None,
    }




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
        print(f"Processing case: {case_name}")
        pred_path = os.path.join(pred_dir, pred_file)
        gt_path = find_gt_path(gt_root, case_name)

        if gt_path is None:
            print(f"[Warning] GT not found for {case_name}")
            continue


        # 计算 Hard Dice
        dice = compute_dice_from_nifti(pred_path, gt_path)
        # dice = compute_dice_from_nifti_braTS_style(pred_path, gt_path)
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

    
    
    


# TODO: HD95 Computing


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
        
        # BraTS 2020 Validation
        # experiment_index = 71
        # fold_index = 1
        # gt_root = './data/BraTS2020/MICCAI_BraTS2020_TrainingData'
        # pred_dir = f'./Pred/BraTS2020/validation_dataset/BraTS2020_val_pred_exp{experiment_index}/val_fold{fold_index}_pred'
        # prob_dir = f'./Pred/BraTS2020/validation_dataset/BraTS2020_val_prob_folds_exp{experiment_index}/fold{fold_index}'
        # metric_obj = BratsDiceMetric()
        # metadata_json_path = f'./Pred/BraTS2020/validation_dataset/BraTS2020_val_prob_folds_exp{experiment_index}/metadata.json'

        # batch_compute_metrics(
        #     pred_dir=pred_dir,
        #     gt_root=gt_root,
        #     prob_dir=prob_dir, # 概率图
        #     is_resemble=False, # 是否是5折重叠的概率图
        #     metric_obj=metric_obj, # BratsDiceMetric 实例
        #     compute_hd95=False,
        #     compute_sensitivity_specificity=False,
        #     metadata_json_path=metadata_json_path
        #     )
        

        # BraTS 2020 Test
        experiment_index = 71
        gt_root = './data/BraTS2020/MICCAI_BraTS2020_TrainingData'
        pred_dir = f'./Pred/BraTS2020/test_dataset/test_pred_soft_ensemble_exp{experiment_index}'
        prob_dir = None # f'./Pred/BraTS2020/test_dataset/test_prob_soft_ensemble_exp{experiment_index}'
        metric_obj = None # BratsDiceMetric()

        batch_compute_metrics(
            pred_dir=pred_dir,
            gt_root=gt_root,
            prob_dir=prob_dir, # 概率图
            is_resemble=True, # 是否是5折重叠的概率图
            metric_obj=metric_obj, # BratsDiceMetric 实例
            compute_hd95=True,
            compute_sensitivity_specificity=True
            )


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