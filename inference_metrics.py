import os
os.chdir(os.path.dirname(__file__))
import torch
import nibabel as nib
import numpy as np
from surface_distance import compute_surface_distances, compute_robust_hausdorff
from inference_utils import convert_label_to_onehot

        
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