import nibabel as nib
import numpy as np
import torch
import os
from surface_distance import compute_surface_distances, compute_robust_hausdorff

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



def batch_compute_dice(pred_dir, gt_dir):
    all_dice_scores = []

    for pred_file in os.listdir(pred_dir):
        if not pred_file.endswith('_pred_mask.nii.gz'):
            continue

        # 获取病例名
        case_name = pred_file.replace('_pred_mask.nii.gz', '')

        # 构造文件路径
        pred_path = os.path.join(pred_dir, pred_file)
        gt_path = os.path.join(gt_dir, case_name + '.nii.gz')

        if not os.path.exists(gt_path):
            print(f"[Warning] Ground truth not found for {case_name}, skipping.")
            continue

        # 计算 Dice
        dice = compute_dice_from_nifti(pred_path, gt_path)
        all_dice_scores.append(dice)

        print(f"{case_name}: {dice}")

    # 计算平均值
    dice_tc = [d["Dice_TC"] for d in all_dice_scores]
    dice_wt = [d["Dice_WT"] for d in all_dice_scores]
    dice_et = [d["Dice_ET"] for d in all_dice_scores]
    mean_dice = [d["Mean_Dice"] for d in all_dice_scores]

    print("\n=== Average Dice Scores ===")
    print(f"Dice_TC Mean: {np.mean(dice_tc):.4f}")
    print(f"Dice_WT Mean: {np.mean(dice_wt):.4f}")
    print(f"Dice_ET Mean: {np.mean(dice_et):.4f}")
    print(f"Mean_Dice:   {np.mean(mean_dice):.4f}")


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

    print("Sum TC:", pred_tc.sum(), "GT TC:", gt_tc.sum())
    print("Sum WT:", pred_wt.sum(), "GT WT:", gt_wt.sum())
    print("Sum ET:", pred_et.sum(), "GT ET:", gt_et.sum())
    print('Sum NCR/NET:', (pred == 1).sum(), "GT NCR/NET:", (gt == 1).sum())
    print('Sum ED:', (pred == 2).sum(), "GT ED:", (gt == 2).sum())
    print('Sum BG:', (pred == 0).sum(), "GT BG:", (gt == 0).sum())

    return {
        "HD95_TC": round(hd95_tc, 4) if not np.isnan(hd95_tc) else None,
        "HD95_WT": round(hd95_wt, 4) if not np.isnan(hd95_wt) else None,
        "HD95_ET": round(hd95_et, 4) if not np.isnan(hd95_et) else None,
        "Mean_HD95": round(hd95_mean, 4) if not np.isnan(hd95_mean) else None,
    }
    

def batch_compute_hd95(pred_dir, gt_dir, spacing=(1.0, 1.0, 1.0)):
    all_hd95_scores = []

    for pred_file in os.listdir(pred_dir):
        if not pred_file.endswith('_pred_mask.nii.gz'):
            continue

        case_name = pred_file.replace('_pred_mask.nii.gz', '')
        pred_path = os.path.join(pred_dir, pred_file)
        gt_path = os.path.join(gt_dir, case_name + '.nii.gz')

        if not os.path.exists(gt_path):
            print(f"[Warning] Ground truth not found for {case_name}, skipping.")
            continue

        hd95 = compute_hd95_from_nifti(pred_path, gt_path, spacing)
        all_hd95_scores.append(hd95)

        print(f"{case_name}: {hd95}")

    # 计算平均值（忽略 None）
    hd95_tc = [d["HD95_TC"] for d in all_hd95_scores if d["HD95_TC"] is not None]
    hd95_wt = [d["HD95_WT"] for d in all_hd95_scores if d["HD95_WT"] is not None]
    hd95_et = [d["HD95_ET"] for d in all_hd95_scores if d["HD95_ET"] is not None]
    mean_hd95 = [d["Mean_HD95"] for d in all_hd95_scores if d["Mean_HD95"] is not None]

    print("\n=== Average HD95 Scores ===")
    print(f"HD95_TC Mean: {np.mean(hd95_tc):.4f}" if hd95_tc else "HD95_TC Mean: N/A")
    print(f"HD95_WT Mean: {np.mean(hd95_wt):.4f}" if hd95_wt else "HD95_WT Mean: N/A")
    print(f"HD95_ET Mean: {np.mean(hd95_et):.4f}" if hd95_et else "HD95_ET Mean: N/A")
    print(f"Mean_HD95:    {np.mean(mean_hd95):.4f}" if mean_hd95 else "Mean_HD95: N/A")



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


def batch_compute_metrics(pred_dir, gt_root, compute_hd95=False):
    pred_files = sorted([f for f in os.listdir(pred_dir) if f.endswith(".nii.gz")])
    all_dice_scores = []
    all_hd95_scores = []

    for pred_file in pred_files:
        case_name = pred_file.replace("_pred_mask.nii.gz", "")
        print(f"Processing case: {case_name}")
        pred_path = os.path.join(pred_dir, pred_file)
        gt_path = find_gt_path(gt_root, case_name)

        if gt_path is None:
            print(f"[Warning] GT not found for {case_name}")
            continue


        # 计算 Dice
        dice = compute_dice_from_nifti(pred_path, gt_path)
        all_dice_scores.append(dice)

        # 计算 HD95
        if compute_hd95:
            hd95 = compute_hd95_from_nifti(pred_path, gt_path)
            all_hd95_scores.append(hd95)

        print(f"{case_name}: dice: {dice} | HD95: {hd95 if compute_hd95 else 'N/A'}")

    # 计算平均值
    dice_tc = [d["Dice_TC"] for d in all_dice_scores]
    dice_wt = [d["Dice_WT"] for d in all_dice_scores]
    dice_et = [d["Dice_ET"] for d in all_dice_scores]
    mean_dice = [d["Mean_Dice"] for d in all_dice_scores]

    print("\n=== Average Dice Scores ===")
    print(f"Dice_TC Mean: {np.mean(dice_tc):.4f}")
    print(f"Dice_WT Mean: {np.mean(dice_wt):.4f}")
    print(f"Dice_ET Mean: {np.mean(dice_et):.4f}")
    print(f"Mean_Dice:   {np.mean(mean_dice):.4f}")
    
    # 计算平均值（忽略 None）
    hd95_tc = [d["HD95_TC"] for d in all_hd95_scores if d["HD95_TC"] is not None]
    hd95_wt = [d["HD95_WT"] for d in all_hd95_scores if d["HD95_WT"] is not None]
    hd95_et = [d["HD95_ET"] for d in all_hd95_scores if d["HD95_ET"] is not None]
    mean_hd95 = [d["Mean_HD95"] for d in all_hd95_scores if d["Mean_HD95"] is not None]

    print("\n=== Average HD95 Scores ===")
    print(f"HD95_TC Mean: {np.mean(hd95_tc):.4f}" if hd95_tc else "HD95_TC Mean: N/A")
    print(f"HD95_WT Mean: {np.mean(hd95_wt):.4f}" if hd95_wt else "HD95_WT Mean: N/A")
    print(f"HD95_ET Mean: {np.mean(hd95_et):.4f}" if hd95_et else "HD95_ET Mean: N/A")
    print(f"Mean_HD95:    {np.mean(mean_hd95):.4f}" if mean_hd95 else "Mean_HD95: N/A")


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
        
        # # BraTS 2020 Validation
        # gt_root = './data/BraTS2020/MICCAI_BraTS2020_TrainingData'
        # pred_dir = './Pred/BraTS2020/validation_dataset/BraTS2020_val_pred_exp69/val_fold5_pred'
        # batch_compute_metrics(pred_dir, gt_root, compute_hd95=False)

        # BraTS 2020 Test
        gt_root = './data/BraTS2020/MICCAI_BraTS2020_TrainingData'
        pred_dir = './Pred/BraTS2020/test_dataset/test_pred_soft_ensemble_exp69'
        batch_compute_metrics(pred_dir, gt_root, compute_hd95=True)


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