def batch_compute_metrics(
    pred_dir, 
    gt_root,
    prob_dir=None,
    metric_obj=None,
    compute_hd95=False,
    compute_sensitivity_specificity=False,
    folded_prob_dir=False,
):
    pred_files = sorted([f for f in os.listdir(pred_dir) if f.endswith(".nii.gz")])
    all_dice_scores = []
    all_soft_dice_scores = []
    all_hd95_scores = []
    all_sensitivity_specificity_scores = []

    for pred_file in pred_files:
        case_name = pred_file.replace("_pred_mask.nii.gz", "")
        print(f"\nProcessing case: {case_name}")
        pred_path = os.path.join(pred_dir, pred_file)
        gt_path = find_gt_path(gt_root, case_name)

        if gt_path is None:
            print(f"[Warning] GT not found for {case_name}")
            continue

        # ---- Hard Dice ----
        dice = compute_dice_from_nifti(pred_path, gt_path)
        all_dice_scores.append(dice)

        # ---- Soft Dice ----
        if prob_dir is not None and metric_obj is not None:
            if folded_prob_dir:
                prob_path = None
                # 搜索5折中的匹配文件
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

            prob_np = np.load(prob_path)  # shape (C, D, H, W)
            gt_tensor = load_nifti_as_tensor(gt_path).long()
            C = prob_np.shape[0]
            gt_onehot = torch.nn.functional.one_hot(gt_tensor, num_classes=C).permute(3, 0, 1, 2).float()
            prob_tensor = torch.from_numpy(prob_np).float()
            soft_dice = compute_soft_dice(prob_tensor, gt_onehot, metric_obj)
            all_soft_dice_scores.append(soft_dice)

        # ----
