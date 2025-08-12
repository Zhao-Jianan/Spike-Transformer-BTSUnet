def run_inference_soft_single(case_dir, save_dir, prob_save_dir, model, inference_engine, device, center_crop=True, verbose=False):
    os.makedirs(save_dir, exist_ok=True)
    if prob_save_dir:
        os.makedirs(prob_save_dir, exist_ok=True)
    
    # ==== Step 1: 推理并获取概率图与标签 ====
    prob, pred_tensor, gt_tensor, raw_gt_tensor_hwd, metadata = pred_single_case(
        case_dir, model, inference_engine, device, center_crop=center_crop
    )
    case_name = os.path.basename(case_dir)

    # ==== Step 2: 保存概率图 & metadata ====
    if prob_save_dir:
        prob_save_path = os.path.join(prob_save_dir, f"{case_name}_prob.npy")
        np.save(prob_save_path, prob)
        if verbose:
            print(f"[INFO] Saved probability map: {prob_save_path}")

        prob_base_dir = os.path.dirname(prob_save_dir.rstrip("/"))
        metadata_json_path = os.path.join(prob_base_dir, "metadata.json")

        # 读取并更新 metadata
        all_metadata = {}
        if os.path.exists(metadata_json_path):
            with open(metadata_json_path, "r") as f:
                all_metadata = json.load(f)
        all_metadata[case_name] = {
            "original_shape": metadata["original_shape"],
            "crop_start": metadata["crop_start"]
        }
        with open(metadata_json_path, "w") as f:
            json.dump(all_metadata, f, indent=2)

    # ==== Step 3: 计算 Dice（原始预测） ====
    dice_dict = dice_score_braTS_per_sample_avg(pred_tensor, gt_tensor)
    pred_tensor_style = pred_tensor.squeeze(0)  # [3, D, H, W]
    gt_tensor_style = gt_tensor.squeeze(0)
    dice_dict_style2 = dice_score_braTS_style(pred_tensor_style, gt_tensor_style)

    if verbose:
        print(f"[Dice] Case {case_name} | TC: {dice_dict['TC']:.4f}, WT: {dice_dict['WT']:.4f}, ET: {dice_dict['ET']:.4f}")
        print(f"[Dice Style] Case {case_name} | TC: {dice_dict_style2[0]:.4f}, WT: {dice_dict_style2[1]:.4f}, ET: {dice_dict_style2[2]:.4f}")

    # ==== Step 4: 转换预测结果为标签并还原到原始形状 ====
    label_np = convert_prediction_to_label_suppress_fp(prob)  # (D, H, W)
    label_tensor = torch.from_numpy(label_np)

    if center_crop:
        restored_label = restore_to_original_shape(
            label_tensor, metadata["original_shape"], metadata["crop_start"]
        )
    else:
        restored_label = label_tensor

    # ==== Step 5: 后处理 & 保存 ====
    # 转为 (H, W, D)
    final_label = restored_label.permute(1, 2, 0)
    final_label = postprocess_brats_label_nnstyle(final_label)

    ref_nii_path = os.path.join(case_dir, f"{case_name}_{cfg.modalities[cfg.modalities.index('t1ce')]}.nii")
    ref_nii = nib.load(ref_nii_path)
    pred_nii = nib.Nifti1Image(final_label, affine=ref_nii.affine, header=ref_nii.header)
    nib.save(pred_nii, os.path.join(save_dir, f"{case_name}_pred_mask.nii.gz"))

    # ==== Step 6: 后处理后 Dice ====
    # 确保是 (D, H, W)
    if final_label.shape[0] != raw_gt_tensor_hwd.shape[2]:
        final_label = np.transpose(final_label, (2, 0, 1))

    final_label_onehot = convert_label_to_onehot(torch.from_numpy(final_label).long()).unsqueeze(0)
    final_gt_onehot = convert_label_to_onehot(raw_gt_tensor_hwd.permute(2, 0, 1)).unsqueeze(0)

    dice_dict_post = dice_score_braTS_per_sample_avg(final_label_onehot, final_gt_onehot)
    dice_dict_style2_post = dice_score_braTS_style(final_label_onehot.squeeze(0), final_gt_onehot.squeeze(0))

    if verbose:
        print(f"[Dice Post] Case {case_name} | TC: {dice_dict_post['TC']:.4f}, WT: {dice_dict_post['WT']:.4f}, ET: {dice_dict_post['ET']:.4f}")
        print(f"[Dice Style Post] Case {case_name} | TC: {dice_dict_style2_post[0]:.4f}, WT: {dice_dict_style2_post[1]:.4f}, ET: {dice_dict_style2_post[2]:.4f}")

    return dice_dict, dice_dict_style2, dice_dict_post, dice_dict_style2_post
