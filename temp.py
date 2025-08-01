def pred_single_case(case_dir, model, inference_engine, device):
    case_name = os.path.basename(case_dir)
    print(f"Processing case: {case_name}")
    image_paths = [os.path.join(case_dir, f"{case_name}_{mod}.nii") for mod in cfg.modalities]

    # 获取预处理输出和原图信息（包括原始shape和crop位置）
    x_batch, metadata = preprocess_for_inference(image_paths, return_metadata=True)  # 修改点1
    x_batch = x_batch.to(device)
    B, C, D, H, W = x_batch.shape
    brain_width = np.array([[0, 0, 0], [D - 1, H - 1, W - 1]])

    with torch.no_grad():
        output = inference_engine(x_batch, model)

    output_prob = torch.sigmoid(output).squeeze(0).cpu().numpy()  # [C, D, H, W]
    return output_prob, metadata


def restore_to_original_shape(cropped_label, original_shape, crop_start):
    """
    将中心裁剪过的预测结果还原回原图大小。
    """
    restored = np.zeros(original_shape, dtype=cropped_label.dtype)
    z, y, x = crop_start
    dz, dy, dx = cropped_label.shape
    restored[z:z+dz, y:y+dy, x:x+dx] = cropped_label
    return restored


def run_inference_soft_single(case_dir, save_dir, model, inference_engine, device):
    os.makedirs(save_dir, exist_ok=True)
    
    prob, metadata = pred_single_case(case_dir, model, inference_engine, device)

    label_np = convert_prediction_to_label_suppress_fp(prob)  # (D, H, W)
    label_np = np.transpose(label_np, (1, 2, 0))  # (H, W, D)

    # 还原原始空间（前提是 metadata 包含 'original_shape' 和 'crop_start'）
    restored_label = restore_to_original_shape(
        label_np,
        metadata["original_shape"],  # e.g., (H, W, D)
        metadata["crop_start"]       # e.g., (z, y, x)
    )

    case_name = os.path.basename(case_dir)
    ref_nii_path = os.path.join(case_dir, f"{case_name}_{cfg.modalities[cfg.modalities.index('t1ce')]}.nii")
    ref_nii = nib.load(ref_nii_path)
    pred_nii = nib.Nifti1Image(restored_label, affine=ref_nii.affine, header=ref_nii.header)

    save_path = os.path.join(save_dir, f"{case_name}_pred_mask.nii.gz")
    nib.save(pred_nii, save_path)


def preprocess_for_inference(image_paths, return_metadata=False):
    # 假设原图为 (H_orig, W_orig, D_orig)
    # 假设你做了中心裁剪后的 shape 是 (144, 144, 144)

    # 示例实现：
    image, original_shape, crop_start = center_crop_and_normalize(image_paths)
    x_batch = torch.from_numpy(image).unsqueeze(0)  # shape: [1, C, D, H, W]

    if return_metadata:
        return x_batch, {
            "original_shape": original_shape,  # e.g., (H, W, D)
            "crop_start": crop_start           # e.g., (z, y, x)
        }
    else:
        return x_batch
