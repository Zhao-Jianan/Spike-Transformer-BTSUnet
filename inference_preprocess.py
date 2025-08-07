import os
os.chdir(os.path.dirname(__file__))
import torch
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, NormalizeIntensityd,
    Orientationd, Spacingd, ToTensord, ConvertToMultiChannelBasedOnBratsClassesd
    )



def preprocess_for_inference_valid(image_paths, label_path, center_crop=True, crop_size=(144, 144, 144)):
    """
    同时加载和预处理 BraTS 图像和标签（支持中心裁剪 + 重采样）

    Args:
        image_paths (list[str]): [t1, t1ce, t2, flair] 路径
        label_path (str): seg 路径
        center_crop (bool): 是否进行中心裁剪
        crop_size (tuple): (D, H, W)，裁剪大小

    Returns:
        image_tensor: (1, 4, D, H, W)
        label_tensor: (1, 3, D, H, W)
        meta: {
            "original_shape": (D, H, W),
            "crop_start": (sd, sh, sw)
        }
    """
    
    data_dict = {
        "image": image_paths,
        "label": label_path,
    }
    
    # Step 1: Load & Channel First
    load_transform = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
    ])
    data = load_transform(data_dict)
    
    # Permute image: (4, D, H, W)
    data["image"] = data["image"].permute(0, 3, 1, 2).contiguous()
    data["label"] = data["label"].permute(0, 3, 1, 2).contiguous()
    print("Loaded image shape:", data["image"].shape)
    print("Loaded label shape:", data["label"].shape)
    
    img_meta = data["image"].meta
    img_spacing = img_meta.get("pixdim", None)

    # Step 2: Spatial Normalization (Orientation + Spacing)
    need_orientation_or_spacing = False
    if img_meta.get("spatial_shape") is None:
        need_orientation_or_spacing = True
    else:
        if not torch.allclose(torch.tensor(img_spacing[:3]), torch.tensor([1.0, 1.0, 1.0])):
            need_orientation_or_spacing = True
        if not (img_meta.get("original_channel_dim", None) == 0 and img_meta.get("original_affine", None) is not None):
            need_orientation_or_spacing = True
    
    if need_orientation_or_spacing:
        print("DO PREPROCESS!!!")
        preprocess = Compose([
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode={"image": "bilinear", "label": "nearest"}),
        ])
        data = preprocess(data)
    
    # Step 3: Center Crop
    def _center_crop_fn(img: torch.Tensor, crop_size=(144, 144, 144)):
        _, D, H, W = img.shape
        cd, ch, cw = crop_size
        sd = (D - cd) // 2
        sh = (H - ch) // 2
        sw = (W - cw) // 2
        cropped = img[:, sd:sd+cd, sh:sh+ch, sw:sw+cw]
        crop_start = (sd, sh, sw)
        return cropped, (D, H, W), crop_start
    
    if center_crop:
        print("Applying center crop...")
        data["image"], original_shape, crop_start = _center_crop_fn(data["image"])
        data["label"], _, _ = _center_crop_fn(data["label"], crop_size)
    else:
        print("No center crop applied.")
        original_shape = data["image"].shape[1:]  # (D, H, W)
        crop_start = (0, 0, 0)
    
    
    # Step 4: Intensity Normalization
    normalize = NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True)
    data = normalize(data)
    
    # Step 5: ToTensor
    to_tensor = ToTensord(keys=["image", "label"])
    data = to_tensor(data)
    
    # Step 6: Add batch dimension
    image_tensor = data["image"].unsqueeze(0)  # (1, 4, D, H, W)
    label_tensor = data["label"].unsqueeze(0)  # (1, 3, D, H, W)
    
    print("Final image shape:", image_tensor.shape)
    print("Final label shape:", label_tensor.shape)
    
    return image_tensor, label_tensor, {
        "original_shape": original_shape,
        "crop_start": crop_start,
    }
    
    
def preprocess_for_inference_test(image_paths, center_crop=True):
    """
    image_paths: list of 4 modality paths [t1, t1ce, t2, flair]
    
    Returns:
        x_seq: torch.Tensor, shape (B=1, C, D, H, W)
    """
    data_dict = {"image": image_paths}
    
    # Step 1: Load + Channel First
    load_transform = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
    ])
    data = load_transform(data_dict)
    data["image"] = data["image"].permute(0, 3, 1, 2).contiguous()
    print("Loaded image shape:", data["image"].shape)  # (C, D, H, W)
    
    img_meta = data["image"].meta
    img_spacing = img_meta.get("pixdim", None)

    # Step 2: Spatial Normalization (Orientation + Spacing)
    need_orientation_or_spacing = False
    if img_meta.get("spatial_shape") is None:
        need_orientation_or_spacing = True
    else:
        if not torch.allclose(torch.tensor(img_spacing[:3]), torch.tensor([1.0, 1.0, 1.0])):
            need_orientation_or_spacing = True
        if not (img_meta.get("original_channel_dim", None) == 0 and img_meta.get("original_affine", None) is not None):
            need_orientation_or_spacing = True
    
    if need_orientation_or_spacing:
        print("DO PREPROCESS!!!")
        preprocess = Compose([
            Orientationd(keys=["image"], axcodes="RAS"),
            Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
        ])
        data = preprocess(data)
    
    # Step 3: Center Crop
    def _center_crop_fn(img: torch.Tensor, crop_size=(144,144,144)):
        _, D, H, W = img.shape
        cd, ch, cw = crop_size
        sd = (D - cd) // 2
        sh = (H - ch) // 2
        sw = (W - cw) // 2
        cropped = img[:, sd:sd+cd, sh:sh+ch, sw:sw+cw]
        crop_start = (sd, sh, sw)
        return cropped, (D, H, W), crop_start
    
    if center_crop:
        print("Applying center crop...")
        data["image"], original_shape, crop_start = _center_crop_fn(data["image"])
    else:
        print("No center crop applied.")
        original_shape = data["image"].shape[1:]  # (D, H, W)
        crop_start = (0, 0, 0)
    
    
    # Step 4: Intensity Normalization
    normalize = NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True)
    data = normalize(data)
    
    # Step 5: ToTensor
    to_tensor = ToTensord(keys=["image"])
    data = to_tensor(data)
    
    # Step 6: Add batch dimension
    img = data["image"]  # shape: (C, D, H, W)
    img = img.unsqueeze(0) # (B=1, C, D, H, W)
    
    print("Preprocessed image shape:", img.shape)  # (B=1, C, D, H, W)
    
    return img, {
        "original_shape": original_shape,  # (D, H, W)
        "crop_start": crop_start           # (sd, sh, sw)
    }
    
    
    