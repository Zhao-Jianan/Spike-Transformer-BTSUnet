import os
os.chdir(os.path.dirname(__file__))
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import nibabel as nib
import numpy as np
from spike_former_unet_model import spike_former_unet3D_8_384
# from simple_unet_model import spike_former_unet3D_8_384
import torch.nn.functional as F
from config import config as cfg
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, NormalizeIntensityd,
    Orientationd, Spacingd, ToTensord
    )
import time
from scipy.ndimage import binary_dilation, binary_opening, label, generate_binary_structure
from inference_helper import TemporalSlidingWindowInference, TemporalSlidingWindowInferenceWithROI
from tqdm import tqdm


def preprocess_for_inference(image_paths):
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
    
    # Step 3: Intensity Normalization
    normalize = NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True)
    data = normalize(data)
    
    # Step 4: ToTensor
    to_tensor = ToTensord(keys=["image"])
    data = to_tensor(data)
    
    # Step 5: Add batch dimension
    img = data["image"]  # shape: (C, D, H, W)
    img = img.unsqueeze(0) # (B=1, C, D, H, W)
    
    return img



def convert_prediction_to_label(mean_prob: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    BraTS标签转换，输入 mean_prob 顺序：TC, WT, ET
    返回标签：0=BG, 1=TC(NCR/NET), 2=ED, 4=ET
    """
    assert mean_prob.shape[0] == 3, "Expected 3 channels: TC, WT, ET"
    tc_prob, wt_prob, et_prob = mean_prob[0], mean_prob[1], mean_prob[2]

    # 阈值化各通道
    et = (et_prob > threshold).astype(np.uint8)
    tc = (tc_prob > threshold).astype(np.uint8)
    wt = (wt_prob > threshold).astype(np.uint8)

    label = np.zeros_like(tc, dtype=np.uint8)

    label[wt == 1] = 2
    label[tc == 1] = 1
    label[et == 1] = 4  # ET优先级最高

    return label


def postprocess_brats_label(pred_mask: np.ndarray) -> np.ndarray:
    """
    BraTS预测标签后处理：
    - ET (4): 向外扩张一圈，只吸收外部的NCR和ED，不吞噬ET内部的NCR
    - NCR/NET (1): 外部NCR做开运算，ET内部NCR保持原样
    - ED (2): 保持原状
    """

    structure = generate_binary_structure(3, 1)

    # 原始标签
    et_mask = (pred_mask == 4)
    ncr_mask = (pred_mask == 1)
    edema_mask = (pred_mask == 2)

    print("Before Postprocessing:")
    print("Sum ET:", np.sum(et_mask))
    print("Sum ED:", np.sum(edema_mask))
    print("Sum NCR:", np.sum(ncr_mask))

    # Step 1: 分离ET内部的NCR（要保护的）与ET外部的NCR（可处理的）
    ncr_inside_et = ncr_mask & et_mask
    ncr_outside_et = ncr_mask & (~et_mask)

    # Step 2: 对外部NCR做开运算
    ncr_outside_processed = binary_opening(ncr_outside_et, structure=structure, iterations=1)
    ncr_processed = ncr_outside_processed | ncr_inside_et

    # 被剥掉的外部NCR边缘
    ncr_removed = ncr_outside_et & (~ncr_outside_processed)

    # Step 3: 构造ET的“外壳”：从ET外面包一圈，不含ET原始区域
    et_outer_shell = binary_dilation(et_mask, structure=structure, iterations=1) & (~et_mask)

    # Step 4: 只允许ET扩张到其“外壳”中满足条件的区域（NCR外部边缘 or ED）
    et_expand_target = et_outer_shell & (ncr_removed | edema_mask)

    # Step 5: 最终ET = 原始ET + 允许扩张区域（外壳目标）
    et_final = et_mask | et_expand_target

    # Step 6: 构建最终mask
    new_mask = np.zeros_like(pred_mask, dtype=np.uint8)
    new_mask[et_final] = 4
    new_mask[(edema_mask) & (new_mask == 0)] = 2
    new_mask[(ncr_processed) & (new_mask == 0)] = 1

    # Step 7: 剩余被删掉的NCR边缘如果未被覆盖，强制转ET（避免留背景）
    ncr_remaining = ncr_removed & (new_mask == 0)
    new_mask[ncr_remaining] = 4

    print("Postprocessing results:")
    print("Sum ET:", np.sum(new_mask == 4))
    print("Sum ED:", np.sum(new_mask == 2))
    print("Sum NCR:", np.sum(new_mask == 1))

    return new_mask



def pred_single_case(case_dir, model, inference_engine, device):
    case_name = os.path.basename(case_dir)
    print(f"Processing case: {case_name}")
    image_paths = [os.path.join(case_dir, f"{case_name}_{mod}.nii") for mod in cfg.modalities]

    x_batch = preprocess_for_inference(image_paths).to(device)
    B, C, D, H, W = x_batch.shape
    brain_width = np.array([[0, 0, 0], [D-1, H-1, W-1]])
    
    with torch.no_grad():
        # output = inference_engine(x_batch, brain_width, model)
        output = inference_engine(x_batch, model)

    output_prob = torch.sigmoid(output).squeeze(0).cpu().numpy()  # [C, D, H, W]

    return output_prob




def run_inference_soft_single(case_dir, save_dir, model, inference_engine, device):
    os.makedirs(save_dir, exist_ok=True)
    
    prob = pred_single_case(case_dir, model, inference_engine, device)

    label_np = convert_prediction_to_label(prob)  # shape: (D, H, W)
    label_np = np.transpose(label_np, (1, 2, 0))  # to (H, W, D)

    case_name = os.path.basename(case_dir)
    ref_nii_path = os.path.join(case_dir, f"{case_name}_{cfg.modalities[cfg.modalities.index('t1ce')]}.nii")
    ref_nii = nib.load(ref_nii_path)
    pred_nii = nib.Nifti1Image(label_np, affine=ref_nii.affine, header=ref_nii.header)

    save_path = os.path.join(save_dir, f"{case_name}_pred_mask.nii.gz")
    nib.save(pred_nii, save_path)

def run_inference_folder_single(case_root, save_dir, model, inference_engine, device, whitelist=None):
    os.makedirs(save_dir, exist_ok=True)
    
    # # For other dataset
    # all_case_dirs = sorted([
    #     os.path.join(case_root, name) for name in os.listdir(case_root)
    #     if os.path.isdir(os.path.join(case_root, name))
    # ])
         
    # For BraTS2018 dataset
    all_case_dirs = []
    for subfolder in ['HGG', 'LGG']:
        sub_dir = os.path.join(case_root, subfolder)
        if not os.path.isdir(sub_dir):
            continue
        case_dirs = sorted([
            os.path.join(sub_dir, name) for name in os.listdir(sub_dir)
            if os.path.isdir(os.path.join(sub_dir, name))
        ])
        all_case_dirs.extend(case_dirs)

    if whitelist is not None:
        whitelist_set = set(whitelist)
        case_dirs = [d for d in all_case_dirs if os.path.basename(d) in whitelist_set]
    else:
        case_dirs = all_case_dirs

    print(f"Found {len(case_dirs)} cases to run.")

    for case_dir in tqdm(case_dirs, desc="Single Model Inference"):
        run_inference_soft_single(case_dir, save_dir, model, inference_engine, device)


def build_model(ckpt_path):
    model = spike_former_unet3D_8_384(
        num_classes=cfg.num_classes,
        T=cfg.T,
        # norm_type=cfg.norm_type,
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


def main():
    # BraTS2018 inference
    ckpt_path = "/hpc/ajhz839/checkpoint/experiment_56/best_model_fold2.pth"
    case_dir = "/hpc/ajhz839/data/BraTS2018/train/"
    output_dir = "/hpc/ajhz839/validation/val_fold2_pred/"
    
    brats2018_case_whitelist = [
        "Brats18_2013_12_1",
        "Brats18_2013_22_1",
        "Brats18_2013_2_1",
        "Brats18_2013_3_1",
        "Brats18_2013_5_1",
        "Brats18_2013_7_1",
        "Brats18_CBICA_ABE_1",
        "Brats18_CBICA_ALU_1",
        "Brats18_CBICA_ANP_1",
        "Brats18_CBICA_ANZ_1",
        "Brats18_CBICA_AQR_1",
        "Brats18_CBICA_AQU_1",
        "Brats18_CBICA_ASH_1",
        "Brats18_CBICA_ASN_1",
        "Brats18_CBICA_ATP_1",
        "Brats18_CBICA_AVG_1",
        "Brats18_CBICA_AVV_1",
        "Brats18_CBICA_AYA_1",
        "Brats18_CBICA_AZD_1",
        "Brats18_CBICA_BFP_1",
        "Brats18_TCIA01_180_1",
        "Brats18_TCIA01_186_1",
        "Brats18_TCIA01_190_1",
        "Brats18_TCIA01_201_1",
        "Brats18_TCIA01_221_1",
        "Brats18_TCIA01_231_1",
        "Brats18_TCIA01_335_1",
        "Brats18_TCIA01_412_1",
        "Brats18_TCIA01_425_1",
        "Brats18_TCIA02_198_1",
        "Brats18_TCIA02_222_1",
        "Brats18_TCIA02_300_1",
        "Brats18_TCIA02_314_1",
        "Brats18_TCIA02_321_1",
        "Brats18_TCIA02_374_1",
        "Brats18_TCIA03_121_1",
        "Brats18_TCIA03_133_1",
        "Brats18_TCIA03_375_1",
        "Brats18_TCIA04_149_1",
        "Brats18_TCIA04_437_1",
        "Brats18_TCIA08_113_1",
        "Brats18_TCIA08_205_1",
        "Brats18_2013_15_1",
        "Brats18_2013_29_1",
        "Brats18_TCIA09_177_1",
        "Brats18_TCIA09_451_1",
        "Brats18_TCIA10_103_1",
        "Brats18_TCIA10_241_1",
        "Brats18_TCIA10_408_1",
        "Brats18_TCIA10_449_1",
        "Brats18_TCIA10_639_1",
        "Brats18_TCIA12_298_1",
        "Brats18_TCIA12_466_1",
        "Brats18_TCIA13_621_1",
        "Brats18_TCIA13_642_1",
        "Brats18_TCIA13_650_1",
        "Brats18_TCIA13_653_1"
    ]
    
    model = build_model(ckpt_path)
    inference_engine = build_inference_engine()
    run_inference_folder_single(case_dir, output_dir, model, inference_engine, cfg.device, brats2018_case_whitelist)


    # BraTS2023 inference
    # ckpt_path = "/hpc/ajhz839/checkpoint/experiment_33/best_model_fold2.pth"
    # case_dir = "/hpc/ajhz839/data/BraTS2023/train/"
    # output_dir = "/hpc/ajhz839/validation/val_fold2_pred/" 
       
    brats2023_case_whitelist = [
        "BraTS-GLI-00002-000",
        "BraTS-GLI-00005-000",
        "BraTS-GLI-00006-000",
        "BraTS-GLI-00008-001",
        "BraTS-GLI-00009-000",
        "BraTS-GLI-00012-000",
        "BraTS-GLI-00021-000",
        "BraTS-GLI-00025-000",
        "BraTS-GLI-00031-000",
        "BraTS-GLI-00032-000",
        "BraTS-GLI-00035-000",
        "BraTS-GLI-00036-001",
        "BraTS-GLI-00043-000",
        "BraTS-GLI-00044-000",
        "BraTS-GLI-00045-001",
        "BraTS-GLI-00058-000",
        "BraTS-GLI-00059-000",
        "BraTS-GLI-00059-001",
        "BraTS-GLI-00066-000",
        "BraTS-GLI-00070-000",
        "BraTS-GLI-00072-001",
        "BraTS-GLI-00077-000",
        "BraTS-GLI-00085-000",
        "BraTS-GLI-00097-001",
        "BraTS-GLI-00100-000",
        "BraTS-GLI-00105-000",
        "BraTS-GLI-00115-000",
        "BraTS-GLI-00133-000",
        "BraTS-GLI-00137-000",
        "BraTS-GLI-00146-000",
        "BraTS-GLI-00156-000",
        "BraTS-GLI-00157-000",
        "BraTS-GLI-00165-000",
        "BraTS-GLI-00177-000",
        "BraTS-GLI-00178-000",
        "BraTS-GLI-00193-000",
        "BraTS-GLI-00199-000",
        "BraTS-GLI-00214-000",
        "BraTS-GLI-00221-000",
        "BraTS-GLI-00222-000",
        "BraTS-GLI-00227-000",
        "BraTS-GLI-00238-000",
        "BraTS-GLI-00239-000",
        "BraTS-GLI-00243-000",
        "BraTS-GLI-00247-000",
        "BraTS-GLI-00249-000",
        "BraTS-GLI-00259-000",
        "BraTS-GLI-00271-000",
        "BraTS-GLI-00273-000",
        "BraTS-GLI-00275-000",
        "BraTS-GLI-00285-000",
        "BraTS-GLI-00290-000",
        "BraTS-GLI-00291-000",
        "BraTS-GLI-00296-000",
        "BraTS-GLI-00299-000",
        "BraTS-GLI-00303-000",
        "BraTS-GLI-00304-000",
        "BraTS-GLI-00309-000",
        "BraTS-GLI-00310-000",
        "BraTS-GLI-00322-000",
        "BraTS-GLI-00352-000",
        "BraTS-GLI-00359-000",
        "BraTS-GLI-00364-000",
        "BraTS-GLI-00367-000",
        "BraTS-GLI-00377-000",
        "BraTS-GLI-00378-000",
        "BraTS-GLI-00383-000",
        "BraTS-GLI-00389-000",
        "BraTS-GLI-00395-000",
        "BraTS-GLI-00407-000",
        "BraTS-GLI-00413-000",
        "BraTS-GLI-00417-000",
        "BraTS-GLI-00418-000",
        "BraTS-GLI-00423-000",
        "BraTS-GLI-00425-000",
        "BraTS-GLI-00433-000",
        "BraTS-GLI-00444-000",
        "BraTS-GLI-00446-000",
        "BraTS-GLI-00469-001",
        "BraTS-GLI-00477-001",
        "BraTS-GLI-00478-001",
        "BraTS-GLI-00479-001",
        "BraTS-GLI-00480-000",
        "BraTS-GLI-00494-001",
        "BraTS-GLI-00526-000",
        "BraTS-GLI-00532-000",
        "BraTS-GLI-00533-000",
        "BraTS-GLI-00539-000",
        "BraTS-GLI-00540-000",
        "BraTS-GLI-00544-000",
        "BraTS-GLI-00545-000",
        "BraTS-GLI-00549-000",
        "BraTS-GLI-00551-000",
        "BraTS-GLI-00559-001",
        "BraTS-GLI-00563-000",
        "BraTS-GLI-00568-000",
        "BraTS-GLI-00583-000",
        "BraTS-GLI-00588-000",
        "BraTS-GLI-00599-000",
        "BraTS-GLI-00605-000",
        "BraTS-GLI-00620-000",
        "BraTS-GLI-00623-000",
        "BraTS-GLI-00628-000",
        "BraTS-GLI-00630-001",
        "BraTS-GLI-00639-000",
        "BraTS-GLI-00642-000",
        "BraTS-GLI-00649-001",
        "BraTS-GLI-00674-001",
        "BraTS-GLI-00679-000",
        "BraTS-GLI-00684-000",
        "BraTS-GLI-00689-000",
        "BraTS-GLI-00703-000",
        "BraTS-GLI-00703-001",
        "BraTS-GLI-00706-000",
        "BraTS-GLI-00715-000",
        "BraTS-GLI-00716-000",
        "BraTS-GLI-00729-000",
        "BraTS-GLI-00730-001",
        "BraTS-GLI-00734-000",
        "BraTS-GLI-00735-000",
        "BraTS-GLI-00740-000",
        "BraTS-GLI-00744-000",
        "BraTS-GLI-00747-000",
        "BraTS-GLI-00751-000",
        "BraTS-GLI-00756-000",
        "BraTS-GLI-00759-000",
        "BraTS-GLI-00773-000",
        "BraTS-GLI-00775-001",
        "BraTS-GLI-00780-000",
        "BraTS-GLI-00781-000",
        "BraTS-GLI-00787-000",
        "BraTS-GLI-00789-000",
        "BraTS-GLI-00792-000",
        "BraTS-GLI-00804-000",
        "BraTS-GLI-00807-000",
        "BraTS-GLI-00810-000",
        "BraTS-GLI-00840-000",
        "BraTS-GLI-00999-000",
        "BraTS-GLI-01001-000",
        "BraTS-GLI-01004-000",
        "BraTS-GLI-01024-000",
        "BraTS-GLI-01025-000",
        "BraTS-GLI-01026-000",
        "BraTS-GLI-01027-001",
        "BraTS-GLI-01041-000",
        "BraTS-GLI-01044-000",
        "BraTS-GLI-01045-000",
        "BraTS-GLI-01046-000",
        "BraTS-GLI-01048-000",
        "BraTS-GLI-01051-000",
        "BraTS-GLI-01061-000",
        "BraTS-GLI-01062-000",
        "BraTS-GLI-01063-000",
        "BraTS-GLI-01066-000",
        "BraTS-GLI-01067-000",
        "BraTS-GLI-01068-000",
        "BraTS-GLI-01074-000",
        "BraTS-GLI-01079-000",
        "BraTS-GLI-01081-000",
        "BraTS-GLI-01092-000",
        "BraTS-GLI-01093-000",
        "BraTS-GLI-01094-000",
        "BraTS-GLI-01102-000",
        "BraTS-GLI-01104-000",
        "BraTS-GLI-01109-000",
        "BraTS-GLI-01110-000",
        "BraTS-GLI-01115-000",
        "BraTS-GLI-01122-000",
        "BraTS-GLI-01126-000",
        "BraTS-GLI-01127-000",
        "BraTS-GLI-01149-000",
        "BraTS-GLI-01152-000",
        "BraTS-GLI-01162-000",
        "BraTS-GLI-01164-000",
        "BraTS-GLI-01172-000",
        "BraTS-GLI-01174-000",
        "BraTS-GLI-01177-000",
        "BraTS-GLI-01180-000",
        "BraTS-GLI-01183-000",
        "BraTS-GLI-01185-000",
        "BraTS-GLI-01191-000",
        "BraTS-GLI-01194-000",
        "BraTS-GLI-01200-000",
        "BraTS-GLI-01201-000",
        "BraTS-GLI-01202-000",
        "BraTS-GLI-01211-000",
        "BraTS-GLI-01225-000",
        "BraTS-GLI-01229-000",
        "BraTS-GLI-01236-000",
        "BraTS-GLI-01241-000",
        "BraTS-GLI-01252-000",
        "BraTS-GLI-01253-000",
        "BraTS-GLI-01255-000",
        "BraTS-GLI-01270-000",
        "BraTS-GLI-01287-000",
        "BraTS-GLI-01289-000",
        "BraTS-GLI-01293-000",
        "BraTS-GLI-01295-000",
        "BraTS-GLI-01309-000",
        "BraTS-GLI-01310-000",
        "BraTS-GLI-01314-000",
        "BraTS-GLI-01321-000",
        "BraTS-GLI-01328-000",
        "BraTS-GLI-01331-000",
        "BraTS-GLI-01333-000",
        "BraTS-GLI-01335-000",
        "BraTS-GLI-01346-000",
        "BraTS-GLI-01351-000",
        "BraTS-GLI-01356-000",
        "BraTS-GLI-01358-000",
        "BraTS-GLI-01363-000",
        "BraTS-GLI-01364-000",
        "BraTS-GLI-01367-000",
        "BraTS-GLI-01370-000",
        "BraTS-GLI-01372-000",
        "BraTS-GLI-01374-000",
        "BraTS-GLI-01378-000",
        "BraTS-GLI-01381-000",
        "BraTS-GLI-01383-000",
        "BraTS-GLI-01386-000",
        "BraTS-GLI-01393-000",
        "BraTS-GLI-01399-000",
        "BraTS-GLI-01402-000",
        "BraTS-GLI-01412-000",
        "BraTS-GLI-01417-000",
        "BraTS-GLI-01426-000",
        "BraTS-GLI-01434-000",
        "BraTS-GLI-01446-000",
        "BraTS-GLI-01452-000",
        "BraTS-GLI-01457-000",
        "BraTS-GLI-01462-000",
        "BraTS-GLI-01468-000",
        "BraTS-GLI-01475-000",
        "BraTS-GLI-01479-000",
        "BraTS-GLI-01480-000",
        "BraTS-GLI-01488-000",
        "BraTS-GLI-01490-000",
        "BraTS-GLI-01492-000",
        "BraTS-GLI-01493-000",
        "BraTS-GLI-01494-000",
        "BraTS-GLI-01499-000",
        "BraTS-GLI-01500-000",
        "BraTS-GLI-01507-000",
        "BraTS-GLI-01516-000",
        "BraTS-GLI-01517-000",
        "BraTS-GLI-01520-000",
        "BraTS-GLI-01532-000",
        "BraTS-GLI-01536-000",
        "BraTS-GLI-01658-000",
        "BraTS-GLI-01664-000"
    ]


    # model = build_model(ckpt_path)
    # inference_engine = build_inference_engine()
    # run_inference_folder_single(case_dir, output_dir, model, inference_engine, cfg.device, brats2023_case_whitelist)
    
    
    
    

    
    # # Clinical data inference
    # prob_base_dir = "/hpc/ajhz839/inference/pred/test_prob_folds/"
    # ensemble_output_dir = "/hpc/ajhz839/inference/pred/test_pred_soft_ensemble/"
    # case_dir = "/hpc/ajhz839/inference/clinical_data/"
    # ckpt_dir = "./checkpoint/"

    # soft_ensemble(prob_base_dir, case_dir, ckpt_dir)
    # ensemble_soft_voting(prob_base_dir, case_dir, ensemble_output_dir)
    
    
    print("Single-model inference completed.")

if __name__ == "__main__":
    main()




