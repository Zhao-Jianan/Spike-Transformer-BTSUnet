import os
import shutil
import json
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

# ---------- 配置 ----------
source_case_dirs = ['/hpc/ajhz839/data/BraTS2020/MICCAI_BraTS2020_TrainingData/']
output_base_dir = './nnUNet_raw/Dataset601_BraTS2020' 
fold_txt_dir = './val_cases'

modalities = ['t1', 't1ce', 't2', 'flair']
mod_suffix_map = {'t1': '0000', 't1ce': '0001', 't2': '0002', 'flair': '0003'}

imagesTr_dir = os.path.join(output_base_dir, 'imagesTr')
labelsTr_dir = os.path.join(output_base_dir, 'labelsTr')
imagesTs_dir = os.path.join(output_base_dir, 'imagesTs')
os.makedirs(imagesTr_dir, exist_ok=True)
os.makedirs(labelsTr_dir, exist_ok=True)
os.makedirs(imagesTs_dir, exist_ok=True)

# ---------- 标签转换 ----------
def convert_brats_labels_to_nnUNet(in_file: str, out_file: str):
    img = sitk.ReadImage(in_file)
    img_npy = sitk.GetArrayFromImage(img)
    if not set(np.unique(img_npy)).issubset({0, 1, 2, 4}):
        raise ValueError(f"{in_file} contains unexpected labels: {np.unique(img_npy)}")
    new_seg = np.zeros_like(img_npy)
    new_seg[img_npy == 4] = 3  # ET
    new_seg[img_npy == 2] = 1  # ED
    new_seg[img_npy == 1] = 2  # NCR/NET
    new_img = sitk.GetImageFromArray(new_seg)
    new_img.CopyInformation(img)
    sitk.WriteImage(new_img, out_file)

### 新增：反转换（nnU-Net -> BraTS 原始标签）
def convert_labels_back_to_BraTS(seg: np.ndarray):
    new_seg = np.zeros_like(seg)
    new_seg[seg == 1] = 2  # ED
    new_seg[seg == 3] = 4  # ET
    new_seg[seg == 2] = 1  # NCR/NET
    return new_seg

def convert_folder_with_preds_back_to_BraTS(input_folder: str, output_folder: str):
    os.makedirs(output_folder, exist_ok=True)
    nii_files = [f for f in os.listdir(input_folder) if f.endswith('.nii') or f.endswith('.nii.gz')]
    for f in tqdm(nii_files, desc="Converting preds to BraTS format"):
        img = sitk.ReadImage(os.path.join(input_folder, f))
        arr = sitk.GetArrayFromImage(img)
        arr_new = convert_labels_back_to_BraTS(arr)
        out_img = sitk.GetImageFromArray(arr_new)
        out_img.CopyInformation(img)
        sitk.WriteImage(out_img, os.path.join(output_folder, f))

# ---------- 拷贝图像和标签 ----------
def copy_case_to_folder(case, case_dir, target_dir, copy_label=True):
    for mod in modalities:
        in_file = os.path.join(case_dir, f"{case}_{mod}.nii")
        if not os.path.exists(in_file):
            raise FileNotFoundError(f"模态缺失：{in_file}")
        out_file = os.path.join(target_dir, f"{case}_{mod_suffix_map[mod]}.nii")
        shutil.copy(in_file, out_file)
    if copy_label:
        label_file = os.path.join(case_dir, f"{case}_seg.nii")
        if os.path.exists(label_file):
            out_label = os.path.join(labelsTr_dir, f"{case}.nii")
            convert_brats_labels_to_nnUNet(label_file, out_label)

def prepare_data_for_nnUNet(train_cases, val_cases, test_cases):
    print("正在整理数据为 nnU-Net 格式...")
    all_dirs = {}
    for base_dir in source_case_dirs:
        for d in os.listdir(base_dir):
            full_path = os.path.join(base_dir, d)
            if os.path.isdir(full_path):
                all_dirs[d] = full_path
    for case in sorted(set(train_cases + val_cases)):
        copy_case_to_folder(case, all_dirs[case], imagesTr_dir, copy_label=True)
    for case in test_cases:
        copy_case_to_folder(case, all_dirs[case], imagesTs_dir, copy_label=False)
    print("数据整理完成！")

# ---------- 生成 json ----------
def generate_training_entry(case_id):
    return {
        "image": f"./imagesTr/{case_id}.nii",
        "label": f"./labelsTr/{case_id}.nii"
    }

def generate_dataset_json(num_folds=5):
    dataset_base = {
        "channel_names": {"0": "T1", "1": "T1ce", "2": "T2", "3": "FLAIR"},
        "labels": {"background": 0, "whole tumor": [1, 2, 3], "tumor core": [2, 3], "enhancing tumor": [3]},
        "regions_class_order": [1, 2, 3],
        "file_ending": ".nii"
    }
    all_cases = set()
    for fold in range(num_folds):
        train_txt = os.path.join(fold_txt_dir, f"train_cases_fold{fold+1}.txt")
        val_txt = os.path.join(fold_txt_dir, f"val_cases_fold{fold+1}.txt")
        with open(train_txt) as f:
            train_cases = [line.strip() for line in f if line.strip()]
        with open(val_txt) as f:
            val_cases = [line.strip() for line in f if line.strip()]
        all_cases.update(train_cases + val_cases)
        dataset = dataset_base.copy()
        dataset["training"] = [generate_training_entry(c) for c in train_cases]
        dataset["validation"] = [generate_training_entry(c) for c in val_cases]
        dataset["numTraining"] = len(train_cases)
        dataset["numValidation"] = len(val_cases)
        json_path = os.path.join(output_base_dir, f"dataset_fold{fold}.json")
        with open(json_path, 'w') as f:
            json.dump(dataset, f, indent=4)
        print(f"已生成: {json_path}")
    full_dataset = dataset_base.copy()
    full_dataset["training"] = [generate_training_entry(c) for c in sorted(all_cases)]
    full_dataset["numTraining"] = len(all_cases)
    with open(os.path.join(output_base_dir, "dataset.json"), 'w') as f:
        json.dump(full_dataset, f, indent=4)
    print("已生成完整 dataset.json")

# ---------- 主入口 ----------
def main():
    num_folds = 5
    test_txt = os.path.join(fold_txt_dir, "test_cases.txt")
    with open(test_txt) as f:
        test_cases = [line.strip() for line in f if line.strip()]
    train_cases_all = []
    val_cases_all = []
    for fold in range(num_folds):
        with open(os.path.join(fold_txt_dir, f"train_cases_fold{fold+1}.txt")) as f:
            train_cases_all.extend([line.strip() for line in f if line.strip()])
        with open(os.path.join(fold_txt_dir, f"val_cases_fold{fold+1}.txt")) as f:
            val_cases_all.extend([line.strip() for line in f if line.strip()])
    prepare_data_for_nnUNet(train_cases_all, val_cases_all, test_cases)
    generate_dataset_json(num_folds=num_folds)
    print("\n推理命令示例：")
    print(f"nnUNetv2_predict -i {imagesTs_dir} -o ./preds -d 601 -c 3d_fullres -f 0")
    print("\n反转换命令示例（推理后执行）：")
    print(f"convert_folder_with_preds_back_to_BraTS('./preds', './preds_BraTS')")

if __name__ == "__main__":
    main()
