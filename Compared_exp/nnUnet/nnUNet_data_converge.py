import os
import shutil
import json
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

# ---------- 路径配置 ----------
source_case_dir = '/hpc/ajhz839/data/BraTS2020/MICCAI_BraTS2020_TrainingData/'  # 原始按 case 组织的数据目录
output_base_dir = './nnUNet_raw/Dataset501_BraTS2020'  # 输出的 nnU-Net 格式目录
fold_txt_dir = './val_cases'  # 5折 txt 文件的目录

imagesTr_dir = os.path.join(output_base_dir, 'imagesTr')
labelsTr_dir = os.path.join(output_base_dir, 'labelsTr')
os.makedirs(imagesTr_dir, exist_ok=True)
os.makedirs(labelsTr_dir, exist_ok=True)

modalities = ['t1', 't1ce', 't2', 'flair']
mod_suffix_map = {'t1': '0000', 't1ce': '0001', 't2': '0002', 'flair': '0003'}

# ---------- 标签转换函数 ----------
def convert_brats_labels_to_nnUNet(in_file: str, out_file: str):
    img = sitk.ReadImage(in_file)
    img_npy = sitk.GetArrayFromImage(img)

    # Check validity
    if not set(np.unique(img_npy)).issubset({0, 1, 2, 4}):
        raise ValueError(f"{in_file} contains unexpected labels: {np.unique(img_npy)}")

    new_seg = np.zeros_like(img_npy)
    new_seg[img_npy == 4] = 3  # ET
    new_seg[img_npy == 2] = 1  # ED
    new_seg[img_npy == 1] = 2  # NCR/NET

    new_img = sitk.GetImageFromArray(new_seg)
    new_img.CopyInformation(img)
    sitk.WriteImage(new_img, out_file)

# ---------- Step 1: 拷贝图像和标签 ----------
print("正在整理数据为 nnU-Net 格式...")
for case in tqdm(sorted(os.listdir(source_case_dir))):
    case_dir = os.path.join(source_case_dir, case)
    if not os.path.isdir(case_dir):
        continue

    for mod in modalities:
        in_file = os.path.join(case_dir, f"{case}_{mod}.nii")
        out_file = os.path.join(imagesTr_dir, f"{case}_{mod_suffix_map[mod]}.nii")
        shutil.copy(in_file, out_file)

    label_file = os.path.join(case_dir, f"{case}_seg.nii")
    if os.path.exists(label_file):
        out_label = os.path.join(labelsTr_dir, f"{case}.nii")  # 官方用 .nii，不带 .gz
        convert_brats_labels_to_nnUNet(label_file, out_label)

print("数据整理完成！")

# ---------- Step 2: 生成 dataset_fold{i}.json ----------
print("正在生成 dataset_fold{i}.json 文件...")

dataset_base = {
    "channel_names": {
        "0": "T1",
        "1": "T1ce",
        "2": "T2",
        "3": "FLAIR"
    },
    "labels": {
        "background": 0,
        "whole tumor": [1, 2, 3],
        "tumor core": [2, 3],
        "enhancing tumor": [3]
    },
    "regions_class_order": [1, 2, 3],
    "file_ending": ".nii"
}

def generate_training_entry(case_id):
    return {
        "image": f"./imagesTr/{case_id}.nii",  # nnU-Net 会自动拼接通道后缀
        "label": f"./labelsTr/{case_id}.nii"
    }

all_cases = set()

for fold in range(5):
    train_txt = os.path.join(fold_txt_dir, f"train_cases_fold{fold+1}.txt")
    with open(train_txt, 'r') as f:
        train_cases = [line.strip() for line in f if line.strip()]
        all_cases.update(train_cases)

    entries = [generate_training_entry(case) for case in train_cases]
    dataset = dataset_base.copy()
    dataset["training"] = entries
    dataset["numTraining"] = len(train_cases)

    json_path = os.path.join(output_base_dir, f"dataset_fold{fold}.json")
    with open(json_path, 'w') as f:
        json.dump(dataset, f, indent=4)
    print(f"已生成: {json_path}")

# ---------- Step 3: 生成完整 dataset.json（合并所有fold） ----------
print("\n 正在生成完整 dataset.json ...")
all_cases = sorted(all_cases)
full_entries = [generate_training_entry(case) for case in all_cases]

full_dataset = dataset_base.copy()
full_dataset["training"] = full_entries
full_dataset["numTraining"] = len(full_entries)

full_json_path = os.path.join(output_base_dir, "dataset.json")
with open(full_json_path, 'w') as f:
    json.dump(full_dataset, f, indent=4)

print(f"已生成完整 dataset.json，包含样本数：{len(full_entries)}")

# ---------- Step 4: 输出训练建议 ----------
print("\n 建议的训练命令如下：\n")
for fold in range(5):
    print(f"cp dataset_fold{fold}.json dataset.json")
    print(f"nnUNetv2_train 501 3d_fullres {fold}\n")
