import os
import shutil
import json

# ==== 修改这两个路径为你自己的 ====
source_case_dir = '/hpc/ajhz839/data/BraTS2020/MICCAI_BraTS2020_TrainingData/'  # 原始按 case 组织的数据目录
output_base_dir = './nnUNet_raw/Dataset501_BraTS2020'  # 输出的 nnU-Net 格式目录

fold_txt_dir = './val_cases'  # 5折 txt 文件的目录
# ====================================

imagesTr_dir = os.path.join(output_base_dir, 'imagesTr')
labelsTr_dir = os.path.join(output_base_dir, 'labelsTr')
os.makedirs(imagesTr_dir, exist_ok=True)
os.makedirs(labelsTr_dir, exist_ok=True)

modalities = ['t1', 't1ce', 't2', 'flair']
mod_suffix_map = {'t1': '0000', 't1ce': '0001', 't2': '0002', 'flair': '0003'}

# Step 1: 拷贝图像和标签
print(" 正在整理数据为 nnU-Net 格式...")
for case in sorted(os.listdir(source_case_dir)):
    case_dir = os.path.join(source_case_dir, case)
    if not os.path.isdir(case_dir):
        continue

    for mod in modalities:
        in_file = os.path.join(case_dir, f"{case}_{mod}.nii")
        out_file = os.path.join(imagesTr_dir, f"{case}_{mod_suffix_map[mod]}.nii.gz")
        shutil.copy(in_file, out_file)

    label_file = os.path.join(case_dir, f"{case}_seg.nii")
    if os.path.exists(label_file):
        out_label = os.path.join(labelsTr_dir, f"{case}.nii.gz")
        shutil.copy(label_file, out_label)

print(" 数据整理完成！")

# Step 2: 生成 dataset_fold{i}.json
print(" 正在生成 dataset_fold{i}.json 文件...")

dataset_base = {
    "channel_names": {
        "0": "T1",
        "1": "T1ce",
        "2": "T2",
        "3": "FLAIR"
    },
    "labels": {
        "0": "background",
        "1": "NCR/NET",
        "2": "ED",
        "4": "ET"
    },
    "file_ending": ".nii.gz"
}

def generate_training_entry(case_id):
    return {
        "image": f"./imagesTr/{case_id}.nii.gz",  # nnU-Net 会自动拼接通道后缀
        "label": f"./labelsTr/{case_id}.nii.gz"
    }

for fold in range(5):
    train_txt = os.path.join(fold_txt_dir, f"train_cases_fold{fold}.txt")
    with open(train_txt, 'r') as f:
        train_cases = [line.strip() for line in f if line.strip()]

    entries = [generate_training_entry(case) for case in train_cases]
    dataset = dataset_base.copy()
    dataset["training"] = entries
    dataset["test"] = []  # 可选：你也可以加测试集 case

    json_path = os.path.join(output_base_dir, f"dataset_fold{fold}.json")
    with open(json_path, 'w') as f:
        json.dump(dataset, f, indent=4)
    print(f"已生成: {json_path}")

print("\n 所有准备工作完成！")

# Step 3: 输出训练指令
print("\n 建议的训练命令如下：\n")
for fold in range(5):
    print(f"cp dataset_fold{fold}.json dataset.json")
    print(f"nnUNetv2_train 501 3d_fullres {fold}\n")
