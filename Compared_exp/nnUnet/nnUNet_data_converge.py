import os
import json
import shutil
import SimpleITK as sitk
import numpy as np
from tqdm import tqdm


# ---------- 配置 ----------
# source_case_dirs = ['/hpc/ajhz839/data/BraTS2020/MICCAI_BraTS2020_TrainingData/']
# output_base_dir = './nnUNet_raw/Dataset501_BraTS2020'
# fold_txt_dir = './val_cases'

source_case_dirs = ['/hpc/ajhz839/data/BraTS2018/train/HGG', '/hpc/ajhz839/data/BraTS2018/train/LGG']
output_base_dir = './nnUNet_raw/Dataset032_BraTS2018'
fold_txt_dir = './val_cases'

imagesTr_dir = os.path.join(output_base_dir, 'imagesTr')
labelsTr_dir = os.path.join(output_base_dir, 'labelsTr')
os.makedirs(imagesTr_dir, exist_ok=True)
os.makedirs(labelsTr_dir, exist_ok=True)

modalities = ['t1', 't1ce', 't2', 'flair']
mod_suffix_map = {'t1': '0000', 't1ce': '0001', 't2': '0002', 'flair': '0003'}


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


# ---------- 拷贝图像和标签 ----------
def prepare_data_for_nnUNet():
    print("正在整理数据为 nnU-Net 格式...")

    case_dirs = []
    for base_dir in source_case_dirs:
        case_dirs += [
            os.path.join(base_dir, d)
            for d in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, d))
        ]

    for case_dir in tqdm(sorted(case_dirs)):
        case = os.path.basename(case_dir)

        # 检查所有模态是否都存在，否则直接报错退出
        for mod in modalities:
            in_file = os.path.join(case_dir, f"{case}_{mod}.nii")
            if not os.path.exists(in_file):
                raise FileNotFoundError(f"模态缺失：{in_file}，程序终止。")

        # 所有模态齐全，开始复制
        for mod in modalities:
            in_file = os.path.join(case_dir, f"{case}_{mod}.nii")
            out_file = os.path.join(imagesTr_dir, f"{case}_{mod_suffix_map[mod]}.nii")
            shutil.copy(in_file, out_file)

        label_file = os.path.join(case_dir, f"{case}_seg.nii")
        if os.path.exists(label_file):
            out_label = os.path.join(labelsTr_dir, f"{case}.nii")
            convert_brats_labels_to_nnUNet(label_file, out_label)
        else:
            print(f"缺失标签：{label_file}，跳过")

    print("数据整理完成！")



# ---------- 生成 json ----------
def generate_training_entry(case_id):
    return {
        "image": f"./imagesTr/{case_id}.nii",  # 自动拼接通道后缀
        "label": f"./labelsTr/{case_id}.nii"
    }

def generate_dataset_json(manual_split=True, num_folds=5):
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

    all_cases = set()

    if manual_split:
        print("正在生成 dataset_fold{i}.json 文件...")
        for fold in range(num_folds):
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

    else:
        print("未手动划分折数，将自动生成完整 dataset.json ...")
        all_cases = sorted([
            fname.split('.')[0] for fname in os.listdir(labelsTr_dir) if fname.endswith('.nii')
        ])

    # 输出完整 dataset.json
    full_entries = [generate_training_entry(case) for case in sorted(all_cases)]
    full_dataset = dataset_base.copy()
    full_dataset["training"] = full_entries
    full_dataset["numTraining"] = len(full_entries)

    full_json_path = os.path.join(output_base_dir, "dataset.json")
    with open(full_json_path, 'w') as f:
        json.dump(full_dataset, f, indent=4)

    print(f"\n已生成完整 dataset.json，包含样本数：{len(full_entries)}")

    if manual_split:
        print("\n建议的训练命令如下：\n")
        for fold in range(num_folds):
            print(f"cp dataset_fold{fold}.json dataset.json")
            print(f"nnUNetv2_train 501 3d_fullres {fold}\n")


# ---------- 主入口 ----------
def main():
    manual_split=False  # 是否使用手动划分的五折数据集
    prepare_data_for_nnUNet()
    generate_dataset_json(manual_split=manual_split)


# ---------- 调用 ----------
if __name__ == "__main__":
    # 设置为 False 表示不使用手动五折划分
    main()
