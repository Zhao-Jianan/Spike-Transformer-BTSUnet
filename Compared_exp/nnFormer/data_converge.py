import numpy as np
from collections import OrderedDict
from batchgenerators.utilities.file_and_folder_operations import *
from nnformer.paths import nnFormer_raw_data
import SimpleITK as sitk
import shutil
import random


import numpy as np
from collections import OrderedDict
from batchgenerators.utilities.file_and_folder_operations import *
from nnformer.paths import nnFormer_raw_data
import SimpleITK as sitk
import shutil

def copy_BraTS_segmentation_and_convert_labels(in_file, out_file):
    """Convert BraTS labels from {0,1,2,4} to {0,1,2,3}."""
    img = sitk.ReadImage(in_file)
    img_npy = sitk.GetArrayFromImage(img)

    uniques = np.unique(img_npy)
    for u in uniques:
        if u not in [0, 1, 2, 4]:
            raise RuntimeError(f"Unexpected label {u} in {in_file}")

    seg_new = np.zeros_like(img_npy)
    seg_new[img_npy == 4] = 3
    seg_new[img_npy == 2] = 1
    seg_new[img_npy == 1] = 2

    img_corr = sitk.GetImageFromArray(seg_new)
    img_corr.CopyInformation(img)
    sitk.WriteImage(img_corr, out_file)


if __name__ == "__main__":
    """
    Convert BraTS2020 dataset to nnFormer format:
    - training list: contains all train+val cases
    - test list: contains test cases (with labels in this experiment)
    """

    # ====== 配置 ======
    task_name = "Task05_BraTS2020"
    train_txt_path = "/hpc/ajhz839/compared_models/nnFormer/project/val_cases/train_cases.txt"   # train+val 病例列表
    test_txt_path = "/hpc/ajhz839/compared_models/nnFormer/project/val_cases/test_cases.txt"     # test 病例列表（有标签）
    data_dir = "/hpc/ajhz839/data/BraTS2020/MICCAI_BraTS2020_TrainingData/"  # BraTS2020 数据路径

    # ====== 输出路径 ======
    target_base = join(nnFormer_raw_data, task_name)
    target_imagesTr = join(target_base, "imagesTr")
    target_labelsTr = join(target_base, "labelsTr")
    target_imagesTs = join(target_base, "imagesTs")
    target_labelsTs = join(target_base, "labelsTs")

    maybe_mkdir_p(target_imagesTr)
    maybe_mkdir_p(target_labelsTr)
    maybe_mkdir_p(target_imagesTs)
    maybe_mkdir_p(target_labelsTs)

    # ====== 读取病例列表 ======
    with open(train_txt_path, "r") as f:
        train_cases = [line.strip() for line in f if line.strip()]
    with open(test_txt_path, "r") as f:
        test_cases = [line.strip() for line in f if line.strip()]

    print(f"Train+Val cases: {len(train_cases)}, Test cases: {len(test_cases)}")

    # ====== 处理训练+验证集 ======
    for p in train_cases:
        patdir = join(data_dir, p)
        assert isdir(patdir), f"Folder not found: {patdir}"

        t1 = join(patdir, p + "_t1.nii")
        t1c = join(patdir, p + "_t1ce.nii")
        t2 = join(patdir, p + "_t2.nii")
        flair = join(patdir, p + "_flair.nii")
        seg = join(patdir, p + "_seg.nii")

        assert all(map(isfile, [t1, t1c, t2, flair, seg])), f"Missing files for {p}"

        shutil.copy(t1, join(target_imagesTr, p + "_0000.nii.gz"))
        shutil.copy(t1c, join(target_imagesTr, p + "_0001.nii.gz"))
        shutil.copy(t2, join(target_imagesTr, p + "_0002.nii.gz"))
        shutil.copy(flair, join(target_imagesTr, p + "_0003.nii.gz"))
        copy_BraTS_segmentation_and_convert_labels(seg, join(target_labelsTr, p + ".nii.gz"))

    # ====== 处理测试集（有标签） ======
    for p in test_cases:
        patdir = join(data_dir, p)
        assert isdir(patdir), f"Folder not found: {patdir}"

        t1 = join(patdir, p + "_t1.nii")
        t1c = join(patdir, p + "_t1ce.nii")
        t2 = join(patdir, p + "_t2.nii")
        flair = join(patdir, p + "_flair.nii")
        seg = join(patdir, p + "_seg.nii")

        assert all(map(isfile, [t1, t1c, t2, flair, seg])), f"Missing files for {p}"

        shutil.copy(t1, join(target_imagesTs, p + "_0000.nii.gz"))
        shutil.copy(t1c, join(target_imagesTs, p + "_0001.nii.gz"))
        shutil.copy(t2, join(target_imagesTs, p + "_0002.nii.gz"))
        shutil.copy(flair, join(target_imagesTs, p + "_0003.nii.gz"))
        copy_BraTS_segmentation_and_convert_labels(seg, join(target_labelsTs, p + ".nii.gz"))

    # ====== 生成 dataset.json ======
    json_dict = OrderedDict()
    json_dict['name'] = "BraTS2020"
    json_dict['description'] = "BraTS2020 train+val and test split for nnFormer"
    json_dict['tensorImageSize'] = "4D"
    json_dict['reference'] = "see BraTS2020"
    json_dict['licence'] = "see BraTS2020 license"
    json_dict['release'] = "0.0"
    json_dict['modality'] = {
        "0": "T1",
        "1": "T1ce",
        "2": "T2",
        "3": "FLAIR"
    }
    json_dict['labels'] = {
        "0": "background",
        "1": "edema",
        "2": "non-enhancing",
        "3": "enhancing"
    }
    json_dict['numTraining'] = len(train_cases)
    json_dict['numTest'] = len(test_cases)
    json_dict['training'] = [
        {'image': f"./imagesTr/{i}.nii.gz", "label": f"./labelsTr/{i}.nii.gz"} for i in train_cases
    ]
    json_dict['test'] = [
        {'image': f"./imagesTs/{i}.nii.gz", "label": f"./labelsTs/{i}.nii.gz"} for i in test_cases
    ]

    save_json(json_dict, join(target_base, "dataset.json"))

