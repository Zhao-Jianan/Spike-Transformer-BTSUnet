import os
import shutil

# 路径配置
dataset_dir = '/hpc/ajhz839/compared_models/nnUnet/project/nnUNet_raw/Dataset501_BraTS2020/'
imagesTr_dir = os.path.join(dataset_dir, 'imagesTr')
labelsTr_dir = os.path.join(dataset_dir, 'labelsTr')
imagesTs_dir = os.path.join(dataset_dir, 'imagesTs')
labelsTs_dir = os.path.join(dataset_dir, 'labelsTs')

test_cases_file = '/hpc/ajhz839/compared_models/nnUnet/project/val_cases/test_cases.txt'

# 创建目标目录（如果不存在）
os.makedirs(imagesTs_dir, exist_ok=True)
os.makedirs(labelsTs_dir, exist_ok=True)

# 读取测试case列表
with open(test_cases_file, 'r') as f:
    test_cases = [line.strip() for line in f if line.strip()]

print(f"共读取 {len(test_cases)} 个测试case.")

# 移动对应文件
for case in test_cases:
    # 1. imagesTr 文件
    # nnUNet格式默认文件名形如：BraTS20_XXX_0000.nii.gz, BraTS20_XXX_0001.nii.gz ...
    # 可能有4个模态，0000,0001,0002,0003
    moved_any = False
    for modality_idx in range(4):
        img_file = f"{case}_{modality_idx:04d}.nii"
        src_img_path = os.path.join(imagesTr_dir, img_file)
        dst_img_path = os.path.join(imagesTs_dir, img_file)
        if os.path.exists(src_img_path):
            shutil.move(src_img_path, dst_img_path)
            moved_any = True
            print(f"移动图像: {src_img_path} -> {dst_img_path}")
        else:
            print(f"警告，图像文件不存在: {src_img_path}")

    # 2. label 文件
    label_file = f"{case}.nii"
    src_label_path = os.path.join(labelsTr_dir, label_file)
    dst_label_path = os.path.join(labelsTs_dir, label_file)
    if os.path.exists(src_label_path):
        shutil.move(src_label_path, dst_label_path)
        moved_any = True
        print(f"移动标签: {src_label_path} -> {dst_label_path}")
    else:
        print(f"警告，标签文件不存在: {src_label_path}")

    if not moved_any:
        print(f"警告，{case} 未找到任何文件移动。")

print("移动操作完成。")
