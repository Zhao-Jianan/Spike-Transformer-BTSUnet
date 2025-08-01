import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import os
from matplotlib import gridspec
import matplotlib.patches as mpatches

# 加载 NIfTI 图像并返回 numpy 数组
def load_nifti_image(path):
    nii = nib.load(path)
    return nii.get_fdata()

# 创建 RGB mask（1: Blue, 2: Green, 4: Red）
def create_rgba_mask(mask):
    rgba = np.zeros(mask.shape + (4,), dtype=np.uint8)  # RGBA format
    rgba[mask == 0] = [0, 0, 0, 0]       # fully transparent
    rgba[mask == 1] = [0, 0, 255, 255]     # necrotic and non-enhancing tumor core (NCR/NET — label 1) # Blue, opaque
    rgba[mask == 2] = [0, 255, 0, 255]   # peritumoral edema (ED — label 2)  Green, opaque
    # rgba[mask == 3] = [255, 0, 0, 255]     # Enhancing - Red, opaque
    rgba[mask == 4] = [255, 0, 0, 255]     # GD-enhancing tumor (ET — label 4) # Red, opaque
    return rgba


def select_best_slice(mask):
    best_score = (0, 0)  # (标签种类数量, 标签面积)
    best_slice_idx = 0
    for i in range(mask.shape[2]):
        slice_mask = mask[:, :, i]
        labels = np.unique(slice_mask)
        labels = labels[labels != 0]  # 忽略背景
        label_count = len(labels)
        label_area = np.sum(slice_mask > 0)
        score = (label_count, label_area)
        if score > best_score:
            best_score = score
            best_slice_idx = i
    return best_slice_idx



# 可视化函数

def plot_modalities_with_masks(t1, t1ce, t2, flair, gt_mask, pred_mask, case_name, slice_idx=80, save_path='output.png'):
    modalities = [t1, t1ce, t2, flair]
    modality_names = ['T1', 'T1ce', 'T2', 'FLAIR']
    row_labels = ['Original Image', 'Ground Truth', 'Predict Result']

    fig = plt.figure(figsize=(14, 12))  # 放大整体画布尺寸
    # gridspec 4行4列，最后一行放图例，第一列留给左侧文字标签
    gs = gridspec.GridSpec(4, 5, width_ratios=[0.2,1,1,1,1], height_ratios=[1,1,1,0.15])
    gs.update(wspace=0.01, hspace=0.05)  # 缩小行间距

    # 左侧文字标签行（3行）
    for row in range(3):
        ax_label = plt.subplot(gs[row, 0])
        ax_label.text(0.5, 0.5, row_labels[row], rotation=90, fontsize=14,
                      va='center', ha='center')
        ax_label.axis('off')

    # 显示图像和mask，列从1开始（0留给文字）
    for row in range(3):
        for col, (modality, name) in enumerate(zip(modalities, modality_names), start=1):
            ax = plt.subplot(gs[row, col])
            slice_img = modality[:, :, slice_idx]
            slice_img = np.rot90(slice_img, k=1)
            ax.imshow(slice_img, cmap='gray')

            if row == 1:
                mask = gt_mask[:, :, slice_idx]
                mask = np.rot90(mask, k=1)
                ax.imshow(create_rgba_mask(mask))
            elif row == 2:
                mask = pred_mask[:, :, slice_idx]
                mask = np.rot90(mask, k=1)
                ax.imshow(create_rgba_mask(mask))

            if row == 0:
                ax.set_title(name, fontsize=14)

            ax.axis('off')

    # 图例部分占整行宽度，列从0到4
    ax_legend = plt.subplot(gs[3, :])
    ax_legend.axis('off')
    legend_patches = [
        mpatches.Patch(color='blue', label='NCR/NET (label 1)'),
        mpatches.Patch(color='green', label='ED (label 2)'),
        mpatches.Patch(color='red', label='ET (label 4)'),
    ]
    ax_legend.legend(handles=legend_patches, loc='center', ncol=3, fontsize=14)

    plt.subplots_adjust(left=0.03, right=0.97, top=0.93, bottom=0.07)

    plt.suptitle('Multimodal MRI with Ground Truth and Prediction Masks', fontsize=18, y=0.98)
    plt.text(0.5, 0.95, f'Case: {case_name}', fontsize=12, ha='center', transform=fig.transFigure)

    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()


def main():
    # 设置数据目录和文件路径
    
    # # BraTS 2018 Training set
    # data_dir = './data/BraTS2018/MICCAI_BraTS_2018_Data_Training/LGG/Brats18_TCIA13_621_1'
    # pred_dir = './Pred/val_fold2_pred_experiment56'
    # case_name = os.path.basename(data_dir)
    
    # t1_path = os.path.join(data_dir, f'{case_name}_t1.nii')
    # t1ce_path = os.path.join(data_dir, f'{case_name}_t1ce.nii')
    # t2_path = os.path.join(data_dir, f'{case_name}_t2.nii')
    # flair_path = os.path.join(data_dir, f'{case_name}_flair.nii')
    # gt_mask_path = os.path.join(data_dir, f'{case_name}_seg.nii')     # ground truth
    # pred_mask_path = os.path.join(pred_dir, f'{case_name}_pred_mask.nii.gz') # model prediction
    
    
    # # BraTS 2018 val dataset
    # data_dir = './Data/BraTS2020/MICCAI_BraTS2020_TrainingData/Brats18_WashU_W033_1'
    # gt_dir = './Pred/nnUNetTrainer'
    # pred_dir = './Pred/test_pred_experiment56'
    # case_name = os.path.basename(data_dir)

    # t1_path = os.path.join(data_dir,f't1.nii.gz')
    # t1ce_path = os.path.join(data_dir, f't1ce.nii.gz')
    # t2_path = os.path.join(data_dir, f't2.nii.gz')
    # flair_path = os.path.join(data_dir, f'flair.nii.gz')
    # gt_mask_path = os.path.join(gt_dir, f'{case_name}.nii.gz')     # ground truth
    # pred_mask_path = os.path.join(pred_dir, f'{case_name}_pred_mask.nii.gz') # model prediction  

    
    # BraTS 2020 Training set
    data_dir = './data/BraTS2020/MICCAI_BraTS2020_TrainingData/BraTS20_Training_360'
    pred_dir = './Pred/BraTS2020_val_pred_exp65/val_fold5_pred'
    case_name = os.path.basename(data_dir)
    
    t1_path = os.path.join(data_dir, f'{case_name}_t1.nii')
    t1ce_path = os.path.join(data_dir, f'{case_name}_t1ce.nii')
    t2_path = os.path.join(data_dir, f'{case_name}_t2.nii')
    flair_path = os.path.join(data_dir, f'{case_name}_flair.nii')
    gt_mask_path = os.path.join(data_dir, f'{case_name}_seg.nii')     # ground truth
    pred_mask_path = os.path.join(pred_dir, f'{case_name}_pred_mask.nii.gz') # model prediction   
    
    save_dir = './visualise/BraTS_2020/val'
    flag = 'exp65'


    # # BraTS 2023 Training set
    # data_dir = './Data/BraTS2023/BraTS-GLI-00006-000'
    # pred_dir = './Pred/BraTS23_val_fold2_pred'
    # case_name = os.path.basename(data_dir)
    
    # t1_path = os.path.join(data_dir, f'{case_name}-t1n.nii.gz')
    # t1ce_path = os.path.join(data_dir, f'{case_name}-t1c.nii.gz')
    # t2_path = os.path.join(data_dir, f'{case_name}-t2w.nii.gz')
    # flair_path = os.path.join(data_dir, f'{case_name}-t2f.nii.gz')
    # gt_mask_path = os.path.join(data_dir, f'{case_name}-seg.nii.gz')     # ground truth
    # pred_mask_path = os.path.join(pred_dir, f'{case_name}_pred_mask.nii.gz') # model prediction
    
    
    # # Clinical Data
    # data_dir = 'C:/Users/ajhz839/code/Python_Projects/Spike-Transformer-BTSUnet/Pred/clinical_data/clinical_data/20220111_pre_OP'
    # pred_dir = 'C:/Users/ajhz839/code/Python_Projects/Spike-Transformer-BTSUnet/Pred/clinical_data/test_pred_soft_ensemble'
    # prefix = '20220114_35320313_BSR'
    # case_name = os.path.basename(data_dir)

    # t1_path = os.path.join(data_dir, f't1.nii.gz')
    # t1ce_path = os.path.join(data_dir, f't1ce.nii.gz')
    # t2_path = os.path.join(data_dir, f't2.nii.gz')
    # flair_path = os.path.join(data_dir, f'flair.nii.gz')
    # gt_mask_path = os.path.join(pred_dir, f'20220111_pre_OP_pred_mask.nii.gz')     # ground truth
    # pred_mask_path = os.path.join(pred_dir, f'20220111_pre_OP_pred_mask.nii.gz') # model prediction
    

    # 加载图像数据
    t1 = load_nifti_image(t1_path)
    t1ce = load_nifti_image(t1ce_path)
    t2 = load_nifti_image(t2_path)
    flair = load_nifti_image(flair_path)
    gt_mask = load_nifti_image(gt_mask_path).astype(np.uint8)
    pred_mask = load_nifti_image(pred_mask_path).astype(np.uint8)
    save_dir = './visualise/BraTS_2020/val'
    save_path = os.path.join(save_dir, f'{case_name}_{flag}.png')


    # 可视化中间层 (中间 slice 通常是肿瘤区域)
    best_slice = select_best_slice(gt_mask)
    plot_modalities_with_masks(t1, t1ce, t2, flair, gt_mask, pred_mask, case_name, slice_idx=best_slice, save_path=save_path)
    print(f"Visualization completed and saved to {save_path}")
    
if __name__ == '__main__':
    main()
    
