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
    

def visulize(case_name, t1_path, t1ce_path, t2_path, flair_path, gt_mask_path, pred_mask_path, save_dir, experiment_index): 
    # 加载图像数据
    t1 = load_nifti_image(t1_path)
    t1ce = load_nifti_image(t1ce_path)
    t2 = load_nifti_image(t2_path)
    flair = load_nifti_image(flair_path)
    gt_mask = load_nifti_image(gt_mask_path).astype(np.uint8)
    pred_mask = load_nifti_image(pred_mask_path).astype(np.uint8)
    save_path = os.path.join(save_dir, f'{case_name}_exp{experiment_index}.png')
    
    # 可视化中间层 (中间 slice 通常是肿瘤区域)
    best_slice = select_best_slice(gt_mask)
    plot_modalities_with_masks(t1, t1ce, t2, flair, gt_mask, pred_mask, case_name, slice_idx=best_slice, save_path=save_path)
    print(f"Visualization completed and saved to {save_path}")
    
        




def visulize_for_brats18(experiment_index, case_name, mode='val', fold=1):
    # BraTS 2018 Training set
    def find_case_dir(base_dir, case_name):
        for grade in ['LGG', 'HGG']:
            case_dir = os.path.join(base_dir, grade, case_name)
            if os.path.exists(case_dir):
                return case_dir
        raise FileNotFoundError(f"Case {case_name} not found in LGG or HGG directories.")
    
    base_dir = './data/BraTS2018/MICCAI_BraTS_2018_Data_Training'
    data_dir = find_case_dir(base_dir, case_name)
    if mode == 'val':
        pred_dir = f'./Pred/val_fold{fold}_pred_experiment{experiment_index}'
        save_dir = './visualise/BraTS_2018/val'
    
        t1_path = os.path.join(data_dir, f'{case_name}_t1.nii')
        t1ce_path = os.path.join(data_dir, f'{case_name}_t1ce.nii')
        t2_path = os.path.join(data_dir, f'{case_name}_t2.nii')
        flair_path = os.path.join(data_dir, f'{case_name}_flair.nii')
        gt_mask_path = os.path.join(data_dir, f'{case_name}_seg.nii')     # ground truth
        pred_mask_path = os.path.join(pred_dir, f'{case_name}_pred_mask.nii.gz') # model prediction
                
    elif mode == 'test':
        pred_dir = f'./Pred/test_pred_experiment{experiment_index}'
        gt_dir = './Pred/nnUNetTrainer'
        save_dir = './visualise/BraTS_2018/test'
        
        t1_path = os.path.join(data_dir,f't1.nii.gz')
        t1ce_path = os.path.join(data_dir, f't1ce.nii.gz')
        t2_path = os.path.join(data_dir, f't2.nii.gz')
        flair_path = os.path.join(data_dir, f'flair.nii.gz')
        gt_mask_path = os.path.join(gt_dir, f'{case_name}.nii.gz')     # ground truth
        pred_mask_path = os.path.join(pred_dir, f'{case_name}_pred_mask.nii.gz') # model prediction  
    visulize(case_name, t1_path, t1ce_path, t2_path, flair_path, gt_mask_path, pred_mask_path, save_dir, experiment_index)


    
def visulize_for_brats20(experiment_index, case_name, mode='val'):
    data_dir = f'./data/BraTS2020/MICCAI_BraTS2020_TrainingData/{case_name}'
    if mode == 'val':
        # 自动查找包含该 case_name 的 fold 目录
        pred_base_dir = f'./Pred/BraTS2020/validation_dataset/BraTS2020_val_pred_exp{experiment_index}'
        found = False
        for fold_dir in os.listdir(pred_base_dir):
            fold_path = os.path.join(pred_base_dir, fold_dir)
            if os.path.isdir(fold_path):
                pred_mask_path = os.path.join(fold_path, f'{case_name}_pred_mask.nii.gz')
                if os.path.exists(pred_mask_path):
                    pred_dir = fold_path
                    found = True
                    break
        if not found:
            raise FileNotFoundError(f"Prediction for case {case_name} not found in any val_foldX_pred directory.")
        save_dir = './visualise/BraTS_2020/val'
    elif mode == 'test':
        pred_dir = f'./Pred/BraTS2020/test_dataset/test_pred_soft_ensemble_exp{experiment_index}'
        save_dir = './visualise/BraTS_2020/test'
        
    t1_path = os.path.join(data_dir, f'{case_name}_t1.nii')
    t1ce_path = os.path.join(data_dir, f'{case_name}_t1ce.nii')
    t2_path = os.path.join(data_dir, f'{case_name}_t2.nii')
    flair_path = os.path.join(data_dir, f'{case_name}_flair.nii')
    gt_mask_path = os.path.join(data_dir, f'{case_name}_seg.nii')     # ground truth
    pred_mask_path = os.path.join(pred_dir, f'{case_name}_pred_mask.nii.gz') # model prediction   
        
    visulize(case_name, t1_path, t1ce_path, t2_path, flair_path, gt_mask_path, pred_mask_path, save_dir, experiment_index)
    
    
def visulize_for_brats23(experiment_index, case_name, mode='val', fold=1):
    # BraTS 2023 val
    data_dir = f'./Data/BraTS2023/{case_name}'
    pred_dir = f'./Pred/BraTS23_val_fold{fold}_pred'
    save_dir = './visualise/BraTS_2023/val'
    
    t1_path = os.path.join(data_dir, f'{case_name}-t1n.nii.gz')
    t1ce_path = os.path.join(data_dir, f'{case_name}-t1c.nii.gz')
    t2_path = os.path.join(data_dir, f'{case_name}-t2w.nii.gz')
    flair_path = os.path.join(data_dir, f'{case_name}-t2f.nii.gz')
    gt_mask_path = os.path.join(data_dir, f'{case_name}-seg.nii.gz')     # ground truth
    pred_mask_path = os.path.join(pred_dir, f'{case_name}_pred_mask.nii.gz') # model prediction
        
    visulize(case_name, t1_path, t1ce_path, t2_path, flair_path, gt_mask_path, pred_mask_path, save_dir, experiment_index)


def visulize_for_clinical_data(experiment_index, time_point, patient_name):
    data_dir = f'C:/Users/ajhz839/code/Python_Projects/Spike-Transformer-BTSUnet/Pred/clinical_data/clinical_data/{time_point}'
    pred_dir = f'C:/Users/ajhz839/code/Python_Projects/Spike-Transformer-BTSUnet/Pred/clinical_data/test_pred_soft_ensemble'
    save_dir = f'C:/Users/ajhz839/code/Python_Projects/Spike-Transformer-BTSUnet/visualise/clinical_data'
    case_name = f'{patient_name}_{time_point}'

    t1_path = os.path.join(data_dir, f't1.nii.gz')
    t1ce_path = os.path.join(data_dir, f't1ce.nii.gz')
    t2_path = os.path.join(data_dir, f't2.nii.gz')
    flair_path = os.path.join(data_dir, f'flair.nii.gz')
    gt_mask_path = os.path.join(pred_dir, f'{time_point}_mask.nii.gz')     # ground truth
    pred_mask_path = os.path.join(pred_dir, f'{time_point}_pred_mask.nii.gz') # model prediction
        
    visulize(case_name, t1_path, t1ce_path, t2_path, flair_path, gt_mask_path, pred_mask_path, save_dir, experiment_index)


def main():
    # # BraTS 2018 Training set & val set
    # mode = 'val'  # 'val' or 'test'
    # experiment_index = '56'
    # case_name = 'Brats18_TCIA13_621_1'
    # fold = 1
    # visulize_for_brats18(experiment_index, case_name,  mode=mode, fold=fold)

    
    # BraTS 2020 val and test
    mode = 'test'  # 'val' or 'test'
    experiment_index = 76
    case_name = 'BraTS20_Training_306'
    visulize_for_brats20(experiment_index, case_name,  mode=mode)



    # # BraTS 2023 Training set
    # case_name = 'BraTS-GLI-00006-000'
    # fold = 1
    # experiment_index = 'exp65'
    # visulize_for_brats23(experiment_index, case_name, mode='val', fold=1)

    
    # # Clinical Data
    # time_point = '20220111_pre_OP'
    # patient_name = '20220114_35320313_BSR'
    # experiment_index = 'exp65'
    # visulize_for_clinical_data(experiment_index, time_point, patient_name)

    
if __name__ == '__main__':
    main()
    
