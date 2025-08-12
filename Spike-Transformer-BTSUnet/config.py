import torch

class Config:
    def __init__(self):
        self.gpu_name = 'cuda:0'
        self.device = torch.device(self.gpu_name if torch.cuda.is_available() else "cpu")
        self.seed =  42 # 42, 3407
        self.use_amp = True  # 使用自动混合精度训练
        self.split_data = True  # 是否划分测试集 2020-True, 2023-False
        self.test_ratio = 0.1  # 测试集比例
        
        # # BraTS2018
        # self.root_dirs = ['/hpc/ajhz839/data/BraTS2018/train/HGG', '/hpc/ajhz839/data/BraTS2018/train/LGG']   
        # self.modalities = ['t1', 't1ce', 't2', 'flair']
        # self.modality_separator = "_"
        # self.image_suffix = ".nii"
        # self.et_label = 4
        
        # BraTS2020
        self.root_dirs = ['/hpc/ajhz839/data/BraTS2020/MICCAI_BraTS2020_TrainingData'] 
        self.modalities = ['t1', 't1ce', 't2', 'flair']
        self.modality_separator = "_"
        self.image_suffix = ".nii"
        self.et_label = 4
        
        # # BraTS2021
        # self.root_dirs = ['/hpc/ajhz839/data/BraTS2021_Training_Data']
        # self.modalities = ['t1', 't1ce', 't2', 'flair']
        # self.modality_separator = "_"
        # self.image_suffix = ".nii.gz"
        # self.et_label = 4
        
        # # BraTS2023
        # self.root_dirs = ['/hpc/ajhz839/data/BraTS2023/train/']
        # self.modalities = ['t1n', 't1c', 't2w', 't2f']
        # self.modality_separator = "-" 
        # self.image_suffix = ".nii.gz"    
        # self.et_label = 3
        
        # # BraTS2025 SSA
        # self.root_dirs = ['/hpc/ajhz839/data/BraTS2023-SSA-V2/']
        # self.modalities = ['t1n', 't1c', 't2w', 't2f']
        # self.modality_separator = "-" 
        # self.image_suffix = ".nii.gz"    
        # self.et_label = 3
        
        self.encode_method = 'none'  # poisson, latency, weighted_phase, none

        self.tumor_crop_ratio = 0.75 # 肿瘤区域裁剪比例
        self.patch_size = [64, 64, 64] # [128, 128, 128]
        self.inference_patch_size = [128, 128, 128]  # 推理时的patch大小


        self.num_classes = 3
        self.model_type = 'spike_former_unet3D_8_384'  # spike_former_unet3D_8_384, spike_former_unet3D_8_512, spike_former_unet3D_8_768
        self.T = 4
        self.norm_type = 'group'  # group, batch
        # self.num_norm_groups = [8, 12, 24, 32]
        self.num_epochs = 600
        self.batch_size = 4
        self.k_folds = 5
        
        self.loss_function = 'dice' # dice, focal, dice_with_fp_penalty, tversky, adaptive_regional
        self.loss_weights = [2.0, 1.0, 4.0] # [2.0, 1.0, 4.0] [1.0, 1.0, 1.0]
        self.double_crop = False  # 是否使用双重裁剪（肿瘤区域和全脑区域）
        self.train_crop_mode = "tumor_aware_random"  # tumor_aware_random, warmup_weighted_random, random, tumor_center
        self.val_crop_mode = 'tumor_aware_random' # tumor_aware_random, sliding_window, random, tumor_center
        self.sliding_window_val = False # 是否在best_patch_dice_score的epoch使用滑动窗口验证
        self.overlap = 0.125
        self.num_workers = 8

        self.compute_hd = False

        self.scheduler = 'polynomial' # cosine, polynomial
        self.power = 2.0  # 300-2.0
        self.num_warmup_epochs = -1  # -1表示不使用warmup
        self.early_stop_patience = 80
        
        self.base_lr = 5e-4 # 1e-3
        self.min_lr = 1e-6

        self.step_mode = 'm'


# 全局单例
config = Config()
