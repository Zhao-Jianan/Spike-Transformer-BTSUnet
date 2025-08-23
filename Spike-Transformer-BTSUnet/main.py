import os
os.chdir(os.path.dirname(__file__))
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import torch.optim as optim
from sklearn.model_selection import KFold, train_test_split
from model.spike_former_unet_model import spike_former_unet3D_8_384, spike_former_unet3D_8_512, spike_former_unet3D_8_768
# from model.simple_unet_model import spike_former_unet3D_8_384, spike_former_unet3D_8_512, spike_former_unet3D_8_768
from training.losses import BratsDiceLoss, BratsDiceLossOptimized, BratsFocalLoss, BratsDiceLosswithFPPenalty, BratsTverskyLoss, AdaptiveRegionalLoss
from training.utils import init_weights, save_metrics_to_file, save_case_list
from training.train import train_fold, get_scheduler, EarlyStopping
from training.plot import plot_metrics
from training.data_loader import get_data_loaders
from config import config as cfg        
from glob import glob
import random
import numpy as np
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy("file_system")

# ================= 工具函数 =================
def setseed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # set_determinism(seed)


def get_loss_function(cfg):
    """Return the loss function based on cfg.loss_function."""
    loss_map = {
        'focal': lambda: BratsFocalLoss(alpha=0.25, gamma=2.0, reduction='mean'),
        'dice': lambda: BratsDiceLossOptimized( # BratsDiceLoss
            smooth_nr=0, smooth_dr=1e-5, squared_pred=True,
            sigmoid=True, weights=cfg.loss_weights),
        'dice_with_fp_penalty': lambda: BratsDiceLosswithFPPenalty(
            smooth_nr=0, smooth_dr=1e-5, squared_pred=True,
            sigmoid=True, weights=cfg.loss_weights),
        'tversky': lambda: BratsTverskyLoss(
            alpha=0.7, beta=0.3, smooth_nr=0.0, smooth_dr=1e-5,
            sigmoid=True, weights=cfg.loss_weights),
        'adaptive_regional': lambda: AdaptiveRegionalLoss(
            global_weight=0.7, regional_weight=0.3, smooth=1e-6, pool_size=8)
    }

    if cfg.loss_function not in loss_map:
        raise ValueError(f"Unsupported loss function: {cfg.loss_function}")
    return loss_map[cfg.loss_function]().to(cfg.device)


def get_model(cfg):
    """Return the model instance based on cfg.model_type."""
    model_map = {
        'spike_former_unet3D_8_384': spike_former_unet3D_8_384,
        'spike_former_unet3D_8_512': spike_former_unet3D_8_512,
        'spike_former_unet3D_8_768': spike_former_unet3D_8_768
    }

    if cfg.model_type not in model_map:
        raise ValueError(f"Unsupported model type: {cfg.model_type}")

    return model_map[cfg.model_type](
        num_classes=cfg.num_classes,
        T=cfg.T,
        norm_type=cfg.norm_type,
        step_mode=cfg.step_mode
    ).to(cfg.device)
    
    
def collect_case_dirs(root_dirs):
    """Collect case directories from given root directories."""
    case_dirs = []
    for root in root_dirs:
        if not os.path.isdir(root):
            raise FileNotFoundError(f"Root directory '{root}' does not exist or is not a directory.")
        case_dirs += sorted(glob(os.path.join(root, '*')))
    return case_dirs



# 主执行流程：5折交叉验证
def main():
    # 设置随机种子
    setseed(cfg.seed)
    
    # Collect cases
    case_dirs = collect_case_dirs(cfg.root_dirs)
            
    # 打印配置名
    print(cfg.device)
    print("CUDA_VISIBLE_DEVICES:", os.environ["CUDA_VISIBLE_DEVICES"])

    # 设置损失函数
    criterion = get_loss_function(cfg)
    
    # 训练验证集 + 测试集划分
    if cfg.split_data:
        train_val_dirs, test_dirs = train_test_split(
            case_dirs, test_size=cfg.test_ratio, random_state=cfg.seed, shuffle=True
        )
        save_case_list(train_val_dirs, name='train_val_cases', fold=None)
        save_case_list(test_dirs, name='test_cases', fold=None)
    else:
        train_val_dirs = case_dirs
        test_dirs = []
    # 打印数据集信息
    print(f"Total cases: {len(case_dirs)} | Train+Val: {len(train_val_dirs)} | Test: {len(test_dirs)}")

    # 交叉验证
    kf = KFold(n_splits=cfg.k_folds, shuffle=True, random_state=cfg.seed)

    for fold, (train_idx, val_idx) in enumerate(kf.split(train_val_dirs)):
        model = get_model(cfg)
        optimizer = optim.AdamW(model.parameters(), lr=cfg.base_lr, eps=1e-8, weight_decay=1e-4)
        scheduler = get_scheduler(optimizer, cfg.num_warmup_epochs, cfg.num_epochs, 
                                  cfg.base_lr, cfg.min_lr, cfg.scheduler, cfg.power)
        early_stopping = EarlyStopping(patience=cfg.early_stop_patience, delta=0)

        # 根据交叉验证划分数据集
        train_case_dirs = [train_val_dirs[i] for i in train_idx]
        val_case_dirs = [train_val_dirs[i] for i in val_idx]
        
        # 保存验证集名单（供推理用）
        save_case_list(train_case_dirs, name='train_cases', fold=fold)
        save_case_list(val_case_dirs, name='val_cases', fold=fold)

        # 训练和验证数据加载器
        train_loader, val_loader, sliding_window_val_loader = get_data_loaders(
            train_case_dirs, val_case_dirs, cfg.patch_size, cfg.batch_size, cfg.T,
            cfg.encode_method, cfg.num_workers, cfg.modalities,
            cfg.modality_separator, cfg.image_suffix, cfg.sliding_window_val
        )


        # 调用训练函数
        train_losses, val_losses, val_dices, val_mean_dices, val_hd95s, lr_history = train_fold(
            train_loader, val_loader, sliding_window_val_loader, model, optimizer,
            criterion, cfg.device, cfg.num_epochs, fold, cfg.compute_hd,
            cfg, scheduler, early_stopping, cfg.use_amp, cfg.use_grad_accum, cfg.accumulation_steps
        )
        
        # 保存指标
        save_metrics_to_file(train_losses, val_losses, val_dices, val_mean_dices, val_hd95s, lr_history, fold)

        # 绘制训练过程的图形
        plot_metrics(
            train_losses, val_losses,  val_dices, val_mean_dices, val_hd95s, lr_history, fold
        )

    print("\nTraining and Validation completed across all folds.")

if __name__ == "__main__":
    main()