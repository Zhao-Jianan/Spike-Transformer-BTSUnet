import os
os.chdir(os.path.dirname(__file__))
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import torch.optim as optim
from sklearn.model_selection import KFold, train_test_split
from spike_former_unet_model import spike_former_unet3D_8_384, spike_former_unet3D_8_512, spike_former_unet3D_8_768
# from simple_unet_model import spike_former_unet3D_8_384, spike_former_unet3D_8_512, spike_former_unet3D_8_768
from losses import BratsDiceLoss, BratsFocalLoss, BratsDiceLosswithFPPenalty, BratsTverskyLoss, AdaptiveRegionalLoss
from utils import init_weights, save_metrics_to_file,save_case_list
from train import train_fold, get_scheduler, EarlyStopping
from plot import plot_metrics
from data_loader import get_data_loaders
from config import config as cfg        
from glob import glob
import random
import numpy as np
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy("file_system")


def setseed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # set_determinism(seed)

# 主执行流程：5折交叉验证
def main():
    # 设置随机种子
    setseed(cfg.seed)
    
    case_dirs = []
    for root in cfg.root_dirs: 
        if not os.path.isdir(root):
            raise FileNotFoundError(f"Root directory '{root}' does not exist or is not a directory.")
        case_dirs += sorted(glob(os.path.join(root, '*')))
        
    
    # 打印配置名
    print(cfg.device)
    print("CUDA_VISIBLE_DEVICES:", os.environ["CUDA_VISIBLE_DEVICES"])

    # 设置损失函数和优化器
    if cfg.loss_function == 'focal':
        criterion = BratsFocalLoss(
            alpha=0.25,
            gamma=2.0,
            reduction='mean').to(cfg.device)
    elif cfg.loss_function == 'dice':
        criterion = BratsDiceLoss(
            smooth_nr=0,
            smooth_dr=1e-5,
            squared_pred=True,
            sigmoid=True,
            weights=cfg.loss_weights).to(cfg.device)
    elif cfg.loss_function == 'dice_with_fp_penalty':
        criterion = BratsDiceLosswithFPPenalty(
            smooth_nr=0,
            smooth_dr=1e-5,
            squared_pred=True,
            sigmoid=True,
            weights=cfg.loss_weights).to(cfg.device)
    elif cfg.loss_function == 'tversky':
        criterion = BratsTverskyLoss(
            alpha=0.7,
            beta=0.3,
            smooth_nr=0.0,
            smooth_dr=1e-5,
            sigmoid=True,
            weights=cfg.loss_weights).to(cfg.device)      
    elif cfg.loss_function == 'adaptive_regional':    
        criterion = AdaptiveRegionalLoss(
            global_weight=0.7, 
            regional_weight=0.3, 
            smooth=1e-6, 
            pool_size=8).to(cfg.device)
    else:
        raise ValueError(f"Unsupported loss function: {cfg.loss_function}")
    
    # 训练验证集 + 测试集划分
    if cfg.split_data:
        train_val_dirs, test_dirs = train_test_split(
            case_dirs, test_size=cfg.test_ratio, random_state=cfg.seed, shuffle=True
        )
        save_case_list(test_dirs, name='test_cases', fold=None)
    else:
        train_val_dirs = case_dirs
        test_dirs = []
    # 打印数据集信息
    print(f"Total cases: {len(case_dirs)} | Train+Val: {len(train_val_dirs)} | Test: {len(test_dirs)}")


    kf = KFold(n_splits=cfg.k_folds, shuffle=True, random_state=cfg.seed)
    # print("weights device:", criterion.weights.device)
    # 开始交叉验证
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_val_dirs)):
        if cfg.model_type == 'spike_former_unet3D_8_384':
            model = spike_former_unet3D_8_384(
                num_classes=cfg.num_classes,
                T=cfg.T,
                norm_type=cfg.norm_type,
                step_mode=cfg.step_mode).to(cfg.device)  # 模型
        elif cfg.model_type == 'spike_former_unet3D_8_512':
            model = spike_former_unet3D_8_512(
                num_classes=cfg.num_classes,
                T=cfg.T,
                norm_type=cfg.norm_type,
                step_mode=cfg.step_mode).to(cfg.device)
        elif cfg.model_type == 'spike_former_unet3D_8_768':
            model = spike_former_unet3D_8_768(
                num_classes=cfg.num_classes,
                T=cfg.T,
                norm_type=cfg.norm_type,
                step_mode=cfg.step_mode).to(cfg.device)
        else:
            raise ValueError(f"Unsupported model type: {cfg.model_type}")
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
            train_case_dirs, val_case_dirs, cfg.patch_size, cfg.batch_size, cfg.T, cfg.encode_method, cfg.num_workers
            )


        # 调用训练函数
        train_losses, val_losses, val_dices, val_mean_dices, val_hd95s, lr_history = train_fold(
            train_loader, val_loader, sliding_window_val_loader, model, optimizer, criterion, cfg.device, cfg.num_epochs, \
                fold, cfg.compute_hd, scheduler, early_stopping, cfg.use_amp
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