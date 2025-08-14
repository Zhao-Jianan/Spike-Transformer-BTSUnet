import os
import torch
import numpy as np
from .metrics import compute_hd95, dice_score_braTS_per_sample_avg, dice_score_nnunet_multiclass_torch_batch
import time
from inference.inference_helper import TemporalSlidingWindowInference
from config import config as cfg
from torch.amp import autocast, GradScaler
import math

# 训练配置
# 学习率调度器
def get_scheduler(
    optimizer,
    num_warmup_epochs,
    num_total_epochs,
    base_lr,
    min_lr=1e-6,
    scheduler='cosine',
    power=2.0
    ):
    # 参数合法性检查
    if base_lr <= min_lr:
        raise ValueError(f"base_lr ({base_lr}) must be greater than min_lr ({min_lr}).")
    if num_warmup_epochs >= num_total_epochs:
        raise ValueError("num_warmup_epochs must be less than num_total_epochs.")

    if scheduler == 'cosine':
        # 余弦退火 + 线性warmup
        def lr_lambda(epoch):
            if epoch < num_warmup_epochs:
                return epoch / max(1, num_warmup_epochs)

            progress = (epoch - num_warmup_epochs) / max(1, num_total_epochs - num_warmup_epochs)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress)) 
            decayed = (1 - min_lr / base_lr) * cosine_decay + min_lr / base_lr
            return decayed
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    if scheduler == 'polynomial':
        # 多项式衰减
        def lr_lambda(epoch):
            if epoch < num_warmup_epochs:
                return float(epoch) / float(max(1, num_warmup_epochs))
            else:
                progress = float(epoch) / float(max(1, num_total_epochs))
                poly_decay = (1 - progress) ** power
                decayed = (1 - min_lr / base_lr) * poly_decay + min_lr / base_lr
                return decayed
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler}")
    
    
# 早停机制 
class EarlyStopping:
    def __init__(self, patience=cfg.early_stop_patience, delta=0.0, start_epoch=200):
        self.patience = patience
        self.delta = delta
        self.start_epoch = start_epoch
        self.best_score = None
        self.best_epoch = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_dice_mean, current_epoch):
            """
            Args:
                val_dice_mean (float): 验证集上的 Dice 平均分（作为早停判断依据）
                current_epoch (int): 当前 epoch 数
            """
            if current_epoch < self.start_epoch:
                return  # 尚未达到启用早停的轮次

            score = val_dice_mean

            if self.best_score is None:
                self.best_score = score
                self.best_epoch = current_epoch
            elif score < self.best_score + self.delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
                    print(f"[EarlyStopping] Early stopping triggered at epoch {current_epoch}.")
            else:
                self.best_score = score
                self.best_epoch = current_epoch
                self.counter = 0



# 训练和验证函数
def train(train_loader, model, optimizer, criterion, device, debug=False, compute_time=False, 
          use_amp=True, use_grad_accum=False, accumulation_steps=1):
    model.train()
    
    running_loss = 0.0
    scaler = GradScaler(enabled=use_amp)  # 混合精度缩放器
    print('Train -------------->>>>>>>')

    for step, (x_seq, y) in enumerate(train_loader, start=1):
        if compute_time:
            torch.cuda.synchronize()
            start_time = time.time()
            
        if debug:
            if torch.isnan(x_seq).any() or torch.isinf(x_seq).any():
                print(f"[FATAL] x_seq contains NaN/Inf at batch, stopping.")
                break
            if torch.isnan(y).any() or torch.isinf(y).any():
                print(f"[FATAL] y contains NaN/Inf at batch, stopping.")
                break
        
        x_seq = x_seq.permute(1, 0, 2, 3, 4, 5).contiguous().to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        
        if compute_time:
            torch.cuda.synchronize()
            time1 = time.time()
            print(f"[DEBUG] Permute time: {time1 - start_time:.4f} seconds")
        
        # 梯度累积模式下，每 accumulation_steps 才清零一次梯度
        if use_grad_accum:
            if (step - 1) % accumulation_steps == 0:
                optimizer.zero_grad(set_to_none=True)
        else:
            optimizer.zero_grad(set_to_none=True)  # 普通训练每 batch 清零

        with autocast(device_type='cuda', enabled=use_amp):
            output = model(x_seq)

            if compute_time:
                time2 = time.time()
                print(f"[DEBUG] Model forward time: {time2 - time1:.4f} seconds")

            if debug:
                if torch.isnan(output).any() or torch.isinf(output).any():
                    print(f"[FATAL] model output NaN/Inf at batch, stopping.")
                    break

            loss = criterion(output, y)

            # 如果使用梯度累积，则 loss 除以 accumulation_steps
            if use_grad_accum:
                loss = loss / accumulation_steps
        
        if compute_time:
            torch.cuda.synchronize()
            time4 = time.time()
            print(f"[DEBUG] Loss computation time: {time4 - time2:.4f} seconds")
        
        if debug and torch.isnan(loss):
            print(f"[FATAL] loss NaN at batch, stopping at epoch")
            break

        scaler.scale(loss).backward()

        # 梯度累积时，只有到达 accumulation_steps 才更新一次
        if not use_grad_accum or step % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()

        running_loss += loss.item() * (accumulation_steps if use_grad_accum else 1)

    avg_loss = running_loss / len(train_loader)
    return avg_loss


# 初始化滑动窗口推理器
val_inferencer = TemporalSlidingWindowInference(
    patch_size=cfg.inference_patch_size,
    overlap=cfg.overlap,
    sw_batch_size=8,
    mode="constant",
    encode_method=cfg.encode_method,
    T=cfg.T,
    num_classes=cfg.num_classes
)


def validate(val_loader, model, criterion, device, val_crop_mode='crop', compute_hd=False, debug=False, use_amp=True):
    model.eval()
    
    total_loss = 0.0
    total_dice = {'TC': 0.0, 'WT': 0.0, 'ET': 0.0}
    hd95s = []
    
    print(f'[Validation] Mode: {val_crop_mode} -------------->>>>>>>')
    with torch.no_grad():
        for i, (x_seq, y) in enumerate(val_loader):
            if debug:
                # 数据检查：检查输入 x_seq 和标签 y 是否包含 NaN 或 Inf
                if torch.isnan(x_seq).any() or torch.isinf(x_seq).any():
                    print(f"[FATAL] x_seq contains NaN/Inf at batch, stopping.")
                    break
                if torch.isnan(y).any() or torch.isinf(y).any():
                    print(f"[FATAL] y contains NaN/Inf at batch, stopping.")
                    break
            
            if val_crop_mode == 'sliding_window':
                x_seq = x_seq.to(device, non_blocking=True)
            else:
                x_seq = x_seq.permute(1, 0, 2, 3, 4, 5).to(device, non_blocking=True) # [T, B, C, D, H, W]
            y_onehot = y.float().to(device, non_blocking=True)
            with autocast(device_type='cuda', enabled=use_amp):
                if val_crop_mode == 'sliding_window':
                    output = val_inferencer(x_seq, model)
                else:
                    output = model(x_seq)  # [B, C, D, H, W]，未过 softmax

                loss = criterion(output, y_onehot)
            
            if debug:
                # 检查 output 是否为 NaN 或 Inf
                if torch.isnan(output).any() or torch.isinf(output).any():
                    print(f"[FATAL] model output NaN/Inf at batch, stopping.")
                    print(f"Output: {output}")
                    break

            dice = dice_score_nnunet_multiclass_torch_batch(output, y_onehot) # dict: {'TC':..., 'WT':..., 'ET':...}
            # 累加各类别的dice值
            for key in total_dice.keys():
                total_dice[key] += dice[key]
                
            if compute_hd:
                hd95 = compute_hd95(output, y_onehot)
                # if np.isnan(hd95):
                #     print(f"[Warning] NaN in HD95")
                hd95s.append(hd95)

            total_loss += loss.item()
            
    num_batches = len(val_loader)
    avg_loss = total_loss / num_batches
    avg_dice = {k: v / num_batches for k, v in total_dice.items()}
    avg_hd95 = np.nanmean(hd95s) if compute_hd else np.nan

    return avg_loss, avg_dice, avg_hd95


def train_one_fold(
    train_loader, val_loader, sliding_window_val_loader, model, optimizer, 
    criterion, device, num_epochs, fold, compute_hd, cfg,
    scheduler=None, early_stopping=None, use_amp=True,
    sliding_window_threshold=0.80, use_grad_accum=False, accumulation_steps=1):
    
    train_losses = []
    patch_val_losses = []
    patch_val_dices = []
    patch_val_mean_dices = []
    patch_val_hd95s = []
    
    entire_val_losses = []
    entire_val_dices = []
    entire_val_mean_dices = []
    entire_val_hd95s = []

    lr_history = []

    patch_best_dice = 0.0
    entire_best_dice = 0.0
    min_dice_threshold = 0.70
    
    warmup_epochs = cfg.num_warmup_epochs
    train_crop_mode = cfg.train_crop_mode


    for epoch in range(num_epochs):
        print(f'----------[Fold {fold}] Epoch {epoch+1}/{num_epochs} ----------')
        if train_crop_mode == 'warmup_weighted_random':
        # 计算当前中心 crop 概率（线性衰减）
            if epoch < warmup_epochs:
                prob = 1.0 - epoch / warmup_epochs  # 从1.0线性下降到0.0
            else:
                prob = 0.0

            if hasattr(train_loader.dataset, 'center_crop_prob'):
                train_loader.dataset.center_crop_prob = prob
                if prob > 0:
                    print(f"Epoch {epoch+1}: center crop prob = {prob:.2f}")
                    
        # ===== 训练阶段 =====    
        train_start_time = time.time()
        train_loss = train(train_loader, model, optimizer, criterion, device, use_amp=use_amp,
                           use_grad_accum=use_grad_accum, accumulation_steps=accumulation_steps)
        # 计时结束
        train_end_time = time.time()
        train_elapsed_time = train_end_time - train_start_time
        print(f"[Fold {fold}] Epoch {epoch+1} training time: {train_elapsed_time:.2f} seconds")
        
        train_losses.append(train_loss)
        
        # ===== 验证阶段 (Patch-based) =====        
        patch_val_start_time = time.time()        
        patch_val_loss, patch_val_dice, patch_val_hd95 = validate(
            val_loader, model, criterion, device, 
            val_crop_mode=cfg.val_crop_mode, compute_hd=compute_hd, use_amp=use_amp)

        # 计时结束
        patch_val_end_time = time.time()
        patch_val_elapsed_time = patch_val_end_time - patch_val_start_time
        print(f"[Fold {fold}] Epoch {epoch+1} val time: {patch_val_elapsed_time:.2f} seconds")
        
        num_classes = len(patch_val_dice)
        patch_val_mean_dice = sum(patch_val_dice.values()) / num_classes
        patch_val_losses.append(patch_val_loss)
        patch_val_dices.append(patch_val_dice)
        patch_val_mean_dices.append(patch_val_mean_dice)

        if compute_hd:
            patch_val_hd95s.append(patch_val_hd95)
            
        # 日志输出
        val_dice_str = " | ".join([f"{k}: {v:.4f}" for k, v in patch_val_dice.items()])

        print(f"[Fold {fold}] Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {patch_val_loss:.4f}")
        print(f"Dice: {val_dice_str} | Mean: {patch_val_mean_dice:.4f}")
        if compute_hd:
            print(f"95HD: {patch_val_hd95:.4f}")
                  
        # 保存检查点
        ckpt_dir = "checkpoints"
        os.makedirs(ckpt_dir, exist_ok=True)
        if patch_val_mean_dice > patch_best_dice and patch_val_mean_dice >= min_dice_threshold:
            patch_best_dice = patch_val_mean_dice
            checkpoint_path = os.path.join(ckpt_dir, f'best_model_fold{fold}.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f"[Fold {fold}] Epoch {epoch+1}: New best Patch Dice = {patch_val_mean_dice:.4f}, model saved.")

            # 滑动窗口验证
            if cfg.val_crop_mode != 'sliding_window' and cfg.sliding_window_val:
                if  patch_val_mean_dice >= sliding_window_threshold:
                    print(f"[Fold {fold}] Epoch {epoch+1}: Performing sliding window validation...")
                    # 计时开始
                    entire_val_start_time = time.time()
                    entire_val_loss, entire_val_dice, entire_val_hd95 = validate(
                        sliding_window_val_loader, model, criterion, device, 
                        val_crop_mode='sliding_window', compute_hd=compute_hd, use_amp=use_amp)
                    entire_val_end_time = time.time()
                    entire_val_elapsed_time = entire_val_end_time - entire_val_start_time
                    print(f"[Fold {fold}] Epoch {epoch+1} entire val time: {entire_val_elapsed_time:.2f} seconds")
                    num_classes = len(entire_val_dice) 
                    entire_val_mean_dice = sum(entire_val_dice.values()) / num_classes
                    entire_val_losses.append(entire_val_loss)
                    entire_val_dices.append(entire_val_dice)
                    entire_val_mean_dices.append(entire_val_mean_dice)

                    if compute_hd:
                        entire_val_hd95s.append(entire_val_hd95)

                    val_dice_str = " | ".join([f"{k}: {v:.4f}" for k, v in entire_val_dice.items()])

                    print(f"Sliding Window Validation Results | ")
                    print(f"Dice: {val_dice_str} | Mean: {entire_val_mean_dice:.4f}")
                    if compute_hd:
                        print(f"95HD: {entire_val_hd95:.4f}")
                        
                    if entire_val_mean_dice > entire_best_dice:
                        entire_best_dice = entire_val_mean_dice
                        torch.save(model.state_dict(), os.path.join(ckpt_dir, f'best_model_fold{fold}_entire.pth'))
                        print(f"[Fold {fold}] Epoch {epoch+1}: Sliding Window New best Dice = {entire_val_mean_dice:.4f}, model saved.")
                    
        # ===== 学习率调度 =====
        if scheduler is not None:
            scheduler.step()  # 更新学习率
            # 打印当前学习率
            current_lrs = [param_group['lr'] for param_group in optimizer.param_groups]
            print(f"Epoch {epoch+1} learning rate(s): {current_lrs[0]}")
            lr_history.append(current_lrs[0]) 
            
        # ===== 早停机制 =====            
        if early_stopping is not None:
            early_stopping(patch_val_mean_dice, epoch+1)
            if early_stopping.early_stop:
                print(f"[Fold {fold}] Early stopping at epoch {epoch+1}")
                break

    return train_losses, patch_val_losses, patch_val_dices, patch_val_mean_dices, patch_val_hd95s, lr_history


# 折训练函数
def train_fold(train_loader, val_loader, sliding_window_val_loader, 
               model, optimizer, criterion, device, 
               num_epochs, fold, compute_hd, cfg,
               scheduler, early_stopping, use_amp,
               use_grad_accum, accumulation_steps):
    print(f"\n[Fold {fold+1}] Training Started")
    
    train_losses, val_losses, val_dices, val_mean_dices, val_hd95s, lr_history = train_one_fold(
        train_loader, val_loader, sliding_window_val_loader, model, optimizer, criterion, device, 
        num_epochs, fold+1, compute_hd, cfg,
        scheduler=scheduler, early_stopping=early_stopping, use_amp=use_amp,
        use_grad_accum=use_grad_accum, accumulation_steps=accumulation_steps
    )
    
    print(f"[Fold {fold+1}] Training Completed")
    
    return train_losses, val_losses, val_dices, val_mean_dices, val_hd95s, lr_history

