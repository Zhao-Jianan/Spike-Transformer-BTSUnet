import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceCrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, dice_weight=1.0, smooth=1e-6):
        """
        :param weight: class weights for CrossEntropyLoss
        :param dice_weight: scaling factor for Dice loss component
        :param smooth: smoothing to avoid division by zero
        """
        super(DiceCrossEntropyLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(weight=weight)
        self.dice_weight = dice_weight
        self.smooth = smooth

    def forward(self, pred, target):
        """
        :param pred: model prediction logits of shape [B, C, D, H, W]
        :param target: ground truth labels of shape [B, D, H, W]
        """
        ce = self.ce_loss(pred, target)

        # Convert target to one-hot format: shape [B, C, D, H, W]
        num_classes = pred.shape[1]
        target_onehot = F.one_hot(target, num_classes).permute(0, 4, 1, 2, 3).contiguous().float()

        # Apply softmax to logits to get probabilities
        pred_soft = F.softmax(pred, dim=1)

        # Compute Dice loss
        dims = (0, 2, 3, 4)  # batch and spatial dims
        intersection = torch.sum(pred_soft * target_onehot, dims)
        cardinality = torch.sum(pred_soft + target_onehot, dims)
        dice_per_class = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        dice_loss = 1. - dice_per_class[1:].mean()

        return ce + self.dice_weight * dice_loss
    
  
    
class BratsDiceLoss(nn.Module):
    def __init__(self, 
                 smooth_nr=0.0, 
                 smooth_dr=1e-5, 
                 squared_pred=True, 
                 sigmoid=True, 
                 weights=None, 
                 include_background=True,
                 batch=False,
                 reduction='mean'):
        super().__init__()
        self.smooth_nr = smooth_nr
        self.smooth_dr = smooth_dr
        self.squared_pred = squared_pred
        self.sigmoid = sigmoid
        self.include_background = include_background
        self.reduction = reduction
        self.batch = batch

        if weights is None:
            weights = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
        else:
            weights = torch.tensor(weights, dtype=torch.float32)
            weights = weights / weights.sum()
            
        self.register_buffer("weights", weights)

    def forward(self, pred, target):
        """
        pred: [B, 3, D, H, W] 模型输出 logits 或概率
        target: [B, 3, D, H, W] one-hot 标签 [TC, WT, ET]
        """

        if self.sigmoid:
            pred = torch.sigmoid(pred)          
        n_pred_ch = pred.shape[1]
        n_target_ch = target.shape[1]
        
        # 如果pred带背景通道，忽略背景通道，取通道1开始
        if not self.include_background:
            if n_pred_ch == 1:
                print("[Warning] single channel prediction, `include_background=False` ignored.")
            elif n_pred_ch <= 3:
                # No background channel to remove, assume input is only [TC, WT, ET]
                pass
            else:
                if n_target_ch > 1:
                    target = target[:, 1:]
                pred = pred[:, 1:]
        else:
            if n_pred_ch != 3:
                print("[Warning] `include_background=True` but input has no background channel.")

        if self.squared_pred:
            pred_sq = pred ** 2
            target_sq = target ** 2
        else:
            pred_sq = pred
            target_sq = target

        dims = (0, 2, 3, 4) if self.batch else (2, 3, 4)
       
        intersection = torch.sum(pred * target, dims)
        cardinality = torch.sum(pred_sq + target_sq, dims)

        dice = (2. * intersection + self.smooth_nr) / (cardinality + self.smooth_dr)
        loss_per_channel = 1 - dice  # shape [3]

        weights = self.weights.to(pred.device)

        if self.reduction == 'mean':
            weighted_loss = (loss_per_channel * weights).mean()
        elif self.reduction == 'sum':
            weighted_loss = (loss_per_channel * weights).sum()
        else:
            raise ValueError(f"Unsupported reduction type: {self.reduction}")

        return weighted_loss
 
 
    
class BratsDiceLossOptimized(nn.Module):
    def __init__(self, 
                 smooth_nr=0.0, 
                 smooth_dr=1e-5, 
                 squared_pred=True, 
                 sigmoid=True, 
                 weights=None, 
                 include_background=True,
                 batch=False,
                 reduction='mean'):
        super().__init__()
        self.smooth_nr = smooth_nr
        self.smooth_dr = smooth_dr
        self.squared_pred = squared_pred
        self.sigmoid = sigmoid
        self.include_background = include_background
        self.reduction = reduction
        self.batch = batch

        if weights is None:
            weights = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
        else:
            weights = torch.tensor(weights, dtype=torch.float32)
            weights = weights / weights.sum()
            
        self.register_buffer("weights", weights)

    def forward(self, pred, target):
        """
        pred:   [B, C, D, H, W] 模型输出 logits 或概率
        target: [B, C, D, H, W] 或 [B, 1, D, H, W]  (label map 或 one-hot)
        """
        if self.sigmoid:
            pred = torch.sigmoid(pred)          

        # ===== 显存优化：标签统一处理为 one-hot (bool) =====
        with torch.no_grad():
            if pred.shape != target.shape:
                # 从整型标签生成 one-hot，布尔节省显存
                target_onehot = torch.zeros_like(pred, dtype=torch.bool)
                target_onehot.scatter_(1, target.long(), 1)
            else:
                target_onehot = target.bool()

            if not self.include_background:
                target_onehot = target_onehot[:, 1:]
            elif target_onehot.shape[1] != pred.shape[1]:
                # 若 include_background=True 但 target 没有背景通道
                pass  

        # ===== 预测通道对齐 =====
        if not self.include_background:
            pred = pred[:, 1:]

        # ===== 计算平方项 =====
        pred_sq = pred ** 2 if self.squared_pred else pred
        target_sq = target_onehot.float() ** 2 if self.squared_pred else target_onehot.float()

        dims = (0, 2, 3, 4) if self.batch else (2, 3, 4)

        # ===== 显存优化：提前 no_grad 计算 sum_gt =====
        with torch.no_grad():
            sum_gt = target_sq.sum(dims)

        intersection = torch.sum(pred * target_onehot.float(), dims)
        sum_pred = pred_sq.sum(dims)

        dice = (2. * intersection + self.smooth_nr) / torch.clamp_min(sum_pred + sum_gt + self.smooth_dr, 1e-8)
        loss_per_channel = 1 - dice

        # ===== 通道加权 =====
        weights = self.weights.to(pred.device)
        if self.reduction == 'mean':
            weighted_loss = (loss_per_channel * weights).mean()
        elif self.reduction == 'sum':
            weighted_loss = (loss_per_channel * weights).sum()
        else:
            raise ValueError(f"Unsupported reduction type: {self.reduction}")

        return weighted_loss    
    

class BratsDiceLosswithFPPenalty(nn.Module):
    def __init__(self, 
                 smooth_nr=0.0, 
                 smooth_dr=1e-5, 
                 squared_pred=True, 
                 sigmoid=True, 
                 weights=None, 
                 include_background=True,
                 batch=False,
                 reduction='mean',
                 fp_lambda=0.5):  # 添加假阳性惩罚系数
        super().__init__()
        self.smooth_nr = smooth_nr
        self.smooth_dr = smooth_dr
        self.squared_pred = squared_pred
        self.sigmoid = sigmoid
        self.include_background = include_background
        self.reduction = reduction
        self.batch = batch
        self.fp_lambda = fp_lambda

        if weights is None:
            weights = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
        else:
            weights = torch.tensor(weights, dtype=torch.float32)
            weights = weights / weights.sum()
            
        self.register_buffer("weights", weights)

    def forward(self, pred, target):
        """
        pred: [B, 3, D, H, W] 模型输出 logits 或概率
        target: [B, 3, D, H, W] one-hot 标签 [TC, WT, ET]
        """
        if self.sigmoid:
            pred = torch.sigmoid(pred)

        n_pred_ch = pred.shape[1]

        if not self.include_background:
            if n_pred_ch == 1:
                print("Warning: single channel prediction, `include_background=False` ignored.")
            elif n_pred_ch <= 3:
                # No background channel to remove, assume input is only [TC, WT, ET]
                pass
            else:
                target = target[:, 1:]
                pred = pred[:, 1:]
        else:
            if n_pred_ch != 3:
                print("Warning: `include_background=True` but input has no background channel.")

        if self.squared_pred:
            pred_sq = pred ** 2
            target_sq = target ** 2
        else:
            pred_sq = pred
            target_sq = target

        dims = (0, 2, 3, 4) if self.batch else (2, 3, 4)

        intersection = torch.sum(pred * target, dims)
        cardinality = torch.sum(pred_sq + target_sq, dims)

        dice = (2. * intersection + self.smooth_nr) / (cardinality + self.smooth_dr)
        loss_per_channel = 1 - dice  # shape [3]

        weights = self.weights.detach()
        if self.reduction == 'mean':
            weighted_loss = (loss_per_channel * weights).mean()
        elif self.reduction == 'sum':
            weighted_loss = (loss_per_channel * weights).sum()
        else:
            raise ValueError(f"Unsupported reduction type: {self.reduction}")

        # -------------------------------
        # 假阳性惩罚项（基于 FP Rate）
        # -------------------------------
        with torch.no_grad():
            pred_bin = (pred > 0.5).float()  # hard prediction
        target_float = target.float()
        false_positive = pred_bin * (1 - target_float)  # pred=1, gt=0
        pred_positive = pred_bin  # pred=1

        fp = false_positive.sum(dims)          # shape [C] or [B, C]
        pp = pred_positive.sum(dims) + 1e-6    # 防止除0
        fp_rate = fp / pp                      # shape [C]

        fp_penalty = (fp_rate * weights).mean()  # 加权平均

        # -------------------------------
        # 加权组合总 loss
        # -------------------------------
        total_loss = weighted_loss + self.fp_lambda * fp_penalty
        return total_loss
    

class BratsTverskyLoss(nn.Module):
    def __init__(self,
                 alpha=0.7,
                 beta=0.3,
                 smooth_nr=0.0,
                 smooth_dr=1e-5,
                 sigmoid=True,
                 weights=None,
                 include_background=True,
                 batch=False,
                 reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth_nr = smooth_nr
        self.smooth_dr = smooth_dr
        self.sigmoid = sigmoid
        self.include_background = include_background
        self.reduction = reduction
        self.batch = batch

        if weights is None:
            weights = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
        else:
            weights = torch.tensor(weights, dtype=torch.float32)
            weights = weights / weights.sum()

        self.register_buffer("weights", weights)

    def forward(self, pred, target):
        """
        pred: [B, 3, D, H, W] 模型输出 logits 或概率
        target: [B, 3, D, H, W] one-hot 标签 [TC, WT, ET]
        """
        if self.sigmoid:
            pred = torch.sigmoid(pred)

        n_pred_ch = pred.shape[1]

        if not self.include_background:
            if n_pred_ch == 1:
                print("Warning: single channel prediction, `include_background=False` ignored.")
            elif n_pred_ch <= 3:
                # No background channel to remove, assume input is only [TC, WT, ET]
                pass
            else:
                target = target[:, 1:]
                pred = pred[:, 1:]
        else:
            if n_pred_ch != 3:
                print("Warning: `include_background=True` but input has no background channel.")
                
        pred = pred.float()
        target = target.float()

        if self.batch:
            dims = (0, 2, 3, 4)
        else:
            dims = (2, 3, 4)

        tp = torch.sum(pred * target, dims)
        fp = torch.sum(pred * (1 - target), dims)
        fn = torch.sum((1 - pred) * target, dims)

        tversky = (tp + self.smooth_nr) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth_dr
        )
        loss_per_channel = 1 - tversky  # shape [3]

        weights = self.weights.detach()

        if self.reduction == 'mean':
            weighted_loss = (loss_per_channel * weights).mean()
        elif self.reduction == 'sum':
            weighted_loss = (loss_per_channel * weights).sum()
        else:
            raise ValueError(f"Unsupported reduction type: {self.reduction}")

        return weighted_loss 
 
     
   
class BratsFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(BratsFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target):
        """
        inputs: (B, C, D, H, W) - raw logits
        targets: (B, C, D, H, W) - one-hot labels with values in {0, 1}
        """
        assert pred.shape == target.shape, "Input and target must have the same shape"
        target = target.float()
        probs = torch.sigmoid(pred)
        ce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        p_t = probs * target + (1 - probs) * (1 - target)
        focal_weight = (1 - p_t) ** self.gamma

        loss = self.alpha * focal_weight * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss

   
 
   
def adaptive_regional_loss(y_true, y_pred):
    """
    Optimized adaptive regional loss for 3D segmentation with faster training and robust NaN handling
    PyTorch implementation
    
    Args:
        y_true: Ground truth tensor of shape (batch, channels, depth, height, width)
        y_pred: Predicted tensor of shape (batch, channels, depth, height, width)
    
    Returns:
        final_loss: Scalar tensor containing the computed loss
    """
    
    # Basic type conversion and clipping
    y_true = y_true.float()
    y_pred = torch.clamp(y_pred.float(), min=1e-7, max=1.0)
    
    # Global dice calculation for 3D (dim=[2, 3, 4] for depth, height, width)
    smooth = 1e-6
    intersection = torch.sum(y_true * y_pred, dim=[2, 3, 4])
    union = torch.sum(y_true, dim=[2, 3, 4]) + torch.sum(y_pred, dim=[2, 3, 4])
    
    # Avoid zero denominators
    union = torch.where(union == 0, torch.ones_like(union) * smooth, union)
    
    global_dice = (2.0 * intersection + smooth) / (union + smooth)
    dice_loss = 1.0 - torch.mean(global_dice)
    
    # Regional dice with 3D average pooling
    # PyTorch expects (batch, channels, depth, height, width) format
    pooled_true = F.avg_pool3d(y_true, kernel_size=8, stride=8)
    pooled_pred = F.avg_pool3d(y_pred, kernel_size=8, stride=8)
    
    # Calculate regional dice for 3D
    region_intersection = torch.sum(pooled_true * pooled_pred, dim=[2, 3, 4])
    region_union = torch.sum(pooled_true, dim=[2, 3, 4]) + torch.sum(pooled_pred, dim=[2, 3, 4])
    
    # Avoid zero denominators in regional dice
    region_union = torch.where(region_union == 0, torch.ones_like(region_union) * smooth, region_union)
    
    region_dice = (2.0 * region_intersection + smooth) / (region_union + smooth)
    region_loss = 1.0 - torch.mean(region_dice)
    
    # Weighted final loss
    final_loss = 0.7 * dice_loss + 0.3 * region_loss
    
    # Final NaN check: replace NaNs with 0
    final_loss = torch.where(torch.isnan(final_loss), torch.tensor(0.0, device=final_loss.device), final_loss)
    
    return final_loss

# class AdaptiveRegionalLoss(nn.Module):
#     """
#     PyTorch module implementation of adaptive regional loss
#     """
#     def __init__(self, global_weight=0.7, regional_weight=0.3, smooth=1e-6, pool_size=8):
#         super(AdaptiveRegionalLoss, self).__init__()
#         self.global_weight = global_weight
#         self.regional_weight = regional_weight
#         self.smooth = smooth
#         self.pool_size = pool_size
    
#     def forward(self, y_pred, y_true):
#         """
#         Forward pass
        
#         Args:
#             y_pred: Predicted tensor of shape (batch, channels, depth, height, width)
#             y_true: Ground truth tensor of shape (batch, channels, depth, height, width)
        
#         Returns:
#             final_loss: Scalar tensor containing the computed loss
#         """
#         # Basic type conversion and clipping
#         y_true = y_true.float()
#         y_pred = torch.clamp(y_pred.float(), min=1e-7, max=1.0)
        
#         # Global dice calculation
#         intersection = torch.sum(y_true * y_pred, dim=[2, 3, 4])
#         union = torch.sum(y_true, dim=[2, 3, 4]) + torch.sum(y_pred, dim=[2, 3, 4])
        
#         # Avoid zero denominators
#         union = torch.where(union == 0, torch.ones_like(union) * self.smooth, union)
        
#         global_dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
#         dice_loss = 1.0 - torch.mean(global_dice)
        
#         # Regional dice with 3D average pooling
#         pooled_true = F.avg_pool3d(y_true, kernel_size=self.pool_size, stride=self.pool_size)
#         pooled_pred = F.avg_pool3d(y_pred, kernel_size=self.pool_size, stride=self.pool_size)
        
#         # Calculate regional dice
#         region_intersection = torch.sum(pooled_true * pooled_pred, dim=[2, 3, 4])
#         region_union = torch.sum(pooled_true, dim=[2, 3, 4]) + torch.sum(pooled_pred, dim=[2, 3, 4])
        
#         # Avoid zero denominators in regional dice
#         region_union = torch.where(region_union == 0, torch.ones_like(region_union) * self.smooth, region_union)
        
#         region_dice = (2.0 * region_intersection + self.smooth) / (region_union + self.smooth)
#         region_loss = 1.0 - torch.mean(region_dice)
        
#         # Weighted final loss
#         final_loss = self.global_weight * dice_loss + self.regional_weight * region_loss
        
#         # Final NaN check
#         final_loss = torch.where(torch.isnan(final_loss), torch.tensor(0.0, device=final_loss.device), final_loss)
        
#         return final_loss  
   



def adaptive_regional_loss(y_true, y_pred):
    """
    y_true: (B, C, D, H, W) ground truth
    y_pred: (B, C, D, H, W) prediction (already passed through sigmoid/softmax if needed)
    """
    # Ensure float and clamp values
    y_true = y_true.float()
    y_pred = y_pred.float().clamp(1e-7, 1.0)

    smooth = 1e-6
    
    # =================== GLOBAL DICE ===================
    intersection = torch.sum(y_true * y_pred, dim=[1, 2, 3, 4])
    union = torch.sum(y_true, dim=[1, 2, 3, 4]) + torch.sum(y_pred, dim=[1, 2, 3, 4])
    union = torch.where(union == 0, torch.ones_like(union) * smooth, union)

    global_dice = (2.0 * intersection + smooth) / (union + smooth)
    dice_loss = 1.0 - torch.mean(global_dice)

    # =================== REGIONAL DICE ===================
    pooled_true = F.avg_pool3d(y_true, kernel_size=8, stride=8, padding=0)
    pooled_pred = F.avg_pool3d(y_pred, kernel_size=8, stride=8, padding=0)

    region_intersection = torch.sum(pooled_true * pooled_pred, dim=[1, 2, 3, 4])
    region_union = torch.sum(pooled_true, dim=[1, 2, 3, 4]) + torch.sum(pooled_pred, dim=[1, 2, 3, 4])
    region_union = torch.where(region_union == 0, torch.ones_like(region_union) * smooth, region_union)

    region_dice = (2.0 * region_intersection + smooth) / (region_union + smooth)

    # =================== FRAGMENTATION DETECTION ===================
    region_dice_mean = torch.mean(region_dice)
    global_dice_mean = torch.mean(global_dice)

    fragmentation_ratio = global_dice_mean / (region_dice_mean + smooth)
    fragmentation_threshold = 1.3
    is_fragmented = fragmentation_ratio > fragmentation_threshold

    # =================== ADAPTIVE REGIONAL AGGREGATION ===================
    region_loss_normal = 1.0 - region_dice_mean

    batch_size = region_dice.shape[0]
    k = max(1, int(batch_size * 0.1))
    top_k_region_dice, _ = torch.topk(region_dice, k=k)
    region_loss_fragmented = 1.0 - torch.mean(top_k_region_dice)

    if is_fragmented:
        region_loss = region_loss_fragmented
    else:
        region_loss = region_loss_normal

    # =================== ADAPTIVE DIFFICULTY WEIGHTING ===================
    prediction_confidence = torch.abs(y_pred - 0.5) * 2.0
    difficulty_weight = 1.0 - prediction_confidence

    if is_fragmented:
        difficulty_multiplier = 0.2
    else:
        difficulty_multiplier = 0.5

    weighted_dice_loss = dice_loss * (1.0 + torch.mean(difficulty_weight) * difficulty_multiplier)

    # =================== ADAPTIVE FINAL WEIGHTING ===================
    if is_fragmented:
        global_weight = 0.8
    else:
        global_weight = 0.7

    regional_weight = 1.0 - global_weight

    final_loss = global_weight * weighted_dice_loss + regional_weight * region_loss

    # Avoid NaNs
    if torch.isnan(final_loss):
        final_loss = torch.tensor(0.0, dtype=torch.float32, device=y_true.device)

    return final_loss





class AdaptiveRegionalLoss(nn.Module):
    def __init__(self, 
                 smooth_nr=0.0, 
                 smooth_dr=1e-6, 
                 squared_pred=False, 
                 sigmoid=True, 
                 weights=None, 
                 include_background=True,
                 batch=False,
                 reduction='mean',
                 pool_size=8,
                 fragmentation_threshold=1.3,
                 topk_ratio=0.1):
        super().__init__()
        self.smooth_nr = smooth_nr
        self.smooth_dr = smooth_dr
        self.squared_pred = squared_pred
        self.sigmoid = sigmoid
        self.include_background = include_background
        self.reduction = reduction
        self.batch = batch
        self.pool_size = pool_size
        self.fragmentation_threshold = fragmentation_threshold
        self.topk_ratio = topk_ratio

        if weights is None:
            weights = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
        else:
            weights = torch.tensor(weights, dtype=torch.float32)
            weights = weights / weights.sum()
        self.register_buffer("weights", weights)

    def forward(self, pred, target):
        """
        pred:   [B, C, D, H, W] 模型输出 (logits 或概率)
        target: [B, C, D, H, W] one-hot 标签 [TC, WT, ET]
        """
        target = target.float()
        pred = pred.float()
        
        if self.sigmoid:
            pred = torch.sigmoid(pred)

        n_pred_ch = pred.shape[1]
        n_target_ch = target.shape[1]

        # ---------- 通道处理 ----------
        if not self.include_background:
            if n_pred_ch == 1:
                print("[Warning] single channel prediction, `include_background=False` ignored.")
            elif n_pred_ch <= 3:
                pass
            else:
                if n_target_ch > 1:
                    target = target[:, 1:]
                pred = pred[:, 1:]
        else:
            if n_pred_ch != 3:
                print("[Warning] `include_background=True` but input has no background channel.")

        if self.squared_pred:
            pred = pred ** 2
            target = target ** 2

        # ========= Global Dice =========
        dims = (0, 2, 3, 4) if self.batch else (2, 3, 4)
        intersection = torch.sum(pred * target, dim=dims)
        union = torch.sum(pred, dim=dims) + torch.sum(target, dim=dims)
        union = torch.where(union == 0, torch.ones_like(union) * self.smooth_dr, union)
        global_dice = (2.0 * intersection + self.smooth_nr) / (union + self.smooth_dr)
        dice_loss = 1.0 - torch.mean(global_dice)

        # ========= Regional Dice =========
        pooled_true = F.avg_pool3d(target, kernel_size=self.pool_size, stride=self.pool_size, padding=0)
        pooled_pred = F.avg_pool3d(pred, kernel_size=self.pool_size, stride=self.pool_size, padding=0)

        region_intersection = torch.sum(pooled_true * pooled_pred, dim=[1, 2, 3, 4])
        region_union = torch.sum(pooled_true, dim=[1, 2, 3, 4]) + torch.sum(pooled_pred, dim=[1, 2, 3, 4])
        region_union = torch.where(region_union == 0, torch.ones_like(region_union) * self.smooth_dr, region_union)
        region_dice = (2.0 * region_intersection + self.smooth_nr) / (region_union + self.smooth_dr)

        region_dice_mean = torch.mean(region_dice)
        global_dice_mean = torch.mean(global_dice)

        # Fragmentation detection
        fragmentation_ratio = global_dice_mean / (region_dice_mean + self.smooth_dr)
        is_fragmented = fragmentation_ratio > self.fragmentation_threshold

        # Adaptive regional aggregation
        region_loss_normal = 1.0 - region_dice_mean
        batch_size = region_dice.shape[0]
        k = max(1, int(batch_size * self.topk_ratio))
        top_k_region_dice, _ = torch.topk(region_dice, k=k)
        region_loss_fragmented = 1.0 - torch.mean(top_k_region_dice)
        region_loss = region_loss_fragmented if is_fragmented else region_loss_normal

        # Adaptive difficulty weighting
        prediction_confidence = torch.abs(pred - 0.5) * 2.0
        difficulty_weight = 1.0 - prediction_confidence
        difficulty_multiplier = 0.2 if is_fragmented else 0.5
        weighted_dice_loss = dice_loss * (1.0 + torch.mean(difficulty_weight) * difficulty_multiplier)

        # Adaptive global vs regional weighting
        global_weight = 0.8 if is_fragmented else 0.7
        regional_weight = 1.0 - global_weight
        final_loss = global_weight * weighted_dice_loss + regional_weight * region_loss

        if torch.isnan(final_loss):
            final_loss = torch.tensor(0.0, dtype=torch.float32, device=pred.device)

        return final_loss
   

def main():
    # loss_fn = BratsDiceLoss(squared_pred=True, sigmoid=True,include_background=True, batch=False, reduction='mean')
    # pred = torch.randn(3, 3, 128, 128, 128)  # logits
    # target = torch.randint(0, 1, (3, 3, 128, 128, 128)).float()  # one-hot mask
    # loss = loss_fn(pred, target)
    # print("Loss:", loss.item())
    
    # loss_fn = BratsFocalLoss(alpha=0.25, gamma=2.0)
    # pred = torch.randn(1, 3, 128, 128, 128)  # logits
    # target = torch.randint(0, 1, (1, 3, 128, 128, 128)).float()  # one-hot mask
    # loss = loss_fn(pred, target)  # pred_logits: raw output before sigmoid
    # print("Loss:", loss.item())
    
    loss_fn = AdaptiveRegionalLoss(global_weight=0.7, regional_weight=0.3, smooth=1e-6, pool_size=8)
    pred = torch.randn(1, 3, 128, 128, 128)  # logits
    target = torch.randint(0, 1, (1, 3, 128, 128, 128)).float()  # one-hot mask
    loss = loss_fn(pred, target)  # pred_logits: raw output before sigmoid
    print("Loss:", loss.item())    
    
    
if __name__ == "__main__":
    main()