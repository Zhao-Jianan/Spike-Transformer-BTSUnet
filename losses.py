import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceLoss as MonaiDiceLoss

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
    

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        """
        :param smooth: smoothing to avoid division by zero
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.weights = {
            1: 2.0,   # Tumor Core (TC)
            2: 1.0,   # Edema
            3: 2.5    # Enhancing Tumor (ET)
        }

    def forward(self, pred, target):
        """
        :param pred: model prediction logits of shape [B, C, D, H, W]
        :param target: ground truth labels of shape [B, D, H, W]
        """
        num_classes = pred.shape[1]
        
        # Convert target to one-hot format: shape [B, C, D, H, W]
        target_onehot = F.one_hot(target, num_classes).permute(0, 4, 1, 2, 3).contiguous().float()

        # Apply softmax to logits to get probabilities
        pred_soft = F.softmax(pred, dim=1)

        # Compute Dice loss
        dims = (0, 2, 3, 4)  # batch and spatial dims
        intersection = torch.sum(pred_soft * target_onehot, dims)
        cardinality = torch.sum(pred_soft + target_onehot, dims)
        dice_per_class = (2. * intersection + self.smooth) / (cardinality + self.smooth)

        # Skip background class if needed (i.e., dice_per_class[1:])
        loss = 0.0
        total_weight = 0.0
        for cls in range(1, num_classes):
            weight = self.weights.get(cls, 1.0)  # 若未设定权重，默认权重为 1.0
            loss += weight * (1.0 - dice_per_class[cls])
            total_weight += weight

        return loss / total_weight
    
    
    
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
        
        # 如果pred带背景通道，忽略背景通道，取通道1开始
        if not self.include_background:
            if n_pred_ch == 1:
                print("single channel prediction, `include_background=False` ignored.")
            else:
                # if skipping background, removing first channel
                target = target[:, 1:]
                pred = pred[:, 1:]

        if self.squared_pred:
            pred_sq = pred ** 2
            target_sq = target ** 2
        else:
            pred_sq = pred
            target_sq = target

        if self.batch:
            dims = (0, 2, 3, 4)
        else:
            dims = (2, 3, 4)
       
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

class AdaptiveRegionalLoss(nn.Module):
    """
    PyTorch module implementation of adaptive regional loss
    """
    def __init__(self, global_weight=0.7, regional_weight=0.3, smooth=1e-6, pool_size=8):
        super(AdaptiveRegionalLoss, self).__init__()
        self.global_weight = global_weight
        self.regional_weight = regional_weight
        self.smooth = smooth
        self.pool_size = pool_size
    
    def forward(self, y_pred, y_true):
        """
        Forward pass
        
        Args:
            y_pred: Predicted tensor of shape (batch, channels, depth, height, width)
            y_true: Ground truth tensor of shape (batch, channels, depth, height, width)
        
        Returns:
            final_loss: Scalar tensor containing the computed loss
        """
        # Basic type conversion and clipping
        y_true = y_true.float()
        y_pred = torch.clamp(y_pred.float(), min=1e-7, max=1.0)
        
        # Global dice calculation
        intersection = torch.sum(y_true * y_pred, dim=[2, 3, 4])
        union = torch.sum(y_true, dim=[2, 3, 4]) + torch.sum(y_pred, dim=[2, 3, 4])
        
        # Avoid zero denominators
        union = torch.where(union == 0, torch.ones_like(union) * self.smooth, union)
        
        global_dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - torch.mean(global_dice)
        
        # Regional dice with 3D average pooling
        pooled_true = F.avg_pool3d(y_true, kernel_size=self.pool_size, stride=self.pool_size)
        pooled_pred = F.avg_pool3d(y_pred, kernel_size=self.pool_size, stride=self.pool_size)
        
        # Calculate regional dice
        region_intersection = torch.sum(pooled_true * pooled_pred, dim=[2, 3, 4])
        region_union = torch.sum(pooled_true, dim=[2, 3, 4]) + torch.sum(pooled_pred, dim=[2, 3, 4])
        
        # Avoid zero denominators in regional dice
        region_union = torch.where(region_union == 0, torch.ones_like(region_union) * self.smooth, region_union)
        
        region_dice = (2.0 * region_intersection + self.smooth) / (region_union + self.smooth)
        region_loss = 1.0 - torch.mean(region_dice)
        
        # Weighted final loss
        final_loss = self.global_weight * dice_loss + self.regional_weight * region_loss
        
        # Final NaN check
        final_loss = torch.where(torch.isnan(final_loss), torch.tensor(0.0, device=final_loss.device), final_loss)
        
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