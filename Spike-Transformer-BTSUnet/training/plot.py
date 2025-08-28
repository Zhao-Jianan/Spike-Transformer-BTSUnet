import matplotlib.pyplot as plt
import os
from utilities.logger import logger

def plot_metrics(
    train_losses, 
    val_losses, 
    val_dices, 
    val_mean_dices, 
    val_hd95s, 
    lr_history, 
    fold_number):
    
    epochs = range(1, len(train_losses) + 1)

    # 拆分 val_dices 和 val_dices_style2 为多个类别
    val_dices_wt = [d['WT'] for d in val_dices]
    val_dices_tc = [d['TC'] for d in val_dices]
    val_dices_et = [d['ET'] for d in val_dices]

    # 判断是否绘制 HD95 曲线
    has_hd95 = len(val_hd95s) == len(epochs)
    rows = 3 if has_hd95 else 2
    cols = 2

    plt.figure(figsize=(12, rows * 4))

    # Subplot 1: Learning Rate
    plt.subplot(rows, cols, 1)
    plt.plot(epochs, lr_history, label='Learning Rate', color='orange')
    plt.title("Learning Rate Schedule")
    plt.xlabel("Epoch")
    plt.ylabel("LR")
    plt.grid(True)

    # Subplot 2: Loss
    plt.subplot(rows, cols, 2)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Val Loss')
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # Subplot 3: Dice Scores (Style 1)
    plt.subplot(rows, cols, 3)
    plt.plot(epochs, val_dices_wt, label='WT Dice')
    plt.plot(epochs, val_dices_tc, label='TC Dice')
    plt.plot(epochs, val_dices_et, label='ET Dice')
    plt.plot(epochs, val_mean_dices, label='Mean Dice', linestyle='--', color='black')
    plt.title("Validation Dice Scores (Style 1)")
    plt.xlabel("Epoch")
    plt.ylabel("Dice")
    plt.legend()
    plt.grid(True)

    # Subplot 4 (optional): HD95
    if has_hd95:
        plt.subplot(rows, cols, 4)
        plt.plot(epochs, val_hd95s, 'r', label='Val 95HD')
        plt.title("Validation 95% Hausdorff Distance")
        plt.xlabel("Epoch")
        plt.ylabel("95HD")
        plt.legend()
        plt.grid(True)
    else:
        logger.warning(f"[Warning] Skipping 95HD plot: val_hd95s has length {len(val_hd95s)}, expected {len(epochs)}")

    plt.tight_layout()

    os.makedirs("visualise", exist_ok=True)
    save_path = f"visualise/metrics_fold{fold_number+1}.png"
    plt.savefig(save_path)
    plt.close()

    
    
    
