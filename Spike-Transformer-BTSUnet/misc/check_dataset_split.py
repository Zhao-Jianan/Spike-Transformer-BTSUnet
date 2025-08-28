import os
os.chdir(os.path.dirname(__file__))
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import torch
from sklearn.model_selection import KFold
from config import config as cfg        
from glob import glob
import random
import numpy as np
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy("file_system")
from utilities.logger import logger


def setseed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # set_determinism(seed)

# 主执行流程：5折交叉验证
def main():
    setseed(cfg.seed)
    
    # 汇总所有case路径
    case_dirs = []
    for root in cfg.root_dirs:
        if not os.path.isdir(root):
            raise FileNotFoundError(f"Root directory '{root}' does not exist or is not a directory.")
        case_dirs += sorted(glob(os.path.join(root, '*')))

    logger.info(f"Using device: {cfg.device}")
    logger.info(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not Set')}")

    # K-fold cross validation
    kf = KFold(n_splits=cfg.k_folds, shuffle=True, random_state=cfg.seed)

    for fold, (train_idx, val_idx) in enumerate(kf.split(case_dirs)):
        logger.info(f"\n====== Fold {fold + 1}/{cfg.k_folds} ======")
        
        # 打印当前fold的验证集case路径
        val_case_dirs = [case_dirs[i] for i in val_idx]
        logger.info(f"Validation set ({len(val_case_dirs)} cases):")
        for val_case in val_case_dirs:
            logger.info(f"{os.path.basename(val_case)}")


    logger.info("\n Finished showing all validation cases:")


if __name__ == "__main__":
    main()