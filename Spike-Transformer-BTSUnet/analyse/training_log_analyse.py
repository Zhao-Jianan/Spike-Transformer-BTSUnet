import json
from utilities.logger import logger

def find_best_val_mean_dice(json_path):
    # 读取整个 JSON 文件（是一个大字典）
    with open(json_path, 'r') as f:
        data = json.load(f)

    # 提取所有字段
    train_losses = data["train_losses"]
    val_losses = data["val_losses"]
    val_dices_wt = data["val_dices_wt"]
    val_dices_tc = data["val_dices_tc"]
    val_dices_et = data["val_dices_et"]
    val_mean_dices = data["val_mean_dices"]
    if "val_mean_dices_style2" in data:
        val_dices_wt_style2 = data["val_dices_wt_style2"]
        val_dices_tc_style2 = data["val_dices_tc_style2"]
        val_dices_et_style2 = data["val_dices_et_style2"]
        val_mean_dices_style2 = data["val_mean_dices_style2"]

    # 找到 val_mean_dices 的最大值索引
    best_idx = val_mean_dices.index(max(val_mean_dices))

    # 打印对应值
    logger.info(f"Best index: {best_idx + 1}")
    logger.info(f"train_losses: {train_losses[best_idx]}")
    logger.info(f"val_losses: {val_losses[best_idx]}")
    logger.info(f"val_dices_wt: {val_dices_wt[best_idx]}")
    logger.info(f"val_dices_tc: {val_dices_tc[best_idx]}")
    logger.info(f"val_dices_et: {val_dices_et[best_idx]}")
    logger.info(f"val_mean_dices: {val_mean_dices[best_idx]}")
    
    
    if "val_mean_dices_style2" in data:
        best_idx_style2 = val_mean_dices_style2.index(max(val_mean_dices_style2))
        logger.info(f"\n===================================")
        logger.info(f"======Best performance style2======")
        logger.info(f"Best index: {best_idx_style2 + 1}")
        logger.info(f"train_losses: {train_losses[best_idx_style2]}")
        logger.info(f"val_losses: {val_losses[best_idx_style2]}")
        logger.info(f"val_dices_wt_style2: {val_dices_wt_style2[best_idx_style2]}")
        logger.info(f"val_dices_tc_style2: {val_dices_tc_style2[best_idx_style2]}")
        logger.info(f"val_dices_et_style2: {val_dices_et_style2[best_idx_style2]}")
        logger.info(f"val_mean_dices_style2: {val_mean_dices_style2[best_idx_style2]}")


def main():
    root_path = "././Project/Result/"
    experiment_name = "109-bra20_simpleunet_144c64p4b_1e3_1e6_poly20_diceloss111_addskip_seed42_aware075_stoploss"
    fold_num = "5"
    file_path = f"{root_path}{experiment_name}/fold_{fold_num}_metrics.json"

    find_best_val_mean_dice(file_path)

if __name__ == "__main__":
    main()
