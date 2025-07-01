import json

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

    # 找到 val_mean_dices 的最大值索引
    best_idx = val_mean_dices.index(max(val_mean_dices))

    # 打印对应值
    print(f"Best index: {best_idx + 1}")
    print(f"train_losses: {train_losses[best_idx]}")
    print(f"val_losses: {val_losses[best_idx]}")
    print(f"val_dices_wt: {val_dices_wt[best_idx]}")
    print(f"val_dices_tc: {val_dices_tc[best_idx]}")
    print(f"val_dices_et: {val_dices_et[best_idx]}")
    print(f"val_mean_dices: {val_mean_dices[best_idx]}")


def main():
    root_path = "./Result/"
    experiment_name = "49-entile18_T4_aware_1e3_1e6_poly20_214_noposembed_groupnorm_aug_2drop"
    fold_num = "1"
    file_path = f"{root_path}{experiment_name}/fold_{fold_num}_metrics.json"

    find_best_val_mean_dice(file_path)

if __name__ == "__main__":
    main()
