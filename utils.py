import torch
import torch.nn.functional as F
import torch.nn as nn
import os, json

def downsample_label(label, size):
    label = label.unsqueeze(1).float()  # [B,1,D,H,W]
    label_down = F.interpolate(label, size=size, mode='nearest')
    return label_down.squeeze(1).long()  # [B,D,H,W]


def init_weights(module):
    """
    针对使用LIFNode的网络，采用Xavier初始化Conv3d和Linear层权重，BatchNorm层权重初始化为1，偏置初始化为0。
    """
    if isinstance(module, nn.Conv3d):
        # Xavier初始化，适合LIFNode激活
        nn.init.xavier_normal_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.Linear):
        nn.init.xavier_normal_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.BatchNorm3d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.MultiheadAttention):
        # 对于 MultiheadAttention 手动初始化其包含的 Linear 层
        nn.init.xavier_normal_(module.in_proj_weight)
        if module.in_proj_bias is not None:
            nn.init.constant_(module.in_proj_bias, 0)
        nn.init.xavier_normal_(module.out_proj.weight)
        if module.out_proj.bias is not None:
            nn.init.constant_(module.out_proj.bias, 0)


def save_metrics_to_file(train_losses, val_losses, val_dices, val_mean_dices, val_hd95s, lr_history, fold, output_dir="metrics"):
    os.makedirs(output_dir, exist_ok=True)

    # 拆分 val_dices 字典数组为独立的列表
    val_dices_wt = [d['WT'] for d in val_dices]
    val_dices_tc = [d['TC'] for d in val_dices]
    val_dices_et = [d['ET'] for d in val_dices]

    data = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_dices_wt": val_dices_wt,
        "val_dices_tc": val_dices_tc,
        "val_dices_et": val_dices_et,
        "val_mean_dices": val_mean_dices,
        "val_hd95s": val_hd95s,
        "lr_history": lr_history
    }

    filepath = os.path.join(output_dir, f"fold_{fold+1}_metrics.json")
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)
        
        
def save_val_case_list(val_case_dirs, fold, save_dir=None):
    """
    保存验证集case名单到txt文件。

    参数：
    - val_case_dirs: list，验证集文件夹路径列表
    - fold: int，当前fold编号
    - save_dir: str或None，保存目录路径，默认当前路径下的'val_cases'文件夹

    返回：
    - val_list_path: str，保存的txt文件完整路径
    """
    if save_dir is None:
        save_dir = os.path.join(os.getcwd(), 'val_cases')  # 当前工作目录下的val_cases文件夹
    os.makedirs(save_dir, exist_ok=True)

    val_case_ids = [os.path.basename(p) for p in val_case_dirs]
    val_list_path = os.path.join(save_dir, f"val_cases_fold{fold}.txt")

    with open(val_list_path, 'w') as f:
        for case_id in val_case_ids:
            f.write(case_id + '\n')

    print(f"[Fold {fold}] Validation case list saved to: {val_list_path}")
    return val_list_path


def save_case_list(case_dirs, name, fold=None, save_dir=None):
    """
    保存指定 case 名单到 txt 文件。

    参数：
    - case_dirs: list[str]，case 路径列表
    - name: str，文件名前缀，例如 'val_cases' 或 'test_cases'
    - fold: int 或 None，当前fold编号；如果为 None，不加 fold 编号
    - save_dir: str 或 None，保存目录路径，默认当前路径下的 'val_cases' 文件夹

    返回：
    - case_list_path: str，保存的 txt 文件完整路径
    """
    if save_dir is None:
        save_dir = os.path.join(os.getcwd(), 'val_cases')
    os.makedirs(save_dir, exist_ok=True)

    case_ids = [os.path.basename(p) for p in case_dirs]
    if fold is not None:
        filename = f"{name}_fold{fold}.txt"
    else:
        filename = f"{name}.txt"

    case_list_path = os.path.join(save_dir, filename)

    with open(case_list_path, 'w') as f:
        for case_id in case_ids:
            f.write(case_id + '\n')

    print(f"[{name}] Case list saved to: {case_list_path}")
    return case_list_path