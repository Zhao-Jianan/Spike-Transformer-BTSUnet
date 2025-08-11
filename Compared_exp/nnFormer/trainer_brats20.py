import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from batchgenerators.transforms import Compose, MirrorTransform, SpatialTransform, GammaTransform, BrightnessMultiplicativeTransform
from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from nnunet.training.dataloading.dataset_loading import unpack_dataset
from nnunet.training.learning_rate.poly_lr import poly_lr
from batchgenerators.utilities.file_and_folder_operations import join
import numpy as np
import torch.backends.cudnn as cudnn

class nnFormerTrainerV2_nnformer_tumor(nnUNetTrainerV2):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None,
                 batch_dice=True, stage=None, unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory,
                         batch_dice, stage, unpack_data, deterministic, fp16)
        self.initial_lr = 1e-2
        self.max_num_epochs = 1000
        self.max_num_epochs = 1000
        self.do_ds = True  # deep supervision
        self.weight_decay = 3e-5
        self.optimizer_name = "SGD"
        self.optimizer_kwargs = {'lr': self.initial_lr, 'weight_decay': self.weight_decay,
                                 'momentum': 0.99, 'nesterov': True}
        self.deep_supervision_scales = None
        self.deep_supervision_weights = None

    # ========【修改1】替换原有的 do_split()，改成用 txt 划分 BraTS2020 =========
    def do_split(self):
        """
        使用已准备好的 txt 文件进行训练集和验证集划分
        train_cases_fold{x}.txt / val_cases_fold{x}.txt
        """
        if self.fold == "all":
            return  # 全量训练模式，不做划分

        # 获取当前 fold
        fold_id = self.fold

        # txt 文件路径
        train_file = join(self.dataset_directory, f"train_cases_fold{fold_id}.txt")
        val_file = join(self.dataset_directory, f"val_cases_fold{fold_id}.txt")

        # 读取病例 ID
        with open(train_file, 'r') as f:
            self.tr_keys = [line.strip() for line in f if line.strip()]

        with open(val_file, 'r') as f:
            self.val_keys = [line.strip() for line in f if line.strip()]

        print(f"[do_split] Using predefined BraTS2020 split for fold {fold_id}: "
              f"{len(self.tr_keys)} training cases, {len(self.val_keys)} validation cases.")
    # ===========================================================================

    def setup_DA_params(self):
        self.deep_supervision_scales = list(np.array(self.net_num_pool_op_kernel_sizes) / np.array(self.net_num_pool_op_kernel_sizes[0]))
        self.deep_supervision_weights = np.array([1 / (2 ** i) for i in range(len(self.deep_supervision_scales))])
        self.deep_supervision_weights[-1] = 0
        self.deep_supervision_weights = self.deep_supervision_weights / self.deep_supervision_weights.sum()

        # Data augmentation settings
        self.data_aug_params = {
            'rotation_x': (-np.pi, np.pi),
            'rotation_y': (-np.pi, np.pi),
            'rotation_z': (-np.pi, np.pi),
            'scale_range': (0.7, 1.4),
            'random_crop': False,
            'do_elastic': False,
            'p_el_per_sample': 0.2,
            'alpha': (0., 900.),
            'sigma': (9., 13.),
            'p_rot_per_sample': 0.5,
            'p_scale_per_sample': 0.5,
            'p_do_mirror': 0.5,
            'mirror_axes': (0, 1, 2),
            'do_additive_brightness': True,
            'p_additive_brightness': 0.3,
            'additive_brightness_mu': 0.0,
            'additive_brightness_sigma': 0.1,
            'p_gamma': 0.3,
            'gamma_range': (0.7, 1.5),
            'gamma_retain_stats': True
        }

    def load_pretrained_weights(self, pretrained_weights):
        checkpoint = torch.load(pretrained_weights, map_location=torch.device('cpu'))
        if 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']
        self.network.load_state_dict(checkpoint, strict=False)
        print(f"Pretrained weights loaded from {pretrained_weights}")

    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False):
        data_dict = next(data_generator)
        data = data_dict['data']
        target = data_dict['target']
        data = data.to(self.device)
        target = target.to(self.device)

        self.optimizer.zero_grad()
        output = self.network(data)
        del data

        if self.do_ds:
            loss = self.loss(output, target)
        else:
            loss = self.loss(output[0], target)

        if do_backprop:
            loss.backward()
            clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

        if run_online_evaluation:
            self.run_online_evaluation(output, target)

        del target
        return loss.detach().cpu().numpy()

    def validate(self, do_mirroring: bool = True, use_sliding_window: bool = True,
                 step_size: float = 0.5, save_softmax: bool = True, use_gaussian: bool = True, overwrite: bool = True):
        self.network.eval()
        self.do_ds = False
        super().validate(do_mirroring, use_sliding_window, step_size, save_softmax, use_gaussian, overwrite)
        self.do_ds = True

    def on_train_start(self):
        # 可在这里加载预训练权重
        pretrained_path = None
        self.load_pretrained_weights(pretrained_path)
