import os
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

class BraTSDataset(Dataset):
    def __init__(self, case_list, data_root, transforms=None):
        self.case_list = case_list
        self.data_root = data_root
        self.transforms = transforms
        self.modalities = ['t1', 't1ce', 't2', 'flair']

    def __len__(self):
        return len(self.case_list)

    def __getitem__(self, idx):
        case = self.case_list[idx]
        case_dir = os.path.join(self.data_root, case)

        images = []
        for mod in self.modalities:
            img_path = os.path.join(case_dir, f"{case}_{mod}.nii")
            img = nib.load(img_path).get_fdata().astype(np.float32)
            images.append(img)
        image = np.stack(images, axis=0)  # [C, D, H, W]

        label_path = os.path.join(case_dir, f"{case}_seg.nii")
        label = nib.load(label_path).get_fdata().astype(np.uint8)

        image = torch.from_numpy(image)
        label = torch.from_numpy(label)

        if self.transforms:
            image, label = self.transforms(image, label)

        return image, label
