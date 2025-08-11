import gzip
import shutil
from pathlib import Path

root = Path("/hpc/ajhz839/compared_models/nnFormer/project/nnFormer_raw/nnFormer_raw_data/Task05_BraTS2020/imagesTr/") 
for nii_file in root.glob("*.nii"):
    gz_file = nii_file.with_suffix(".nii.gz")  # 直接改成 .nii.gz 后缀
    with open(nii_file, 'rb') as f_in:
        with gzip.open(gz_file, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)