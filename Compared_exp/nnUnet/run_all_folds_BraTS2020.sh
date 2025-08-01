#!/bin/bash

export CUDA_VISIBLE_DEVICES=1
export nnUNet_raw="/hpc/ajhz839/compared_models/nnUnet/project/nnUNet_raw/"
export nnUNet_preprocessed="/hpc/ajhz839/compared_models/nnUnet/project/nnUNet_preprocessed/"
export RESULTS_FOLDER="/hpc/ajhz839/compared_models/nnunet/project/nnUNet_results/"

cd $nnUNet_raw/Dataset501_BraTS2020

for i in {0..4}
do
    echo "开始 Fold $i 的训练..."
    cp dataset_fold${i}.json dataset.json

    nnUNetv2_train 501 3d_fullres $i > fold${i}_train.log 2>&1

    echo "Fold $i 训练完成，日志保存在 fold${i}_train.log"
done

echo "所有 Fold 训练完成！"


