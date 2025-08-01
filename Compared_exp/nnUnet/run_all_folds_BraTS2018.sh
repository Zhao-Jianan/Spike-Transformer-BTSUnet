#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export nnUNet_raw="/hpc/ajhz839/compared_models/nnUnet/project/nnUNet_raw/"
export nnUNet_preprocessed="/hpc/ajhz839/compared_models/nnUnet/project/nnUNet_preprocessed/"
export RESULTS_FOLDER="/hpc/ajhz839/compared_models/nnunet/project/nnUNet_results/"

cd $nnUNet_raw/Dataset032_BraTS2018

for i in {0..4}
do
    echo "==============================="
    echo "开始 Fold $i 的训练..."
    echo "==============================="
    
    nnUNetv2_train 032 3d_fullres $i > fold${i}_train.log 2>&1

    echo "Fold $i 训练完成，日志保存在 fold${i}_train.log"
done

echo "所有 Fold 训练完成！"
