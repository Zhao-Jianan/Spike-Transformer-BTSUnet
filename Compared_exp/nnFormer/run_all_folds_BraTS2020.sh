#!/bin/bash

export CUDA_VISIBLE_DEVICES=1
export nnFormer_raw_data_base="/hpc/ajhz839/compared_models/nnFormer/project/nnFormer_raw/"
export nnFormer_preprocessed="/hpc/ajhz839/compared_models/nnFormer/project/nnFormer_preprocessed/"
export RESULTS_FOLDER="/hpc/ajhz839/compared_models/nnFormer/project/nnFormer_results/"

TASK_ID=501
DATASET_NAME="BraTS2020"
TASK_DIR="$nnFormer_raw_data_base/Dataset${TASK_ID}_${DATASET_NAME}"
cd $TASK_DIR

for i in {0..4}
do
    echo "开始 Fold $i 的训练..."

    cp dataset_fold${i}.json dataset.json

    python -u /hpc/ajhz839/compared_models/nnFormer/nnFormer/nnformer/run/run_training.py \
      3d_fullres nnFormerTrainerV2 ${TASK_ID} $i \
      > ${RESULTS_FOLDER}/fold${i}_train.log 2>&1

    echo "Fold $i 训练完成，日志保存在 fold${i}_train.log"
done

echo "所有 Fold 训练完成！"
