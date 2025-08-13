#!/bin/bash
# 一键跑五折训练+预测+融合
# 使用 CUDA 0, Task005_BraTS2020

# 固定参数
CUDA=0
TASK_ID=005
TASK_NAME=Task${TASK_ID}_BraTS2020
NAME=nnformer_bra20

echo "===== 参数 ====="
echo "CUDA: $CUDA"
echo "任务: $TASK_NAME"
echo "实验名: $NAME"
echo "================"

####################
# 1. 五折训练
####################
cd /hpc/ajhz839/compared_models/nnFormer/nnFormer/
echo "===== 开始五折训练 ====="
CUDA_VISIBLE_DEVICES=${CUDA} nnFormer_train 3d_fullres nnFormerTrainerV2 ${TASK_ID} all

####################
# 2. 五折预测
####################
cd /hpc/ajhz839/compared_models/nnFormer/project/nnFormer_raw/nnFormer_raw_data/${TASK_NAME}/
echo "===== 开始五折预测 ====="
for fold in 0 1 2 3 4; do
    echo "---- Fold ${fold} ----"
    CUDA_VISIBLE_DEVICES=${CUDA} nnFormer_predict \
        -i imagesTs \
        -o inferTs/${NAME}_fold${fold} \
        -m 3d_fullres \
        -t ${TASK_NAME} \
        -f $fold \
        -chk model_best \
        -tr nnFormerTrainerV2
done

####################
# 3. 五折结果融合
####################
echo "===== 开始融合五折预测结果 ====="
mkdir -p inferTs/${NAME}_ensemble

CUDA_VISIBLE_DEVICES=${CUDA} nnUNet_ensemble \
    -f \
    inferTs/${NAME}_fold0 \
    inferTs/${NAME}_fold1 \
    inferTs/${NAME}_fold2 \
    inferTs/${NAME}_fold3 \
    inferTs/${NAME}_fold4 \
    -o inferTs/${NAME}_ensemble \
    -pp None

echo "===== 全部完成 ====="
