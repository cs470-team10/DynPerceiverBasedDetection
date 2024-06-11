#!/bin/bash
CONFIG_NAME="retinanet_dyn_perceiver-800MF_fpn_1x_ilsvrc_single-"
CONFIG_PATH="model_configs/ilsvrc/ablation"
WORK_DIR_PATH="ilsvrc_single/ablation"
LOG_PATH="model_configs/ilsvrc/ablation/scripts/logs"
IMPROVEMENT_CHECKPOINT="work_dirs/ilsvrc_single/improvements/retinanet_dyn_perceiver-800MF_fpn_1x_ilsvrc_single-dyn-theta_115e-4-lambda_1/epoch_12.pth"
BASE_CHECKPOINT="work_dirs/ilsvrc_single/baseline/retinanet_dyn_perceiver-800MF_fpn_1x_ilsvrc_single-base/epoch_12.pth"

test_config() {
    local file_config=$1
    local test_file_config=$2
    if [ -f "work_dirs/${WORK_DIR_PATH}/${CONFIG_NAME}${file_config}/${CONFIG_NAME}${file_config}.py" ]; then
        echo "${file_config} already finished."
    else 
        start_time=$(TZ=Asia/Seoul date '+%Y-%m-%d %H:%M:%S (KST)')
        echo "[${start_time}] Start ${file_config}"
        python3 tools/test.py \
        ${CONFIG_PATH}/${CONFIG_NAME}${file_config}.py \
        ${test_file_config} > ${LOG_PATH}/${file_config}.out
        mv work_dirs/${CONFIG_NAME}${file_config} work_dirs/${WORK_DIR_PATH}
        end_time=$(TZ=Asia/Seoul date '+%Y-%m-%d %H:%M:%S (KST)')
        echo "[${end_time}] End ${file_config}"
    fi
}

train_config() {
    local file_config=$1
    if [ -f "work_dirs/${WORK_DIR_PATH}/${CONFIG_NAME}${file_config}/epoch_12.pth" ]; then
        echo "${file_config} already finished."
    else 
        start_time=$(TZ=Asia/Seoul date '+%Y-%m-%d %H:%M:%S (KST)')
        echo "[${start_time}] Start ${file_config}"
        python3 tools/train.py \
        ${CONFIG_PATH}/${CONFIG_NAME}${file_config}.py \
        --resume --auto-scale-lr > ${LOG_PATH}/${file_config}.out
        for number in {1..11}; do
            rm work_dirs/${CONFIG_NAME}${file_config}/epoch_${number}.pth
        done
        mv work_dirs/${CONFIG_NAME}${file_config} work_dirs/${WORK_DIR_PATH}
        end_time=$(TZ=Asia/Seoul date '+%Y-%m-%d %H:%M:%S (KST)')
        echo "[${end_time}] End ${file_config}"
    fi
}

mkdir -p work_dirs
mkdir -p work_dirs/${WORK_DIR_PATH}
mkdir -p ${LOG_PATH}

test_config dyn-theta_115e-4-lambda_1-random-exiting ${IMPROVEMENT_CHECKPOINT}
test_config dyn-theta_115e-4-lambda_1-zero-padding ${IMPROVEMENT_CHECKPOINT}
test_config base-dyn-evaluation ${BASE_CHECKPOINT}
train_config dyn-theta_115e-4-lambda_1-no-freeze