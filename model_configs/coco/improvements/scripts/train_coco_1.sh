#!/bin/bash
CONFIGS=(
    "theta_factor_7e-2"
    "theta_factor_4e-2"
    "theta_factor_5e-1"
    "theta_factor_5e-2"
    "theta_factor_6e-2"
)
CONFIG_NAME="retinanet_dyn_perceiver-800MF_fpn_1x_coco_single-"
CONFIG_PATH="model_configs/coco/improvements"
WORK_DIR_PATH="coco_single/improvements"
LOG_PATH="model_configs/coco/improvements/scripts/logs"

run_config() {
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

for config in "${CONFIGS[@]}"; do
    run_config $config
done
