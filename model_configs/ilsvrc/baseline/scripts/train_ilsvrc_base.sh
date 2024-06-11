#!/bin/bash
CONFIGS=(
    "base"
)
CONFIG_NAME="retinanet_dyn_perceiver-800MF_fpn_1x_ilsvrc_single-"
CONFIG_PATH="model_configs/ilsvrc/baseline"
WORK_DIR_PATH="ilsvrc_single/baseline"
LOG_PATH="model_configs/ilsvrc/baseline/scripts/logs"

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
