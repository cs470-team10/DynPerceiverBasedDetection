#!/bin/bash
CONFIGS=("theta_1-lambda_1" \
         "theta_5e-1-lambda_1" \
         "theta_3e-1-lambda_1")

run_config() {
    local file_config=$1
    start_time=$(TZ=Asia/Seoul date '+%Y-%m-%d %H:%M:%S (KST)')
    echo "[${start_time}] Start ${file_config}"
    python3 tools/train.py \
    improvements/ilsvrc/candidates/retinanet_dyn_perceiver-800MF_fpn_1x_ilsvrc_single-dyn-${file_config}.py \
    --resume --auto-scale-lr > experiment-1-sub-exp/ilsrvc/${file_config}.out
    for number in {1..11}; do
        rm work_dirs/retinanet_dyn_perceiver-800MF_fpn_1x_ilsvrc_single-dyn-${file_config}/epoch_${number}.pth
    done
    end_time=$(TZ=Asia/Seoul date '+%Y-%m-%d %H:%M:%S (KST)')
    echo "[${end_time}] End ${file_config}"
}

for config in "${CONFIGS[@]}"; do
    run_config $config
done
