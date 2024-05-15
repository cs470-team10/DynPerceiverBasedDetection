# RegNet-X 800MF Based RetinaNet

nohup python3 tools/train.py \
    baselines/regnetx_800mf_wo_dyn_perceiver/retinanet_regnetx-800MF_fpn_1x_coco.py \
    --auto-scale-lr > regnetx-800mf-train.out &
nohup python3 tools/test.py \
    baselines/regnetx_800mf_wo_dyn_perceiver/retinanet_regnetx-800MF_fpn_1x_coco.py \
    baselines/regnetx_800mf_wo_dyn_perceiver/retinanet_regnetx-800MF_fpn_1x_coco.py > regnetx-800mf-test.out &

python3 tools/analysis_tools/get_flops.py baselines/regnetx_800mf_wo_dyn_perceiver/retinanet_regnetx-800MF_fpn_1x_coco.py

# RegNet-Y 800MF Based RetinaNet

nohup python3 tools/train.py \
    baselines/regnety_800mf_wo_dyn_perceiver/retinanet_regnety-800MF_fpn_1x_coco.py \
    --auto-scale-lr > regnety-800mf-train.out &
nohup python3 tools/test.py \
    baselines/regnety_800mf_wo_dyn_perceiver/retinanet_regnety-800MF_fpn_1x_coco.py \
    baselines/regnety_800mf_wo_dyn_perceiver/retinanet_regnety-800MF_fpn_1x_coco.pth > regnety-800mf-test.out &

python3 tools/analysis_tools/get_flops.py baselines/regnety_800mf_wo_dyn_perceiver/retinanet_regnety-800MF_fpn_1x_coco.py

# Dyn-Perceiver(RegNet-Y 800MF) Based RetinaNet

nohup python3 tools/train.py \
    baselines/regnety_800mf_with_dyn_perceiver/retinanet_dyn_perceiver-800MF_fpn_1x_coco.py \
    --auto-scale-lr > dyn-800mf-train.out &
nohup python3 tools/test.py \
    baselines/regnety_800mf_with_dyn_perceiver/retinanet_dyn_perceiver-800MF_fpn_1x_coco.py \
    baselines/regnety_800mf_with_dyn_perceiver/retinanet_dyn_perceiver-800MF_fpn_1x_coco.pth > dyn-800mf-test.out &

python3 tools/analysis_tools/get_flops.py baselines/regnety_800mf_with_dyn_perceiver/retinanet_dyn_perceiver-800MF_fpn_1x_coco.py


# ######################################### Experiment 1 #########################################

# Train improvements

nohup python3 tools/train.py \
    improvements/retinanet_dyn_perceiver-800MF_fpn_1x_coco_single-theta_factor_3e-1.py \
    --auto-scale-lr > theta_factor_3e-1-train.out &

nohup python3 tools/train.py \
    improvements/retinanet_dyn_perceiver-800MF_fpn_1x_coco_single-theta_factor_3e-1.py \
    --auto-scale-lr > theta_factor_3e-1-train-3.out &

nohup python3 tools/train.py \
    improvements/retinanet_dyn_perceiver-800MF_fpn_1x_coco_single-theta_factor_1e-1.py \
    --auto-scale-lr > theta_factor_1e-1-train.out &

nohup python3 tools/train.py \
    improvements/retinanet_dyn_perceiver-800MF_fpn_1x_coco_single-theta_factor_5e-2.py \
    --auto-scale-lr > theta_factor_5e-2-train.out &

nohup python3 tools/train.py \
    improvements/retinanet_dyn_perceiver-800MF_fpn_1x_coco_single-theta_factor_1e-2.py \
    --auto-scale-lr > theta_factor_1e-2-train.out &

nohup python3 tools/train.py \
    improvements/retinanet_dyn_perceiver-800MF_fpn_1x_coco_single-theta_factor_5e-3.py \
    --auto-scale-lr > theta_factor_5e-3-train.out &

nohup python3 tools/train.py \
    improvements/retinanet_dyn_perceiver-800MF_fpn_1x_coco_single-theta_factor_1e-3.py \
    --auto-scale-lr > theta_factor_1e-3-train.out &

nohup python3 tools/train.py \
    improvements/retinanet_dyn_perceiver-800MF_fpn_1x_coco_single-theta_factor_5e-4.py \
    --auto-scale-lr > theta_factor_5e-4-train.out &

nohup python3 tools/train.py \
    improvements/retinanet_dyn_perceiver-800MF_fpn_1x_coco_single-theta_factor_1e-4.py \
    --auto-scale-lr > theta_factor_1e-4-train.out &

# Training baseline
nohup python3 tools/train.py \
    improvements/retinanet_regnety-800MF_fpn_1x_coco_single.py \
    --auto-scale-lr > baseline-regnety-train.out &

nohup python3 tools/train.py \
    improvements/retinanet_dyn_perceiver-800MF_fpn_1x_coco_single-baseline.py \
    --auto-scale-lr > baseline-dyn-train.out &