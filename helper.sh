# Demo

python3 demo/image_demo.py \
    data/coco/val2017/000000446703.jpg \
    baselines/regnety_800mf_wo_dyn_perceiver/retinanet_regnety-800MF_fpn_1x_coco.py \
    --weights baselines/regnety_800mf_wo_dyn_perceiver/retinanet_regnety-800MF_fpn_1x_coco.pth \
    --device cpu

# Convert weight
python3 tools/model_converters/dyn2mmdet.py \
    ./baselines/regnety_800mf_with_dyn_perceiver/reg800m_perceiver_t128.pth \
    ./baselines/regnety_800mf_with_dyn_perceiver/reg800m_perceiver_t128_converted.pth

# Analysis Log

python3 tools/cs470/analysis_log.py \
    --src ./work_dirs/retinanet_regnetx-800MF_fpn_1x_imageNet/20240410_065852/20240410_065852.log \
    --dest ./

