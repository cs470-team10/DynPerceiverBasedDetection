nohup python3 tools/test.py \
    models/retinanet_regnetx-1.6GF_fpn_1x_coco.py \
    models/retinanet_regnetx-1.6GF_fpn_1x_coco_20200517_191403-37009a9d.pth > retina1.6.out &

nohup python3 tools/test.py \
    models/retinanet_regnetx-3.2GF_fpn_1x_coco.py \
    models/retinanet_regnetx-3.2GF_fpn_1x_coco_20200520_163141-cb1509e8.pth > retina3.2.out &

nohup python3 tools/test.py \
    models/retinanet_regnetx-800MF_fpn_1x_coco.py \
    models/retinanet_regnetx-800MF_fpn_1x_coco_20200517_191403-f6f91d10.pth > retina800.out &

python3 tools/analysis_tools/get_flops.py models/retinanet_regnetx-1.6GF_fpn_1x_coco.py
python3 tools/analysis_tools/get_flops.py models/retinanet_regnetx-3.2GF_fpn_1x_coco.py

python3 tools/train.py models/retinanet_regnetx-1.6GF_fpn_1x_coco.py
python3 tools/train.py models/retinanet_regnetx-3.2GF_fpn_1x_coco.py