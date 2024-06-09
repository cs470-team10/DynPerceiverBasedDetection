_base_ = '../improvements/retinanet_dyn_perceiver-800MF_fpn_1x_coco_single-theta_factor_115e-4.py'

test_cfg = dict(threshold_distribution=[0.5, 1, 1, 0.85])
