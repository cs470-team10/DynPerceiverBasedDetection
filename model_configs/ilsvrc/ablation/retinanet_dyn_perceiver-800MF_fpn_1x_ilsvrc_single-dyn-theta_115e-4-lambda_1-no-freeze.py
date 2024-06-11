_base_ ='../improvements/retinanet_dyn_perceiver-800MF_fpn_1x_ilsvrc_single-dyn-theta_115e-4-lambda_1.py'

model = dict(backbone=dict(test_num=1))
