_base_ ='../baseline/retinanet_dyn_perceiver-800MF_fpn_1x_ilsvrc_single-base.py' 
dynamic_evaluate_epoch = [12]
dynamic_evaluate_on_test = True

val_cfg = dict(dynamic_evaluate_epoch=dynamic_evaluate_epoch)
test_cfg = dict(dynamic_evaluate=dynamic_evaluate_on_test)