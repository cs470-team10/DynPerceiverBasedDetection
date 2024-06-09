_base_ = '../improvements/retinanet_dyn_perceiver-800MF_fpn_1x_coco_single-dyn-base.py'
dynamic_evaluate_epoch = [12] # Training 때 dynamic evaluation을 할 epoch. 안할거면 [], 다할거면 [i + 1 for i in range(12)]
theta_factor = 115e-4
lambda_factor = 1
dynamic_evaluate_on_test = True
NUM_IMAGES = [[412, 139, 20, 22], [381, 150, 25, 37], [356, 162, 31, 44], [332, 172, 37, 52], [307, 180, 45, 61], [285, 179, 55, 74], [257, 192, 58, 86], [237, 185, 64, 107], [215, 187, 69, 122], [199, 183, 70, 141], [186, 173, 78, 156], [179, 158, 85, 171], [166, 153, 89, 185], [156, 149, 91, 197], [144, 148, 92, 209], [128, 145, 95, 225], [114, 141, 99, 239], [114, 141, 99, 239], [108, 131, 101, 253], [98, 131, 97, 267], [87, 130, 98, 278], [80, 120, 97, 296], [69, 118, 96, 310], [59, 117, 90, 327], [48, 109, 98, 338], [44, 92, 98, 359], [40, 88, 89, 376], [27, 82, 95, 389], [22, 70, 89, 412], [19, 59, 84, 431], [11, 48, 74, 460], [9, 38, 70, 476], [6, 26, 65, 496], [0, 0, 0, 593]]

model = dict(
    bbox_head=dict(
        loss_dyn=dict(theta_factor=theta_factor, lambda_factor=lambda_factor)
    )
)
val_cfg = dict(dynamic_evaluate_epoch=dynamic_evaluate_epoch)
test_cfg = dict(type='DynamicTestLoopRandomExiting', num_images=NUM_IMAGES, dynamic_evaluate=dynamic_evaluate_on_test) # Random exiting
