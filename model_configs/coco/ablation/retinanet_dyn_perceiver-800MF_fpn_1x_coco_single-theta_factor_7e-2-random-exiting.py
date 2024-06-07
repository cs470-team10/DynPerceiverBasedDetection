_base_ = '../baseline/retinanet_dyn_perceiver-800MF_fpn_1x_coco_single-base.py'
dynamic_evaluate_epoch = [12] # Training 때 dynamic evaluation을 할 epoch. 안할거면 [], 다할거면 [i + 1 for i in range(12)]
theta_factor = 7e-2
lambda_factor = 1-theta_factor
dynamic_evaluate_on_test = True
NUM_IMAGES = [[365, 184, 21, 23], [348, 184, 28, 33], [322, 195, 29, 47], [301, 195, 33, 64], [280, 194, 45, 74], [262, 190, 50, 91], [244, 186, 58, 105], [227, 171, 73, 122], [207, 170, 76, 140], [191, 163, 84, 155], [176, 156, 89, 172], [162, 154, 96, 181], [153, 153, 97, 190], [140, 152, 96, 205], [134, 145, 96, 218], [120, 141, 98, 234], [109, 137, 97, 250], [109, 137, 97, 250], [103, 128, 99, 263], [93, 126, 94, 280], [86, 121, 89, 297], [76, 114, 80, 323], [62, 107, 79, 345], [56, 100, 84, 353], [50, 88, 85, 370], [43, 77, 88, 385], [36, 64, 89, 404], [32, 55, 81, 425], [25, 51, 69, 448], [18, 46, 60, 469], [14, 37, 54, 488], [10, 31, 49, 503], [6, 21, 48, 518], [0, 0, 0, 593]]

model = dict(
    bbox_head=dict(
        loss_dyn=dict(theta_factor=theta_factor,
                      lambda_factor=lambda_factor,
                      with_kd=False,
                      type='DynLoss')
    )
)
val_cfg = dict(dynamic_evaluate_epoch=dynamic_evaluate_epoch)
test_cfg = dict(type='DynamicTestLoopRandomExiting', num_images=NUM_IMAGES, dynamic_evaluate=dynamic_evaluate_on_test) # Random exiting
