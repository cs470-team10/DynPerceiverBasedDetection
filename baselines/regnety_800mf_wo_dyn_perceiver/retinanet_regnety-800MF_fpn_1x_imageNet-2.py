_base_ = '../../configs/regnet/retinanet_regnetx-3.2GF_fpn_1x_coco.py'
custom_imports = dict(
    imports=['mmdet.models.backbones.regnety_800mf'],
    allow_failed_imports=False)
model = dict(
    backbone=dict(
        type='RegNetY800MF',
        init_cfg=dict(type='Pretrained', 
                      checkpoint='./baselines/regnety_800mf_wo_dyn_perceiver/regnet_y_800mf-converted.pth')),
    neck=dict(
        type='FPN',
        # in_channels=[64, 128, 288, 672],
        in_channels=[64, 144, 320, 784],
        out_channels=256,
        num_outs=5))

custom_hooks = [
    dict(type='WandbLoggerHook',
         init_kwargs=dict(project='cs470', entity='plasma3365'),
         interval=10,
         log_checkpoint=True,
         log_model=True)
]
