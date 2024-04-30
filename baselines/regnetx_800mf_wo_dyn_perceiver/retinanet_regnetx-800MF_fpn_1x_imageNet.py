_base_ = '../../configs/regnet/retinanet_regnetx-3.2GF_fpn_1x_coco.py'
model = dict(
    backbone=dict(
        type='RegNet',
        arch='regnetx_800mf',
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(
            type='Pretrained', checkpoint='open-mmlab://regnetx_800mf')),
    neck=dict(
        type='FPN',
        in_channels=[64, 128, 288, 672],
        out_channels=256,
        num_outs=5))

custom_hooks = [
    dict(type='WandbLoggerHook',
         init_kwargs=dict(project='cs470', entity='plasma3365'),
         interval=10,
         log_checkpoint=True,
         log_model=True)
]
