_base_ = '../../configs/regnet/retinanet_regnetx-3.2GF_fpn_1x_coco.py'

# [CS470] 강우성, [CS470] 이정완: dynamic_evaluate를 True로 변경하면 Dynamic Evaluation이 실행됩니다.
dynamic_evaluate = False # Dynamic Evaluate를 사용할거면 true로 변경.

custom_imports = dict(
    imports=['mmdet.models.backbones.dyn_perceiver_regnet_baseline'],
    allow_failed_imports=False)
model = dict(
    backbone=dict(
        type='DynPerceiverBaseline',
        test_num=1,
        init_cfg=dict(type='Pretrained', 
                      checkpoint='./baselines/regnety_800mf_with_dyn_perceiver/reg800m_perceiver_t128_converted.pth')),
    neck=dict(
        type='FPN',
        # in_channels=[64, 128, 288, 672],
        in_channels=[64, 144, 320, 784],
        out_channels=256,
        num_outs=5),
    type='DynRetinaNet',
    dynamic_evaluate=dynamic_evaluate
)

custom_hooks = [
    dict(type='WandbLoggerHook',
         init_kwargs=dict(project='cs470', entity='plasma3365'),
         interval=10,
         log_checkpoint=True,
         log_model=True)
]
val_cfg = dict(type='DynamicValLoop', dynamic_evaluate=dynamic_evaluate)
test_cfg = dict(type='DynamicTestLoop', dynamic_evaluate=dynamic_evaluate)