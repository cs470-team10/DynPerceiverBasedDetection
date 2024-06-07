_base_ = '../configs/regnet/retinanet_regnetx-3.2GF_fpn_1x_coco.py'
custom_imports = dict(
    imports=['mmdet.models.backbones.regnety_800mf'],
    allow_failed_imports=False)
model = dict(
    backbone=dict(
        type='RegNetY800MF',
        init_cfg=dict(type='Pretrained', 
                      checkpoint='./baselines/regnety_800mf_wo_dyn_perceiver/regnet_y_800mf-converted.pth')),
    neck=dict(in_channels=[64, 144, 320, 784],
              add_extra_convs='on_output')
)

custom_hooks = [
    dict(type='WandbLoggerHook',
         init_kwargs=dict(project='cs470', entity='plasma3365'),
         interval=10,
         log_checkpoint=True,
         log_model=True)
]
test_dataloader = dict(
    dataset=dict(
        ann_file='annotations/singlebox_instances_val2017.json',
        data_prefix=dict(img='val_single/')))
test_evaluator = dict(
    ann_file='data/coco/annotations/singlebox_instances_val2017.json')
train_dataloader = dict(
    batch_sampler=dict(drop_last=True),
    dataset=dict(
        ann_file='annotations/singlebox_instances_train2017.json',
        backend_args=None,
        data_prefix=dict(img='train_single/')))
val_dataloader = dict(
    dataset=dict(
        ann_file='annotations/singlebox_instances_val2017.json',
        data_prefix=dict(img='val_single/')))
val_evaluator = dict(
    ann_file='data/coco/annotations/singlebox_instances_val2017.json')

