_base_ = '../../configs/regnet/retinanet_regnetx-3.2GF_fpn_1x_coco.py'

dynamic_evaluate = True # Dynamic Evaluate를 사용할거면 true로 변경.

# 여기부터
dataset_type = 'CocoDataset'
data_root = 'data/coco/'
# [CS470] 강우성, [CS470] 이정완: dynamic_evaluate를 True로 변경하면 Dynamic Evaluation이 실행됩니다.

backend_args = None

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
] # 여기까지

custom_imports = dict(
    imports=['mmdet.models.backbones.dyn_perceiver_regnet_zeromap'],
    allow_failed_imports=False)
model = dict(
    backbone=dict(
        type='DynPerceiverZeromap',
        test_num=4,
        init_cfg=dict(type='Pretrained', 
                      checkpoint='./baselines/regnety_800mf_with_dyn_perceiver/reg800m_perceiver_t128_converted.pth')),
    neck=dict(
        type='FPN',
        # in_channels=[64, 128, 288, 672],
        in_channels=[64, 144, 320, 784],
        out_channels=256,
        num_outs=5),
    bbox_head=dict(
        loss_dyn=None,
        type='DynRetinaHead'),
    type='DynRetinaNet'
)

#여기부터
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/singlebox_instances_train2017.json', 
        data_prefix=dict(img='train2017/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args))
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/singlebox_instances_val2017.json',
        data_prefix=dict(img='val2017/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/singlebox_instances_val2017.json',
    metric='bbox',
    format_only=False,
    backend_args=backend_args) 
test_dataloader = val_dataloader
test_evaluator = val_evaluator # 여기까지

custom_hooks = [
    dict(type='WandbLoggerHook',
         init_kwargs=dict(project='cs470', entity='plasma3365'),
         interval=10,
         log_checkpoint=True,
         log_model=True)
]

val_cfg = dict(type='DynamicValLoop', dynamic_evaluate=dynamic_evaluate)
test_cfg = dict(type='DynamicTestLoop', dynamic_evaluate=dynamic_evaluate)
test_dataloader = dict(
    dataset=dict(
        ann_file='annotations/singlebox_instances_val2017.json',
        data_prefix=dict(img='val_single/')))
test_evaluator = dict(
    ann_file='data/coco/annotations/singlebox_instances_val2017.json')
train_dataloader = dict(
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