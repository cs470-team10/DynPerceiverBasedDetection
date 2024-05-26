_base_ = '../configs/regnet/retinanet_regnetx-3.2GF_fpn_1x_coco.py'

custom_imports = dict(
    imports=['mmdet.models.backbones.dyn_perceiver_regnet_zeromap',
             'mmdet.datasets.custom_coco_dataset',
             'mmdet.core.evaluation.custom_evaluation_hook'],
    allow_failed_imports=False)

model = dict(
    backbone=dict(
        type='DynPerceiverZeromap',
        test_num=2,
        num_classes=80,
        init_cfg=dict(type='Pretrained', 
                      checkpoint='./baselines/regnety_800mf_with_dyn_perceiver/reg800m_perceiver_t128_converted.pth')),
    neck=dict(in_channels=[64, 144, 320, 784]),
    bbox_head=dict(
        loss_dyn=None,
        type='DynRetinaHead'),
    type='DynRetinaNet'
)

custom_hooks = [
    dict(type='WandbLoggerHook',
         init_kwargs=dict(project='cs470', entity='plasma3365'),
         interval=10,
         log_checkpoint=True,
         log_model=True),
    dict(type='CustomEvaluationHook')
]

# dataset_type 수정
dataset_type = 'CustomCocoDataset'

data = dict(
    train=dict(
        type=dataset_type,
        ann_file='annotations/singlebox_instances_train2017.json',
        data_prefix=dict(img='train_single/')),
    val=dict(
        type=dataset_type,
        ann_file='annotations/singlebox_instances_val2017.json',
        data_prefix=dict(img='val_single/')),
    test=dict(
        type=dataset_type,
        ann_file='annotations/singlebox_instances_val2017.json',
        data_prefix=dict(img='val_single/'))
)

test_dataloader = dict(
    dataset=dict(
        type=dataset_type,  # 추가된 부분
        ann_file='annotations/singlebox_instances_val2017.json',
        data_prefix=dict(img='val_single/')))

val_dataloader = dict(
    dataset=dict(
        type=dataset_type,  # 추가된 부분
        ann_file='annotations/singlebox_instances_val2017.json',
        data_prefix=dict(img='val_single/')))
