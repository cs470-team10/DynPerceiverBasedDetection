_base_ ='../baseline/retinanet_dyn_perceiver-800MF_fpn_1x_ilsvrc_single-base.py'  # '../configs/regnet/retinanet_regnetx-3.2GF_fpn_1x_coco.py'

model = dict(
    neck=dict(type = "DynFPN",
        add_extra_convs='on_output'),
    bbox_head=dict(
        loss_dyn=dict(type='DynLoss')
    )
)