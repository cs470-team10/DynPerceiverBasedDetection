_base_ = '../baseline/retinanet_dyn_perceiver-800MF_fpn_1x_coco_single-base.py'

model = dict(
    neck=dict(type = "DynFPN",
              add_extra_convs='on_output'),
    bbox_head=dict(
        loss_dyn=dict(with_kd=False, type='DynLoss')
    )
)