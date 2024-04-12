from dyn_perceiver.dyn_perceiver_regnet_model import DynPerceiver
import torch.nn as nn
from mmdet.registry import MODELS
from mmengine.model import BaseModule

@MODELS.register_module()
class DynPerceiverBaseline(BaseModule):
    def __init__(self, init_cfg, **args):
        super(DynPerceiverBaseline, self).__init__(init_cfg)
        self.dyn_perceiver = DynPerceiver(
            num_latents=128,
            cnn_arch='regnet_y_800mf',
            depth_factor=[1,1,1,2],
            spatial_reduction=True,
            with_last_CA=True,
            SA_widening_factor=4,
            with_x2z=True,
            with_dwc=True,
            with_z2x=True,
            with_isc=True)
        if (init_cfg == None or init_cfg['type'] != 'Pretrained' or init_cfg['checkpoint'] == None or not isinstance(init_cfg['checkpoint'], str)):
            raise 'A pretrained model must be provided.'
        

    def forward(self, x):
        _y_early3, _y_att, _y_cnn, _y_merge, outs = self.dyn_perceiver.forward(x)
        # torch.Size([2, 64, 200, 304])
        # torch.Size([2, 144, 100, 152])
        # torch.Size([2, 320, 50, 76])
        # torch.Size([2, 784, 25, 38])
        return outs