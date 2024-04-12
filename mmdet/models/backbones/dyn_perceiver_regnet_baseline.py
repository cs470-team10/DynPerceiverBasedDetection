from dyn_perceiver.dyn_perceiver_regnet_model import DynPerceiver
import torch.nn as nn
import torch
from mmdet.registry import MODELS

@MODELS.register_module()
class DynPerceiverBaseline(nn.Module):
    def __init__(self, pretrained_file: str, **args):
        super().__init__()
        self.model = DynPerceiver(
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
        
        checkpoint = torch.load(pretrained_file)
        self.model.load_state_dict(checkpoint['model'])
        print("[CS470] backbone checkpoint loaded.")

    def forward(self, x):
        _y_early3, _y_att, _y_cnn, _y_merge, outs = self.model.forward(x)
        # torch.Size([2, 64, 200, 304])
        # torch.Size([2, 144, 100, 152])
        # torch.Size([2, 320, 50, 76])
        # torch.Size([2, 784, 25, 38])
        return outs