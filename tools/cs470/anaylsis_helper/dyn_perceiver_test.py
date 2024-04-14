from dyn_perceiver.dyn_perceiver_regnet_model import DynPerceiver
import torch

class DynPerceiverTest:
    def __init__(self, pretrained_file):
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