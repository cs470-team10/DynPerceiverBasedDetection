from mmdet.registry import MODELS
from mmengine.model import BaseModule
from dyn_perceiver.cnn_core import regnet_y_800mf
from cs470_logger.cs470_debug_print import cs470_debug_print

@MODELS.register_module()
class RegNetY800MF(BaseModule):
    def __init__(self, init_cfg, **args):
        super(RegNetY800MF, self).__init__(init_cfg)
        cnn = regnet_y_800mf()
        self.cnn_stem = cnn.stem
        self.cnn_body = cnn.trunk_output
        if (init_cfg == None or init_cfg['type'] != 'Pretrained' or init_cfg['checkpoint'] == None or not isinstance(init_cfg['checkpoint'], str)):
            raise 'A pretrained model must be provided.'
        self._freeze_stages()

    def train(self, mode=True):
        self.cnn_stem.train(mode=mode)
        self.cnn_body.train(mode=mode)
        self._freeze_stages()

    def forward(self, x):
        outs = []
        x = self.cnn_stem(x)
        x = self.cnn_body.block1(x)
        outs.append(x)
        x = self.cnn_body.block2(x)
        outs.append(x)
        x = self.cnn_body.block3(x)
        outs.append(x)
        x = self.cnn_body.block4(x)
        outs.append(x)
        return outs
    
    def _freeze_stages(self):
        # freeze stages
        for name, param in self.named_parameters():
            # cnn stem and conv block
            if 'cnn_stem' in name:
                cs470_debug_print(f"{name} freezed!")
                param.requires_grad = False
            if f"cnn_body.block1" in name:
                cs470_debug_print(f"{name} freezed!")
                param.requires_grad = False
        self.cnn_stem.eval()
        cs470_debug_print("cnn_stem evaluation mode!")
        self.cnn_body.block1.eval()
        cs470_debug_print("cnn_body.block1 evaluation mode!")