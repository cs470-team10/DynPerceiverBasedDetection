from dyn_perceiver.dyn_perceiver_regnet_model import DynPerceiver
import torch.nn as nn
from mmdet.registry import MODELS
from mmengine.model import BaseModule

@MODELS.register_module()
class DynPerceiverZeromap(BaseModule):
    def __init__(self, init_cfg, test_num, **args):
        super(DynPerceiverZeromap, self).__init__(init_cfg)
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
        self.test_num = test_num
        self._freeze_stages()
        self.threshold = None

    def forward(self, x):
        y_early3, y_att, y_cnn, y_merge, outs = self.dyn_perceiver.forward(x, threshold=self.threshold)
        # torch.Size([2, 64, 200, 304])
        # torch.Size([2, 144, 100, 152])
        # torch.Size([2, 320, 50, 76])
        # torch.Size([2, 784, 25, 38])
        # [CS470] 김남우, [CS470] 이찬규: [TODO] 여기서 classifiers의 output은 [Batch Size, 80(COCO의 # of classes)]로 바뀌어야 합니다.
        # 그래야 정완님이 threshold 비교할 때 사용할 수 있습니다.
        # 남우님이랑 찬규님 중에 누가 할지 몰라서 일단 두 분 다 적었습니다.
        return outs, y_early3, y_att, y_cnn, y_merge
    
    def set_threshold(self, threshold):
        self.threshold = threshold
    
    def unset_threshold(self):
        self.threshold = None
    
    def train(self, mode=True):
        self.dyn_perceiver.train(mode)
        self._freeze_stages()
    
    def _freeze_stages(self):
        # freeze stages
        test_num = self.test_num
        for name, param in self.dyn_perceiver.named_parameters():
            # cnn stem and conv block
            if 'cnn_stem' in name:
                print(f"{name} freezed!")
                param.requires_grad = False
            if f"cnn_body.block1" in name:
                print(f"{name} freezed!")
                param.requires_grad = False
            
            # classification branch(x2z, z2x, self attention, token mixer, expander; ~ stage 1 범위)
            if test_num == 2:
                # stage 1
                i = 1
                if (f"dwc{str(i)}_x2z" in name) or \
                        (f"cross_att{str(i)}_x2z" in name) or \
                        (f"self_att{str(i)}" in name) or \
                        (f"token_mixer.{str(i - 1)}." in name) or \
                        (f"token_expander.{str(i - 1)}." in name) or \
                        (f"cross_att{str(i + 1)}_z2x" in name):
                    print(f"{name} freezed!")
                    param.requires_grad = False
            
            # classification branch(x2z, z2x, self attention, token mixer, expander, latent; ~ stage 1 범위)
            if test_num == 3:
                if "latent" in name:
                    print(f"{name} freezed!")
                    param.requires_grad = False
                # stage 1
                i = 1
                if (f"dwc{str(i)}_x2z" in name) or \
                        (f"cross_att{str(i)}_x2z" in name) or \
                        (f"self_att{str(i)}" in name) or \
                        (f"token_mixer.{str(i - 1)}." in name) or \
                        (f"token_expander.{str(i - 1)}." in name) or \
                        (f"cross_att{str(i + 1)}_z2x" in name):
                    print(f"{name} freezed!")
                    param.requires_grad = False
            
            # classification branch(x2z, z2x, self attention, token mixer, expander, latent; 전체 범위)
            if test_num == 4:
                if "latent" in name:
                    print(f"{name} freezed!")
                    param.requires_grad = False
                # stage 1
                i = 1
                if (f"dwc{str(i)}_x2z" in name) or \
                        (f"cross_att{str(i)}_x2z" in name) or \
                        (f"self_att{str(i)}" in name) or \
                        (f"token_mixer.{str(i - 1)}." in name) or \
                        (f"token_expander.{str(i - 1)}." in name) or \
                        (f"cross_att{str(i + 1)}_z2x" in name):
                    print(f"{name} freezed!")
                    param.requires_grad = False
                # stage 2
                i = 2
                if (f"dwc{str(i)}_x2z" in name) or \
                        (f"cross_att{str(i)}_x2z" in name) or \
                        (f"self_att{str(i)}" in name) or \
                        (f"token_mixer.{str(i - 1)}." in name) or \
                        (f"token_expander.{str(i - 1)}." in name) or \
                        (f"cross_att{str(i + 1)}_z2x" in name):
                    print(f"{name} freezed!")
                    param.requires_grad = False
                # stage 3
                i = 3
                if (f"dwc{str(i)}_x2z" in name) or \
                        (f"cross_att{str(i)}_x2z" in name) or \
                        (f"self_att{str(i)}" in name) or \
                        (f"token_mixer.{str(i - 1)}." in name) or \
                        (f"token_expander.{str(i - 1)}." in name) or \
                        (f"cross_att{str(i + 1)}_z2x" in name):
                    print(f"{name} freezed!")
                    param.requires_grad = False
                # stage 4
                i = 4
                if (f"dwc{str(i)}_x2z" in name) or \
                        (f"cross_att{str(i)}_x2z" in name) or \
                        (f"self_att{str(i)}" in name) or \
                        (f"last_cross_att_z2x" in name):
                    print(f"{name} freezed!")
                    param.requires_grad = False
            
        self.dyn_perceiver.cnn_stem.eval()
        print("cnn_stem evaluation mode!")
        self.dyn_perceiver.cnn_body.block1.eval()
        print("cnn_body.block1 evaluation mode!")
        if test_num == 2 or test_num == 3:
            # stage 1
            self.dyn_perceiver.dwc1_x2z.eval()
            print("dwc1_x2z evaluation mode!")
            self.dyn_perceiver.cross_att1_x2z.eval()
            print("cross_att1_x2z evaluation mode!")
            self.dyn_perceiver.self_att1.eval()
            print("self_att1 evaluation mode!")
            self.dyn_perceiver.token_mixer[0].eval()
            print("token_mixer.0 evaluation mode!")
            self.dyn_perceiver.token_expander[0].eval()
            print("token_expander.0 evaluation mode!")
            self.dyn_perceiver.cross_att2_z2x.eval()
            print("cross_att2_z2x evaluation mode!")
        if test_num == 4:
            # stage 1
            self.dyn_perceiver.dwc1_x2z.eval()
            print("dwc1_x2z evaluation mode!")
            self.dyn_perceiver.cross_att1_x2z.eval()
            print("cross_att1_x2z evaluation mode!")
            self.dyn_perceiver.self_att1.eval()
            print("self_att1 evaluation mode!")
            self.dyn_perceiver.token_mixer[0].eval()
            print("token_mixer.0 evaluation mode!")
            self.dyn_perceiver.token_expander[0].eval()
            print("token_expander.0 evaluation mode!")
            self.dyn_perceiver.cross_att2_z2x.eval()
            print("cross_att2_z2x evaluation mode!")

            # stage 2
            self.dyn_perceiver.dwc2_x2z.eval()
            print("dwc2_x2z evaluation mode!")
            self.dyn_perceiver.cross_att2_x2z.eval()
            print("cross_att2_x2z evaluation mode!")
            self.dyn_perceiver.self_att2.eval()
            print("self_att2 evaluation mode!")
            self.dyn_perceiver.token_mixer[1].eval()
            print("token_mixer.1 evaluation mode!")
            self.dyn_perceiver.token_expander[1].eval()
            print("token_expander.1 evaluation mode!")
            self.dyn_perceiver.cross_att3_z2x.eval()
            print("cross_att3_z2x evaluation mode!")

            # stage 3
            self.dyn_perceiver.dwc3_x2z.eval()
            print("dwc3_x2z evaluation mode!")
            self.dyn_perceiver.cross_att3_x2z.eval()
            print("cross_att3_x2z evaluation mode!")
            self.dyn_perceiver.self_att3.eval()
            print("self_att3 evaluation mode!")
            self.dyn_perceiver.token_mixer[2].eval()
            print("token_mixer.2 evaluation mode!")
            self.dyn_perceiver.token_expander[2].eval()
            print("token_expander.2 evaluation mode!")
            self.dyn_perceiver.cross_att4_z2x.eval()
            print("cross_att4_z2x evaluation mode!")

            # stage 4
            self.dyn_perceiver.dwc4_x2z.eval()
            print("dwc3_x2z evaluation mode!")
            self.dyn_perceiver.cross_att4_x2z.eval()
            print("cross_att4_x2z evaluation mode!")
            self.dyn_perceiver.self_att4.eval()
            print("self_att4 evaluation mode!")
            self.dyn_perceiver.last_cross_att_z2x.eval()
            print("last_cross_att_z2x evaluation mode!")
