from dyn_perceiver.dyn_perceiver_regnet_model import DynPerceiver
import torch.nn as nn
from mmdet.registry import MODELS
from mmengine.model import BaseModule

@MODELS.register_module()
class DynPerceiverBaseline(BaseModule):
    def __init__(self, init_cfg, test_num, **args):
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

        # freeze stages
        i = 1
        for name, param in self.dyn_perceiver.named_parameters():
            # cnn stem and conv block
            if 'cnn_stem' in name:
                print(f"{name} freezed!")
                param.requires_grad = False
            if f"cnn_body.block{str(i)}" in name:
                print(f"{name} freezed!")
                param.requires_grad = False
            # classification branch(x2z, z2x, self attention, token mixer, expander; ~ stage 1 범위)
            if test_num == 2:
                if (f"dwc{str(i)}_x2z" in name) or (f"cross_att{str(i)}_x2z" in name) or (f"self_att{str(i)}" in name) or (f"token_mixer.{str(i - 1)}." in name) or (f"token_expander.{str(i - 1)}." in name) or (f"cross_att{str(i + 1)}_z2x" in name):
                    print(f"{name} freezed!")
                    param.requires_grad = False
            # classification branch(x2z, z2x, self attention, token mixer, expander, latent; ~ stage 1 범위)
            if test_num == 3:
                if ("latent" in name) or (f"dwc{str(i)}_x2z" in name) or (f"cross_att{str(i)}_x2z" in name) or (f"self_att{str(i)}" in name) or (f"token_mixer.{str(i - 1)}." in name) or (f"token_expander.{str(i - 1)}." in name) or (f"cross_att{str(i + 1)}_z2x" in name):
                    print(f"{name} freezed!")
                    param.requires_grad = False
            # classification branch(x2z, z2x, self attention, token mixer, expander, latent; 전체 범위)
            if test_num == 4:
                pass
                # if ("latent" in name) or (f"dwc{str(i)}_x2z" in name) or (f"cross_att{str(i)}_x2z" in name) or (f"self_att{str(i)}" in name) or (f"token_mixer.{str(i - 1)}." in name) or (f"token_expander.{str(i - 1)}." in name) or (f"cross_att{str(i + 1)}_z2x" in name):
                #     print(f"{name} freezed!")
                #     param.requires_grad = False
            
        self.dyn_perceiver.cnn_stem.eval()
        print("cnn_stem evaluation mode!")
        self.dyn_perceiver.cnn_body.block1.eval()
        print("cnn_body.block1 evaluation mode!")
        if test_num == 2 or test_num == 3:
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
            print("cross_att2_z2x.0 evaluation mode!")
        if test_num == 4:
            pass
        

    def forward(self, x):
        _y_early3, _y_att, _y_cnn, _y_merge, outs = self.dyn_perceiver.forward(x)
        # torch.Size([2, 64, 200, 304])
        # torch.Size([2, 144, 100, 152])
        # torch.Size([2, 320, 50, 76])
        # torch.Size([2, 784, 25, 38])
        return outs
    

# cnn_body.block1.block1-0.proj.0.weight is freezed!
# cnn_body.block1.block1-0.proj.1.weight is freezed!
# cnn_body.block1.block1-0.proj.1.bias is freezed!
# cnn_body.block1.block1-0.f.a.0.weight is freezed!
# cnn_body.block1.block1-0.f.a.1.weight is freezed!
# cnn_body.block1.block1-0.f.a.1.bias is freezed!
# cnn_body.block1.block1-0.f.b.0.weight is freezed!
# cnn_body.block1.block1-0.f.b.1.weight is freezed!
# cnn_body.block1.block1-0.f.b.1.bias is freezed!
# cnn_body.block1.block1-0.f.se.fc1.weight is freezed!
# cnn_body.block1.block1-0.f.se.fc1.bias is freezed!
# cnn_body.block1.block1-0.f.se.fc2.weight is freezed!
# cnn_body.block1.block1-0.f.se.fc2.bias is freezed!
# cnn_body.block1.block1-0.f.c.0.weight is freezed!
# cnn_body.block1.block1-0.f.c.1.weight is freezed!
# cnn_body.block1.block1-0.f.c.1.bias is freezed!
# dwc1_x2z.weight is freezed!
# dwc1_x2z.bias is freezed!
# cross_att1_x2z.cross_attn.q_norm.weight is freezed!
# cross_att1_x2z.cross_attn.q_norm.bias is freezed!
# cross_att1_x2z.cross_attn.kv_norm.weight is freezed!
# cross_att1_x2z.cross_attn.kv_norm.bias is freezed!
# cross_att1_x2z.cross_attn.attention.relative_position_bias is freezed!
# cross_att1_x2z.cross_attn.attention.q_proj.weight is freezed!
# cross_att1_x2z.cross_attn.attention.k_proj.weight is freezed!
# cross_att1_x2z.cross_attn.attention.v_proj.weight is freezed!
# cross_att1_x2z.cross_attn.attention.o_proj.weight is freezed!
# cross_att1_x2z.cross_attn.attention.o_proj.bias is freezed!
# cross_att1_x2z.mlp.layernorm.weight is freezed!
# cross_att1_x2z.mlp.layernorm.bias is freezed!
# cross_att1_x2z.mlp.fc1.weight is freezed!
# cross_att1_x2z.mlp.fc1.bias is freezed!
# cross_att1_x2z.mlp.fc2.weight is freezed!
# cross_att1_x2z.mlp.fc2.bias is freezed!
# self_att1.0.self_attn.norm.weight is freezed!
# self_att1.0.self_attn.norm.bias is freezed!
# self_att1.0.self_attn.attention.q_proj.weight is freezed!
# self_att1.0.self_attn.attention.k_proj.weight is freezed!
# self_att1.0.self_attn.attention.v_proj.weight is freezed!
# self_att1.0.self_attn.attention.o_proj.weight is freezed!
# self_att1.0.self_attn.attention.o_proj.bias is freezed!
# self_att1.0.mlp.layernorm.weight is freezed!
# self_att1.0.mlp.layernorm.bias is freezed!
# self_att1.0.mlp.fc1.weight is freezed!
# self_att1.0.mlp.fc1.bias is freezed!
# self_att1.0.mlp.fc2.weight is freezed!
# self_att1.0.mlp.fc2.bias is freezed!
# self_att1.1.self_attn.norm.weight is freezed!
# self_att1.1.self_attn.norm.bias is freezed!
# self_att1.1.self_attn.attention.q_proj.weight is freezed!
# self_att1.1.self_attn.attention.k_proj.weight is freezed!
# self_att1.1.self_attn.attention.v_proj.weight is freezed!
# self_att1.1.self_attn.attention.o_proj.weight is freezed!
# self_att1.1.self_attn.attention.o_proj.bias is freezed!
# self_att1.1.mlp.layernorm.weight is freezed!
# self_att1.1.mlp.layernorm.bias is freezed!
# self_att1.1.mlp.fc1.weight is freezed!
# self_att1.1.mlp.fc1.bias is freezed!
# self_att1.1.mlp.fc2.weight is freezed!
# self_att1.1.mlp.fc2.bias is freezed!
# self_att1.2.self_attn.norm.weight is freezed!
# self_att1.2.self_attn.norm.bias is freezed!
# self_att1.2.self_attn.attention.q_proj.weight is freezed!
# self_att1.2.self_attn.attention.k_proj.weight is freezed!
# self_att1.2.self_attn.attention.v_proj.weight is freezed!
# self_att1.2.self_attn.attention.o_proj.weight is freezed!
# self_att1.2.self_attn.attention.o_proj.bias is freezed!
# self_att1.2.mlp.layernorm.weight is freezed!
# self_att1.2.mlp.layernorm.bias is freezed!
# self_att1.2.mlp.fc1.weight is freezed!
# self_att1.2.mlp.fc1.bias is freezed!
# self_att1.2.mlp.fc2.weight is freezed!
# self_att1.2.mlp.fc2.bias is freezed!
# cross_att2_z2x.cross_attn.q_norm.weight is freezed!
# cross_att2_z2x.cross_attn.q_norm.bias is freezed!
# cross_att2_z2x.cross_attn.kv_norm.weight is freezed!
# cross_att2_z2x.cross_attn.kv_norm.bias is freezed!
# cross_att2_z2x.cross_attn.attention.q_proj.weight is freezed!
# cross_att2_z2x.cross_attn.attention.k_proj.weight is freezed!
# cross_att2_z2x.cross_attn.attention.v_proj.weight is freezed!
# cross_att2_z2x.cross_attn.attention.o_proj.weight is freezed!
# cross_att2_z2x.cross_attn.attention.o_proj.bias is freezed!
# cross_att2_z2x.mlp.layernorm.weight is freezed!
# cross_att2_z2x.mlp.layernorm.bias is freezed!
# cross_att2_z2x.mlp.fc1.weight is freezed!
# cross_att2_z2x.mlp.fc1.bias is freezed!
# cross_att2_z2x.mlp.fc2.weight is freezed!
# cross_att2_z2x.mlp.fc2.bias is freezed!
# token_expander.0.0.weight is freezed!
# token_expander.0.0.bias is freezed!
# token_expander.0.1.weight is freezed!
# token_expander.0.1.bias is freezed!
# token_mixer.0.0.weight is freezed!
# token_mixer.0.0.bias is freezed!
# token_mixer.0.1.weight is freezed!
# token_mixer.0.1.bias is freezed!