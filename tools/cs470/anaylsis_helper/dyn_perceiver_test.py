from dyn_perceiver.dyn_perceiver_regnet_model import DynPerceiver
import torch
from PIL import Image
from torchvision import transforms
import torch.nn as nn

class DynPerceiverTest:
    def __init__(self, base_dir, pretrained_file, coco):
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
        checkpoint = torch.load(base_dir + pretrained_file)
        self.model.load_state_dict(checkpoint['model'])
        self.coco = coco
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # 이미지를 텐서로 변환
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # 정규화
        ])
        self.model.eval()
        self.softmax = nn.Softmax(dim=0)
    
    def forward(self, image: Image):
        image_tensor = self.transform(image).unsqueeze(0)
        with torch.no_grad():
            y_early3, y_att, y_cnn, y_merge, _ = self.model.forward(image_tensor)
        y_early3 = self.softmax(y_early3.squeeze())
        y_att = self.softmax(y_att.squeeze())
        y_cnn = self.softmax(y_cnn.squeeze())
        y_merge = self.softmax(y_merge.squeeze())
        return y_early3, y_att, y_cnn, y_merge
    
    def analysis_threshold(self, outs, T):
        for i in range(3):
            if (T[i] < torch.max(outs[i])):
                return i + 1, torch.argmax(outs[i]).item()
        return 4, torch.argmax(outs[3]).item()