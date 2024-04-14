from dyn_perceiver.dyn_perceiver_regnet_model import DynPerceiver
import torch
from PIL import Image
from torchvision import transforms
import torch.nn as nn

# Threshold는 아래에서 따왔습니다.
# tensor([ 4.4334e-01,  3.3758e-01,  2.4245e-01, -1.0000e+08])
# valid acc: 0.070, test acc: 77.326, test flops: 0.68M
# acc of each exit: tensor([0.8582, 0.6986, 0.4244, 0.2431])

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
        self.T = [4.4334e-01,  3.3758e-01,  2.4245e-01, -1.0000e+08]
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # 이미지를 텐서로 변환
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # 정규화
        ])
        self.model.eval()
        self.softmax = nn.Softmax(dim=0)


    def forward(self, image: Image, w: int, h: int):
        image_tensor = self.transform(image).unsqueeze(0)
        with torch.no_grad():
            y_early3, y_att, y_cnn, y_merge, _ = self.model.forward(image_tensor)
        y_early3 = self.softmax(y_early3.squeeze())
        y_att = self.softmax(y_att.squeeze())
        y_cnn = self.softmax(y_cnn.squeeze())
        y_merge = self.softmax(y_merge.squeeze())
        if (self.T[0] < torch.max(y_early3)):
            return 1, torch.argmax(y_early3).item()
        elif (self.T[1] < torch.max(y_att)):
            return 2, torch.argmax(y_att).item()
        elif (self.T[2] < torch.max(y_cnn)):
            return 3, torch.argmax(y_cnn).item()
        else:
            return 4, torch.argmax(y_merge).item()