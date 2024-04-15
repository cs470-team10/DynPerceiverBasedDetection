from dyn_perceiver.dyn_perceiver_regnet_model import DynPerceiver
import torch
from PIL import Image
from torchvision import transforms
import torch.nn as nn
import os
import json

class DynPerceiverTest:
    def __init__(self, base_dir, output_dir, pretrained_file, save_frequency = 10):
        self.checkpoint_path = base_dir + pretrained_file
        self.cache_path = output_dir + "/cache.json"
        self.model = None
        self.transform = None
        self.softmax = None
        self.save_frequency = save_frequency
        self.load_cache()

    def load_model(self):
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
        checkpoint = torch.load(self.checkpoint_path)
        self.model.load_state_dict(checkpoint['model'])
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # 이미지를 텐서로 변환
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # 정규화
        ])
        self.model.eval()
        self.softmax = nn.Softmax(dim=0)

    def id_forward(self, image_id):
        cache = self.cache.get(str(image_id))
        if (cache is None):
            return False, [], []
        value = cache['value']
        index = cache['index']
        return True, value, index
    
    def model_forward(self, image: Image):
        image_tensor = self.transform(image).unsqueeze(0)
        with torch.no_grad():
            y_early3, y_att, y_cnn, y_merge, _ = self.model.forward(image_tensor)
        value = []
        index = []
        value, index = self.formatting(y_early3, value, index)
        value, index = self.formatting(y_att, value, index)
        value, index = self.formatting(y_cnn, value, index)
        value, index = self.formatting(y_merge, value, index)
        return value, index
    
    def formatting(self, result, value, index):
        result = self.softmax(result.squeeze())
        value.append(torch.max(result).item())
        index.append(torch.argmax(result).item())
        return value, index
    
    def analysis_threshold(self, value, index, T):
        for i in range(3):
            if (T[i] < value[i]):
                return i + 1, index[i]
        return 4, index[3]
    
    # Cache operation
    def load_cache(self):
        self.cache = {}
        self.save_frequency_count = 0
        if os.path.exists(self.cache_path):
            with open(self.cache_path, 'r') as json_file:
                json_data = json.load(json_file)
                self.cache = json_data['cache']
    
    def save_cache(self):
        with open(self.cache_path, 'w') as f:
            json.dump({'cache' : self.cache}, f, indent = 2)
        self.save_frequency_count = 0
    
    def forward(self, image_id, image_path, require_image):
        found, value, index = self.id_forward(image_id)
        if (found):
            image = Image.open(image_path) if require_image else None
        else:
            if (self.model is None):
                self.load_model()
            image = Image.open(image_path)
            value, index = self.model_forward(image)
            self.append_cache(image_id, value, index)
        return image, value, index
    
    def append_cache(self, image_id, value, index):
        self.cache[f"{image_id}"] = {'value': value, 'index': index}
        self.save_frequency_count += 1
        if (self.save_frequency_count >= self.save_frequency):
            self.save_cache()