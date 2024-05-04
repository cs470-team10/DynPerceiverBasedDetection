from torch.utils.data import DataLoader
from typing import Tuple
from mmengine.runner.amp import autocast
from torch import Tensor
import torch

# train_dataloader: COCO Trainset의 dataloader
# model: DynRetinaNet 모델
# fp16: 신경 안써도 됨. autocast만 켜고 하셈.

def get_threshold(each_exit = False) -> Tensor:
    # [CS470] 강우성: [TODO] train_dataloader랑 model(DynRetinaNet 참고) 보면서 threshold 구하는 method 만들기.
    probs_list = []
    if each_exit:
        for i in range(4):
            probs = torch.zeros(4, dtype=torch.float)
            probs[i] = 1
            probs_list.append(probs)
    else:
        p_list = torch.zeros(34)
        for i in range(17):
            p_list[i] = (i + 4) / 20
            p_list[33 - i] = 20 / (i + 4)
            
        k = [0.85, 1, 0.5, 1]
        for i in range(33):
            probs = torch.exp(torch.log(p_list[i]) * torch.range(1, 4))
            probs /= probs.sum()
            for j in range(3):
                probs[j] *= k[j]
                probs[j+1:4] = (1 - probs[0:j+1].sum()) * probs[j+1:4] / probs[j+1:4].sum()
            probs_list.append(probs)
    return probs_list # size : 34 * 4