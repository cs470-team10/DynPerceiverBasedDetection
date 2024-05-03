from torch.utils.data import DataLoader
from typing import Tuple
from mmengine.runner.amp import autocast
from torch import Tensor
import torch

# train_dataloader: COCO Trainset의 dataloader
# model: DynRetinaNet 모델
# fp16: 신경 안써도 됨. autocast만 켜고 하셈.

def get_threshold(train_dataloader: DataLoader, model, fp16: bool) -> Tensor:
    # [CS470] 강우성: [TODO] train_dataloader랑 model(DynRetinaNet 참고) 보면서 threshold 구하는 method 만들기.
    with autocast(enabled=fp16):
        thresholds = torch.tensor([[0.1, 0.2, 0.3, 0.4] for i in range(30)])
    return thresholds