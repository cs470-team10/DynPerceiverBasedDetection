import warnings

import torch
import torch.nn as nn

from mmdet.registry import MODELS
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

@MODELS.register_module()
class DynLoss(nn.Module):

    def __init__(self,
                 theta_factor,
                 mixup_fn=None,
                 smoothing_=0,
                 loss_cnn_factor=0.25,
                 loss_att_factor=0.25,
                 loss_merge_factor=0.5
                 ):
        super(DynLoss, self).__init__()
        self.mixup_fn = mixup_fn
        self.smoothing = smoothing_
        self.loss_cnn_factor = loss_cnn_factor
        self.loss_att_factor = loss_att_factor
        self.loss_merge_factor = loss_merge_factor
        self.theta_factor = theta_factor

        if self.mixup_fn is not None:
            self.criterion = SoftTargetCrossEntropy()
        elif self.smoothing > 0.:
            self.criterion = LabelSmoothingCrossEntropy(smoothing=self.smoothing)
        else:
            self.criterion = torch.nn.CrossEntropyLoss()
    
    def forward(self, 
                pred, 
                target):
        loss_early3 = self.criterion(pred[0], target)
        loss_att = self.criterion(pred[1], target)
        loss_cnn = self.criterion(pred[2], target)
        loss_merge = self.criterion(pred[3], target)

        loss = self.loss_cnn_factor*loss_cnn + self.loss_att_factor*(loss_att+loss_early3) + self.loss_merge_factor*loss_merge
        
        return loss