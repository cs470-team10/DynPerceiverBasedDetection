import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.registry import MODELS
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

@MODELS.register_module()
class DynLoss(nn.Module):

    def __init__(self,
                 theta_factor,
                 lambda_factor,
                 mixup_fn=None,
                 smoothing_=0,
                 loss_cnn_factor=0.25,
                 loss_att_factor=0.25,
                 loss_merge_factor=0.5,
                 with_kd=True,
                 T_kd=1.0,
                 alpha_kd=0.5
                 ):
        super(DynLoss, self).__init__()
        self.mixup_fn = mixup_fn
        self.smoothing = smoothing_
        self.loss_cnn_factor = loss_cnn_factor
        self.loss_att_factor = loss_att_factor
        self.loss_merge_factor = loss_merge_factor
        self.theta_factor = theta_factor
        self.lambda_factor = lambda_factor
        self.with_kd = with_kd
        self.T_kd=T_kd
        self.alpha_kd=alpha_kd

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
        
        if self.with_kd:
            out_teacher = pred[3].detach()
        
            kd_loss = F.kl_div(F.log_softmax(pred[0]/self.T_kd, dim=1),F.softmax(out_teacher/self.T_kd, dim=1), reduction='batchmean') * self.T_kd**2 + \
                    F.kl_div(F.log_softmax(pred[1]/self.T_kd, dim=1),F.softmax(out_teacher/self.T_kd, dim=1), reduction='batchmean') * self.T_kd**2 + \
                    F.kl_div(F.log_softmax(pred[2]/self.T_kd, dim=1),F.softmax(out_teacher/self.T_kd, dim=1), reduction='batchmean') * self.T_kd**2

            loss += self.alpha_kd * kd_loss

        return loss