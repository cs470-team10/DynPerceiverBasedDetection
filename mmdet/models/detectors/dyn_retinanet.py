# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .single_stage import SingleStageDetector

from typing import List, Tuple, Union, Dict, OrderedDict

from torch import Tensor
import torch

from mmdet.registry import MODELS
from mmdet.structures import OptSampleList, SampleList
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmengine.utils import is_list_of

from mmdet.structures import SampleList
from mmengine.analysis import get_model_complexity_info
from mmengine.analysis.print_helper import _format_size
import numpy as np
from functools import partial
from copy import deepcopy
from cs470_logger.cs470_print import cs470_print

@MODELS.register_module()
class DynRetinaNet(SingleStageDetector):
    def __init__(self,
                 backbone: ConfigType,
                 neck: ConfigType,
                 bbox_head: ConfigType,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)
        if self.bbox_head.loss_dyn is not None:
            self.loss_earlyexit = True
            self.theta_factor = self.bbox_head.loss_dyn.theta_factor
        else:
            self.loss_earlyexit = False
            self.theta_factor = 0
        self.metrics = []
    
    # Loss functions
    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        x, y_early3, y_att, y_cnn, y_merge = self.extract_feat(batch_inputs)
        if self.loss_earlyexit:
            earlyexit_preds = (y_early3, y_att, y_cnn, y_merge)
            losses = self.bbox_head.loss(x, batch_data_samples, earlyexit_preds)
        else:
            losses = self.bbox_head.loss(x, batch_data_samples)
        return losses
    
    def parse_losses(
        self, losses: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        log_vars = []
        earlyexit_loss = 0
        earlyexit_loss_name = ''
        for loss_name, loss_value in losses.items():
            if "earlyexit" in loss_name:
                earlyexit_loss_name = loss_name
                if isinstance(loss_value, torch.Tensor):
                    earlyexit_loss = loss_value.mean()
                elif is_list_of(loss_value, torch.Tensor):
                    earlyexit_loss = sum(_loss.mean() for _loss in loss_value)
            elif isinstance(loss_value, torch.Tensor):
                log_vars.append([loss_name, loss_value.mean()])
            elif is_list_of(loss_value, torch.Tensor):
                log_vars.append(
                    [loss_name,
                     sum(_loss.mean() for _loss in loss_value)])
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')
        loss = sum(value for key, value in log_vars if 'loss' in key)

        if earlyexit_loss != 0:
            log_vars.append([earlyexit_loss_name, earlyexit_loss])
            loss = (1 - self.theta_factor) * loss + self.theta_factor * earlyexit_loss

        log_vars.insert(0, ['loss', loss])
        log_vars = OrderedDict(log_vars)

        return loss, log_vars

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        x, _y_early3, _y_att, _y_cnn, _y_merge = self.extract_feat(batch_inputs)
        results_list = self.bbox_head.predict(
            x, batch_data_samples, rescale=rescale)
        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples
    
    def _forward(
            self,
            batch_inputs: Tensor,
            batch_data_samples: OptSampleList = None) -> Tuple[List[Tensor]]:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

         Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (list[:obj:`DetDataSample`]): Each item contains
                the meta information of each image and corresponding
                annotations.

        Returns:
            tuple[list]: A tuple of features from ``bbox_head`` forward.
        """
        x, _y_early3, _y_att, _y_cnn, _y_merge = self.extract_feat(batch_inputs)
        results = self.bbox_head.forward(x)
        return results

    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        """Extract features.

        Args:
            batch_inputs (Tensor): Image tensor with shape (N, C, H ,W).

        Returns:
            tuple[Tensor]: Multi-level features that may have
            different resolutions.
        """
        x, y_early3, y_att, y_cnn, y_merge = self.backbone(batch_inputs)
        # cs470_print(f"Exited in {self.backbone.get_last_exited_stage()}")
        # for i in range(len(x)):
        #     cs470_print(f"Featuremap {str(i + 1)} size: {str(x[i].size())}{', zero' if torch.all(x[i] == 0) else ''}")
        if self.with_neck:
            x = self.neck(x)
        return x, y_early3, y_att, y_cnn, y_merge
    
    # Helper Functions
    def get_dynamic_flops(self, data_loader, num_images=10):
        thresholds = torch.tensor([
            [-1, 10, 10, 10],
            [10, -1, 10, 10],
            [10, 10, -1, 10],
            [10, 10, 10, -1]
        ])
        flops_list = []
        model = deepcopy(self)
        model.eval()
        model.forward = partial(model._forward)

        for threshold in thresholds:
            model.set_threshold(threshold)
            avg_flops = []
            for idx, data_batch in enumerate(data_loader):
                if idx == num_images:
                    break
                data = model.data_preprocessor(data_batch)
                outputs = get_model_complexity_info(
                    model,
                    None,
                    inputs=data['inputs'],
                    show_table=False,
                    show_arch=False)
                avg_flops.append(outputs['flops'])

            mean_flops = int(np.average(avg_flops))
            flops_list.append(mean_flops)
            model.unset_threshold()
        flops = torch.tensor(flops_list)
        cs470_print("Flops per early exiting stages: " + str(flops.tolist()))
        return flops
    
    def set_threshold(self, threshold):
        self.backbone.set_threshold(threshold)
    
    def unset_threshold(self):
        self.backbone.unset_threshold()