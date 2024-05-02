# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .single_stage import SingleStageDetector

from typing import List, Tuple, Union

from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import OptSampleList, SampleList
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig


@MODELS.register_module()
class DynRetinaNet(SingleStageDetector):
    def __init__(self,
                 backbone: ConfigType,
                 neck: ConfigType,
                 bbox_head: ConfigType,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 dynamic_evaluate: bool = False) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)
        self.dynamic_evaluate = dynamic_evaluate

    # [CS470] 강우성: 너가 찾아둔 thresholds는 여기로 들어가서 정완님이 사용
    # [CS470] 이정완: 우성이 것 참고하시면 됩니다.
    def set_thresholds(self, thresholds):
        self.thresholds = thresholds
        print("Thresholds: " + str(self.thresholds))
        
    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        x = self.extract_feat(batch_inputs, True)
        losses = self.bbox_head.loss(x, batch_data_samples)
        return losses

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        if self.dynamic_evaluate:
            # [CS470] 이정완: [TODO] 여기 보면 x가 fpn의 featuremap result고, 뒤에 4개는 dynamic perceiver의 classifier의 값입니다.(아마 남우님이 뽑아주신 값이 나오지 않을까 싶습니다. [B, 80]?)
            # 여기서 threshold랑 잘 비교해서 fpn의 output을 잘 0으로 패딩한 후에 bbox_head.predict에 넣어주면 됩니다.
            # Threshold 구하는법: self.thresholds에 있습니다. 근데 어떤 데이터구조로 받을지는 우성이랑 정하면 될 듯 합니다.
            x, y_early3, y_att, y_cnn, y_merge = self.extract_feat(batch_inputs, False)

            results_list = self.bbox_head.predict(
                x, batch_data_samples, rescale=rescale)
            
            batch_data_samples = self.add_pred_to_datasample(
                batch_data_samples, results_list)
            return batch_data_samples, y_early3, y_att, y_cnn, y_merge
        else:
            x, _y_early3, _y_att, _y_cnn, _y_merge= self.extract_feat(batch_inputs, False)
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
        x = self.extract_feat(batch_inputs, True)
        results = self.bbox_head.forward(x)
        return results

    def extract_feat(self, batch_inputs: Tensor, isTraining: bool) -> Tuple[Tensor]:
        """Extract features.

        Args:
            batch_inputs (Tensor): Image tensor with shape (N, C, H ,W).

        Returns:
            tuple[Tensor]: Multi-level features that may have
            different resolutions.
        """

        x, y_early3, y_att, y_cnn, y_merge = self.backbone(batch_inputs)
        # for a in x:
        #     print("Featuremap out: " + str(list(a.size())))
        if self.with_neck:
            x = self.neck(x)
            # for a in x:
            #     print("neck out: " + str(list(a.size())))
        
        if isTraining:
            return x
        else:
            return x, y_early3, y_att, y_cnn, y_merge