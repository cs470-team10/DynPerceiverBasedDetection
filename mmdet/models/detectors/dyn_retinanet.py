# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .single_stage import SingleStageDetector

from typing import List, Tuple, Union

from torch import Tensor
import torch

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
    def set_threshold(self, threshold):
        self.threshold = threshold
        
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
            # Threshold 구하는법: self.threshold에 있습니다. 근데 어떤 데이터구조로 받을지는 우성이랑 정하면 될 듯 합니다.
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
        # [CS470] 이정완: 아래의 코드는 threshold를 backbone 단에 넣어주고 싶을 때 사용합니다.
        #               만약 우리가 논의했던 "2번째 방법"이라면 아래의 코드를 지우면 됩니다.
        #               아래의 코드가 있다면 backbone에서 feature map을 zero로 만들어줘서 출력하므로
        #               (남우가 작성한 코드가 돌아가므로) 정완님이 따로 threshold를 비교할 필요가 없어집니다.
        # [CS470] 김남우: 아래의 코드는 threshold를 backbone 단에 넣어주고 싶을 때 사용합니다.
        if not isTraining and self.dynamic_evaluate:
            self.backbone.set_threshold(self.threshold)
        # ------------------------------------------------------------------------

        x, y_early3, y_att, y_cnn, y_merge = self.backbone(batch_inputs)


        # [CS470] 이정완: 아래의 코드는 threshold를 backbone 단에 넣어주고 싶을 때 사용합니다.
        #               만약 우리가 논의했던 "2번째 방법"이라면 아래의 코드를 지우면 됩니다.
        #               아래의 코드가 있다면 backbone에서 feature map을 zero로 만들어줘서 출력하므로
        #               (남우가 작성한 코드가 돌아가므로) 정완님이 따로 threshold를 비교할 필요가 없어집니다.
        # [CS470] 김남우: 아래의 코드는 threshold를 backbone 단에 넣어주고 싶을 때 사용합니다.
        if not isTraining and self.dynamic_evaluate:
            self.backbone.unset_threshold()
        # ------------------------------------------------------------------------


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
        

    def get_dynamic_flops(self):
        # [CS470] 김남우, [CS470] 이찬규: [TODO]
        # 이건 뭐 별거는 아닌데, 우린 결국 flops별 accuarcy를 비교하는 것이니까
        # 이에 대한 내장 함수 하나 있어도 좋을 것 같다는 생각이 들어 일단 파놨습니다.
        # 각각 early exiting되는 stage에 따라 flops 계산하는 로직 넣어두면
        # 좋지 않을까요?
        # 
        # tools/analysis_tools/get_flops.py 참고해서 작성하면 좋을 듯 합니다.
        return torch.tensor([1,2,3])