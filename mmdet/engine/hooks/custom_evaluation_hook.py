from mmengine.registry import HOOKS
from mmengine.hooks import Hook
import torch
import numpy as np
from mmdet.evaluation.functional import bbox_overlaps

@HOOKS.register_module()
class CustomEvaluationHook(Hook):
    def __init__(self, interval=1, **kwargs):
        super().__init__(**kwargs)
        self.interval = interval

    def after_train_epoch(self, runner):
        runner.logger.info("CustomEvaluationHook after_train_epoch is called")
        if not self.every_n_epochs(runner, self.interval):
            return
        runner.logger.info("Interval check passed")
        print("Yes I am here Do not worry!!!")
        # 모델을 평가 모드로 설정
        runner.model.eval()
        print("Yes I am here Do not worry2222!!!")
        # 데이터 로더 가져오기
        dataloader = runner.val_dataloader
        dataset = dataloader.dataset
        print("Yes I am here Do not worry33333!!!")
        results = []
        for i, data in enumerate(dataloader):
            with torch.no_grad():
                result = runner.model(return_loss=False, rescale=True, **data)
            results.extend(result)
        print("Yes I am here Do not worry444444!!!")
        # 평가 메트릭 계산
        image_mAPs = []
        for i, result in enumerate(results):
            gt_bboxes = dataset.get_ann_info(i)['bboxes']
            pred_bboxes = result[0]

            if gt_bboxes.shape[0] == 0:
                image_mAPs.append(0)
                runner.logger.info(f'Image {i}: No ground truth bounding boxes, mAP = 0')
                continue

            ious = bbox_overlaps(pred_bboxes, gt_bboxes)
            iou_thr = 0.5
            max_iou = ious.max(axis=1)
            match_indices = max_iou >= iou_thr

            tp = np.sum(match_indices)
            fp = np.sum(~match_indices)
            fn = gt_bboxes.shape[0] - tp

            precision = tp / (tp + fp) if tp + fp > 0 else 0
            recall = tp / (tp + fn) if tp + fn > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

            image_mAPs.append(f1)
            runner.logger.info(f'Image {i}: Precision = {precision}, Recall = {recall}, mAP = {f1}')
        print("Yes I am here Do not worry55555!!!")
    