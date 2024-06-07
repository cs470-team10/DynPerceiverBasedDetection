from mmdet.registry import DATASETS
from mmdet.datasets.coco import CocoDataset
from mmdet.evaluation.functional import bbox_overlaps
import numpy as np
import logging

@DATASETS.register_module()
class CustomCocoDataset(CocoDataset):
    def evaluate(self, results, metric='bbox', logger=None, jsonfile_prefix=None, **kwargs):
        # 로깅 설정
        if logger is None:
            logger = logging.getLogger()
        
        # 각 이미지에 대한 mAP 계산 로직 작성
        image_mAPs = []
        for i, result in enumerate(results):
            gt_bboxes = self.get_ann_info(i)['bboxes']
            gt_labels = self.get_ann_info(i)['labels']
            pred_bboxes = result[0]
            pred_labels = result[1]
            pred_scores = result[2]

            if gt_bboxes.shape[0] == 0:
                image_mAPs.append(0)
                logger.info(f'Image {i}: No ground truth bounding boxes, mAP = 0')
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
            logger.info(f'Image {i}: Precision = {precision}, Recall = {recall}, mAP = {f1}')

        return dict(mAP=image_mAPs)
