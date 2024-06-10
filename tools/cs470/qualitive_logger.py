from os import path
import os
from cs470_logger.cs470_print import cs470_print
from typing import Sequence
from mmdet.structures import DetDataSample

class QualitiveLogger:
    def __init__(self, log_dir, threshold_number):
        self.log_dir = path.join(log_dir, "cs470_log")
        os.makedirs(self.log_dir, exist_ok=True)
        self.csv_file_dir = path.join(self.log_dir, f"threshold_{threshold_number}_qualitive.csv")
        self.fieldnames = ['image_id', 'bbox', 'label', 'score', 'early_exit', 'batch_idx']
        self.content = []

    def process(self, batch_idx:int , data_batch: dict, outputs: Sequence[DetDataSample], early_exit_stage: int):
        for output in outputs:
            img_id = os.path.basename(output.metainfo['img_path'])
            bboxes = output.pred_instances.bboxes.tolist() if output.pred_instances.bboxes.numel() > 0 else []
            labels = output.pred_instances.labels.tolist() if output.pred_instances.labels.numel() > 0 else []
            scores = output.pred_instances.scores.tolist() if output.pred_instances.scores.numel() > 0 else []

            for bbox, label, score in zip(bboxes, labels, scores):
                self.content.append({
                    'image_id': img_id,
                    'bbox': bbox,
                    'label': label,
                    'score': score,
                    'early_exit' : early_exit_stage,
                    'batch_idx' : batch_idx
                })
        
    
    def save_info(self):
        csv_file = open(self.csv_file_dir, "w")
        csv_file.write(",".join(self.fieldnames) + "\n")
        for content in self.content:
            csv_file.write(",".join(str(content[fieldname]) for fieldname in self.fieldnames) + "\n")
        csv_file.close()
        cs470_print(self.csv_file_dir + " saved.")
        return

