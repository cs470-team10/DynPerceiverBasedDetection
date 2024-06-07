from os import path
import os
import torch
from cs470_logger.cs470_print import cs470_print

class DynamicEvaluationLogger:
    def __init__(self, log_dir, flops, filename = "test_info.csv"):
        self.log_dir = path.join(log_dir, "cs470_log")
        os.makedirs(self.log_dir, exist_ok=True)
        self.csv_file_dir = path.join(self.log_dir, filename)
        self.flops = flops
        self.num_exiting_images = torch.tensor([0, 0, 0, 0])
        self.flops_info = []
        self.mAP_info = []
        self.mAP_50_info = []
        self.mAP_75_info = []
        self.mAP_s_info = []
        self.mAP_m_info = []
        self.mAP_l_info = []
        self.image_ratio_info = []
        self.thresholds_info = []
        self.classification_accuracy = []
        self.flops_unit = 1e9
        self.is_coco = None
        self.correct_classify_images = 0

    def _set_is_coco(self, metrics):
        if self.is_coco is not None:
            return
        for key in metrics.keys():
            if "coco/bbox_mAP" in key:
                self.is_coco = True
                return
            
        self.is_coco = False
        return

    def save_info(self, metrics, thresholds):
        self._set_is_coco(metrics)
        self.flops_info.append(self.get_average_flops())
        self.image_ratio_info.append(self.get_ratio_of_exiting_stages())
        if self.is_coco:
            self.mAP_info.append(metrics['coco/bbox_mAP'])
            self.mAP_50_info.append(metrics['coco/bbox_mAP_50'])
            self.mAP_75_info.append(metrics['coco/bbox_mAP_75'])
            self.mAP_s_info.append(metrics['coco/bbox_mAP_s'])
            self.mAP_m_info.append(metrics['coco/bbox_mAP_m'])
            self.mAP_l_info.append(metrics['coco/bbox_mAP_l'])
        else:
            self.mAP_info.append(metrics['pascal_voc/mAP'])
            self.mAP_50_info.append(metrics['pascal_voc/AP50'])
        self.thresholds_info.append(thresholds)
        self.classification_accuracy.append(self.get_classification_accuracy())
        self.num_exiting_images = torch.tensor([0, 0, 0, 0])
        self.correct_classify_images = 0

    def append(self, exiting_stage):
        self.num_exiting_images[exiting_stage - 1] += 1

    def append_classifier(self, correct):
        self.correct_classify_images += 1 if correct else 0

    def get_average_flops(self):
        return torch.sum((self.num_exiting_images / torch.sum(self.num_exiting_images)) * self.flops).item() / self.flops_unit
    
    def get_ratio_of_exiting_stages(self):
        return (self.num_exiting_images / torch.sum(self.num_exiting_images)).tolist()
    
    def get_classification_accuracy(self):
        cs470_print(f"Classification Accuracy: {str(self.correct_classify_images * 1.0 / torch.sum(self.num_exiting_images).item())}")
        return self.correct_classify_images * 1.0 / torch.sum(self.num_exiting_images).item()
    
    def process(self):
        csv_file = open(self.csv_file_dir, "w")
        label, _ = self._get_mAP(0)
        csv_file.write(f"flops(GF),{label},exiting_in_1,exiting_in_2,exiting_in_3,exiting_in_4,threshold_1,threshold_2,threshold_3,threshold_4,classification_accuracy\n")
        for i in range(len(self.flops_info)):
            _, mAP_info = self._get_mAP(i)
            output = [self.flops_info[i]] + mAP_info + self.image_ratio_info[i] + self.thresholds_info[i] + [self.classification_accuracy[i]]
            csv_file.write(",".join(str(num) for num in output) + "\n")
        csv_file.close()
        cs470_print(self.csv_file_dir + " saved.")
        return
    
    def _get_mAP(self, i):
        if self.is_coco:
            return "bbox_mAP,bbox_mAP_50,bbox_mAP_75,bbox_mAP_s,bbox_mAP_m,bbox_mAP_l", [self.mAP_info[i], self.mAP_50_info[i], self.mAP_75_info[i], self.mAP_s_info[i], self.mAP_m_info[i], self.mAP_l_info[i]]
        else:
            return "pascal_voc_mAP,pascal_voc_AP50", [self.mAP_info[i], self.mAP_50_info[i]]

class DynamicValidationLogger(DynamicEvaluationLogger):
    def __init__(self, log_dir, flops, epoch):
        super().__init__(log_dir, flops, f"epoch_{epoch}_validation_info.csv")
