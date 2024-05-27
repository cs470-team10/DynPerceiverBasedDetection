from os import path
import os
import torch
from cs470_logger.cs470_print import cs470_print

class DynamicEvaluationLogger:
    def __init__(self, log_dir, flops):
        self.log_dir = path.join(log_dir, "cs470_log")
        os.makedirs(self.log_dir, exist_ok=True)
        self.csv_file_dir = path.join(self.log_dir, "test_info.csv")
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
        self.flops_unit = 1e9

    def save_info(self, metrics, thresholds):
        self.flops_info.append(self.get_average_flops())
        self.image_ratio_info.append(self.get_ratio_of_exiting_stages())
        self.mAP_info.append(metrics['coco/bbox_mAP'])
        self.mAP_50_info.append(metrics['coco/bbox_mAP_50'])
        self.mAP_75_info.append(metrics['coco/bbox_mAP_75'])
        self.mAP_s_info.append(metrics['coco/bbox_mAP_s'])
        self.mAP_m_info.append(metrics['coco/bbox_mAP_m'])
        self.mAP_l_info.append(metrics['coco/bbox_mAP_l'])
        self.thresholds_info.append(thresholds)
        self.num_exiting_images = torch.tensor([0, 0, 0, 0])

    def append(self, exiting_stage):
        self.num_exiting_images[exiting_stage - 1] += 1

    def get_average_flops(self):
        return torch.sum((self.num_exiting_images / torch.sum(self.num_exiting_images)) * self.flops).item() / self.flops_unit
    
    def get_ratio_of_exiting_stages(self):
        return (self.num_exiting_images / torch.sum(self.num_exiting_images)).tolist()
    
    def process(self):
        
        csv_file = open(self.csv_file_dir, "w")
        csv_file.write("flops(GF),bbox_mAP,bbox_mAP_50,bbox_mAP_75,bbox_mAP_s,bbox_mAP_m,bbox_mAP_l,exiting_in_1,exiting_in_2,exiting_in_3,exiting_in_4,threshold_1,threshold_2,threshold_3,threshold_4\n")
        for i in range(len(self.flops_info)):
            output = [self.flops_info[i], self.mAP_info[i], self.mAP_50_info[i], self.mAP_75_info[i], self.mAP_s_info[i], self.mAP_m_info[i], self.mAP_l_info[i]] + self.image_ratio_info[i] + self.thresholds_info[i]
            csv_file.write(",".join(str(num) for num in output) + "\n")
        csv_file.close()
        cs470_print(self.csv_file_dir + " saved.")
        return
    

class DynamicValidationLogger(DynamicEvaluationLogger):
    def __init__(self, log_dir, flops, epoch):
        super().__init__(log_dir, flops)
        self.csv_file_dir = path.join(self.log_dir, f"epoch_{epoch}_validation_info.csv")
