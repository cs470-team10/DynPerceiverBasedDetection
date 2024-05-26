from mmengine.runner import ValLoop, TestLoop
from typing import Dict, List, Union

from torch.utils.data import DataLoader
from mmengine.evaluator import Evaluator

from mmdet.registry import LOOPS

from dyn_perceiver.get_threshold import get_threshold as _get_threshold
import torch
from cs470_logger.cs470_print import cs470_print
from tools.cs470.dynamic_evaluation_logger import DynamicEvaluationLogger
import random

NUM_IMAGES = [[365, 184, 21, 23], [348, 184, 28, 33], [322, 195, 29, 47], [301, 195, 33, 64], [280, 194, 45, 74], [262, 190, 50, 91], [244, 186, 58, 105], [227, 171, 73, 122], [207, 170, 76, 140], [191, 163, 84, 155], [176, 156, 89, 172], [162, 154, 96, 181], [153, 153, 97, 190], [140, 152, 96, 205], [134, 145, 96, 218], [120, 141, 98, 234], [109, 137, 97, 250], [109, 137, 97, 250], [103, 128, 99, 263], [93, 126, 94, 280], [86, 121, 89, 297], [76, 114, 80, 323], [62, 107, 79, 345], [56, 100, 84, 353], [50, 88, 85, 370], [43, 77, 88, 385], [36, 64, 89, 404], [32, 55, 81, 425], [25, 51, 69, 448], [18, 46, 60, 469], [14, 37, 54, 488], [10, 31, 49, 503], [6, 21, 48, 518], [0, 0, 0, 593]]

@LOOPS.register_module()
class DynamicTestLoopRandomExiting(TestLoop):
    def __init__(self,
                 runner,
                 dataloader: Union[DataLoader, Dict],
                 evaluator: Union[Evaluator, Dict, List],
                 fp16: bool = False,
                 dynamic_evaluate: bool = False):
        super().__init__(runner, dataloader, evaluator, fp16)
        self.dynamic_evaluate = dynamic_evaluate
        if self.dynamic_evaluate:
            self.get_flops()

    def run(self) -> dict:
        self.runner.call_hook('before_test')
        self.runner.call_hook('before_test_epoch')
        self.runner.model.eval()
        if self.dynamic_evaluate:
            cs470_print("Dynamic Evaluation")
            self.evaluate_logger = DynamicEvaluationLogger(self.runner._log_dir, self.flops)
            self.get_threshold_and_flops()
            for index, _threshold in enumerate(self.thresholds):
                num_exiting_images = NUM_IMAGES[index]
                thresholds = [torch.tensor([-1,1,1,1]),torch.tensor([1,-1,1,1]),torch.tensor([1,1,-1,1]),torch.tensor([1,1,1,-1])]
                applied_thresholds = []
                for new_threshold, count in zip(thresholds, num_exiting_images):
                    for _ in range(count):
                        applied_thresholds.append(new_threshold)
                random.shuffle(applied_thresholds)
                for idx, data_batch in enumerate(self.dataloader):
                    self.set_threshold(applied_thresholds[i])
                    self.run_iter(idx, data_batch)
                    self.evaluate_logger.append(self.get_last_exited_stage())
                    self.evaluate_logger.append_classifier(self.get_last_classifiy_correct())
                    self.unset_threshold()
                metrics = self.evaluator.evaluate(len(self.dataloader.dataset))
                self.runner.model.metrics.append(metrics)
                self.evaluate_logger.save_info(metrics, [0,0,0,0])
            self.runner.call_hook('after_test_epoch', metrics=metrics)
            self.runner.call_hook('after_test')
            self.evaluate_logger.process()
        else:
            for idx, data_batch in enumerate(self.dataloader):
                self.run_iter(idx, data_batch)

            metrics = self.evaluator.evaluate(len(self.dataloader.dataset))
            self.runner.call_hook('after_test_epoch', metrics=metrics)
            self.runner.call_hook('after_test')
        return metrics
    
    @torch.no_grad()
    def get_threshold_and_flops(self):
        self.thresholds = _get_threshold(self.runner.model, self.runner.train_loop.dataloader, self.fp16)
        cs470_print("Thresholds: " + str([threshold.tolist() for threshold in self.thresholds]))
        cs470_print("Flops per early exiting stages: " + str(self.flops.tolist()))
        return
    
    def get_flops(self):
        self.flops = self.runner.model.get_dynamic_flops(data_loader=self.dataloader)
    
    def get_last_exited_stage(self):
        return self.runner.model.backbone.get_last_exited_stage()
    
    def get_last_classifiy_correct(self):
        return self.runner.model.classifiy_correct
    
    def set_threshold(self, threshold):
        self.runner.model.set_threshold(threshold)

    def unset_threshold(self):
        self.runner.model.unset_threshold()
