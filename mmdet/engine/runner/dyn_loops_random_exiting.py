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

@LOOPS.register_module()
class DynamicTestLoopRandomExiting(TestLoop):
    def __init__(self,
                 runner,
                 dataloader: Union[DataLoader, Dict],
                 evaluator: Union[Evaluator, Dict, List],
                 num_images,
                 fp16: bool = False,
                 dynamic_evaluate: bool = False):
        super().__init__(runner, dataloader, evaluator, fp16)
        self.dynamic_evaluate = dynamic_evaluate
        self.num_images = num_images
        if self.dynamic_evaluate:
            self.get_flops()

    def run(self) -> dict:
        self.runner.call_hook('before_test')
        self.runner.call_hook('before_test_epoch')
        self.runner.model.eval()
        if self.dynamic_evaluate:
            self.get_threshold_and_flops()
            for iteraion_num in range(10):
                cs470_print(f"Dynamic Evaluation {str(iteraion_num + 1)}")
                self.evaluate_logger = DynamicEvaluationLogger(self.runner._log_dir, self.flops, f"test_info_{str(iteraion_num + 1)}.csv")
                for index, _threshold in enumerate(self.thresholds):
                    num_exiting_images = self.num_images[index]
                    thresholds = [torch.tensor([-1,1,1,1]),torch.tensor([1,-1,1,1]),torch.tensor([1,1,-1,1]),torch.tensor([1,1,1,-1])]
                    applied_thresholds = []
                    for new_threshold, count in zip(thresholds, num_exiting_images):
                        for _ in range(count):
                            applied_thresholds.append(new_threshold)
                    random.shuffle(applied_thresholds)
                    for idx, data_batch in enumerate(self.dataloader):
                        self.set_threshold(applied_thresholds[idx])
                        self.run_iter(idx, data_batch)
                        self.evaluate_logger.append(self.get_last_exited_stage())
                        self.evaluate_logger.append_classifier(self.get_last_classifiy_correct())
                        self.unset_threshold()
                    metrics = self.evaluator.evaluate(len(self.dataloader.dataset))
                    self.runner.model.metrics.append(metrics)
                    self.evaluate_logger.save_info(metrics, [0,0,0,0])
                self.evaluate_logger.process()
            self.runner.call_hook('after_test_epoch', metrics=metrics)
            self.runner.call_hook('after_test')
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
