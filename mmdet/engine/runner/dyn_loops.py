from mmengine.runner import ValLoop, TestLoop
from typing import Dict, List, Union, Sequence
from mmengine.runner.amp import autocast

from torch.utils.data import DataLoader
from mmengine.evaluator import Evaluator

from mmdet.registry import LOOPS

from dyn_perceiver.get_threshold import get_threshold as _get_threshold
import torch
from cs470_logger.cs470_print import cs470_print
from tools.cs470.dynamic_evaluation_logger import DynamicEvaluationLogger, DynamicValidationLogger
from tools.cs470.qualitive_logger import QualitiveLogger

@LOOPS.register_module()
class DynamicValLoop(ValLoop):
    def __init__(self,
                 runner,
                 dataloader: Union[DataLoader, Dict],
                 evaluator: Union[Evaluator, Dict, List],
                 fp16: bool = False,
                 dynamic_evaluate_epoch: List[int] = [],
                 threshold_distribution: List[float] = [0.85, 1, 0.5, 1]
                 ):
        super().__init__(runner, dataloader, evaluator, fp16)
        self.dynamic_evaluate_epoch = dynamic_evaluate_epoch
        self.threshold_distribution = threshold_distribution
        if len(self.dynamic_evaluate_epoch) > 0:
            self.get_flops()

    def run(self) -> dict:
        self.runner.call_hook('before_val')
        self.runner.call_hook('before_val_epoch')
        self.runner.model.eval()
        self.get_dynamic_evalute()
        if self.dynamic_evaluate:
            cs470_print("Dynamic Evaluation")
            self.evaluate_logger = DynamicValidationLogger(self.runner._log_dir, self.flops, self.runner.train_loop.epoch)
            self.get_threshold_and_flops()
            for _index, threshold in enumerate(self.thresholds):
                cs470_print("Thresholds for output: " + str(threshold.tolist()))
                self.set_threshold(threshold)
                for idx, data_batch in enumerate(self.dataloader):
                    self.run_iter(idx, data_batch)
                    self.evaluate_logger.append(self.get_last_exited_stage())
                    self.evaluate_logger.append_classifier(self.get_last_classifiy_correct())
                self.unset_threshold()
                metrics = self.evaluator.evaluate(len(self.dataloader.dataset))
                self.runner.model.metrics.append(metrics)
                self.evaluate_logger.save_info(metrics, threshold.tolist())
            self.runner.call_hook('after_val_epoch', metrics=metrics)
            self.runner.call_hook('after_val')
            self.evaluate_logger.process()
        else:
            for idx, data_batch in enumerate(self.dataloader):
                self.run_iter(idx, data_batch)

            metrics = self.evaluator.evaluate(len(self.dataloader.dataset))
            self.runner.call_hook('after_val_epoch', metrics=metrics)
            self.runner.call_hook('after_val')
        return metrics
    
    @torch.no_grad()
    def get_threshold_and_flops(self):
        self.thresholds = _get_threshold(self.runner.model, self.runner.train_loop.dataloader, self.threshold_distribution, self.fp16)
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

    def get_dynamic_evalute(self):
        self.dynamic_evaluate = self.runner.train_loop.epoch in self.dynamic_evaluate_epoch

@LOOPS.register_module()
class DynamicTestLoop(TestLoop):
    def __init__(self,
                 runner,
                 dataloader: Union[DataLoader, Dict],
                 evaluator: Union[Evaluator, Dict, List],
                 fp16: bool = False,
                 dynamic_evaluate: bool = False,
                 threshold_distribution: List[float] = [0.85, 1, 0.5, 1],
                 use_qualitive_logger: bool = False):
        super().__init__(runner, dataloader, evaluator, fp16)
        self.dynamic_evaluate = dynamic_evaluate
        self.threshold_distribution = threshold_distribution
        self.use_qualitive_logger = use_qualitive_logger and dynamic_evaluate
        self.qualitive_logger = None
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
            for index, threshold in enumerate(self.thresholds):
                cs470_print("Thresholds for output: " + str(threshold.tolist()))
                if self.use_qualitive_logger:
                    self.qualitive_logger = QualitiveLogger(self.runner._log_dir, index + 1)
                self.set_threshold(threshold)
                for idx, data_batch in enumerate(self.dataloader):
                    self.run_iter(idx, data_batch)
                    self.evaluate_logger.append(self.get_last_exited_stage())
                    self.evaluate_logger.append_classifier(self.get_last_classifiy_correct())
                self.unset_threshold()
                metrics = self.evaluator.evaluate(len(self.dataloader.dataset))
                self.runner.model.metrics.append(metrics)
                if self.use_qualitive_logger:
                    self.qualitive_logger.save_info()
                    self.qualitive_logger = None
                self.evaluate_logger.save_info(metrics, threshold.tolist())
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
    def run_iter(self, idx, data_batch: Sequence[dict]) -> None:
        self.runner.call_hook(
            'before_test_iter', batch_idx=idx, data_batch=data_batch)
        # predictions should be sequence of BaseDataElement
        with autocast(enabled=self.fp16):
            outputs = self.runner.model.test_step(data_batch)
        if self.use_qualitive_logger:
            self.qualitive_logger.process(idx, data_batch, outputs, self.get_last_classifiy_correct())
        self.evaluator.process(data_samples=outputs, data_batch=data_batch)
        self.runner.call_hook(
            'after_test_iter',
            batch_idx=idx,
            data_batch=data_batch,
            outputs=outputs)
    
    @torch.no_grad()
    def get_threshold_and_flops(self):
        self.thresholds = _get_threshold(self.runner.model, self.runner.train_loop.dataloader, self.threshold_distribution, self.fp16)
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
