from mmengine.runner import ValLoop, TestLoop
from typing import Dict, List, Union

from torch.utils.data import DataLoader
from mmengine.evaluator import Evaluator

from mmdet.registry import LOOPS

from dyn_perceiver.get_threshold import get_threshold as _get_threshold
import torch
from cs470_logger.cs470_print import cs470_print
from tools.cs470.dynamic_evaluation_logger import DynamicEvaluationLogger, DynamicValidationLogger

# [CS470] 강우성, [CS470] 이정완: 아까 저희가 봤던 Loop을 수정한 버전입니다. Dynamic Evaluation은 여기서 일어난다고 생각하시면 될 것 같습니다.
# Early Exiting을 위해서는 classifier의 결과가 필요한데, 이를 위해서 DynRetinaNet이라는 구조가 추가되었습니다.
# 변경된 점은 DynPerceiver 내의 classifier를 output까지 가져오는 것 하나입니다.
# 물론 이 작업은 training 때는 돌아가지 않고, evaluate 파트에서만 돌아갑니다.
# 지금 보면 ValLoop랑 TestLoop 2개 있는데, 사실상 동일한 코드라 여기서 변경할 점이 있다면 하나 작업한 후에 복붙하시면 될 듯 합니다.

@LOOPS.register_module()
class DynamicValLoop(ValLoop):
    def __init__(self,
                 runner,
                 dataloader: Union[DataLoader, Dict],
                 evaluator: Union[Evaluator, Dict, List],
                 fp16: bool = False,
                 dynamic_evaluate: bool = False):
        super().__init__(runner, dataloader, evaluator, fp16)
        cs470_print("Dynamic Validation")
        self.dynamic_evaluate = dynamic_evaluate
        if self.dynamic_evaluate:
            self.get_flops()

    def run(self) -> dict:
        self.runner.call_hook('before_val')
        self.runner.call_hook('before_val_epoch')
        self.runner.model.eval()
        if self.dynamic_evaluate:
            self.evaluate_logger = DynamicValidationLogger(self.runner._log_dir, self.flops, self.runner.train_loop.epoch)
            # [CS470] 강우성: 아래의 함수에서 threshold 계산해서 연결해줌. 딱히 건들 필요는 ㄴㄴ
            self.get_threshold_and_flops()
            # [CS470] 이정완: [TODO] 여기서 threshold별 mAP 구해야합니다. run_ter 참조.
            # 그리고 하나 구현해두면 ValLoop에서도 동일하니 복붙하시면 됩니다.
            for _index, threshold in enumerate(self.thresholds):
                cs470_print("Thresholds for output: " + str(threshold.tolist()))
                self.set_threshold(threshold)
                for idx, data_batch in enumerate(self.dataloader):
                    self.run_iter(idx, data_batch)
                    self.evaluate_logger.append(self.get_last_exited_stage())
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
        self.thresholds = _get_threshold(self.runner.model, self.runner.train_loop.dataloader, self.fp16)
        cs470_print("Thresholds: " + str([threshold.tolist() for threshold in self.thresholds]))
        cs470_print("Flops per early exiting stages: " + str(self.flops.tolist()))
        self.evaluate_logger.add_flops(self.flops)
        return
    
    def get_flops(self):
        self.flops = self.runner.model.get_dynamic_flops(data_loader=self.dataloader)
    
    def get_last_exited_stage(self):
        return self.runner.model.backbone.get_last_exited_stage()

    def set_threshold(self, threshold):
        self.runner.model.set_threshold(threshold) # [CS470] 이정완: 모델에 threshold를 전달하는 법

    def unset_threshold(self):
        self.runner.model.unset_threshold()

@LOOPS.register_module()
class DynamicTestLoop(TestLoop):
    def __init__(self,
                 runner,
                 dataloader: Union[DataLoader, Dict],
                 evaluator: Union[Evaluator, Dict, List],
                 fp16: bool = False,
                 dynamic_evaluate: bool = False):
        super().__init__(runner, dataloader, evaluator, fp16)
        cs470_print("Dynamic Test")
        self.dynamic_evaluate = dynamic_evaluate
        if self.dynamic_evaluate:
            self.get_flops()

    def run(self) -> dict:
        self.runner.call_hook('before_test')
        self.runner.call_hook('before_test_epoch')
        self.runner.model.eval()
        if self.dynamic_evaluate:
            self.evaluate_logger = DynamicEvaluationLogger(self.runner._log_dir, self.flops)
            # [CS470] 강우성: 아래의 함수에서 threshold 계산해서 연결해줌. 딱히 건들 필요는 ㄴㄴ
            self.get_threshold_and_flops()
            # [CS470] 이정완: [TODO] 여기서 threshold별 mAP 구해야합니다. run_ter 참조.
            # 그리고 하나 구현해두면 ValLoop에서도 동일하니 복붙하시면 됩니다.
            for _index, threshold in enumerate(self.thresholds):
                cs470_print("Thresholds for output: " + str(threshold.tolist()))
                self.set_threshold(threshold)
                for idx, data_batch in enumerate(self.dataloader):
                    self.run_iter(idx, data_batch)
                    self.evaluate_logger.append(self.get_last_exited_stage())
                self.unset_threshold()
                metrics = self.evaluator.evaluate(len(self.dataloader.dataset))
                self.runner.model.metrics.append(metrics)
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
    def get_threshold_and_flops(self):
        self.thresholds = _get_threshold(self.runner.model, self.runner.train_loop.dataloader, self.fp16)
        cs470_print("Thresholds: " + str([threshold.tolist() for threshold in self.thresholds]))
        cs470_print("Flops per early exiting stages: " + str(self.flops.tolist()))
        return
    
    def get_flops(self):
        self.flops = self.runner.model.get_dynamic_flops(data_loader=self.dataloader)
    
    def get_last_exited_stage(self):
        return self.runner.model.backbone.get_last_exited_stage()
    
    def set_threshold(self, threshold):
        self.runner.model.set_threshold(threshold) # [CS470] 이정완: 모델에 threshold를 전달하는 법

    def unset_threshold(self):
        self.runner.model.unset_threshold()
