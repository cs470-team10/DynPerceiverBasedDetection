from mmengine.runner import ValLoop, TestLoop
from mmengine.runner.amp import autocast
from typing import Dict, List, Optional, Sequence, Tuple, Union

from torch.utils.data import DataLoader
from mmengine.evaluator import Evaluator

from mmdet.registry import LOOPS

from dyn_perceiver.get_threshold import get_threshold as _get_threshold
import torch
from torch import Tensor

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
        print("[CS470] Dynamic Validation")
        self.dynamic_evaluate = dynamic_evaluate

    def run(self) -> dict:
        self.runner.call_hook('before_val')
        self.runner.call_hook('before_val_epoch')
        self.runner.model.eval()
        if self.dynamic_evaluate:
            # [CS470] 강우성: 아래의 함수에서 threshold 계산해서 연결해줌. 딱히 건들 필요는 ㄴㄴ
            self.get_threshold_and_flops()
            # [CS470] 이정완: [TODO] 여기서 threshold별 mAP 구해야합니다. run_ter 참조.
            # 그리고 하나 구현해두면 TestLoop에서도 동일하니 복붙하시면 됩니다.
            """
            원래 이 함수가 하나의 threshold에 대해서 하나의 metrics 결과를 리턴하고 그것 바탕으로
            wandb logger가 map 측정해서 그래프 작성하는 것 같은데, 각 threshold별로 map를 측정해서
            한번에 리턴하게 되면 wandb logger의 input dim에서 오류가 발생할 것 같아서 우선 각 threshold
            별로 evaluate까지 한 metrics 결과를 dyn_retinanet에 metrics라는 리스트에 append하는 식으로
            코드를 작성해봤습니다
            """
            for index, tensor in enumerate(self.thresholds):
                for idx, data_batch in enumerate(self.dataloader):
                    self.run_iter(index, idx, data_batch)
                metrics = self.evaluator.evaluate(len(self.dataloader.dataset))
                self.runner.model.metrics.append(metrics)
            print("finish")
            self.runner.call_hook('after_val_epoch', metrics=metrics)
            self.runner.call_hook('after_val')
            print(self.runner.model.metrics)
        else:
            for idx, data_batch in enumerate(self.dataloader):
                self.run_iter(-1, idx, data_batch)

            # compute metrics
            metrics = self.evaluator.evaluate(len(self.dataloader.dataset))
            self.runner.call_hook('after_val_epoch', metrics=metrics)
            self.runner.call_hook('after_val')
        return metrics
    
    @torch.no_grad()
    def get_threshold_and_flops(self):
        # [CS470] 강우성: threshold는 아래의 method를 통해 DynRetinaNet에 전달.
        # [CS470] 이정완: flops는 early exiting stage별 flops를 계산한 결과가 전달됩니다.
        self.thresholds = _get_threshold()
        print("Thresholds: " + str(self.thresholds))
        self.flops = self.runner.model.get_dynamic_flops(data_loader=self.dataloader)
        print("Flops per early exiting stages: " + str(self.flops))
        return
    
    def set_threshold(self, threshold):
        self.runner.model.set_threshold(threshold) # [CS470] 이정완: 모델에 threshold를 전달하는 법

    @torch.no_grad()
    def run_iter(self, index, idx, data_batch: Sequence[dict]):
        self.runner.call_hook(
            'before_val_iter', batch_idx=idx, data_batch=data_batch)
        # outputs should be sequence of BaseDataElement
        #print("Check argument(Val loop):", self.dynamic_evaluate)
        if self.dynamic_evaluate:
            #print("val loop dyn eval in")
            with autocast(enabled=self.fp16):
                # 모델에 threshold를 넣어주는 방법입니다.
                self.set_threshold(self.thresholds[index])
                print("Thresholds for output: " + str(self.thresholds[index]))
                outputs, y_early3, y_att, y_cnn, y_merge = self.runner.model.test_step(data_batch)
            self.evaluator.process(data_samples=outputs, data_batch=data_batch)
            self.runner.call_hook(
                'after_val_iter',
                batch_idx=idx,
                data_batch=data_batch,
                outputs=outputs)
        else:
            with autocast(enabled=self.fp16):
                outputs = self.runner.model.test_step(data_batch)

            self.evaluator.process(data_samples=outputs, data_batch=data_batch)
            self.runner.call_hook(
                'after_val_iter',
                batch_idx=idx,
                data_batch=data_batch,
                outputs=outputs)

@LOOPS.register_module()
class DynamicTestLoop(TestLoop):
    def __init__(self,
                 runner,
                 dataloader: Union[DataLoader, Dict],
                 evaluator: Union[Evaluator, Dict, List],
                 fp16: bool = False,
                 dynamic_evaluate: bool = False):
        super().__init__(runner, dataloader, evaluator, fp16)
        print("[CS470] Dynamic Test")
        self.dynamic_evaluate = dynamic_evaluate

    def run(self) -> dict:
        self.runner.call_hook('before_test')
        self.runner.call_hook('before_test_epoch')
        self.runner.model.eval()
        if self.dynamic_evaluate:
            # [CS470] 강우성: 아래의 함수에서 threshold 계산해서 연결해줌. 딱히 건들 필요는 ㄴㄴ
            self.get_threshold_and_flops()
            # [CS470] 이정완: [TODO] 여기서 threshold별 mAP 구해야합니다. run_ter 참조.
            # 그리고 하나 구현해두면 ValLoop에서도 동일하니 복붙하시면 됩니다.
            for index, tensor in enumerate(self.thresholds):
                print("Thresholds for output: " + str(self.thresholds[index]))
                for idx, data_batch in enumerate(self.dataloader):
                    self.run_iter(index, idx, data_batch)
                metrics = self.evaluator.evaluate(len(self.dataloader.dataset))
                self.runner.model.metrics.append(metrics)
            print("finish")
            self.runner.call_hook('after_test_epoch', metrics=metrics)
            self.runner.call_hook('after_test')
        else:
            for idx, data_batch in enumerate(self.dataloader):
                self.run_iter(-1, idx, data_batch)

            metrics = self.evaluator.evaluate(len(self.dataloader.dataset))
            print("finish")
            self.runner.call_hook('after_test_epoch', metrics=metrics)
            self.runner.call_hook('after_test')
        return metrics
    
    @torch.no_grad()
    def get_threshold_and_flops(self):
        # [CS470] 강우성: threshold는 아래의 method를 통해 DynRetinaNet에 전달.
        # [CS470] 이정완: flops는 early exiting stage별 flops를 계산한 결과가 전달됩니다.
        self.thresholds = _get_threshold(self.runner.model, self.runner.train_loop.dataloader, self.fp16) #self.runner.train_loop.dataloader
        print("Thresholds: " + str(self.thresholds))
        self.flops = self.runner.model.get_dynamic_flops(data_loader=self.dataloader)
        print("Flops per early exiting stages: " + str(self.flops))
        return
    
    def set_threshold(self, threshold):
        self.runner.model.set_threshold(threshold) # [CS470] 이정완: 모델에 threshold를 전달하는 법

    @torch.no_grad()
    def run_iter(self, index, idx, data_batch: Sequence[dict]):
        self.runner.call_hook(
            'before_test_iter', batch_idx=idx, data_batch=data_batch)
        if self.dynamic_evaluate:
            with autocast(enabled=self.fp16):
                # 모델에 threshold를 넣어주는 방법입니다.
                self.set_threshold(self.thresholds[index])
                #print("Thresholds for output: " + str(self.thresholds[index]))
                outputs, y_early3, y_att, y_cnn, y_merge = self.runner.model.test_step(data_batch)
            self.evaluator.process(data_samples=outputs, data_batch=data_batch)
            self.runner.call_hook(
                'after_test_iter',
                batch_idx=idx,
                data_batch=data_batch,
                outputs=outputs)
        else:
            with autocast(enabled=self.fp16):
                outputs = self.runner.model.test_step(data_batch)

            self.evaluator.process(data_samples=outputs, data_batch=data_batch)
            self.runner.call_hook(
                'after_test_iter',
                batch_idx=idx,
                data_batch=data_batch,
                outputs=outputs)
            

def get_label_list(data_batch: Sequence[dict]) -> Tensor:
    labels = []
    
    for t in data_batch['data_samples']:
        labels.append(t.gt_instances.labels.tolist()[0])
    return torch.tensor(labels)