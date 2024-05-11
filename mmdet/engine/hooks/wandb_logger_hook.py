from mmengine.registry import HOOKS
from mmengine.hooks import Hook
import wandb
from mmengine.runner import *
from .dynamic_summary import summary
from io import StringIO
import sys

@HOOKS.register_module()
class WandbLoggerHook(Hook):
    def __init__(self, interval=1, log_checkpoint=True, log_model=True, **kwargs):
        super().__init__()
        self.interval = interval
        self.log_checkpoint = log_checkpoint
        self.log_model = log_model

    def after_train_iter(self, runner, **kwargs):
        if self.every_n_inner_iters(runner.iter, self.interval):
            # MessageHub에서 로그 스칼라의 최신 값을 사용
            metrics = {key: runner.message_hub.get_scalar(key).current()
                    for key in runner.message_hub.log_scalars.keys()}
            wandb.log(metrics, step=runner.iter)

    def after_train_epoch(self, runner, **kwargs):
        work_dir = runner.work_dir  # runner의 작업 디렉토리를 가져옵니다.
        if self.log_checkpoint:
            # find_latest_checkpoint 함수를 사용하여 최근 체크포인트 파일 경로를 얻습니다.
            checkpoint_path = find_latest_checkpoint(work_dir)
            if checkpoint_path:
                wandb.save(checkpoint_path, policy="now")

        if self.log_model and runner.epoch % self.interval == 0:
            # Log a summary of the model using torchsummary
            original_stdout = sys.stdout  # Save a reference to the original standard output
            sys.stdout = StringIO()  # Redirect standard output to a StringIO object.
            
            summary(runner.model, input_size=(3, 1333, 800))
            
            model_summary = sys.stdout.getvalue()  # Retrieve the value written to the StringIO object
            sys.stdout = original_stdout  # Restore the standard output to its original value
            
            wandb.log({"model_summary": model_summary}, step=runner.iter)

