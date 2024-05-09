from mmdet.models.detectors import DynRetinaNet
import argparse
import os
import torch

from mmengine.config import Config
from mmengine.runner import Runner

def parse_args():
    parser = argparse.ArgumentParser(description='Test get_dynamic_flops function')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    # 설정 파일 로드
    cfg = Config.fromfile(args.config)

    # 모델 초기화
    model = DynRetinaNet(
        backbone=cfg.model.backbone,
        neck=cfg.model.neck,
        bbox_head=cfg.model.bbox_head,
        train_cfg=cfg.model.train_cfg,
        test_cfg=cfg.model.test_cfg,
        data_preprocessor=cfg.data_preprocessor,
        init_cfg=cfg.model.init_cfg,
        dynamic_evaluate=True  # 동적 평가 활성화
    )

    # 체크포인트 로드
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])

    # get_dynamic_flops 메서드 테스트
    flops = model.get_dynamic_flops(num_images=10)  # 테스트할 이미지 수를 10으로 설정
    print("Dynamic FLOPS:", flops)

if __name__ == '__main__':
    main()