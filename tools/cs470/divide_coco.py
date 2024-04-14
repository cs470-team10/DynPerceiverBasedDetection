# 이찬규 작성
import os
import shutil
import json
from tqdm import tqdm  # tqdm 라이브러리를 불러옵니다.

# 각 데이터셋 경로 설정
annotation_path = './annotations_trainval2017'
image_paths = {'val': './val2017', 'train': './train2017/train2017'}
# 새로운 폴더 생성
for dataset_type in ['val', 'train']:
    os.makedirs(f'{dataset_type}_single', exist_ok=True)
# 각 .json 파일을 순회하며 처리
for dataset_type, path in image_paths.items():
    annFile = f'{annotation_path}/instances_{dataset_type}2017.json'
    # JSON 파일 읽기
    with open(annFile, 'r') as f:
        coco_data = json.load(f)
    # 이미지 ID별 객체 수 카운트
    img_to_objs = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id in img_to_objs:
            img_to_objs[img_id] += 1
        else:
            img_to_objs[img_id] = 1
    # 한 개의 객체만 포함하는 이미지 식별
    single_obj_imgs = [img_id for img_id, count in img_to_objs.items() if count == 1]
    # 전체 이미지 수와 한 개의 객체만 포함하는 이미지의 수 계산
    total_images = len(coco_data['images'])
    num_single_obj_imgs = len(single_obj_imgs)
    ratio_single_obj_imgs = num_single_obj_imgs / total_images
    # 한 개의 객체만 포함하는 이미지의 비율과 개수 출력
    # print(f'{dataset_type} dataset: {num_single_obj_imgs} out of {total_images} images have a single object. Ratio: {ratio_single_obj_imgs:.2f}')
    # 한 개의 객체만 포함하는 이미지 식별
    single_obj_imgs = [img_id for img_id, count in img_to_objs.items() if count == 1]
    # 해당 이미지 파일을 새 폴더로 복사, tqdm을 적용해 진행 상황 표시
    for img_id in tqdm(single_obj_imgs, desc=f'Copying {dataset_type} images'):
        # 이미지 ID를 사용하여 파일 이름 찾기
        file_name = next((item['file_name'] for item in coco_data['images'] if item['id'] == img_id), None)
        if file_name:
            src = os.path.join(path, file_name)
            dst = os.path.join(f'{dataset_type}_single', file_name)
            shutil.copy(src, dst)
