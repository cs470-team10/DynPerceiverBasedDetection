# 정지용 작성
import pycocotools.coco as coco
import os
from tools.cs470.anaylsis_helper.draw_bbox import draw_bbox, get_size, sanitize_text
from tools.cs470.anaylsis_helper.dyn_perceiver_test import DynPerceiverTest
from tools.cs470.anaylsis_helper.imagenet_mapping import get_imagenet_id
from tqdm import tqdm
from PIL import Image
import re

base_dir = './'
data_dir = "data/coco"
set_name = "val2017"
output_dir = "coco_analysis"
pretrained_file = 'baselines/regnety_800mf_with_dyn_perceiver/reg800m_perceiver_t128.pth'

data_dir = base_dir + data_dir
output_dir = base_dir + output_dir

coco = coco.COCO(f"{data_dir}/annotations/instances_{set_name}.json")
dyn_perceiver = DynPerceiverTest(base_dir, pretrained_file, coco)
os.makedirs(output_dir, exist_ok=True)
os.makedirs(f"{output_dir}/images", exist_ok=True)

def analysis(image_id):
    annotations = coco.loadAnns(coco.getAnnIds(image_id))
    if (len(annotations) == 1):
        image_info = coco.loadImgs(image_id)[0]
        image_width = image_info["width"]
        image_height = image_info["height"]
        image = Image.open(f"{data_dir}/{set_name}/{str(image_id).zfill(12)}.jpg")
        exit_stage, estimated = dyn_perceiver.forward(image, image_width, image_height)
        estimated_class = sanitize_text(get_imagenet_id(estimated))
        draw_bbox(coco, image, output_dir, set_name, image_id, exit_stage, estimated, annotations)
        for annotation in annotations:
            x, y, w, h = [int(b) for b in annotation['bbox']]
            size = get_size(w, h)
            class_name = sanitize_text(coco.loadCats(annotation["category_id"])[0]["name"])
            csv_file.write(f"{image_id},{class_name},{w},{h},{w*h},{size},{exit_stage},{estimated_class}\n")
            csv_file.write(f"{image_id},{w},{h},{size},{class_name}")
        return True
    else:
        return False


csv_file = open(f"{output_dir}/coco_analysis.csv", "w")
csv_file.write("image_id,class_name,bbox_width,bbox_height,bbox_size_1,bbox_size_2,exit_stage,estimated_class\n")
total_length = 0
length = 0

for image_id in tqdm(coco.getImgIds(), desc=f'Analyzing {set_name} images'):
    total_length += 1
    length += 1 if analysis(image_id) else 0

print(f"{length}/{total_length}")

csv_file.close()
