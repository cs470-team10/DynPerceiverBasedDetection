# 정지용 작성
import pycocotools.coco as coco
import os
from tools.cs470.anaylsis_helper.draw_bbox import draw_bbox, get_size
from tools.cs470.anaylsis_helper.dyn_perceiver_test import DynPerceiverTest
from tqdm import tqdm

base_dir = './'
data_dir = "data/coco"
set_name = "val2017"
output_dir = "coco_analysis"
pretrained_file = 'baselines/regnety_800mf_with_dyn_perceiver/reg800m_perceiver_t128.pth'

dyn_perceiver = DynPerceiverTest(base_dir + pretrained_file)

data_dir = base_dir + data_dir
output_dir = base_dir + output_dir

coco = coco.COCO(f"{data_dir}/annotations/instances_{set_name}.json")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(f"{output_dir}/images", exist_ok=True)

def analysis(image_id):
    annotations = coco.loadAnns(coco.getAnnIds(image_id))
    if (len(annotations) == 1):
        draw_bbox(coco, data_dir, output_dir, set_name, image_id, annotations)
        for annotation in annotations:
            x, y, w, h = [int(b) for b in annotation['bbox']]
            size = get_size(w, h)
            class_name = coco.loadCats(annotation["category_id"])[0]["name"]

            csv_file.write(f"{image_id},{w},{h},{size},{class_name}")
        return True
    else:
        return False



csv_file = open(f"{output_dir}/coco_analysis.csv", "w")
csv_file.write("image_id,bbox_width,bbox_height,bbox_size,class_name\n")
total_length = 0
length = 0

for image_id in tqdm(coco.getImgIds(), desc=f'Analyzing {set_name} images'):
    total_length += 1
    length += 1 if analysis(image_id) else 0

print(f"{length}/{total_length}")

csv_file.close()
