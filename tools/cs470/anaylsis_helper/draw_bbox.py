# 정지용 작성
import matplotlib.pyplot as plt
from tools.cs470.anaylsis_helper.imagenet_mapping import get_imagenet_id
import os
import re

small = 32 * 32 # mAP small 최대 사이즈
medium = 96 * 96 # mAP medium 최대 사이즈
color_list = ["pink", "red", "teal", "blue", "orange", "yellow", "black", "magenta","green","aqua"]*10

def draw_bbox(coco, image, output_dir, set_name, image_id, exit_stage, estimated, annotations):
    fig, ax = plt.subplots(figsize=(15,10))
    
    for annotation in annotations:
        x, y, w, h = [int(b) for b in annotation['bbox']]
        size = get_size(w, h)
        class_id = annotation["category_id"]
        class_name = coco.loadCats(class_id)[0]["name"]
        color_ = color_list[class_id]
        rect = plt.Rectangle((x, y), w, h, linewidth=2, edgecolor=color_, facecolor='none')
        t_box=ax.text(x, y, class_name.capitalize(),  color='red', fontsize=20)
        t_box.set_bbox(dict(boxstyle='square, pad=0.2',facecolor='white', alpha=0.6, edgecolor='blue'))
        ax.add_patch(rect)
    ax.axis('off')
    ax.imshow(image)
    ax.set_xlabel('Longitude')
    ax.set_title(f"({size.capitalize()}) {class_name.capitalize()} ({set_name} id: {image_id}) - Exiting in {exit_stage} (Estimated: {get_imagenet_id(estimated).capitalize()})")
    class_name = re.sub('[^0-9a-zA-Z]+', '_', class_name)
    os.makedirs(f'{output_dir}/images/{class_name}', exist_ok=True)
    os.makedirs(f'{output_dir}/images/{class_name}/exit_{exit_stage}', exist_ok=True)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/images/{class_name}/exit_{exit_stage}/{str(image_id).zfill(12)}.jpg')
    plt.close()

def get_size(w, h):
    if w * h < small:
        return "small"
    elif w * h < medium:
        return "medium"
    else:
        return "large"
    
def sanitize_text(text:str):
    return re.sub('[\t]+', '_', re.sub('[,]+', '/', re.sub('[^0-9a-zA-Z|/|\t]+', '_', text)))