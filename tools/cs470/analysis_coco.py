# 정지용 작성

# Threshold는 Dynamic Perceiver에서 dynamic_evaluate에서 따왔습니다.(아래 config 변수에 있음.)
# [0] Accuarcy: 71.012%, Flops: 0.50M
# [1] Accuarcy: 71.524%, Flops: 0.51M
# [2] Accuarcy: 72.062%, Flops: 0.53M
# [3] Accuarcy: 72.566%, Flops: 0.54M
# [4] Accuarcy: 73.110%, Flops: 0.55M
# [5] Accuarcy: 73.584%, Flops: 0.56M
# [6] Accuarcy: 74.172%, Flops: 0.58M
# [7] Accuarcy: 74.778%, Flops: 0.59M
# [8] Accuarcy: 75.408%, Flops: 0.61M
# [9] Accuarcy: 75.956%, Flops: 0.63M
# [10] Accuarcy: 76.436%, Flops: 0.65M
# [11] Accuarcy: 76.900%, Flops: 0.66M
# [12] Accuarcy: 77.334%, Flops: 0.68M
# [13] Accuarcy: 77.820%, Flops: 0.70M
# [14] Accuarcy: 78.144%, Flops: 0.72M
# [15] Accuarcy: 78.466%, Flops: 0.74M
# [16] Accuarcy: 78.684%, Flops: 0.75M
# [17] Accuarcy: 78.684%, Flops: 0.75M
# [18] Accuarcy: 78.970%, Flops: 0.77M
# [19] Accuarcy: 79.130%, Flops: 0.78M
# [20] Accuarcy: 79.284%, Flops: 0.80M
# [21] Accuarcy: 79.418%, Flops: 0.83M
# [22] Accuarcy: 79.530%, Flops: 0.85M
# [23] Accuarcy: 79.658%, Flops: 0.87M
# [24] Accuarcy: 79.720%, Flops: 0.90M
# [25] Accuarcy: 79.772%, Flops: 0.93M
# [26] Accuarcy: 79.794%, Flops: 0.96M
# [27] Accuarcy: 79.776%, Flops: 1.00M
# [28] Accuarcy: 79.810%, Flops: 1.04M
# [29] Accuarcy: 79.836%, Flops: 1.07M
# [30] Accuarcy: 79.858%, Flops: 1.11M
# [31] Accuarcy: 79.836%, Flops: 1.16M
# [32] Accuarcy: 79.802%, Flops: 1.19M

base_dir = './' # 기본 위치
data_dir = "data/coco" # 데이터셋 위치
set_name = "train2017" # 측정하고자 하는 validation name
output_dir = "coco_analysis" # 어디다가 저장할건지?
pretrained_file = 'baselines/regnety_800mf_with_dyn_perceiver/reg800m_perceiver_t128.pth' # dynamic perceiver의 pre-train weight을 가져오면 됨.
draw_bbox_indexes = [] # bbox 그리고 싶은 config의 index를 넣어주면 됩니다. 안그리고 싶으면 비워두면 됨.

# ----------------------------------------------------------------------------------------------------

import pycocotools.coco as _coco
import os
from tools.cs470.anaylsis_helper.draw_bbox import draw_bbox, get_size, sanitize_text
from tools.cs470.anaylsis_helper.dyn_perceiver_test import DynPerceiverTest
from tools.cs470.anaylsis_helper.imagenet_mapping import get_imagenet_id
from tqdm import tqdm
import re
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

data_dir = base_dir + data_dir
output_dir = base_dir + output_dir

os.makedirs(output_dir, exist_ok=True)
output_dir = output_dir + "/" + set_name
os.makedirs(output_dir, exist_ok=True)
os.makedirs(f"{output_dir}/csvs", exist_ok=True)
os.makedirs(f"{output_dir}/graphs", exist_ok=True)
color_list = ["pink", "red", "teal", "blue", "orange", "yellow", "magenta","green","aqua"]

image_ids = []
classes = []
image_widths = []
image_heights = []
bbox_widths = []
bbox_heights = []
bbox_size_1s = []
bbox_size_2s = []
bbox_ratios = []
exit_stages = []
estimated_classes = []

def analysis_image(image_id, model, coco, config):
    annotations = coco.loadAnns(coco.getAnnIds(image_id))
    if (len(annotations) == 1):
        image_info = coco.loadImgs(image_id)[0]
        image_width = image_info["width"]
        image_height = image_info["height"]
        result, image, value, index = model.forward(image_id, f"{data_dir}/{set_name}/{str(image_id).zfill(12)}.jpg", len(draw_bbox_indexes) > 0)
        if (result is False):
            return
        annotation = annotations[0]

        x, y, w, h = [int(b) for b in annotation['bbox']]
        size = get_size(w, h)
        class_name = sanitize_text(coco.loadCats(annotation["category_id"])[0]["name"])
        append_default(image_id,class_name,image_width,image_height,w,h,w*h,size,(w * h) / (image_width * image_height))
        for config_entry in config:
            T = config_entry["threshold"]
            exit_stage, estimated = model.analysis_threshold(value, index, T)
            estimated_class = sanitize_text(get_imagenet_id(estimated))
            append_threshold(config_entry['index'], exit_stage, estimated_class)

        if (len(draw_bbox_indexes) > 0):
            for draw_bbox_index in draw_bbox_indexes:
                T = config[draw_bbox_index]["threshold"]
                exit_stage, estimated = model.analysis_threshold(value, index, T)
                draw_bbox(coco, image, f"{output_dir}/images/{file_name(config[draw_bbox_index], 'images', '')}", set_name, image_id, exit_stage, estimated, annotations)

def append_default(image_id, class_name, image_width, image_height, bbox_width, bbox_height, bbox_size_1, bbox_size_2, bbox_ratio):
    image_ids.append(image_id)
    classes.append(class_name)
    image_widths.append(image_width)
    image_heights.append(image_height)
    bbox_widths.append(bbox_width)
    bbox_heights.append(bbox_height)
    bbox_size_1s.append(bbox_size_1)
    bbox_size_2s.append(bbox_size_2)
    bbox_ratios.append(bbox_ratio)

def append_threshold(i, exit_stage, estimated_class):
    exit_stages[i].append(exit_stage)
    estimated_classes[i].append(estimated_class)

def file_name(config_entry, name = "coco_analysis_accuarcy", posfix=".csv"):
    accuarcy = re.sub("[.]", "_", str(config_entry["accuarcy"]))
    flops = re.sub("[.]", "_", str(config_entry["flops"]))
    return f"{config_entry['index']}_{name}__accuarcy_{accuarcy}__flops_{flops}{posfix}"

def formatting_config_entry(config_entry, path = ''):
    path = f", be saving in {path}" if path != '' else ""
    return f"[{config_entry['index']}] Accuarcy: {config_entry['accuarcy']}%, Flops: {config_entry['flops']}{path}"

def graph_path(i, config_entry, title):
    formatted_title = str(i) + "_" + re.sub("[\t]+", "_", title).lower()
    os.makedirs(f"{output_dir}/graphs/{formatted_title}", exist_ok=True)
    return f"{output_dir}/graphs/{formatted_title}/" + file_name(config_entry, name = re.sub("[\t]+", "_", title).lower(), posfix=".jpg")

def graph_title(config_entry, title):
    return title + "\n" + formatting_config_entry(config_entry)

def draw_graph_entry(x, y, s, color, xticks, title, x_title, y_title, output_dir, label_format = '{:,.0f}', yticks = []):
    plt.figure(figsize=(15, 10))
    plt.scatter(x, y, s=s, color = color)

    plt.xlabel(x_title, fontdict = {'fontsize' : 15})
    plt.xticks(xticks, fontsize = 15)
    if (len(yticks) > 0):
        plt.yticks(yticks, fontsize = 15)
    else:
        plt.yticks(fontsize = 15)
    plt.ylabel(y_title, fontdict = {'fontsize' : 15})
    plt.title(title, weight = 'bold', fontdict = {'fontsize' : 20})
    ticks_loc = plt.gca().get_yticks().tolist()
    plt.gca().yaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
    plt.gca().set_yticklabels([label_format.format(x) for x in ticks_loc])
    plt.tight_layout()
    plt.savefig(output_dir)
    plt.close()

def draw_graph(config_entry, small_total, medium_total, large_total):
    x_title = "Early Exit Stages"
    index = config_entry['index']
    
    i = 1
    color_index = 0

    ## Bbox size per Exit Stage
    exp_title = "Bbox Size per Exit Stage"
    path = graph_path(i, config_entry, exp_title)
    title = graph_title(config_entry, exp_title)
    y_title = "Bbox Size(pixel^2)"
    x = exit_stages[index]
    y = [i * 1 for i in bbox_size_1s]
    s = [80 for i in x]
    color = [color_list[(color_index + i - 1) % len(color_list)] for i in x]
    draw_graph_entry(x, y, s, color, [1,2,3,4], title, x_title, y_title, path, '{:,.0f}')

    i += 1
    color_index += 4

    ## Bbox ratio per Exit Stage
    exp_title = "Bbox Ratio per Exit Stage"
    path = graph_path(i, config_entry, exp_title)
    title = graph_title(config_entry, exp_title)
    y_title = "Bbox Ratio(%)"
    x = exit_stages[index]
    y = [i * 100 for i in bbox_ratios]
    s = [80 for i in x]
    color = [color_list[(color_index + i - 1) % len(color_list)] for i in x]
    draw_graph_entry(x, y, s, color, [1,2,3,4], title, x_title, y_title, path, '{:,.0f}%')

    i += 1
    color_index += 4

    ## mAP size per Exit Stage
    exp_title = "mAP Size per Exit Stage"
    path = graph_path(i, config_entry, exp_title)
    title = graph_title(config_entry, exp_title)
    y_title = "mAP Size(small = 1, medium = 2, large = 3)"
    x = [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]
    y = [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3]
    s = []

    for x_index in [1, 2, 3, 4]:
        for y_index in [1, 2, 3]:
            total = 0
            if (y_index == 1):
                target = 'small'
                divide = small_total
            elif (y_index == 2):
                target = 'medium'
                divide = medium_total
            elif (y_index == 3):
                target = 'large'
                divide = large_total
            for i in range(len(image_ids)):
                if (bbox_size_2s[i] == target and exit_stages[index][i] == x_index):
                    total += 1
            if (divide == 0):
                s.append(0)
            else:
                s.append(20000.0 * (total * 1.0 / divide))
    color = [color_list[(color_index + i - 1) % len(color_list)] for i in y]
    draw_graph_entry(x, y, s, color, [1,2,3,4], title, x_title, y_title, path, '{:,.0f}', yticks=[1,2,3])

    i += 1
    color_index += 4


def analysis(config):
    for i in range(len(config)):
        exit_stages.append([])
        estimated_classes.append([])
    print("Analyze target:")
    for config_entry in config:
        print(formatting_config_entry(config_entry, file_name(config_entry)))
    print("\n")
    if (len(draw_bbox_indexes) > 0):
        os.makedirs(f"{output_dir}/images", exist_ok=True)
        print("Draw bbox enabled for target:")
        for draw_bbox_index in draw_bbox_indexes:
            config_entry = config[draw_bbox_index]
            print(formatting_config_entry(config_entry, "images/" + file_name(config[draw_bbox_index], 'images', '') + "/*"))
            os.makedirs(f"{output_dir}/images/{file_name(config[draw_bbox_index], 'images', '')}", exist_ok=True)
        print("\n")
    coco = _coco.COCO(f"{data_dir}/annotations/instances_{set_name}.json")
    dyn_perceiver = DynPerceiverTest(base_dir, output_dir, pretrained_file)
    for image_id in tqdm(coco.getImgIds(), desc=f'Analyzing {set_name} images'):
        analysis_image(image_id, dyn_perceiver, coco, config)
    
    dyn_perceiver.save_cache()

    small_total = 0
    medium_total = 0
    large_total = 0
    for i in range(len(image_ids)):
        if (bbox_size_2s[i] == 'small'):
            small_total += 1
        if (bbox_size_2s[i] == 'medium'):
            medium_total += 1
        if (bbox_size_2s[i] == 'large'):
            large_total += 1

    for config_entry in tqdm(config, desc=f'Making csv files'):
        i = config_entry['index']
        csv_file = open(f"{output_dir}/csvs/{file_name(config_entry)}", "w")
        csv_file.write("image_id,class_name,image_width,image_height,bbox_width,bbox_height,bbox_size_1,bbox_size_2,bbox_ratio,exit_stage,estimated_class\n")
        for j in range(len(image_ids)):
            csv_file.write(f"{image_ids[j]},{classes[j]},{image_widths[j]},{image_heights[j]},{bbox_widths[j]},{bbox_heights[j]},{bbox_size_1s[j]},{bbox_size_2s[j]},{bbox_ratios[j]},{exit_stages[i][j]},{estimated_classes[i][j]}\n")
        csv_file.close()

    for config_entry in tqdm(config, desc=f'Drawing graphs'):
        draw_graph(config_entry, small_total, medium_total, large_total)
        
    print(f"Total analyzed images: {len(image_ids)}/{len(coco.getImgIds())} (small images: {small_total}, medium images: {medium_total}, large iamges: {large_total})")

config = [{'index': 0, 'accuarcy': '71.012', 'flops': '0.50M', 'threshold': [0.17108260095119476, 0.09215764701366425, 0.08628688752651215, -100000000.0]}, {'index': 1, 'accuarcy': '71.524', 'flops': '0.51M', 'threshold': [0.19014045596122742, 0.11052383482456207, 0.10168169438838959, -100000000.0]}, {'index': 2, 'accuarcy': '72.062', 'flops': '0.53M', 'threshold': [0.21074678003787994, 0.12973493337631226, 0.1147477775812149, -100000000.0]}, {'index': 3, 'accuarcy': '72.566', 'flops': '0.54M', 'threshold': [0.23101598024368286, 0.1498234122991562, 0.12824562191963196, -100000000.0]}, {'index': 4, 'accuarcy': '73.110', 'flops': '0.55M', 'threshold': [0.25348547101020813, 0.16746169328689575, 0.14211776852607727, -100000000.0]}, {'index': 5, 'accuarcy': '73.584', 'flops': '0.56M', 'threshold': [0.2749546766281128, 0.18839630484580994, 0.1570391058921814, -100000000.0]}, {'index': 6, 'accuarcy': '74.172', 'flops': '0.58M', 'threshold': [0.2972123324871063, 0.20938993990421295, 0.17038308084011078, -100000000.0]}, {'index': 7, 'accuarcy': '74.778', 'flops': '0.59M', 'threshold': [0.32199037075042725, 0.23119482398033142, 0.18098518252372742, -100000000.0]}, {'index': 8, 'accuarcy': '75.408', 'flops': '0.61M', 'threshold': [0.34821707010269165, 0.25238943099975586, 0.19185809791088104, -100000000.0]}, {'index': 9, 'accuarcy': '75.956', 'flops': '0.63M', 'threshold': [0.3738722801208496, 0.2738359272480011, 0.2056938111782074, -100000000.0]}, {'index': 10, 'accuarcy': '76.436', 'flops': '0.65M', 'threshold': [0.39778774976730347, 0.29331475496292114, 0.2184578776359558, -100000000.0]}, {'index': 11, 'accuarcy': '76.900', 'flops': '0.66M', 'threshold': [0.42058131098747253, 0.31685400009155273, 0.23203787207603455, -100000000.0]}, {'index': 12, 'accuarcy': '77.334', 'flops': '0.68M', 'threshold': [0.4436093866825104, 0.33770108222961426, 0.24252906441688538, -100000000.0]}, {'index': 13, 'accuarcy': '77.820', 'flops': '0.70M', 'threshold': [0.46794241666793823, 0.3597652316093445, 0.25371596217155457, -100000000.0]}, {'index': 14, 'accuarcy': '78.144', 'flops': '0.72M', 'threshold': [0.49303779006004333, 0.3801296353340149, 0.26543813943862915, -100000000.0]}, {'index': 15, 'accuarcy': '78.466', 'flops': '0.74M', 'threshold': [0.5187497138977051, 0.4020453989505768, 0.27617257833480835, -100000000.0]}, {'index': 16, 'accuarcy': '78.684', 'flops': '0.75M', 'threshold': [0.5404309630393982, 0.42071378231048584, 0.288371741771698, -100000000.0]}, {'index': 17, 'accuarcy': '78.684', 'flops': '0.75M', 'threshold': [0.5404309630393982, 0.42071378231048584, 0.288371741771698, -100000000.0]}, {'index': 18, 'accuarcy': '78.970', 'flops': '0.77M', 'threshold': [0.5642502903938293, 0.4429320693016052, 0.3011114001274109, -100000000.0]}, {'index': 19, 'accuarcy': '79.130', 'flops': '0.78M', 'threshold': [0.5878438949584961, 0.46467655897140503, 0.31212520599365234, -100000000.0]}, {'index': 20, 'accuarcy': '79.284', 'flops': '0.80M', 'threshold': [0.6150196194648743, 0.48851433396339417, 0.3243710994720459, -100000000.0]}, {'index': 21, 'accuarcy': '79.418', 'flops': '0.83M', 'threshold': [0.6423977017402649, 0.5151886343955994, 0.3405746817588806, -100000000.0]}, {'index': 22, 'accuarcy': '79.530', 'flops': '0.85M', 'threshold': [0.6679922342300415, 0.5439590811729431, 0.3553999662399292, -100000000.0]}, {'index': 23, 'accuarcy': '79.658', 'flops': '0.87M', 'threshold': [0.6978371739387512, 0.5731199979782104, 0.37403300404548645, -100000000.0]}, {'index': 24, 'accuarcy': '79.720', 'flops': '0.90M', 'threshold': [0.7249411940574646, 0.6060675382614136, 0.39228442311286926, -100000000.0]}, {'index': 25, 'accuarcy': '79.772', 'flops': '0.93M', 'threshold': [0.7507454752922058, 0.6367663145065308, 0.41207045316696167, -100000000.0]}, {'index': 26, 'accuarcy': '79.794', 'flops': '0.96M', 'threshold': [0.7723860144615173, 0.6686626672744751, 0.4361606538295746, -100000000.0]}, {'index': 27, 'accuarcy': '79.776', 'flops': '1.00M', 'threshold': [0.7967429757118225, 0.6977986693382263, 0.4646255075931549, -100000000.0]}, {'index': 28, 'accuarcy': '79.810', 'flops': '1.04M', 'threshold': [0.8132866024971008, 0.7234163284301758, 0.4953666925430298, -100000000.0]}, {'index': 29, 'accuarcy': '79.836', 'flops': '1.07M', 'threshold': [0.8288894295692444, 0.7450734972953796, 0.5320638418197632, -100000000.0]}, {'index': 30, 'accuarcy': '79.858', 'flops': '1.11M', 'threshold': [0.8466140031814575, 0.7640485763549805, 0.5767571926116943, -100000000.0]}, {'index': 31, 'accuarcy': '79.836', 'flops': '1.16M', 'threshold': [0.8643122911453247, 0.7828938961029053, 0.6226142048835754, -100000000.0]}, {'index': 32, 'accuarcy': '79.802', 'flops': '1.19M', 'threshold': [0.8785117268562317, 0.800524115562439, 0.664090096950531, -100000000.0]}]

analysis(config)