_base_ ='../../../configs/regnet/retinanet_regnetx-3.2GF_fpn_ilsvrc.py'  # '../configs/regnet/retinanet_regnetx-3.2GF_fpn_1x_coco.py'
dynamic_evaluate_epoch = [12] # Training 때 dynamic evaluation을 할 epoch. 안할거면 [], 다할거면 [i + 1 for i in range(12)]
theta_factor = 1
lambda_factor = 1
dynamic_evaluate_on_test = True

custom_imports = dict(
    imports=['mmdet.models.backbones.dyn_perceiver_regnet_zeromap',
             'mmdet.datasets.ilsvrc'],
    allow_failed_imports=False)

num_classes = 200

model = dict(
    backbone=dict(
        type='DynPerceiverZeromap',
        test_num=2,
        num_classes=num_classes,
        init_cfg=dict(type='Pretrained', 
                      checkpoint='./baselines/regnety_800mf_with_dyn_perceiver/reg800m_perceiver_t128_converted.pth')),
    neck=dict(type = "DynFPN",
              in_channels=[64, 144, 320, 784],
              add_extra_convs='on_output'),
    bbox_head=dict(
        loss_dyn=dict(theta_factor=theta_factor,
                      lambda_factor=lambda_factor,
                       type='DynLoss'),
        num_classes=num_classes,
        type='DynRetinaHead'),
    type='DynRetinaNet'
)

custom_hooks = [
    dict(type='WandbLoggerHook',
         init_kwargs=dict(project='cs470', entity='plasma3365'),
         interval=10,
         log_checkpoint=True,
         log_model=True)
]

val_cfg = dict(type='DynamicValLoop', dynamic_evaluate_epoch=dynamic_evaluate_epoch)
test_cfg = dict(type='DynamicTestLoop', dynamic_evaluate=dynamic_evaluate_on_test)

dataset_type = 'ILSVRCDataset' # Custom

test_dataloader = dict(
    dataset=dict(
        ann_file='annotations/filtered_val_xml_files.txt',
        data_prefix=dict(img='val2014_single/')))

train_dataloader = dict(
    batch_sampler=dict(drop_last=True),
    dataset=dict(
        ann_file='annotations/filtered_train_xml_files.txt',
        backend_args=None,
        data_prefix=dict(img='train2014_single/')))

val_dataloader = dict(
    dataset=dict(
        ann_file='annotations/filtered_val_xml_files.txt',
        data_prefix=dict(img='val2014_single/')))


"""
Coco dataset_type을 그대로 사용하는 코드

classes = ('accordion', 'airplane', 'ant', 'antelope', 'apple', 'armadillo', 
    'artichoke', 'axe', 'baby_bed', 'backpack', 'bagel', 'balance_beam',
    'banana', 'band_aid', 'banjo', 'baseball', 'basketball', 'bathing_cap',
    'beaker', 'bear', 'bee', 'bell_pepper', 'bench', 'bicycle', 'binder',
    'bird', 'bookshelf', 'bow_tie', 'bow', 'bowl', 'brassiere', 'burrito',
    'bus', 'butterfly', 'camel', 'can_opener', 'car', 'cart', 'cattle',
    'cello', 'centipede', 'chain_saw', 'chair', 'chime', 'cocktail_shaker',
    'coffee_maker', 'computer_keyboard', 'computer_mouse', 'corkscrew',
    'cream', 'croquet_ball', 'crutch', 'cucumber', 'cup_or_mug', 'diaper',
    'digital_clock', 'dishwasher', 'dog', 'domestic_cat', 'dragonfly',
    'drum', 'dumbbell', 'electric_fan', 'elephant', 'face_powder', 'fig',
    'filing_cabinet', 'flower_pot', 'flute', 'fox', 'french_horn', 'frog',
    'frying_pan', 'giant_panda', 'goldfish', 'golf_ball', 'golfcart',
    'guacamole', 'guitar', 'hair_dryer', 'hair_spray', 'hamburger',
    'hammer', 'hamster', 'harmonica', 'harp', 'hat_with_a_wide_brim',
    'head_cabbage', 'helmet', 'hippopotamus', 'horizontal_bar', 'horse',
    'hotdog', 'iPod', 'isopod', 'jellyfish', 'koala_bear', 'ladle',
    'ladybug', 'lamp', 'laptop', 'lemon', 'lion', 'lipstick', 'lizard',
    'lobster', 'maillot', 'maraca', 'microphone', 'microwave', 'milk_can',
    'miniskirt', 'monkey', 'motorcycle', 'mushroom', 'nail', 'neck_brace',
    'oboe', 'orange', 'otter', 'pencil_box', 'pencil_sharpener', 'perfume',
    'person', 'piano', 'pineapple', 'ping-pong_ball', 'pitcher', 'pizza',
    'plastic_bag', 'plate_rack', 'pomegranate', 'popsicle', 'porcupine',
    'power_drill', 'pretzel', 'printer', 'puck', 'punching_bag', 'purse',
    'rabbit', 'racket', 'ray', 'red_panda', 'refrigerator',
    'remote_control', 'rubber_eraser', 'rugby_ball', 'ruler',
    'salt_or_pepper_shaker', 'saxophone', 'scorpion', 'screwdriver',
    'seal', 'sheep', 'ski', 'skunk', 'snail', 'snake', 'snowmobile',
    'snowplow', 'soap_dispenser', 'soccer_ball', 'sofa', 'spatula',
    'squirrel', 'starfish', 'stethoscope', 'stove', 'strainer',
    'strawberry', 'stretcher', 'sunglasses', 'swimming_trunks', 'swine',
    'syringe', 'table', 'tape_player', 'tennis_ball', 'tick', 'tie',
    'tiger', 'toaster', 'traffic_light', 'train', 'trombone', 'trumpet',
    'turtle', 'tv_or_monitor', 'unicycle', 'vacuum', 'violin',
    'volleyball', 'waffle_iron', 'washer', 'water_bottle', 'watercraft',
    'whale', 'wine_bottle', 'zebra')
  

data = dict(
    train=dict(
        classes=classes,
        ann_file='annotations/ILSVRC_val2014_single.json',
        img_prefix='ILSVRC2014/val2014_single/'),
    val=dict(
        classes=classes,
        ann_file='annotations/ILSVRC_val2014_single.json',
        img_prefix='ILSVRC2014/val2014_single/'),
    test=dict(
        classes=classes,
        ann_file='annotations/ILSVRC_val2014_single.json',
        img_prefix='ILSVRC2014/val2014_single/')) 
"""