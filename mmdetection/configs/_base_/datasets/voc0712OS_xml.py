# dataset settings

from base_dirs import BASE_DATA_FOLDER

dataset_type = 'XMLDataset' ### <<<<<<<<<<---------- Important ---------->>>>>>>>>>
data_root = BASE_DATA_FOLDER+'/VOCdevkit_xml/'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1000, 600), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1000, 600),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

voc_cs_classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person']
voc_os_classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        type='RepeatDataset',
        times=3,
        dataset=dict(
            type=dataset_type,
            classes=voc_cs_classes, # CS
            ann_file=[
                data_root + 'VOC0712/ImageSets/Main_CS/train.txt'
            ],
            img_prefix=[data_root + 'VOC0712/'],
            pipeline=train_pipeline)),
    trainCS=dict(
        type=dataset_type,
        classes=voc_cs_classes, # CS
        ann_file=data_root + 'VOC0712/ImageSets/Main_CS/train.txt',
        img_prefix=data_root + 'VOC0712/',
        pipeline=test_pipeline),
    val=dict(
        type=dataset_type,
        classes=voc_cs_classes, # CS
        ann_file=data_root + 'VOC0712/ImageSets/Main_CS/val.txt',
        img_prefix=data_root + 'VOC0712/',
        pipeline=test_pipeline),
    testCS=dict(
        type=dataset_type,
        classes=voc_cs_classes, # CS
        ann_file=data_root + 'VOC0712/ImageSets/Main_CS/test.txt',
        img_prefix=data_root + 'VOC0712/',
        pipeline=test_pipeline),
    testOS=dict(
        type=dataset_type,
        classes=voc_os_classes, # OS
        ann_file=data_root + 'VOC0712/ImageSets/Main/test.txt',
        img_prefix=data_root + 'VOC0712/',
        pipeline=test_pipeline),
    testOOD=dict(
        type=dataset_type,
        classes=voc_os_classes, # OOD only
        ann_file=data_root + 'VOC0712/ImageSets/Main_CS/test_ood.txt',
        img_prefix=data_root + 'VOC0712/',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='mAP')
