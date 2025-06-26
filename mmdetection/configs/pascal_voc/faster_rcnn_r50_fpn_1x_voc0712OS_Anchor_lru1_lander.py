_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn_voc_wAnchor_lru1_lander.py', '../_base_/datasets/voc0712OS_lru1_lander.py',
    '../_base_/default_runtime.py', 
]
model = dict(roi_head=dict(bbox_head=dict(num_classes=2))) ### <<<<<<<<<<---------- NUM CS CLASSES ---------->>>>>>>>>>
# optimizer
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
# actual epoch = 3 * 3 = 9
lr_config = dict(policy='step', step=[4, 6])
# runtime settings
runner = dict(
    type='EpochBasedRunner', max_epochs=7)
