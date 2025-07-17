#!/bin/bash 
cd ../../mmdetection

CONFIG="configs/pascal_voc/faster_rcnn_r50_fpn_1x_voc0712OS_Anchor_ardea10_lru2.py"
SAVE_DIR="weights/frcnnCEwAnchorVocCS_ardea10_lru2"

python tools/train.py $CONFIG --gpus 1 --work-dir $SAVE_DIR
