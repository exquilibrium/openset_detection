#!/bin/bash 
cd ../../mmdetection

CONFIG="configs/pascal_voc/faster_rcnn_r50_fpn_1x_voc0712OS_Anchor_lru1.py"
SAVE_DIR="weights/frcnnCEwAnchorVocCS_lru1"

python tools/train.py $CONFIG --gpus 1 --work-dir $SAVE_DIR
