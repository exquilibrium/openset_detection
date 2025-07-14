#!/bin/bash 
cd ../../mmdetection

# === SET PATH TO DATASET DIRECTORY!!! ===
MODEL_PATH="/home/chen/openset_detection/scripts/YOLOv8/training/runs/detect/train_lru1_lru2/weights/best.pt"
TRAIN_SET="/media/chen/76AECF8EAECF4579/data/lru1_all/ImageSets/YOLO_CS_lru2/train.txt"
VAL_SET="/media/chen/76AECF8EAECF4579/data/lru1_all/ImageSets/YOLO_CS_lru2/val.txt"
TEST_SET="/media/chen/76AECF8EAECF4579/data/lru1_all/ImageSets/YOLO/test.txt"
OOD_SET="/media/chen/76AECF8EAECF4579/data/lru1_all/ImageSets/YOLO_CS_lru2/test_ood.txt"
# === SET PATH TO DATASET DIRECTORY!!! ===

NUM_CLSS="2"
SAVE_NAME="frcnn_GMMDet_Voc_lru1_lru2_yolo"

./test_standard_yolo.sh $MODEL_PATH $NUM_CLSS $SAVE_NAME $TRAIN_SET $VAL_SET $TEST_SET $OOD_SET