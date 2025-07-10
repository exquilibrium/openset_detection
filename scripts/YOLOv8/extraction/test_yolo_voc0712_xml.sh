#!/bin/bash 
cd ../../mmdetection

# === SET PATH TO DATASET DIRECTORY!!! ===
MODEL_PATH="/home/chen/openset_detection/scripts/YOLOv8/training/runs/detect/train_voc0712/weights/best.pt"
TRAIN_SET="/media/chen/76AECF8EAECF4579/data/VOCdevkit_xml/VOC0712/ImageSets/YOLO_CS/train.txt"
VAL_SET="/media/chen/76AECF8EAECF4579/data/VOCdevkit_xml/VOC0712/ImageSets/YOLO_CS/val.txt"
TEST_SET="/media/chen/76AECF8EAECF4579/data/VOCdevkit_xml/VOC0712/ImageSets/YOLO/test.txt"
# === SET PATH TO DATASET DIRECTORY!!! ===

NUM_CLSS="15"
SAVE_NAME="frcnn_GMMDet_Voc_xml_yolo"

./test_standard_yolo.sh $MODEL_PATH $NUM_CLSS $SAVE_NAME $TRAIN_SET $VAL_SET $TEST_SET