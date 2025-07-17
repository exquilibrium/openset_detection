#!/bin/sh

cd ../../mmdetection

# Get project root: openset_detection/
PROJECT_ROOT=$(realpath "$(dirname "$0")/..")
BASE_DATA_FOLDER=$(python3 -c "import sys; sys.path.insert(0, '${PROJECT_ROOT}'); import base_dirs; print(base_dirs.BASE_DATA_FOLDER)")

# === SET PATH TO DATASET DIRECTORY!!! ===
MODEL_PATH="${PROJECT_ROOT}/scripts/YOLOv8/training/runs/detect/train_voc0712/weights/best.pt"
TRAIN_SET="${BASE_DATA_FOLDER}/VOCdevkit_xml/VOC0712/ImageSets/YOLO_CS/train.txt"
VAL_SET="${BASE_DATA_FOLDER}/VOCdevkit_xml/VOC0712/ImageSets/YOLO_CS/val.txt"
TEST_SET="${BASE_DATA_FOLDER}/VOCdevkit_xml/VOC0712/ImageSets/YOLO/test.txt"
OOD_SET="${BASE_DATA_FOLDER}/VOCdevkit_xml/VOC0712/ImageSets/YOLO_CS/test_ood.txt"
# === SET PATH TO DATASET DIRECTORY!!! ===

NUM_CLSS="15"
SAVE_NAME="frcnn_GMMDet_Voc_xml_yolo"

./test_standard_yolo.sh $MODEL_PATH $NUM_CLSS $SAVE_NAME $TRAIN_SET $VAL_SET $TEST_SET $OOD_SET