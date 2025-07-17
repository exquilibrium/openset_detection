#!/bin/sh

cd ../../mmdetection

# Get project root: openset_detection/
PROJECT_ROOT=$(realpath "$(dirname "$0")/..")
BASE_DATA_FOLDER=$(python3 -c "import sys; sys.path.insert(0, '${PROJECT_ROOT}'); import base_dirs; print(base_dirs.BASE_DATA_FOLDER)")

# === SET PATH TO DATASET DIRECTORY!!! ===
DS=ardea10
CS=lander

MODEL_PATH="${PROJECT_ROOT}/scripts/YOLOv8/training/runs/detect/train_${DS}_${CS}/weights/best.pt"
TRAIN_SET="${BASES_DATA_FOLDER}/ardea10_all/ImageSets/YOLO_CS_${CS}/train.txt"
VAL_SET="${BASES_DATA_FOLDER}/ardea10_all/ImageSets/YOLO_CS_${CS}/val.txt"
TEST_SET="${BASES_DATA_FOLDER}/ardea10_all/ImageSets/YOLO/test.txt"
OOD_SET="${BASES_DATA_FOLDER}/ardea10_all/ImageSets/YOLO_CS_${CS}/test_ood.txt"
# === SET PATH TO DATASET DIRECTORY!!! ===

NUM_CLSS="2"
SAVE_NAME="frcnn_GMMDet_Voc_${DS}_${CS}_yolo"

./test_standard_yolo.sh $MODEL_PATH $NUM_CLSS $SAVE_NAME $TRAIN_SET $VAL_SET $TEST_SET $OOD_SET