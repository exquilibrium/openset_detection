#!/bin/bash 
cd ../../mmdetection

# === SET PATH TO DATASET DIRECTORY!!! ===
#MODEL_PATH="/home/chen/openset_detection/scripts/YOLOv8/training/runs/detect/train_lru1_drone/weights/best.pt"
#TRAIN_SET="/media/chen/76AECF8EAECF4579/data/lru1_all/ImageSets/YOLO_CS_drone/train.txt"
#VAL_SET="/media/chen/76AECF8EAECF4579/data/lru1_all/ImageSets/YOLO_CS_drone/val.txt"
#TEST_SET="/media/chen/76AECF8EAECF4579/data/lru1_all/ImageSets/YOLO/test.txt"
#OOD_SET="/media/chen/76AECF8EAECF4579/data/lru1_all/ImageSets/YOLO_CS_drone/test_ood.txt"

MODEL_PATH="/home/chen_le/openset_detection/scripts/YOLOv8/training/runs/detect/train_lru1_drone/weights/best.pt"
TRAIN_SET="/volume/hot_storage/slurm_data/chen_le/ARCHES/lru1_all/ImageSets/YOLO_CS_drone/train.txt"
VAL_SET="/volume/hot_storage/slurm_data/chen_le/ARCHES/lru1_all/ImageSets/YOLO_CS_drone/val.txt"
TEST_SET="/volume/hot_storage/slurm_data/chen_le/ARCHES/lru1_all/ImageSets/YOLO/test.txt"
OOD_SET="/volume/hot_storage/slurm_data/chen_le/ARCHES/lru1_all/ImageSets/YOLO_CS_drone/test_ood.txt"
# === SET PATH TO DATASET DIRECTORY!!! ===

NUM_CLSS="2"
SAVE_NAME="frcnn_GMMDet_Voc_lru1_drone_yolo"

./test_standard_yolo.sh $MODEL_PATH $NUM_CLSS $SAVE_NAME $TRAIN_SET $VAL_SET $TEST_SET $OOD_SET