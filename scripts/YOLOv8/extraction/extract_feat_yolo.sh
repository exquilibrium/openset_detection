#!/bin/bash



# lru1 classes: lander, lru2, drone
# === SET PATH TO DATASET DIRECTORY!!! ===
MODEL_PATH="/home/chen/TMNF/YOLOv8/training/runs/detect/ardea10_cs/weights/best.pt"
TRAIN_DIR="/media/chen/76AECF8EAECF4579/data/ardea10_run04_labels/VOCdevkit/VOC2007CS_lru2/splits/images/train"
VAL_DIR="/media/chen/76AECF8EAECF4579/data/ardea10_run04_labels/VOCdevkit/VOC2007CS_lru2/splits/images/val"
TEST_DIR="/media/chen/76AECF8EAECF4579/data/ardea10_run04_labels/VOCdevkit/VOC2007/splits/images/test"
CONF_THRESH=0.2
IOU_THRESH=0.5
# === SET PATH TO DATASET DIRECTORY!!! ===



# Path to script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Extract features
python3 "$SCRIPT_DIR/feat_extraction_yolo.py" "$MODEL_PATH" "$TRAIN_DIR" --num_classes 2 --conf_thresh $CONF_THRESH --iou_thresh $IOU_THRESH
python3 "$SCRIPT_DIR/feat_extraction_yolo.py" "$MODEL_PATH" "$VAL_DIR" --num_classes 3 --conf_thresh $CONF_THRESH --iou_thresh $IOU_THRESH
python3 "$SCRIPT_DIR/feat_extraction_yolo.py" "$MODEL_PATH" "$TEST_DIR" --num_classes 3 --conf_thresh $CONF_THRESH --iou_thresh $IOU_THRESH

# Assign predictions
python3 "$SCRIPT_DIR/pred_assignment_yolo.py" "train" --num_classes 2 --conf_thresh $CONF_THRESH --iou_thresh $IOU_THRESH
python3 "$SCRIPT_DIR/pred_assignment_yolo.py" "val" --num_classes 3 --conf_thresh $CONF_THRESH --iou_thresh $IOU_THRESH
python3 "$SCRIPT_DIR/pred_assignment_yolo.py" "test" --num_classes 3 --conf_thresh $CONF_THRESH --iou_thresh $IOU_THRESH


