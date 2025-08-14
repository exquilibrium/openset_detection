#!/bin/bash

# Path to scripts/FRCNN folder
PROJECT_ROOT="/home/chen_le/openset_detection/scripts/YOLOv8"

# Run training script
bash "$PROJECT_ROOT/training/train_yolo_ardea10.sh"
bash "$PROJECT_ROOT/training/train_yolo_ardea10_lander.sh"
bash "$PROJECT_ROOT/training/train_yolo_ardea10_lru1.sh"
bash "$PROJECT_ROOT/training/train_yolo_ardea10_lru2.sh"

# Run testing script
bash "$PROJECT_ROOT/extraction/test_yolo_ardea10.sh"
bash "$PROJECT_ROOT/extraction/test_yolo_ardea10_lander.sh"
bash "$PROJECT_ROOT/extraction/test_yolo_ardea10_lru1.sh"
bash "$PROJECT_ROOT/extraction/test_yolo_ardea10_lru2.sh"
