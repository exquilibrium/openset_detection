#!/bin/bash

# Adjust base_dirs.py
#PROJECT_ROOT="/home/chen/openset_detection/scripts/YOLOv8"
PROJECT_ROOT="/home/chen_le/openset_detection/scripts/YOLOv8"

# Run training script
bash "$PROJECT_ROOT/training/train_yolo_lru1.sh"
#bash "$PROJECT_ROOT/training/train_yolo_lru1_drone.sh"
#bash "$PROJECT_ROOT/training/train_yolo_lru1_lander.sh"
#bash "$PROJECT_ROOT/training/train_yolo_lru1_lru2.sh"

# Run testing script
#bash "$PROJECT_ROOT/extraction/test_yolo_lru1.sh"
#bash "$PROJECT_ROOT/extraction/test_yolo_lru1_drone.sh"
#bash "$PROJECT_ROOT/extraction/test_yolo_lru1_lander.sh"
#bash "$PROJECT_ROOT/extraction/test_yolo_lru1_lru2.sh"
