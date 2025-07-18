#!/bin/bash

# Path to scripts/FRCNN folder
PROJECT_ROOT="/home/chen/openset_detection/scripts/YOLOv8"

# Run training script
#bash "$PROJECT_ROOT/training/train_yolo_voc0712_xml.sh"

# Run testing script
bash "$PROJECT_ROOT/extraction/test_yolo_voc0712_xml.sh"
