#!/bin/bash

# Path to scripts/FRCNN folder
PROJECT_ROOT="/home/chen/openset_detection/scripts/FRCNN"

# Adjust pretrained in _base_/models
# Run training script
#bash "$PROJECT_ROOT/training/train_frcnn_lru1.sh"
#bash "$PROJECT_ROOT/training/train_frcnn_lru1_drone.sh"
#bash "$PROJECT_ROOT/training/train_frcnn_lru1_lander.sh"
#bash "$PROJECT_ROOT/training/train_frcnn_lru1_lru2.sh"

# Run testing script
bash "$PROJECT_ROOT/extraction/test_frcnn_lru1.sh"
bash "$PROJECT_ROOT/extraction/test_frcnn_lru1_drone.sh"
bash "$PROJECT_ROOT/extraction/test_frcnn_lru1_lander.sh"
bash "$PROJECT_ROOT/extraction/test_frcnn_lru1_lru2.sh"
