#!/bin/bash

# Path to scripts/FRCNN folder
PROJECT_ROOT="$(realpath "$(dirname "${BASH_SOURCE[0]}")/../FRCNN")"

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
