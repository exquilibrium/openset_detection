#!/bin/bash

# Adjust base_dirs.py
# PROJECT_ROOT="/home/chen_le/openset_detection/scripts/FRCNN"
PROJECT_ROOT="/home/chen/openset_detection/scripts/FRCNN"

# Run training script
#bash "$PROJECT_ROOT/training/train_gmmDet_voc0712_xml.sh"

# Run testing script
bash "$PROJECT_ROOT/extraction/test_voc0712_xml.sh"
