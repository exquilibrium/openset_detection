#!/bin/bash

# === SET PATH TO DATASET DIRECTORY!!! ===
# DATASET_DIR="/volume/hot_storage/slurm_data/chen_le/ARCHES/lru1_all"
DATASET_DIR="/media/chen/76AECF8EAECF4579/data/lru1_all"
# === SET PATH TO DATASET DIRECTORY!!! ===

# Path to script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Create VOC dataset
python "$SCRIPT_DIR/create_voc_dataset.py" "$DATASET_DIR" --train_split 0.6 --val_split 0.2 --test_split 0.2

# Create balanced subset
#python "$SCRIPT_DIR/create_balanced_voc_subset.py" "$DATASET_DIR" --adjust no --train_split 0.6 --val_split 0.2 --test_split 0.2
#rm -rf "${DATASET_DIR}"
#mv "${DATASET_DIR}_small" "${DATASET_DIR}"

# Create closed-set VOC datasets for each class
python "$SCRIPT_DIR/create_voc_closedset.py" "$DATASET_DIR" "drone"
python "$SCRIPT_DIR/create_voc_closedset.py" "$DATASET_DIR" "lander"
python "$SCRIPT_DIR/create_voc_closedset.py" "$DATASET_DIR" "lru2"

# Create YOLO dataset splits
python "$SCRIPT_DIR/create_yolo_dataset.py" "$DATASET_DIR"

# Create co-heatmap
python "$SCRIPT_DIR/create_coheatmap.py" "$DATASET_DIR"

echo "Data preprocessing complete!"
