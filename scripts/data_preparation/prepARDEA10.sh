#!/bin/bash

# === SET PATH TO DATASET DIRECTORY!!! ===\
#SOURCE_DIR=/home_local/chen_le/ARCHES
SOURCE_DIR=/media/chen/76AECF8EAECF4579/ARCHES
BASE_DATA_FOLDER=$(python3 -c "import sys; sys.path.insert(0, '../..'); import base_dirs; print(base_dirs.BASE_DATA_FOLDER)")
DATASET_DIR="${BASE_DATA_FOLDER}/ardea10_all"
# === SET PATH TO DATASET DIRECTORY!!! ===

# Path to script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Import data
python "$SCRIPT_DIR/import_data.py" "${SOURCE_DIR}/ardea10_run01_labels" "${DATASET_DIR}"
python "$SCRIPT_DIR/import_data.py" "${SOURCE_DIR}/ardea10_run02_labels" "${DATASET_DIR}"
python "$SCRIPT_DIR/import_data.py" "${SOURCE_DIR}/ardea10_run04_labels" "${DATASET_DIR}"
python "$SCRIPT_DIR/import_data.py" "${SOURCE_DIR}/ardea10_run05_labels" "${DATASET_DIR}"

# Create VOC dataset
python "$SCRIPT_DIR/create_voc_dataset.py" "$DATASET_DIR" --train_split 0.6 --val_split 0.2 --test_split 0.2

# Create balanced subset
#python "$SCRIPT_DIR/create_balanced_voc_subset.py" "$DATASET_DIR" --adjust no --train_split 0.6 --val_split 0.2 --test_split 0.2
#rm -rf "${DATASET_DIR}"
#mv "${DATASET_DIR}_small" "${DATASET_DIR}"

# Create closed-set VOC datasets for each class
python "$SCRIPT_DIR/create_voc_closedset.py" "$DATASET_DIR" "lander"
python "$SCRIPT_DIR/create_voc_closedset.py" "$DATASET_DIR" "lru1"
python "$SCRIPT_DIR/create_voc_closedset.py" "$DATASET_DIR" "lru2"

# Create YOLO dataset splits
python "$SCRIPT_DIR/create_yolo_dataset.py" "$DATASET_DIR"

# Create OOD set for Mahalanobis
python create_oodset.py "$DATASET_DIR" "lander"
python create_oodset.py "$DATASET_DIR" "lru1"
python create_oodset.py "$DATASET_DIR" "lru2"

# Create co-heatmap
python "$SCRIPT_DIR/create_coheatmap.py" "$DATASET_DIR"

echo "Data preprocessing complete!"
