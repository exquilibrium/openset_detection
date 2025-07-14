#!/bin/bash

# Converts Pascal VOC2007 and VOC2012 into
# small balanced XMLDataset VOC0712_small for debugging
# How to use: Set DATASET_DIR to folder path containing VOCdevkit with VOC2007, VOC2012

# === SET PATH TO DATASET DIRECTORY!!! ===
DATASET_DIR="/media/chen/76AECF8EAECF4579/data"
# === SET PATH TO DATASET DIRECTORY!!! ===

# Path to script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Combine VOC2007 and VOC2012 to VOC0712 XMLDataset
python "$SCRIPT_DIR/create_voc0712.py" $DATASET_DIR

# Create smaller balanced dataset VOC0712
python "$SCRIPT_DIR/create_balanced_voc_subset.py" "${DATASET_DIR}/VOCdevkit_xml/VOC0712" --adjust int3 --train_split 0.6 --val_split 0.2 --test_split 0.2

# Create closed-set VOC dataset
ython3 "$SCRIPT_DIR/create_voc_closedset.py" "${DATASET_DIR}/VOCdevkit_xml/VOC0712_small" "pottedplant,sheep,sofa,train,tvmonitor"

# Renaming
rm -rf "${DATASET_DIR}/VOCdevkit_xml/VOC0712"
mv "${DATASET_DIR}/VOCdevkit_xml/VOC0712_small" "${DATASET_DIR}/VOCdevkit_xml/VOC0712"
mv "${DATASET_DIR}/VOCdevkit_xml/VOC0712/data_CS_pottedplant,sheep,sofa,train,tvmonitor.yaml" \
   "${DATASET_DIR}/VOCdevkit_xml/VOC0712/data_CS.yaml"
mv "${DATASET_DIR}/VOCdevkit_xml/VOC0712/ImageSets/Main_CS_pottedplant,sheep,sofa,train,tvmonitor" \
   "${DATASET_DIR}/VOCdevkit_xml/VOC0712/ImageSets/Main_CS"

# Create yolo dataset
python3 "$SCRIPT_DIR/create_yolo_dataset.py" "${DATASET_DIR}/VOCdevkit_xml/VOC0712"

# Create OOD set for Mahalanobis
python create_oodset.py "${DATASET_DIR}VOCdevkit_xml/VOC0712/" "pottedplant,sheep,sofa,train,tvmonitor" --shortname

# Utils
python3 "$SCRIPT_DIR/create_coheatmap.py" "${DATASET_DIR}/VOCdevkit_xml/VOC0712"

echo "Data preprocessing complete!"
