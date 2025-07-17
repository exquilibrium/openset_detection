#!/bin/sh

cd ../YOLOv8/training

BASE_DATA_FOLDER=$(python3 -c "import sys; sys.path.insert(0, '../../..'); import base_dirs; print(base_dirs.BASE_DATA_FOLDER)")

yolo detect train \
  model=yolov8n.pt \
  data="${BASE_DATA_FOLDER}/VOCdevkit_xml/VOC0712/data_CS.yaml" \
  epochs=50 \
  imgsz=640 \
  amp=False \
  project="runs/detect" \
  name="train_voc0712"