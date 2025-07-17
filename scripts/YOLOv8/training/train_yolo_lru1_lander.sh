#!/bin/sh

cd ../YOLOv8/training

BASE_DATA_FOLDER=$(python3 -c "import sys; sys.path.insert(0, '../../..'); import base_dirs; print(base_dirs.BASE_DATA_FOLDER)")

python remap_yolo_labels.py "${BASE_DATA_FOLDER}/lru1_all/JPEGImages" 0,1,2 0,2,1

yolo detect train \
  model=yolov8n.pt \
  data="${BASE_DATA_FOLDER}/lru1_all/data_CS_lander.yaml" \
  epochs=50 \
  imgsz=640 \
  amp=False \
  project="runs/detect" \
  name="train_lru1_lander"

python remap_yolo_labels.py "${BASE_DATA_FOLDER}/lru1_all/JPEGImages" 0,1,2 2,0,1 --reverse

