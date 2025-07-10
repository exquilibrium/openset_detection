#!/bin/sh

cd ../YOLOv8/training

# Sometimes slurm just fails, try running a few times
nvidia-smi

python remap_yolo_labels.py "/media/chen/76AECF8EAECF4579/data/lru1_all/JPEGImages" 0,1,2 0,2,1
#data="/volume/hot_storage/slurm_data/chen_le/ARCHES/lru1_all/data_CS_lander.yaml" \
yolo detect train \
  model=yolov8n.pt \
  data="/media/chen/76AECF8EAECF4579/data/lru1_all/data_CS_lander.yaml" \
  epochs=50 \
  imgsz=640 \
  amp=False \
  project="runs/detect" \
  name="train_lru1_lander"
python remap_yolo_labels.py "/media/chen/76AECF8EAECF4579/data/lru1_all/JPEGImages" 0,1,2 2,0,1 --reverse

