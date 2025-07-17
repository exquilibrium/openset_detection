#!/bin/sh

cd ../YOLOv8/training

# Sometimes slurm just fails, try running a few times
nvidia-smi

#python remap_yolo_labels.py "/media/chen/76AECF8EAECF4579/data/ardea10_all/JPEGImages" 0,1,2 2,0,1
python remap_yolo_labels.py "/volume/hot_storage/slurm_data/chen_le/ARCHES/ardea10_all/JPEGImages" 0,1,2 2,0,1

#data="/volume/hot_storage/slurm_data/chen_le/ARCHES/ardea10_all/data.yaml" \
#data="/media/chen/76AECF8EAECF4579/data/ardea10_all/data.yaml" \
yolo detect train \
  model=yolov8n.pt \
  data="/volume/hot_storage/slurm_data/chen_le/ARCHES/ardea10_all/data.yaml" \
  epochs=50 \
  imgsz=640 \
  amp=False \
  project="runs/detect" \
  name="train_ardea10_lander"

#python remap_yolo_labels.py "/media/chen/76AECF8EAECF4579/data/ardea10_all/JPEGImages" 0,1,2 2,0,1 --reverse
python remap_yolo_labels.py "/volume/hot_storage/slurm_data/chen_le/ARCHES/data/ardea10_all/JPEGImages" 0,1,2 2,0,1 --reverse
