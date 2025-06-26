#!/bin/sh

# Sometimes slurm just fails, try running a few times

nvidia-smi

yolo detect train model=yolov8n.pt data="/volume/hot_storage/slurm_data/chen_le/ARCHES/lru1_all/data.yaml" epochs=50 imgsz=640 amp=False
yolo detect train model=yolov8n.pt data="/volume/hot_storage/slurm_data/chen_le/ARCHES/lru1_all/data_CS_drone.yaml" epochs=50 imgsz=640 amp=False
yolo detect train model=yolov8n.pt data="/volume/hot_storage/slurm_data/chen_le/ARCHES/lru1_all/data_CS_lander.yaml" epochs=50 imgsz=640 amp=False
yolo detect train model=yolov8n.pt data="/volume/hot_storage/slurm_data/chen_le/ARCHES/lru1_all/data_CS_lru2.yaml" epochs=50 imgsz=640 amp=False
