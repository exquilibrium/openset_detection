import os
import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path

def compute_mean_std(image_dir):
    image_dir = Path(image_dir)
    image_paths = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))

    if not image_paths:
        print("No images found in directory.")
        return

    means = []
    stds = []

    for img_path in tqdm(image_paths, desc="Processing images"):
        img = cv2.imread(str(img_path))  # BGR format
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        img = img.astype(np.float32) / 255.0  # Scale to [0, 1]

        means.append(np.mean(img, axis=(0, 1)))
        stds.append(np.std(img, axis=(0, 1)))

    mean = np.mean(means, axis=0) * 255  # Rescale to [0, 255]
    std = np.mean(stds, axis=0) * 255

    print("Mean (RGB):", mean.tolist())
    print("Std  (RGB):", std.tolist())

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute mean and std for RGB images.")
    parser.add_argument("image_dir", help="Directory containing images (JPG/PNG).")
    args = parser.parse_args()

    compute_mean_std(args.image_dir)


# python compute_mean_std.py "/media/chen/76AECF8EAECF4579/data/lru1_all_labels/VOCdevkit/VOC2007s/JPEGImages"
# python compute_mean_std.py "/media/chen/76AECF8EAECF4579/data/ardea10_run04_labels/VOCdevkit/VOC2007s/JPEGImages"
