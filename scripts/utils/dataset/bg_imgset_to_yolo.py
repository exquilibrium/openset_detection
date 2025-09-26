from pathlib import Path

# Path to your original txt file
file_path = Path("/volume/hot_storage/slurm_data/chen_le/ARCHES/ardea10_all/ImageSets/Main/bg.txt")

# Base directory where the images are stored
img_dir = "/volume/hot_storage/slurm_data/chen_le/ARCHES/ardea10_all/JPEGImages"

# Construct output file in the same directory, e.g. bg_yolo_yolo.txt
out_path = file_path.with_name(file_path.stem + "_yolo.txt")

with open(file_path, "r") as f:
    lines = f.readlines()

with open(out_path, "w") as f:
    for line in lines:
        name = line.strip()
        if not name.endswith(".jpg"):
            name += ".jpg"
        full_path = f"{img_dir}/{name}"
        f.write(full_path + "\n")

print(f"Saved converted file to {out_path}")
