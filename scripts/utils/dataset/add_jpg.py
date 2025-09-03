file_path = "/volume/hot_storage/slurm_data/chen_le/ARCHES/lru1_all/ImageSets/YOLO/bg_yolo.txt"

# Read all lines
with open(file_path, "r") as f:
    lines = f.readlines()

# Add .jpg if missing
with open(file_path, "w") as f:
    for line in lines:
        line = line.strip()
        if not line.endswith(".jpg"):
            line += ".jpg"
        f.write(line + "\n")
