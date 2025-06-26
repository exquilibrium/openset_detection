import os
from collections import Counter

for split in ['train.txt', 'val.txt', 'test.txt']:
    base_path = "/media/chen/76AECF8EAECF4579/data/VOCdevkit_xml/VOC0712"
    imageset_path = f"{base_path}/ImageSets/YOLO_CS/{split}"
    labels_dir = f"{base_path}/JPEGImages"  # adjust if needed

    class_counts = Counter()

    with open(imageset_path, "r") as f:
        for line in f:
            image_path = line.strip()
            basename = os.path.splitext(os.path.basename(image_path))[0]
            label_path = os.path.join(labels_dir, f"{basename}.txt")

            if not os.path.isfile(label_path):
                print(f"Warning: Label not found for {image_path}")
                continue

            with open(label_path, "r", encoding="utf-8", errors="ignore") as label_file:
                for annotation in label_file:
                    parts = annotation.strip().split()
                    if not parts:
                        continue
                    try:
                        class_id = int(parts[0])
                        class_counts[class_id] += 1
                    except ValueError:
                        print(f"Invalid label format in {label_path}")

    print(f"\nResults for {split}:")
    for class_id in sorted(class_counts):
        print(f"Class {class_id}: {class_counts[class_id]} occurrences")

