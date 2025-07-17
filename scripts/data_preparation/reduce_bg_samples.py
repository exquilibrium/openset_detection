import argparse
from pathlib import Path
import xml.etree.ElementTree as ET
from tqdm import tqdm
import random
import shutil

def parse_annotation(xml_path, class_names=None):
    if not xml_path.exists():
        return False
    tree = ET.parse(xml_path)
    root = tree.getroot()
    objects = root.findall("object")
    if not objects:
        return False
    if class_names is None:
        return True
    for obj in objects:
        name = obj.find("name").text.strip()
        if name in class_names:
            return True
    return False

def reduce_background_entries(train_txt_path, ratio=0.3, class_names=None):
    train_txt_path = Path(train_txt_path)
    dataset_root = train_txt_path.parents[2]  # VOC structure: VOC/ImageSets/Main/train.txt
    annotations_dir = dataset_root / "Annotations"

    with open(train_txt_path, "r") as f:
        image_paths = [line.strip() for line in f.readlines()]

    annotated = []
    background = []

    for img_path in tqdm(image_paths, desc="Scanning annotations"):
        stem = Path(img_path).stem  # handles full path or ID
        xml_file = annotations_dir / f"{stem}.xml"
        if parse_annotation(xml_file, class_names):
            annotated.append(img_path)
        else:
            background.append(img_path)

    max_bg = int(len(annotated) * ratio)
    reduced_background = random.sample(background, min(len(background), max_bg))

    reduced_train_list = sorted(annotated + reduced_background)

    # Backup original file
    backup_path = train_txt_path.with_name(train_txt_path.stem + "_old.txt")
    shutil.move(train_txt_path, backup_path)

    # Save new train.txt in original location
    with open(train_txt_path, "w") as f:
        f.write("\n".join(reduced_train_list) + "\n")

    print(f"\nOriginal annotated: {len(annotated)}")
    print(f"Original background-only: {len(background)}")
    print(f"Reduced background: {len(reduced_background)}")
    print(f"Backed up old train.txt to: {backup_path}")
    print(f"Saved reduced train.txt to: {train_txt_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reduce background-only entries in VOC train.txt")
    parser.add_argument("train_txt", help="Path to VOC-style train.txt")
    parser.add_argument("--ratio", type=float, default=0.3,
                        help="Max ratio of background-only images to annotated ones (default=0.3)")
    args = parser.parse_args()

    reduce_background_entries(args.train_txt, args.ratio)


# python reduce_voc_background.py /media/chen/76AECF8EAECF4579/data/lru1_all/ImageSets/Main_CS_lander/train.txt