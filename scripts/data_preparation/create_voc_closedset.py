import shutil
import argparse
from pathlib import Path
import xml.etree.ElementTree as ET
from tqdm import tqdm
import random

def create_closedset(dataset_dir: str, class_names: str) -> None:
    """
    Creates a closed-set dataset by creating closed set train/val/test txts.

    Args:
        dataset_dir (str): Root directory containing VOC data.
        class_names (str): Class names to exclude (e.g., "drone,lander").
    """
    # Source paths
    dataset_dir = Path(dataset_dir)
    src_images = dataset_dir / "JPEGImages"
    src_annotations = dataset_dir /  "Annotations"
    src_splits = dataset_dir /  "ImageSets" / "Main"

    # Destination paths
    closed_name = f"CS_{class_names}"
    dst_splits = dataset_dir / "ImageSets" / f"Main_{closed_name}"
    dst_splits.mkdir(parents=True, exist_ok=True)

    # Filter annotations
    valid_image_ids = []

    xml_files = list(src_annotations.glob("*.xml"))
    for xml_file in tqdm(xml_files, desc="Filtering images", unit="file"):
        tree = ET.parse(xml_file)
        root = tree.getroot()

        has_class = any((obj.find("name").text) in class_names for obj in root.iter("object"))
        if not has_class:
            image_file = src_images / f"{xml_file.stem}.jpg"
            if image_file.exists():
                valid_image_ids.append(xml_file.stem)

    # Filter and write split txts
    split_names = ["train", "val", "test"]
    valid_set = set(valid_image_ids)

    for split_name in split_names:
        src_split_file = src_splits / f"{split_name}.txt"
        dst_split_file = dst_splits / f"{split_name}.txt"

        if not src_split_file.exists():
            print(f"File {src_split_file} does not exist!")
            continue

        with open(src_split_file, "r") as src_f:
            lines = [line.strip() for line in src_f if line.strip() in valid_set]
        with open(dst_split_file, "w") as dst_f:
            dst_f.write("\n".join(lines) + "\n")

        print(f"Wrote {len(lines)} entries to {dst_split_file}")

def main():
    parser = argparse.ArgumentParser(description='Test with Distance')
    parser.add_argument('dataset_dir', type=str, help='Directory that contains VOCdevkit')
    parser.add_argument('class_names', type=str, help='Comma separated class names to be Known Unknown Class')
    args = parser.parse_args()

    print('Converting VOC to a closed-set form.')
    print(f'Open-Set classes: {args.class_names.split(",")}')
    create_closedset(args.dataset_dir, args.class_names)
    print("Completed creating VOC-CS ImageSets.\n")


if __name__ == "__main__":
    main()


