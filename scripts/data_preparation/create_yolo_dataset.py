import os
import random
import shutil
import argparse
from pathlib import Path
import xml.etree.ElementTree as ET
from typing import Tuple, List
from tqdm import tqdm


# Convert dataset from PASCAL VOC XML format to YOLO format.
# https://github.com/OliLov/python-projects/blob/main/data_extraction/convert_voc_to_yolo.py
def convert_bbox_to_yolo(size: Tuple[int, int], box: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    """
    Convert bounding box coordinates from PASCAL VOC format to YOLO format.

    Args:
        size (Tuple[int, int]): Image size as (width, height).
        box (Tuple[float, float, float, float]): Bounding box in VOC format (xmin, ymin, xmax, ymax).

    Returns:
        Tuple[float, float, float, float]: Bounding box in YOLO format (x_center, y_center, width, height), normalized to [0, 1].
    """
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]

    rel_x_center = (box[0] + box[2]) / 2.0 * dw
    rel_y_center = (box[1] + box[3]) / 2.0 * dh
    rel_width = (box[2] - box[0]) * dw
    rel_height = (box[3] - box[1]) * dh

    return (rel_x_center, rel_y_center, rel_width, rel_height)

def voc_to_yolo(input_file: Path, output_txt: Path, classes: List[str]):
    """
    Convert a single PASCAL VOC XML annotation file to YOLO format.

    Args:
        input_file (Path): Path to the VOC XML file.
        output_txt (Path): Path where the corresponding YOLO .txt annotation will be saved.

    Returns:
        List[str]: Updated list of class names including any new classes found in the input file.
    """
    try:
        tree = ET.parse(input_file)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"Error parsing {input_file}: {e}")
        return  # Skip this file and continue with the next

    size_element = root.find("size")
    image_width = int(size_element.find("width").text)
    image_height = int(size_element.find("height").text)

    with output_txt.open("w") as file:
        for obj in root.iter("object"):
            #if int(obj.find("difficult").text) == 1:
            #    print(f"Difficult skipped")
            #    continue
                
            class_name = obj.find("name").text
            class_id = classes.index(class_name)
            
            xml_box = obj.find("bndbox")
            bbox = (
                float(xml_box.find("xmin").text),
                float(xml_box.find("ymin").text),
                float(xml_box.find("xmax").text),
                float(xml_box.find("ymax").text),
            )
            yolo_bbox = convert_bbox_to_yolo((image_width, image_height), bbox)
            file.write(f"{class_id} {' '.join(map(str, yolo_bbox))}\n")

    return classes
    
def create_yolo_labels(dataset_dir: Path):
    """
    Convert Pascal VOC annotations to YOLO format and generate classes.txt and data.yaml.

    Args:
        dataset_dir (Path): Directory containing the 'Annotations' folder with VOC XML files.
    """
    input_dir = dataset_dir / "Annotations"
    output_dir = dataset_dir / "JPEGImages" # Yolo looks up image/label path from train/val/test txt
    output_dir.mkdir(parents=True, exist_ok=True)

    classes = set()
    xml_files = list(input_dir.glob("*.xml"))

    for xml_file in tqdm(xml_files, desc="Collecting classes", unit="file"):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for obj in root.findall("object"):
            class_name = obj.find("name").text.strip()
            classes.add(class_name)
    unique_sorted_classes= sorted(classes)

    for xml_file in tqdm(xml_files, desc="Converting to YOLO", unit="file"):
        output_txt_path = output_dir / xml_file.with_suffix(".txt").name
        voc_to_yolo(xml_file, output_txt_path, unique_sorted_classes)

    return unique_sorted_classes

def create_yolo_imagesets(dataset_dir: Path, classes: List):
    """
    Convert VOC-style train/val/test splits to YOLO-compatible format and save data.yaml files.

    Args:
        dataset_dir (Path): Path to dataset root containing JPEGImages and ImageSets/Main* folders.
        classes (List[str]): List of class names for YOLO (order matters).
    """
    image_dir = dataset_dir / "JPEGImages"
    splits_dir = dataset_dir / "ImageSets"
    split_names = ["train", "val", "test"]

    # Find folders like Main, Main_lru2, etc.
    imageset_dirs = [d for d in splits_dir.iterdir() if d.is_dir() and d.name.startswith("Main")]

    for imgset_dir in imageset_dirs:
        suffix = imgset_dir.name[len("Main"):]  # e.g., '', '_CS_lru2', '_CS_drone'
        yolo_dir = splits_dir / f"YOLO{suffix}"
        yolo_dir.mkdir(parents=True, exist_ok=True)

        yaml_data = {}

        for split_name in split_names:
            voc_split_file = imgset_dir / f"{split_name}.txt"
            yolo_split_file = yolo_dir / f"{split_name}.txt"

            if voc_split_file.exists():
                with open(voc_split_file, "r") as f:
                    image_ids = [line.strip() for line in f if line.strip()]

                with open(yolo_split_file, "w") as f:
                    for img_id in image_ids:
                        img_path = image_dir / f"{img_id}.jpg"
                        f.write(str(img_path.resolve()) + "\n")
                print(f"Generated: {yolo_split_file}")

                yaml_data[split_name] = str(yolo_split_file.resolve())
            else:
                print(f"Split file not found: {voc_split_file}")

        # Write data{suffix}.yaml in dataset_dir
        yaml_path = dataset_dir / f"data{suffix}.yaml"
        id_classes = classes
        if len(suffix) > 0 and len(suffix[4:]) > 0:
            id_classes = [c for c in classes if c not in suffix[4:].split(',')]
        print(f'Updated classes: {id_classes}')
        yaml_path.write_text(
            f"train: {yolo_dir / 'train.txt'}\n"
            f"val: {yolo_dir / 'val.txt'}\n\n"
            f"nc: {len(id_classes)}\n"
            f"names: {id_classes}\n"
        )
        print(f"Saved YOLO data config: {yaml_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split YOLO dataset into train, val, test sets.")
    parser.add_argument("dataset_dir", type=Path, help="Path to dataset folder")
    args = parser.parse_args()

    print("Converting VOC ImageSets to YOLOv8 splits.")
    classes = create_yolo_labels(args.dataset_dir)
    create_yolo_imagesets(args.dataset_dir, classes)
    print("YOLO split conversion complete.\n")
