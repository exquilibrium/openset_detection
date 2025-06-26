import os
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from tqdm import tqdm

def corrupt_voc_labels(voc_dir, corruption_ratio=0.2, class_list=None):
    ann_dir = Path(voc_dir) / "Annotations"
    xml_files = list(ann_dir.glob("*.xml"))

    if class_list is None:
        raise ValueError("You must provide a list of all class names via `class_list`.")

    # Select files to corrupt
    num_to_corrupt = int(len(xml_files) * corruption_ratio)
    files_to_corrupt = random.sample(xml_files, num_to_corrupt)

    print(f"Corrupting {num_to_corrupt} of {len(xml_files)} annotation files...")

    for xml_file in tqdm(files_to_corrupt, desc="Corrupting"):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        modified = False

        for obj in root.findall("object"):
            name_el = obj.find("name")
            current_class = name_el.text.strip()
            other_classes = [cls for cls in class_list if cls != current_class]

            if other_classes:
                name_el.text = random.choice(other_classes)
                modified = True

        if modified:
            tree.write(xml_file)

    print("Corruption complete.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Corrupt some ratio of VOC annotation labels.")
    parser.add_argument("voc_dir", help="Path to VOC2007 directory")
    parser.add_argument("--classes", nargs="+", required=True, help="List of all class names")
    parser.add_argument("--ratio", type=float, default=0.2, help="Fraction of labels to corrupt (default: 0.2)")
    args = parser.parse_args()

    corrupt_voc_labels(args.voc_dir, corruption_ratio=args.ratio, class_list=args.classes)

# python corrupt_labels.py '/media/chen/76AECF8EAECF4579/data/ardea10_run04_labels/VOCdevkit/VOC2007CS_lru2' --classes lru1 lander --ratio 0.2


