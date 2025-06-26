import os
import xml.etree.ElementTree as ET
from pathlib import Path
from tqdm import tqdm

def rename_class_in_voc_annotations(annotations_dir, old_class, new_class):
    annotations_dir = Path(annotations_dir)
    xml_files = list(annotations_dir.glob("*.xml"))

    renamed_count = 0

    for xml_file in tqdm(xml_files, desc=f"Renaming '{old_class}' to '{new_class}'"):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        modified = False

        for obj in root.findall("object"):
            name_el = obj.find("name")
            if name_el is not None and name_el.text.strip() == old_class:
                name_el.text = new_class
                modified = True
                renamed_count += 1

        if modified:
            tree.write(xml_file)

    print(f"Renamed {renamed_count} instances of '{old_class}' to '{new_class}'.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Rename class label in VOC annotation files.")
    parser.add_argument("annotations_dir", help="Path to VOC Annotations directory")
    parser.add_argument("--old", required=True, help="Old class name to replace (e.g., 'lru2')")
    parser.add_argument("--new", required=True, help="New class name (e.g., 'lru1')")
    args = parser.parse_args()

    rename_class_in_voc_annotations(args.annotations_dir, args.old, args.new)

# python rename_annotation.py '/media/chen/76AECF8EAECF4579/data/ardea10_run04_labels/VOCdevkit/VOC2007/Annotations' --old lru2 --new lru1