import os
import argparse
import xml.etree.ElementTree as ET
from tqdm import tqdm

def count_files_without_objects(annotations_dir):
    xml_files = [f for f in os.listdir(annotations_dir) if f.endswith(".xml")]
    no_object_count = 0

    for filename in tqdm(xml_files, desc="Checking annotations"):
        filepath = os.path.join(annotations_dir, filename)
        tree = ET.parse(filepath)
        root = tree.getroot()

        if not root.findall("object"):
            no_object_count += 1

    return no_object_count, len(xml_files)

def main():
    parser = argparse.ArgumentParser(description="Count annotation files without any objects.")
    parser.add_argument("annotations_dir", help="Path to VOC annotations directory.")
    args = parser.parse_args()

    no_obj_count, total = count_files_without_objects(args.annotations_dir)

    print(f"\nTotal annotation files: {total}")
    print(f"Files without any <object>: {no_obj_count}")
    print(f"Percentage: {no_obj_count / total * 100:.2f}%")

if __name__ == "__main__":
    main()

# python count_unannotated.py "/media/chen/76AECF8EAECF4579/data/ardea10_run04_labels/VOCdevkit/VOC2007/Annotations"
# python count_unannotated.py "/media/chen/76AECF8EAECF4579/data/ardea10_run04_labels/VOCdevkit/VOC2007CS_lru2/Annotations"