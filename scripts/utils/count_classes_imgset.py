import os
import argparse
import xml.etree.ElementTree as ET
from collections import Counter
from tqdm import tqdm


def parse_voc_annotations_from_imageset(imageset_file):
    # Infer VOC root and annotation path
    imageset_file = os.path.abspath(imageset_file)
    dataset_root = os.path.abspath(os.path.join(imageset_file, "../../.."))
    annotations_dir = os.path.join(dataset_root, "Annotations")

    # Load image names (without extensions) from imageset file
    with open(imageset_file, 'r') as f:
        image_ids = [line.strip().split('/')[-1].replace('.jpg', '').replace('.png', '') for line in f if line.strip()]
    
    unique_classes = []
    single_class_counter = Counter()
    combo_class_counter = Counter()
    total = 0
    missing = 0

    for img_id in tqdm(image_ids, desc="Parsing annotations"):
        xml_file = os.path.join(annotations_dir, img_id + ".xml")
        if not os.path.exists(xml_file):
            missing += 1
            continue

        tree = ET.parse(xml_file)
        root = tree.getroot()

        objects = root.findall("object")
        if not objects:
            continue
        total += 1

        class_names = sorted({obj.find("name").text.strip() for obj in objects})
        for name in class_names:
            if name not in unique_classes:
                unique_classes.append(name)
        
        if len(class_names) == 1:
            single_class_counter[class_names[0]] += 1
        else:
            combo_key = "_".join(class_names)
            combo_class_counter[combo_key] += 1

    print(f'Total valid annotated files: {total}')
    print(f'Missing XML files: {missing}')
    return unique_classes, single_class_counter, combo_class_counter


def main():
    parser = argparse.ArgumentParser(description="Count class distributions in VOC annotations (from imageset file).")
    parser.add_argument("imageset_file", help="Path to imageset .txt file (e.g., train.txt).")
    parser.add_argument("--list", action="store_true", help="List all single and combo classes.")
    args = parser.parse_args()

    class_names, single_counts, combo_counts = parse_voc_annotations_from_imageset(args.imageset_file)

    print(f"\nFound {len(class_names)} single classes: {sorted(class_names)}")
    print(f"Found {len(combo_counts)} combo classes")

    if args.list:
        print("\nCounts for single classes:")
        for clsn in sorted(single_counts):
            print(f"{clsn}: {single_counts[clsn]}")
        print("\nCounts for combo classes:")
        for combo in sorted(combo_counts):
            print(f"{combo}: {combo_counts[combo]}")


if __name__ == "__main__":
    main()

# python count_classes_imgset.py "/media/chen/76AECF8EAECF4579/data/lru1_all/ImageSets/Main_CS_lander/train.txt"