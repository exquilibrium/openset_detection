import os
import argparse
import xml.etree.ElementTree as ET
from collections import Counter
from tqdm import tqdm  # Add tqdm for the progress bar

def parse_voc_annotations(annotations_dir):
    unique_classes = []
    single_class_counter = Counter()
    combo_class_counter = Counter()

    xml_files = [f for f in os.listdir(annotations_dir) if f.endswith(".xml")]

    for filename in tqdm(xml_files, desc="Parsing annotations"):
        filepath = os.path.join(annotations_dir, filename)
        tree = ET.parse(filepath)
        root = tree.getroot()

        objects = root.findall("object")
        if not objects:
            continue

        class_names = sorted({obj.find("name").text.strip() for obj in objects})
        for name in class_names:
            if name not in unique_classes:
                unique_classes.append(name)
        
        if len(class_names) == 1:
            single_class_counter[class_names[0]] += 1
        else:
            combo_key = "_".join(class_names)
            combo_class_counter[combo_key] += 1

    return unique_classes, single_class_counter, combo_class_counter

def main():
    parser = argparse.ArgumentParser(description="Count single and combo class images in VOC annotations.")
    parser.add_argument("annotations_dir", help="Path to source directory VOC2007.")
    parser.add_argument("--list", action="store_true", help="List all single and combo classes")
    args = parser.parse_args()

    class_names, single_counts, combo_counts = parse_voc_annotations(args.annotations_dir)

    print(f"\nFound {len(class_names)} single classes: {sorted(class_names)}")
    print(f"Found {len(combo_counts)} combo classes")

    if args.list:
        print("\n Counts for single classes:")
        for clsn in sorted(single_counts):
            print(f"{clsn}: {single_counts[clsn]}")
        print("\n Counts for combo classes:")
        for combo in sorted(combo_counts):
            print(f"{combo}: {combo_counts[combo]}")

if __name__ == "__main__":
    main()


# python count_classes.py "/media/chen/76AECF8EAECF4579/data" --list
# python count_classes.py "/media/chen/76AECF8EAECF4579/data/VOCdevkit_xml/VOC0712/Annotations"
# python count_classes.py "/volume/hot_storage/slurm_data/chen_le/ARCHES/lru1_all/Annotations"


