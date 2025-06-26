import os
import random
import xml.etree.ElementTree as ET
from collections import defaultdict
from tqdm import tqdm

random.seed(42)

def parse_annotations(annotation_dir, image_ids):
    class_to_images = defaultdict(set)
    image_to_classes = {}

    for image_id in tqdm(image_ids, desc="Parsing annotations"):
        xml_path = os.path.join(annotation_dir, image_id + ".xml")
        if not os.path.exists(xml_path):
            continue
        tree = ET.parse(xml_path)
        root = tree.getroot()
        classes = {obj.find('name').text for obj in root.findall('object')}
        image_to_classes[image_id] = classes
        for cls in classes:
            class_to_images[cls].add(image_id)

    return class_to_images, image_to_classes

def balanced_subset(image_ids, class_to_images, image_to_classes, target_size):
    selected = set()
    remaining = set(image_ids)

    while len(selected) < target_size and remaining:
        candidates = list(remaining)
        random.shuffle(candidates)
        for img in candidates:
            if len(selected) >= target_size:
                break
            selected.add(img)
            remaining.remove(img)
    return list(selected)

def reduce_split(split_file, annotation_dir, ratio=0.1):
    with open(split_file) as f:
        image_ids = [line.strip() for line in f.readlines()]

    target_size = max(1, int(len(image_ids) * ratio))
    class_to_images, image_to_classes = parse_annotations(annotation_dir, image_ids)
    selected_ids = balanced_subset(image_ids, class_to_images, image_to_classes, target_size)

    with open(split_file, "w") as f:
        for image_id in selected_ids:
            f.write(image_id + "\n")

    print(f"Overwritten {split_file} with {len(selected_ids)} entries.")
    return set(selected_ids)

def get_all_split_ids(image_sets_dir, splits):
    all_ids = set()
    for split in splits:
        split_file = os.path.join(image_sets_dir, f"{split}.txt")
        if os.path.exists(split_file):
            with open(split_file) as f:
                all_ids.update(line.strip() for line in f)
    return all_ids

def remove_unused_data(voc_root, used_ids):
    annotations_dir = os.path.join(voc_root, "Annotations")
    images_dir = os.path.join(voc_root, "JPEGImages")

    all_annotation_files = set(f[:-4] for f in os.listdir(annotations_dir) if f.endswith(".xml"))
    all_image_files = set(f[:-4] for f in os.listdir(images_dir) if f.endswith(".jpg"))

    unused_ids = (all_annotation_files | all_image_files) - used_ids

    for img_id in tqdm(unused_ids, desc="Removing unused files"):
        xml_path = os.path.join(annotations_dir, img_id + ".xml")
        img_path = os.path.join(images_dir, img_id + ".jpg")
        if os.path.exists(xml_path):
            os.remove(xml_path)
        if os.path.exists(img_path):
            os.remove(img_path)

def reduce_voc(voc_root):
    splits = ["train", "val", "test"]
    image_sets_dir = os.path.join(voc_root, "ImageSets", "Main")
    annotation_dir = os.path.join(voc_root, "Annotations")

    final_used_ids = set()
    for split in splits:
        split_file = os.path.join(image_sets_dir, f"{split}.txt")
        if not os.path.exists(split_file):
            continue
        print(f"\nProcessing {split_file}")
        used_ids = reduce_split(split_file, annotation_dir, ratio=0.1)
        final_used_ids.update(used_ids)

    remove_unused_data(voc_root, final_used_ids)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Reduce VOC dataset splits to and remove unused data.")
    parser.add_argument("voc_path", help="Path to VOC2007 or VOC2012 folder")
    args = parser.parse_args()

    reduce_voc(args.voc_path)
