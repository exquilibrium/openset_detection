import os
import argparse
import shutil
import random
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

def create_balanced_voc_subset(src_voc_path, adjust):
    src_voc = Path(src_voc_path)
    dst_voc = Path(src_voc_path+"_small")

    src_img_dir = src_voc / "JPEGImages"
    src_ann_dir = src_voc / "Annotations"
    dst_img_dir = dst_voc / "JPEGImages"
    dst_ann_dir = dst_voc / "Annotations"

    # Create destination directories
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_ann_dir.mkdir(parents=True, exist_ok=True)

    class_to_files = defaultdict(list)

    # Collect all images with a progress bar
    for xml_file in tqdm(src_ann_dir.glob("*.xml"), desc="Processing annotations"):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        objects = root.findall("object")

        file_id = xml_file.stem
        if len(objects) == 0:
            class_to_files["background"].append(file_id)
        elif len(objects) == 1:
            class_name = objects[0].find("name").text
            class_to_files[class_name].append(file_id)
        else:
            class_names = sorted({obj.find("name").text for obj in objects})
            class_name = "_".join(class_names)
            class_to_files[class_name].append(file_id)

    # Min single class
    filtered_items = {k: v for k, v in class_to_files.items() if "_" not in k}
    minimum_class, minimum_files = min(filtered_items.items(), key=lambda item: len(item[1]))
    max_per_class = len(minimum_files)

    print(f"Class with lowest amount (no '_'): {minimum_class} ({max_per_class} images)")

    # Sample and copy files with tqdm progress bar
    total_selected = sum(min(max_per_class, len(files)) for files in class_to_files.values())
    with tqdm(total=total_selected, desc="Copying files") as pbar:
        for class_name, files in class_to_files.items():
            if "int" in adjust:
                num = adjust.replace("int", "")
                adjusted_max = max_per_class if class_name.count('_') == 0 else int(num)
            elif adjust == "lin":
                adjusted_max = max_per_class // (1 + class_name.count('_'))
            elif adjust == "exp":
                adjusted_max = max_per_class // (2**class_name.count('_'))
            else: # adjust == "no"
                adjusted_max = max_per_class

            selected = random.sample(files, min(adjusted_max, len(files)))
            for file_id in selected:
                shutil.copy(src_img_dir / f"{file_id}.jpg", dst_img_dir / f"{file_id}.jpg")
                shutil.copy(src_ann_dir / f"{file_id}.xml", dst_ann_dir / f"{file_id}.xml")
                pbar.update(1)

    print(f"Subset created at: {dst_voc}")

def generate_voc_imagesets(dataset_dir: str, train_split: float, val_split: float, test_split: float):
    """
    Generates 'train.txt', 'val.txt', 'test.txt' files for a VOC-style dataset
    by randomly splitting images in 'JPEGImages' using the provided ratios.

    Args:
        dataset_dir (str): Path to the dataset directory containing 'JPEGImages'.
        train_split (float): Proportion of images to allocate to the training set.
        val_split (float): Proportion of images to allocate to the validation set.
        test_split (float): Proportion of images to allocate to the test set. Should satisfy train + val + test = 1.0.

    Notes:
        - A fixed random seed is used for reproducibility.
        - All generated .txt files are written to 'ImageSets/Main'.
    """
    seed = 42
    dataset_dir = Path(dataset_dir)
    image_dir = dataset_dir / "JPEGImages"
    splits_dir = dataset_dir / "ImageSets" / "Main"
    splits_dir.mkdir(parents=True, exist_ok=True)

    # Get all image IDs (stems only)
    all_images = sorted([img.stem for img in image_dir.glob("*.jpg")])
    total = len(all_images)
    random.seed(seed)
    random.shuffle(all_images)

    n_train = int(total * train_split)
    n_val = int(total * val_split)
    n_test = total - n_train - n_val  # Remaining

    train_ids = all_images[:n_train]
    val_ids = all_images[n_train:n_train + n_val]
    test_ids = all_images[n_train + n_val:]

    # Write splits
    def write_split(name, ids):
        split_path = splits_dir / f"{name}.txt"
        print(f"Writing {len(ids)} entries to {name}.txt")
        with open(split_path, "w") as f:
            for img_id in ids:
                f.write(f"{img_id}\n")

    write_split("train", train_ids)
    write_split("val", val_ids)
    write_split("test", test_ids)

def main():
    """
    Create a smaller balanced VOC dataset from a given VOC dataset.
    Every image with 1 class annotation will appear at most as much as the class that has
    the least amount of images with 1 class annotations. Combinations will appear at most 
    as much as that as well.

    Example:
    ----- Before balance -----
    drone: 100
    lander: 150
    lru2: 50
    drone_lander: 32
    lander_lru2: 60

    ----- After balance -----
    drone: 50 (100)
    lander: 50 (150)
    lru2: 50
    drone_lander: 32
    lander_lru2: 50 (60)
    """
    parser = argparse.ArgumentParser(description="Create smaller balanced VOC dataset.")
    parser.add_argument("src_dir", help="Path to the source VOC directory.")
    parser.add_argument("--adjust", default="no", help="Type of adjustment for combo classes: no, int<int>, lin, exp")
    parser.add_argument("--train_split", type=float, default=0.6, help="Ratio of training data (default: 0.7)")
    parser.add_argument("--val_split", type=float, default=0.2, help="Ratio of validation data (default: 0.2)")
    parser.add_argument("--test_split", type=float, default=0.2, help="Ratio of test data (default: 0.1)")
    args = parser.parse_args()

    create_balanced_voc_subset(args.src_dir, args.adjust)
    generate_voc_imagesets(args.src_dir+"_small", args.train_split, args.val_split, args.test_split)

if __name__ == "__main__":
    main()

# python create_balanced_voc_subset.py "/media/chen/76AECF8EAECF4579/data/VOCdevkit_xml/VOC0712" --adjust int3 --train_split 0.6 --val_split 0.2 --test_split 0.2
