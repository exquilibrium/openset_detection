import os
import shutil
import argparse
import cv2
import xml.etree.ElementTree as ET
from pathlib import Path
from tqdm import tqdm
import random

def create_empty_voc_xml(image_path, annotation_path, folder):
    """
    Creates a PASCAL VOC-compatible XML annotation file with no objects.

    Args:
        image_path (Path): Path to the image file.
        annotation_path (Path): Path where the XML file will be saved.
        folder (str): Folder name to include in the annotation metadata.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Warning: Could not read image {image_path}")
        return

    height, width, depth = img.shape

    annotation = ET.Element("annotation")
    ET.SubElement(annotation, "folder").text = folder
    ET.SubElement(annotation, "filename").text = image_path.name
    ET.SubElement(annotation, "path").text = str(image_path)

    source = ET.SubElement(annotation, "source")
    ET.SubElement(source, "database").text = "Unknown"

    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = str(depth)

    ET.SubElement(annotation, "segmented").text = "0"

    tree = ET.ElementTree(annotation)
    tree.write(annotation_path, encoding="utf-8", xml_declaration=True)

def add_missing_annotations(dataset_dir: str):
    """
    Scans a directory for images without corresponding VOC XML annotations 
    and generates empty annotations for them.

    Args:
        dataset_dir (str or Path): Directory containing images and annotations.
    """
    dataset_dir = Path(dataset_dir)
    image_paths = sorted(dataset_dir.glob("*.jpg"))
    count = 0

    for img_path in tqdm(image_paths, desc="Processing images"):
        annotation_path = dataset_dir / f"{img_path.stem}.xml"
        if not annotation_path.exists():
            create_empty_voc_xml(img_path, annotation_path, dataset_dir.name)
            count += 1

    print(f'Created {count} annotations!')

def organize_files(dataset_dir: str):
    """
    Moves all .jpg and .xml files in a directory into the PASCAL VOC structure.

    Args:
        dataset_dir (str): Directory containing flat image and annotation files.
                        After execution, JPEG images go to 'JPEGImages/' and 
                        annotations to 'Annotations/'.
    """
    dataset_dir = Path(dataset_dir)
    images_folder = os.path.join(dataset_dir, "JPEGImages")
    labels_folder = os.path.join(dataset_dir, "Annotations")

    os.makedirs(images_folder, exist_ok=True)
    os.makedirs(labels_folder, exist_ok=True)

    files = os.listdir(dataset_dir)
    for file in tqdm(files, desc="Organizing files"):
        file_path = os.path.join(dataset_dir, file)
        if os.path.isfile(file_path):
            if file.lower().endswith(".jpg"):
                shutil.move(file_path, os.path.join(images_folder, file))
            elif file.lower().endswith(".xml"):
                shutil.move(file_path, os.path.join(labels_folder, file))

    print("All images and XML files have been moved.")

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
    parser = argparse.ArgumentParser(description="Converts raw ARCHES data into PASCAL-VOC style dataset.")
    parser.add_argument("dataset_dir", type=str, help="Directory with images and existing VOC annotations.")
    parser.add_argument("--train_split", type=float, default=0.7, help="Ratio of training data (default: 0.7)")
    parser.add_argument("--val_split", type=float, default=0.2, help="Ratio of validation data (default: 0.2)")
    parser.add_argument("--test_split", type=float, default=0.1, help="Ratio of test data (default: 0.1)")
    args = parser.parse_args()

    print('Creating VOC dataset structure.')
    add_missing_annotations(args.dataset_dir)
    organize_files(args.dataset_dir)
    generate_voc_imagesets(args.dataset_dir, args.train_split, args.val_split, args.test_split)
    print("Finished creating VOC dataset.\n")

if __name__ == "__main__":
    main()
