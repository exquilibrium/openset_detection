import argparse
from pathlib import Path
import xml.etree.ElementTree as ET
from tqdm import tqdm


def create_pure_ood_set(dataset_dir: str, open_set_classes: str, shortname: bool) -> None:
    """
    Creates a pure OOD test set for both VOC and YOLO formats.

    Args:
        dataset_dir (str): Path to VOC dataset root (contains JPEGImages, Annotations, ImageSets).
        open_set_classes (str): Comma-separated open-set class names (e.g., 'drone,lander').
        shortname (bool): If True, uses short directory names.
    """
    dataset_dir = Path(dataset_dir)
    jpegimages_dir = dataset_dir / "JPEGImages"
    annotations_dir = dataset_dir / "Annotations"
    imagesets_dir = dataset_dir / "ImageSets"

    # Folder naming
    class_string = open_set_classes.replace(",", "_")
    voc_subdir = "Main_CS" if shortname else f"Main_CS_{class_string}"
    yolo_subdir = "YOLO_CS" if shortname else f"YOLO_CS_{class_string}"

    voc_output_dir = imagesets_dir / voc_subdir
    yolo_output_dir = imagesets_dir / yolo_subdir
    voc_output_dir.mkdir(parents=True, exist_ok=True)
    yolo_output_dir.mkdir(parents=True, exist_ok=True)

    # Output files
    voc_output_file = voc_output_dir / "test_ood.txt"
    yolo_output_file = yolo_output_dir / "test_ood.txt"

    # Parse open-set classes
    open_set = set(cls.strip() for cls in open_set_classes.split(","))
    pure_ood_ids = []

    for xml_path in tqdm(annotations_dir.glob("*.xml"), desc="Scanning Annotations", unit="file"):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        object_classes = {obj.find("name").text for obj in root.findall("object")}

        # Only OOD classes
        if object_classes and object_classes.issubset(open_set):
            pure_ood_ids.append(xml_path.stem)

    # Write VOC-style (stems only)
    with open(voc_output_file, "w") as f:
        for image_id in sorted(pure_ood_ids):
            f.write(image_id + "\n")

    # Write YOLO-style (absolute image paths)
    with open(yolo_output_file, "w") as f:
        for image_id in sorted(pure_ood_ids):
            image_path = jpegimages_dir / f"{image_id}.jpg"
            f.write(str(image_path.resolve()) + "\n")

    print(f"[VOC]  Created: {voc_output_file} ({len(pure_ood_ids)} entries)")
    print(f"[YOLO] Created: {yolo_output_file} ({len(pure_ood_ids)} entries)")


def main():
    parser = argparse.ArgumentParser(description="Create pure OOD ImageSet for VOC and YOLO.")
    parser.add_argument("dataset_dir", type=str, help="Path to VOC dataset root.")
    parser.add_argument("open_set_classes", type=str, help="Comma-separated open-set class names (e.g., 'drone,lander').")
    parser.add_argument('--shortname', action='store_true', help='Use short folder names (no class string in path)')
    args = parser.parse_args()

    print(f"Creating pure OOD test set with classes: {args.open_set_classes}")
    create_pure_ood_set(args.dataset_dir, args.open_set_classes, args.shortname)


if __name__ == "__main__":
    main()
