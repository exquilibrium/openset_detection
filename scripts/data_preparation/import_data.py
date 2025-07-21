import argparse
from pathlib import Path
import shutil
import random
import xml.etree.ElementTree as ET
from typing import List
from tqdm import tqdm

def has_objects(xml_path: Path) -> bool:
    """Return True if XML file contains at least one <object>."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        return any(obj.tag == "object" for obj in root.findall("object"))
    except Exception as e:
        print(f"Warning: could not parse {xml_path}: {e}")
        return False

def copy_files(filenames: List[str], source: Path, target: Path):
    for name in tqdm(filenames, desc="Copying files", unit="file"):
        jpg_src = source / f"{name}.jpg"
        jpg_dst = target / f"{name}.jpg"
        if jpg_src.exists():
            shutil.copy(jpg_src, jpg_dst)

        xml_src = source / f"{name}.xml"
        xml_dst = target / f"{name}.xml"
        if xml_src.exists():
            shutil.copy(xml_src, xml_dst)

def main():
    parser = argparse.ArgumentParser(description="Copy images")
    parser.add_argument("source", type=Path, help="Source folder containing .jpg and .xml files")
    parser.add_argument("target", type=Path, help="Target folder to copy selected files")
    args = parser.parse_args()

    source = args.source
    target = args.target
    target.mkdir(parents=True, exist_ok=True)

    all_images = set(img.stem for img in source.glob("*.jpg"))

    annotated_with_objects = []
    background_candidates = []

    for name in tqdm(all_images, desc="Filtering", unit="file"):
        xml_path = source / f"{name}.xml"
        if xml_path.exists() and has_objects(xml_path):
            annotated_with_objects.append(name)
        else:
            background_candidates.append(name)

    print(f"Found {len(annotated_with_objects)} annotated images with objects.")
    print(f"Found {len(background_candidates)} background images.")

    copy_files(annotated_with_objects, source, target)
    copy_files(background_candidates, source, target)

    print(f"Copied {len(annotated_with_objects)} annotated and {len(background_candidates)} background images to {target}.")

if __name__ == "__main__":
    main()
