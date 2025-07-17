import argparse
from pathlib import Path
import shutil
import random
from typing import List
from tqdm import tqdm

def copy_files(filenames: List[str], source: Path, target: Path, include_xml: bool = False):
    for name in tqdm(filenames, desc="Copying files", unit="file"):
        jpg_src = source / f"{name}.jpg"
        jpg_dst = target / f"{name}.jpg"
        if jpg_src.exists():
            shutil.copy(jpg_src, jpg_dst)

        if include_xml:
            xml_src = source / f"{name}.xml"
            xml_dst = target / f"{name}.xml"
            if xml_src.exists():
                shutil.copy(xml_src, xml_dst)

def main():
    parser = argparse.ArgumentParser(description="Copy annotated and sampled background images.")
    parser.add_argument("source", type=Path, help="Source folder containing .jpg and .xml files")
    parser.add_argument("target", type=Path, help="Target folder to copy selected files")
    args = parser.parse_args()

    source = args.source
    target = args.target
    target.mkdir(parents=True, exist_ok=True)

    annotated = [xml.stem for xml in source.glob("*.xml")]
    all_images = [img.stem for img in source.glob("*.jpg")]

    unannotated = list(set(all_images) - set(annotated))
    num_background = int(0.3 * len(annotated))
    sampled_background = random.sample(unannotated, min(num_background, len(unannotated)))

    print(f"Found {len(annotated)} annotated images.")
    print(f"Sampling {len(sampled_background)} background images (30%).")

    copy_files(annotated, source, target, include_xml=True)
    copy_files(sampled_background, source, target, include_xml=False)

    print(f"Copied {len(annotated)} annotated and {len(sampled_background)} background images to {target}.")

if __name__ == "__main__":
    main()
