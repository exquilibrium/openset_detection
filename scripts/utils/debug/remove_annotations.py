import argparse
from pathlib import Path

def clean_annotations(voc_dir: Path, image_ext=".jpg"):
    dir_a = voc_dir / "Annotations"
    dir_b = voc_dir / "JPEGImages"

    deleted_count = 0
    for annotation_file in dir_a.glob("*.xml"):
        image_file = dir_b / (annotation_file.stem + image_ext)
        if not image_file.exists():
            annotation_file.unlink()
            print(f"Deleted: {annotation_file}")
            deleted_count += 1

    print(f"\nDone. Deleted {deleted_count} orphan annotation(s).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Delete annotation files without corresponding image files.")
    parser.add_argument("voc_dir", type=Path, help="Path to the root of VOC dataset (containing Annotations/ and JPEGImages/)")
    args = parser.parse_args()

    clean_annotations(args.voc_dir)


# python remove_annotations.py "/home/chen/openset_detection/datasets/data/VOCdevkit/VOC2007CS"
# python remove_annotations.py "/home/chen/openset_detection/datasets/data/VOCdevkit/VOC2012CS"
