import os
import argparse
from pathlib import Path

def generate_imagesets_from_yolo_split(voc_root):
    voc_root = Path(voc_root)
    splits_dir = voc_root / "splits" / "images"
    imgsets_dir = voc_root / "ImageSets" / "Main"
    imgsets_dir.mkdir(parents=True, exist_ok=True)

    split_names = ["train", "val", "test"]

    for split in split_names:
        split_path = splits_dir / split
        if not split_path.exists():
            print(f"Warning: {split_path} does not exist. Skipping.")
            continue

        image_ids = [Path(f).stem for f in os.listdir(split_path) if f.endswith(".jpg")]
        output_file = imgsets_dir / f"{split}.txt"

        with open(output_file, "w") as f:
            for image_id in sorted(image_ids):
                f.write(f"{image_id}\n")

        print(f"Written: {output_file} ({len(image_ids)} images)")

    # Create trainval.txt if train and val exist
    train_file = imgsets_dir / "train.txt"
    val_file = imgsets_dir / "val.txt"
    trainval_file = imgsets_dir / "trainval.txt"

    if train_file.exists() and val_file.exists():
        with open(trainval_file, "w") as out_f:
            train_ids = open(train_file).readlines()
            val_ids = open(val_file).readlines()
            out_f.writelines(sorted(set(train_ids + val_ids)))
        print(f"Written: {trainval_file}")

def main():
    parser = argparse.ArgumentParser(description="Generate ImageSets/Main txts from YOLO-style split folders.")
    parser.add_argument("voc_dir", help="Path to the VOC2007 directory (must contain 'splits/images/').")
    args = parser.parse_args()

    generate_imagesets_from_yolo_split(args.voc_dir)

if __name__ == "__main__":
    main()

# python generate_imagesets_from_yolo.py "/media/chen/76AECF8EAECF4579/data/lru1_all_labels/VOCdevkit/VOC2007/"
# python generate_imagesets_from_yolo.py "/media/chen/76AECF8EAECF4579/data/ardea10_run04_labels/VOCdevkit/VOC2007/"
