import cv2
import xml.etree.ElementTree as ET
from pathlib import Path
import argparse
from tqdm import tqdm
import numpy as np  # required for random color generation

def parse_voc_annotation(xml_path, class_colors):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    objects = []
    for obj in root.findall("object"):
        name = obj.find("name").text
        bbox = obj.find("bndbox")
        xmin = int(float(bbox.find("xmin").text))
        ymin = int(float(bbox.find("ymin").text))
        xmax = int(float(bbox.find("xmax").text))
        ymax = int(float(bbox.find("ymax").text))
        objects.append((name, (xmin, ymin, xmax, ymax)))
        if name not in class_colors:
            class_colors[name] = tuple(int(c) for c in np.random.randint(0, 255, size=3))
    return objects

def voc_imageset_to_video(imageset_path: Path, fps: int = 30) -> None:
    """
    Converts images listed in a VOC-style imageset file to a video with bounding boxes.

    Args:
        imageset_path (Path): Path to train.txt or similar VOC imageset file.
        fps (int): Frames per second for the video.
    """
    imageset_path = Path(imageset_path)
    dataset_root = imageset_path.parents[2]  # <-- infer dataset root

    jpeg_dir = dataset_root / "JPEGImages"
    ann_dir = dataset_root / "Annotations"

    with open(imageset_path, "r") as f:
        image_ids = [line.strip() for line in f if line.strip()]
        
    image_ids = sorted(image_ids)

    if not image_ids:
        print("No image IDs found in the imageset.")
        return

    # Get size from first image
    first_img = cv2.imread(str(jpeg_dir / f"{image_ids[0]}.jpg"))
    if first_img is None:
        print(f"Error reading first image: {image_ids[0]}.jpg")
        return
    height, width = first_img.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_path = imageset_path.parent / f"{imageset_path.stem}_video.mp4"
    video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    class_colors = {}

    for img_id in tqdm(image_ids, desc="Building video", unit="frame"):
        img_path = jpeg_dir / f"{img_id}.jpg"
        ann_path = ann_dir / f"{img_id}.xml"
        img = cv2.imread(str(img_path))

        if img is None:
            print(f"Warning: Cannot read image {img_path}")
            continue

        if ann_path.exists():
            objects = parse_voc_annotation(ann_path, class_colors)
            for label, (xmin, ymin, xmax, ymax) in objects:
                color = class_colors[label]
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)

                # Draw filled rectangle for text background
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                label_w, label_h = label_size
                text_y = ymin - 10 if ymin - 10 > label_h else ymin + label_h + 2

                cv2.rectangle(img, (xmin, text_y - label_h - 2), (xmin + label_w, text_y), color, thickness=-1)
                cv2.putText(img, label, (xmin, text_y - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        video_writer.write(img)

    video_writer.release()
    print(f"Video saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a video from VOC imageset with bounding boxes.")
    parser.add_argument("imageset_path", type=Path, help="Path to VOC train.txt or similar imageset file.")
    parser.add_argument("--fps", type=int, default=20, help="Frames per second for the video.")
    args = parser.parse_args()

    voc_imageset_to_video(args.imageset_path, args.fps)