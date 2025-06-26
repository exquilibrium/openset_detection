import cv2
from pathlib import Path
import argparse
from tqdm import tqdm

def images_to_video(image_dir: Path, fps: int = 30) -> None:
    """
    Converts a directory of images to an MP4 video and saves it as output.mp4 
    in the parent directory of the image folder.

    Args:
        image_dir (Path): Directory containing image files.
        fps (int): Frames per second of the output video.
    """
    image_dir = Path(image_dir)
    output_video = image_dir.parent / "output.mp4"
    images = sorted(image_dir.glob("*.jpg"))

    if not images:
        print("No images found.")
        return

    # Read the first image to get dimensions
    frame = cv2.imread(str(images[0]))
    if frame is None:
        print("Error reading the first image.")
        return

    height, width, _ = frame.shape

    # Define video codec and writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))

    for img_path in tqdm(images, desc="Building video", unit="frame"):
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"Warning: Skipping unreadable file {img_path}")
            continue
        video_writer.write(frame)

    video_writer.release()
    print(f"Video saved to: {output_video}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a directory of images to an MP4 video.")
    parser.add_argument("image_dir", type=Path, help="Directory containing images.")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second for the video.")
    args = parser.parse_args()

    images_to_video(args.image_dir, args.fps)

