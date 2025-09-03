import argparse
import random

def create_mixed_imageset(ood_bg_file: str, bg_file: str, output_file: str):
    """
    Create a new imageset file consisting of OOD samples + a fraction of background images.

    Args:
        ood_bg_file (str): Path to file containing OOD + background images (in that order).
        bg_file (str): Path to file containing only background images.
        output_file (str): Path to save the new imageset file.
        bg_fraction (float): Fraction of background images to include (default: 0.1).
        seed (int): Random seed for reproducibility.
    """

    # Read OOD+BG file
    with open(ood_bg_file, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    # Read background-only file
    with open(bg_file, "r") as f:
        bg_lines = [line.strip() for line in f if line.strip()]

    # OOD samples are those in ood_bg_file but not in background file
    ood_lines = [l for l in lines if l not in bg_lines]

    # Write out OOD first, then background
    with open(output_file, "w") as f:
        f.write("\n".join(ood_lines + bg_lines) + "\n")

    print(f"✅ Created {output_file} with {len(ood_lines)} OOD and {len(bg_lines)} BG.")

print("<", end="")
l = ["YOLO",
 "YOLO_CS_drone",
 "YOLO_CS_lander",
 "YOLO_CS_lru2"]
for x in l:
    bg = f'/volume/hot_storage/slurm_data/chen_le/ARCHES/lru1_all/ImageSets/YOLO/bg_yolo.txt'
    test = f'/volume/hot_storage/slurm_data/chen_le/ARCHES/lru1_all/ImageSets/{x}/test.txt'
    ood = f'/volume/hot_storage/slurm_data/chen_le/ARCHES/lru1_all/ImageSets/{x}/test_ood.txt'
    test_out = f'/volume/hot_storage/slurm_data/chen_le/ARCHES/lru1_all/ImageSets/{x}/test(full_bg).txt'
    ood_out = f'/volume/hot_storage/slurm_data/chen_le/ARCHES/lru1_all/ImageSets/{x}/test_ood(full_bg).txt'
    create_mixed_imageset(ood, bg, ood_out)
    create_mixed_imageset(test, bg, test_out)
    print("|", end="")
print("> finished")

