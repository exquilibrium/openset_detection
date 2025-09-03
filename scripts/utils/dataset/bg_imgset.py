import argparse
import random

def create_mixed_imageset(ood_bg_file: str, bg_file: str, output_file: str, bg_fraction: float = 0.1, seed: int = 42):
    """
    Create a new imageset file consisting of OOD samples + a fraction of background images.

    Args:
        ood_bg_file (str): Path to file containing OOD + background images (in that order).
        bg_file (str): Path to file containing only background images.
        output_file (str): Path to save the new imageset file.
        bg_fraction (float): Fraction of background images to include (default: 0.1).
        seed (int): Random seed for reproducibility.
    """
    random.seed(seed)

    # Read OOD+BG file
    with open(ood_bg_file, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    # Read background-only file
    with open(bg_file, "r") as f:
        bg_lines = [line.strip() for line in f if line.strip()]

    # OOD samples are those in ood_bg_file but not in background file
    ood_lines = [l for l in lines if l not in bg_lines]

    # Sample background deterministically
    num_sample = max(1, int(len(bg_lines) * bg_fraction)) if bg_lines else 0
    sampled_bg = random.sample(bg_lines, num_sample) if num_sample > 0 else []

    # Write out OOD first, then background
    with open(output_file, "w") as f:
        f.write("\n".join(ood_lines + sampled_bg) + "\n")

    print(f"✅ Created {output_file} with {len(ood_lines)} OOD and {len(sampled_bg)} BG (seed={seed}).")

print("<", end="")
l = ["Main",
 "Main_CS_drone",
 "Main_CS_lander",
 "Main_CS_lru2"]
for x in l:
    bg = f'/volume/hot_storage/slurm_data/chen_le/ARCHES/lru1_all/ImageSets/Main/bg.txt'
    test = f'/volume/hot_storage/slurm_data/chen_le/ARCHES/lru1_all/ImageSets/{x}/test.txt'
    ood = f'/volume/hot_storage/slurm_data/chen_le/ARCHES/lru1_all/ImageSets/{x}/test_ood.txt'
    test_out = f'/volume/hot_storage/slurm_data/chen_le/ARCHES/lru1_all/ImageSets/{x}/test(10bg).txt'
    ood_out = f'/volume/hot_storage/slurm_data/chen_le/ARCHES/lru1_all/ImageSets/{x}/test_ood(10bg).txt'
    create_mixed_imageset(ood, bg, ood_out)
    create_mixed_imageset(test, bg, test_out)
    print("|", end="")
print("> finished")

print("<", end="")
l = ["YOLO",
 "YOLO_CS_drone",
 "YOLO_CS_lander",
 "YOLO_CS_lru2"]
for x in l:
    bg = f'/volume/hot_storage/slurm_data/chen_le/ARCHES/lru1_all/ImageSets/YOLO/bg_yolo.txt'
    test = f'/volume/hot_storage/slurm_data/chen_le/ARCHES/lru1_all/ImageSets/{x}/test.txt'
    ood = f'/volume/hot_storage/slurm_data/chen_le/ARCHES/lru1_all/ImageSets/{x}/test_ood.txt'
    test_out = f'/volume/hot_storage/slurm_data/chen_le/ARCHES/lru1_all/ImageSets/{x}/test(10bg).txt'
    ood_out = f'/volume/hot_storage/slurm_data/chen_le/ARCHES/lru1_all/ImageSets/{x}/test_ood(10bg).txt'
    create_mixed_imageset(ood, bg, ood_out)
    create_mixed_imageset(test, bg, test_out)
    print("|", end="")
print("> finished")


#if __name__ == "__main__":
#    parser = argparse.ArgumentParser(description="Create new imageset with OOD + seeded background sample")
#    parser.add_argument("ood_bg_file", help="Path to OOD+background imageset file")
#    parser.add_argument("bg_file", help="Path to background-only imageset file")
#    parser.add_argument("output_file", help="Path for output imageset file")
#    parser.add_argument("--bg_fraction", type=float, default=0.1, help="Fraction of background images (default: 0.1)")
#    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
#    args = parser.parse_args()
#    create_mixed_imageset(args.ood_bg_file, args.bg_file, args.output_file, args.bg_fraction, args.seed)

# python bg_imgset.py /volume/hot_storage/slurm_data/chen_le/ARCHES/lru1_all/ImageSets/Main/"test_ood.txt" /volume/hot_storage/slurm_data/chen_le/ARCHES/lru1_all/ImageSets/Main/bg.txt /volume/hot_storage/slurm_data/chen_le/ARCHES/lru1_all/ImageSets/Main/"test_ood (10bg).txt"