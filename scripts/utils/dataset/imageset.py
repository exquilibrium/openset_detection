from pathlib import Path

def filter_filepaths(a_path, b_path, output_path):
    # Read file paths from B into a set for fast lookup
    with open(b_path, "r") as f:
        b_paths = set(line.strip() for line in f if line.strip())

    # Filter lines from A based on B
    with open(a_path, "r") as f:
        a_paths = [line.strip() for line in f if line.strip()]

    filtered_paths = [path for path in a_paths if path in b_paths]

    # Save the filtered file paths into a new file
    with open(output_path, "w") as f:
        for path in filtered_paths:
            f.write(path + "\n")

    print(f"Filtered {len(filtered_paths)} paths out of {len(a_paths)}.")
    print(f"Result saved to: {output_path}")


if __name__ == "__main__":
    a_txt = "/volume/hot_storage/slurm_data/chen_le/ARCHES/lru1_all/ImageSets/YOLO/test.txt"
    b_txt = "/volume/hot_storage/slurm_data/chen_le/ARCHES/lru1_all/ImageSets/YOLO/test_ood(full_bg).txt"
    output_txt = "/volume/hot_storage/slurm_data/chen_le/ARCHES/lru1_all/ImageSets/YOLO/test_ood.txt"

    filter_filepaths(a_txt, b_txt, output_txt)
