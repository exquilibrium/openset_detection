import os

def prepend_path_to_filenames(input_file, output_file, base_path):
    with open(input_file, 'r') as infile:
        lines = infile.readlines()

    with open(output_file, 'w') as outfile:
        for line in lines:
            filename = line.strip()
            if filename:  # Skip empty lines
                full_path = f"{base_path.rstrip('/')}/{filename}"
                outfile.write(full_path + '\n')

if __name__ == "__main__":
    input_txt = "/volume/hot_storage/slurm_data/chen_le/ARCHES/lru1_all/ImageSets/YOLO/bg.txt"
    output_txt = "/volume/hot_storage/slurm_data/chen_le/ARCHES/lru1_all/ImageSets/YOLO/bg_yolo.txt"
    path_prefix = "/volume/hot_storage/slurm_data/chen_le/ARCHES/lru1_all/JPEGImages"

    prepend_path_to_filenames(input_txt, output_txt, path_prefix)


