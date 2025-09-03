import os

def rename_file(old_path, new_path):
    os.rename(old_path, new_path)
    print(f"Renamed:\n{old_path}\n→ {new_path}")

print("<", end="")
l = ["Main",
 "Main_CS_drone",
 "Main_CS_lander",
 "Main_CS_lru2"]
for x in l:
    test = f'/volume/hot_storage/slurm_data/chen_le/ARCHES/lru1_all/ImageSets/{x}/test.txt'
    ood = f'/volume/hot_storage/slurm_data/chen_le/ARCHES/lru1_all/ImageSets/{x}/test_ood.txt'
    test_out = f'/volume/hot_storage/slurm_data/chen_le/ARCHES/lru1_all/ImageSets/{x}/test(full_bg).txt'
    ood_out = f'/volume/hot_storage/slurm_data/chen_le/ARCHES/lru1_all/ImageSets/{x}/test_ood(full_bg).txt'
    rename_file(test, test_out)
    rename_file(ood, ood_out)
    print("|", end="")
for x in l:
    test = f'/volume/hot_storage/slurm_data/chen_le/ARCHES/lru1_all/ImageSets/{x}/test(10bg).txt'
    ood = f'/volume/hot_storage/slurm_data/chen_le/ARCHES/lru1_all/ImageSets/{x}/test_ood(10bg).txt'
    test_out = f'/volume/hot_storage/slurm_data/chen_le/ARCHES/lru1_all/ImageSets/{x}/test.txt'
    ood_out = f'/volume/hot_storage/slurm_data/chen_le/ARCHES/lru1_all/ImageSets/{x}/test_ood.txt'
    rename_file(test, test_out)
    rename_file(ood, ood_out)
    print("|", end="")
print("> finished")

l = ["YOLO",
 "YOLO_CS_drone",
 "YOLO_CS_lander",
 "YOLO_CS_lru2"]