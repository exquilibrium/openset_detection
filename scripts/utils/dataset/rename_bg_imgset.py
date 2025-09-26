import os

BASE_DIR_FOLDER = os.path.dirname(os.path.realpath(__file__))
if BASE_DIR_FOLDER == '/home/chen/openset_detection':
    BASE_DATA_FOLDER = '/media/chen/76AECF8EAECF4579/data'
    BASE_VOC_FOLDER = '/media/chen/76AECF8EAECF4579/data'
else: # '/home/chen_le/openset_detection' >>> DLR
    BASE_DATA_FOLDER = '/volume/hot_storage/slurm_data/chen_le/ARCHES'
    BASE_VOC_FOLDER = '/volume/hot_storage/slurm_data/chen_le'

def rename_file(old_path, new_path):
    os.rename(old_path, new_path)
    print(f"Renamed:\n{old_path}\n→ {new_path}")

def rename_image_sets(image_sets, full):
    if full:
        stash = "small"
        use = "full"
    else:
        stash = "full"
        use = "small"

    for x in image_sets:
        test = f'{BASE_DATA_FOLDER}/{dataset}/ImageSets/{x}/test.txt'
        ood = f'{BASE_DATA_FOLDER}/{dataset}/ImageSets/{x}/test_ood.txt'
        test_out = f'{BASE_DATA_FOLDER}/{dataset}/ImageSets/{x}/test({stash}_bg).txt'
        ood_out = f'{BASE_DATA_FOLDER}/{dataset}/ImageSets/{x}/test_ood({stash}_bg).txt'
        rename_file(test, test_out)
        rename_file(ood, ood_out)
    for x in image_sets:
        test = f'{BASE_DATA_FOLDER}/{dataset}/ImageSets/{x}/test({use}_bg).txt'
        ood = f'{BASE_DATA_FOLDER}/{dataset}/ImageSets/{x}/test_ood({use}_bg).txt'
        test_out = f'{BASE_DATA_FOLDER}/{dataset}/ImageSets/{x}/test.txt'
        ood_out = f'{BASE_DATA_FOLDER}/{dataset}/ImageSets/{x}/test_ood.txt'
        rename_file(test, test_out)
        rename_file(ood, ood_out)

# SET DATASET
dataset = "lru1_all"
use_full_bg_imgset = True

classes_dict = {
    "lru1_all" : ["drone", "lander", "lru2"],
    "ardea10_all": ["lander", "lru1", "lru2"]
}

image_sets = ["Main"] + [("Main_CS_" + x) for x in classes_dict[dataset]]
rename_image_sets(image_sets, use_full_bg_imgset)

image_sets = ["YOLO"] + [("YOLO_CS_" + x) for x in classes_dict[dataset]]
rename_image_sets(image_sets, use_full_bg_imgset)



