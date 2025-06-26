import os
import shutil
import argparse
from tqdm import tqdm

def copy_files(src_dir, dst_dir, file_list, ext):
    for fname in file_list:
        src = os.path.join(src_dir, fname + ext)
        dst = os.path.join(dst_dir, os.path.basename(src))
        if os.path.exists(src):
            shutil.copy2(src, dst)

def read_set_file(path):
    with open(path, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def write_set_file(path, entries):
    with open(path, 'w') as f:
        for entry in sorted(entries):
            f.write(entry + '\n')

def merge_voc_datasets(input_dir):
    voc2007 = os.path.join(input_dir, 'VOCdevkit/VOC2007')
    voc2012 = os.path.join(input_dir, 'VOCdevkit/VOC2012')
    voc0712 = os.path.join(input_dir, 'VOCdevkit_xml/VOC0712')

    subdirs = ['JPEGImages', 'Annotations', 'ImageSets/Main']
    for sub in subdirs:
        os.makedirs(os.path.join(voc0712, sub), exist_ok=True)

    voc2007_sets = {
        'train': read_set_file(os.path.join(voc2007, 'ImageSets/Main/train.txt')),
        'val':   read_set_file(os.path.join(voc2007, 'ImageSets/Main/val.txt')),
        'test':  read_set_file(os.path.join(voc2007, 'ImageSets/Main/test.txt')),
    }
    voc2012_sets = {
        'train': read_set_file(os.path.join(voc2012, 'ImageSets/Main/train.txt')),
        'val':   read_set_file(os.path.join(voc2012, 'ImageSets/Main/val.txt')),
    }

    voc0712_sets = {
        'train': sorted(set(voc2007_sets['train'] + voc2007_sets['val'] + voc2012_sets['train'])),
        'val': voc2012_sets['val'],
        'test': voc2007_sets['test'],
    }

    for subset, img_ids in voc0712_sets.items():
        for img_id in tqdm(img_ids, desc=f"Copying {subset} images"):
            if img_id in voc2007_sets['train'] or img_id in voc2007_sets['val'] or img_id in voc2007_sets['test']:
                copy_files(os.path.join(voc2007, 'JPEGImages'), os.path.join(voc0712, 'JPEGImages'), [img_id], '.jpg')
                copy_files(os.path.join(voc2007, 'Annotations'), os.path.join(voc0712, 'Annotations'), [img_id], '.xml')
            elif img_id in voc2012_sets['train'] or img_id in voc2012_sets['val']:
                copy_files(os.path.join(voc2012, 'JPEGImages'), os.path.join(voc0712, 'JPEGImages'), [img_id], '.jpg')
                copy_files(os.path.join(voc2012, 'Annotations'), os.path.join(voc0712, 'Annotations'), [img_id], '.xml')
    
    for subset, img_ids in voc0712_sets.items():
        write_set_file(os.path.join(voc0712, 'ImageSets/Main', f'{subset}.txt'), img_ids)

    print(f'VOC0712 dataset created at {voc0712}')

def main():
    parser = argparse.ArgumentParser(description="Combines VOC2007 and VOC2012 datasets into VOC0712.")
    parser.add_argument("dataset_dir", type=str, help="Path to folder containing VOCdevkit with VOC2007 and VOC2012")
    args = parser.parse_args()

    print('Creating VOC0712 dataset...')
    merge_voc_datasets(args.dataset_dir)
    print("Finished creating VOC0712.\n")

if __name__ == "__main__":
    main()



# python create_voc0712.py "/media/chen/76AECF8EAECF4579/data"
