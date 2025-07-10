import os
import argparse
from tqdm import tqdm

def parse_mapping(from_str, to_str, reverse=False):
    from_ids = list(map(int, from_str.split(',')))
    to_ids = list(map(int, to_str.split(',')))
    if reverse:
        return dict(zip(to_ids, from_ids))
    return dict(zip(from_ids, to_ids))

def remap_labels(path, mapping):
    files = [f for f in os.listdir(path) if f.endswith('.txt')]
    for f in tqdm(files, desc="Remapping labels"):
        with open(os.path.join(path, f), 'r') as infile:
            lines = infile.readlines()
        with open(os.path.join(path, f), 'w') as outfile:
            for line in lines:
                parts = line.strip().split()
                if parts:
                    parts[0] = str(mapping.get(int(parts[0])))
                    outfile.write(' '.join(parts) + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("label_dir", help="Directory containing YOLO .txt label files")
    parser.add_argument("from_ids", help="Comma-separated list of original class IDs (e.g., 0,1,2)")
    parser.add_argument("to_ids", help="Comma-separated list of new class IDs (e.g., 1,2,0)")
    parser.add_argument("--reverse", action="store_true", help="Reverse the mapping (to_ids → from_ids)")
    args = parser.parse_args()

    mapping = parse_mapping(args.from_ids, args.to_ids, reverse=args.reverse)
    remap_labels(args.label_dir, mapping)
