import os
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
import itertools
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm  # Progress bar

def parse_voc_annotations(annotation_dir):
    image_classes = {}  # image_id -> set of class names
    xml_files = list(Path(annotation_dir).glob("*.xml"))
    for xml_file in tqdm(xml_files, desc="Parsing annotations", unit="file"):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        image_id = xml_file.stem
        classes = set()
        for obj in root.iter("object"):
            class_name = obj.find("name").text
            classes.add(class_name)
        image_classes[image_id] = classes
    return image_classes

def build_cooccurrence_matrix(image_classes):
    class_list = sorted({cls for classes in image_classes.values() for cls in classes})
    class_index = {cls: i for i, cls in enumerate(class_list)}
    size = len(class_list)
    matrix = [[0] * size for _ in range(size)]

    for classes in image_classes.values():
        for c1, c2 in itertools.combinations_with_replacement(classes, 2):
            i, j = class_index[c1], class_index[c2]
            matrix[i][j] += 1
            if i != j:
                matrix[j][i] += 1  # symmetry

    return class_list, matrix

def plot_heatmap(class_list, matrix, save_path):
    df = pd.DataFrame(matrix, index=class_list, columns=class_list)
    plt.figure(figsize=(8, 6))
    sns.heatmap(df, annot=True, cmap="YlGnBu", fmt="d")
    plt.title("Class Co-occurrence Heatmap")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved heatmap to {save_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Visualize and save class co-occurrence heatmap from VOC annotations.")
    parser.add_argument("dataset_dir", type=Path, help="Path to VOC dataset root (contains VOCdevkit/VOC2007/Annotations)")
    args = parser.parse_args()

    annotation_dir = args.dataset_dir / "Annotations"
    save_path = args.dataset_dir / "class_cooccurrence_heatmap.png"

    if not annotation_dir.exists():
        print(f"Annotation directory not found: {annotation_dir}")
        return

    image_classes = parse_voc_annotations(annotation_dir)
    class_list, matrix = build_cooccurrence_matrix(image_classes)
    plot_heatmap(class_list, matrix, save_path)
    print("Created coocurrence heatmap!\n")

if __name__ == "__main__":
    main()
