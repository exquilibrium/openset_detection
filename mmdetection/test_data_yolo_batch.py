# Modified from: https://github.com/dimitymiller/openset_detection

import argparse
import os
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import torchvision.ops as ops

from ultralytics import YOLO
from ultralytics.nn.modules.head import Detect
from ultralytics.utils import ops as yolo_ops

import numpy as np
import tqdm
import json
import sys

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from base_dirs import *

### DATASET 
from torch.utils.data import Dataset, DataLoader
class ImagePathDataset(Dataset):
    def __init__(self, imageset_path):
        self.image_paths = [Path(line.strip()) for line in Path(imageset_path).read_text().splitlines()]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = cv2.imread(str(path))  # keep as NumPy
        return path, img

def collate_fn(batch):
    paths, imgs = zip(*batch)  # unpacks list of tuples
    return list(paths), list(imgs)

# Mahalanobis++ extraction
import torch.nn as nn

C_max = 512
C_out = 256
proj = nn.Linear(C_max, C_out).to(device)
proj.eval()

def parse_args():
    parser = argparse.ArgumentParser(description='Test the data and save the raw detections')
    parser.add_argument('model_path', help='Path to object detector weights')
    parser.add_argument('imageset_path',  help='Path to ImageSets .txt file')
    parser.add_argument('--subset', default = None, help='train, val, test, testOOD (Maha++)')
    parser.add_argument('--num_classes',  help='Number of total classes')
    parser.add_argument('--saveNm', default = None, help='name to save results as')
    args = parser.parse_args()
    return args

args = parse_args()

# Used for Mahalanobis++
# Determine which os classes need to be removed from yolo test
suffix = args.saveNm[len("frcnn_GMMDet_Voc_"):] # custom -> xml, lru1 | yolo -> xml_yolo, lru1_yolo
suffix = suffix[:-len("_yolo")] # xml, lru1
num_classes_dict = { # Class IDs of ID classes
    'xml' : ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person'],
    'lru1' : ['drone', 'lander', 'lru2'],
    'lru1_drone' : ['lander', 'lru2'],
    'lru1_lander' : ['drone', 'lru2'],
    'lru1_lru2' : ['drone', 'lander'],
    'ardea10' : ['lander', 'lru1', 'lru2'],
    'ardea10_lander' : ['lru1', 'lru2'],
    'ardea10_lru1' : ['lander', 'lru2'],
    'ardea10_lru2' : ['lander', 'lru1'],
}
id_classes = num_classes_dict[suffix] # CS classes

### Mahalanobis++
class SaveInputOnly:
    def __init__(self):
        self.input = None

    def __call__(self, module, input):
        self.input = input

def load_voc_labels(image_path):
    boxes = []
    labels = []

    image_path = Path(image_path)
    annot_path = image_path.parent.parent / "Annotations" / image_path.with_suffix('.xml').name

    if not annot_path.exists():
        print(f"Warning: Annotation path does not exist: {annot_path}")
        exit()

    tree = ET.parse(annot_path)
    root = tree.getroot()

    for obj in root.findall("object"):
        name = obj.find("name").text
        if name not in id_classes:
            print(f"Warning: Class name '{name}' not found in class list.")
            continue

        cls_id = id_classes.index(name)

        bndbox = obj.find("bndbox")
        xmin = int(float(bndbox.find("xmin").text))
        ymin = int(float(bndbox.find("ymin").text))
        xmax = int(float(bndbox.find("xmax").text))
        ymax = int(float(bndbox.find("ymax").text))

        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(cls_id)

    return boxes, labels

def compute_iou(boxA, boxB):
    # boxA, boxB: [x1, y1, x2, x2]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

###################################################################################################
############## Setup Config #######################################################################
###################################################################################################
class SaveIO:
    """Simple PyTorch hook to save the output of a nn.module."""
    def __init__(self):
        self.input = None
        self.output = None
        
    def __call__(self, module, module_in, module_out):
        self.input = module_in
        self.output = module_out

def load_and_prepare_model(model_path):
    """
    Load YOLO model and register forward hooks.
    Hook intermediate features and raw predictions to reverse-engineer YOLO model outputs.

    Args:
        model_path (YOLO): Path to YOLOv8 model

    Returns:
        model: YOLO model.
        hooks: List of registered hook references
    """
    # Load YOLO model
    model = YOLO(model_path)
    detect = None
    cv2_hooks = None
    cv3_hooks = None
    cv2_pre_hooks = None
    detect_hook = SaveIO()

    # Identify Detect layer (YOLO head) and register forward hook (`detect_hook`)
    for i, module in enumerate(model.model.modules()):
        if type(module) is Detect:
            module.register_forward_hook(detect_hook)
            detect = module

            # Register forward hooks on detection scale's internal convolution layers (`cv2` and `cv3`)
            cv2_hooks = [SaveIO() for _ in range(module.nl)]
            cv3_hooks = [SaveIO() for _ in range(module.nl)]
            cv2_pre_hooks = [SaveInputOnly() for _ in range(module.nl)] # ---> Mahalanobis++
            for i in range(module.nl):
                module.cv2[i].register_forward_hook(cv2_hooks[i])
                module.cv3[i].register_forward_hook(cv3_hooks[i])
                module.cv2[i].register_forward_pre_hook(cv2_pre_hooks[i])
            break
    input_hook = SaveIO()

    # Register top-level forward hook on entire model
    model.model.register_forward_hook(input_hook)
    hooks = [input_hook, detect, detect_hook, cv2_hooks, cv3_hooks, cv2_pre_hooks]

    return model, hooks

def run_predict_batch(imgs_batch, 
                    img_paths_batch,
                    model,
                    hooks,
                    num_classes,
                    conf_threshold=0.2,
                    iou_threshold=0.5,
                    gt_boxes_batch=None,
                    gt_labels_batch=None):
    """
    Run batch prediction with YOLOv8 and extract Mahalanobis++ features.

    Args:
        imgs_batch (list): Batch of images
        img_paths_batch (list): Batch of paths to image file.
        model (YOLO): YOLO model object.
        hooks (list): List of hooks for the model.
        num_classes (int): Number of os classes.
        conf_threshold (float, optional): Confidence threshold for detection. Default is 0.5.
        iou_threshold (float, optional): Intersection over Union (IoU) threshold for NMS. Default is 0.7.
        gt_boxes_batch (optional)
        gt_labels_batch (optional)

    Returns:
        - List of class_results per image
        - List of Mahalanobis++ features per image
    """
    all_results = []
    all_features = []

    input_hook, detect, detect_hook, cv2_hooks, cv3_hooks, cv2_pre_hooks = hooks
    scale_strides = [8, 16, 32]  # P3, P4, P5

    with torch.no_grad():
        model(imgs_batch, verbose=False, half=True)

    # Extract feature maps and flatten outputs
    shape = detect_hook.input[0][0].shape  # BCHW
    x = [torch.cat((cv2_hooks[i].output, cv3_hooks[i].output), 1) for i in range(detect.nl)]
    x_cat = torch.cat([xi.view(shape[0], detect.no, -1) for xi in x], 2)
    box, classes = x_cat.split((detect.reg_max * 4, detect.nc), 1)

    # Store results for each image
    class_results_batch = []
            
    for batch_idx in range(shape[0]):
        # Hook outputs for this image
        xywh_sigmoid = detect_hook.output[0][batch_idx].T      # [N, 4 + C]
        all_logits = classes[batch_idx].T                      # [N, C]
        coords = xywh_sigmoid[:, :4]                           # [N, 4]
        activations = xywh_sigmoid[:, 4:]                      # [N, C]
        logits = all_logits                                    # [N, C]

        # Scale bounding boxes (yolo_ops.scale_boxes is still CPU-per-box)
        coords_cpu = coords.detach().cpu().numpy()
        orig_img_shape = model.predictor.batch[1][batch_idx].shape[:2]
        img_shape = input_hook.input[0][batch_idx].shape[1:3]
        scaled_coords = np.array([
            yolo_ops.scale_boxes(img_shape, coords_cpu[i], orig_img_shape)
            for i in range(coords_cpu.shape[0])
        ])
        scaled_coords = torch.tensor(scaled_coords, dtype=torch.float32, device=coords.device)

        # Convert to [cx, cy, w, h]
        x0, y0, x1, y1 = scaled_coords[:, 0], scaled_coords[:, 1], scaled_coords[:, 2], scaled_coords[:, 3]
        cx = (x0 + x1) / 2
        cy = (y0 + y1) / 2
        w = x1 - x0
        h = y1 - y0
        bbox_xywh = torch.stack([cx, cy, w, h], dim=1)

        # Prepare for NMS
        boxes_for_nms = torch.cat([
            bbox_xywh, activations, activations, logits
        ], dim=1).T.unsqueeze(0)  # [1, 4+3C, N] → [1, N, 4+3C]

        nms_results_batch = yolo_ops.non_max_suppression(
            boxes_for_nms,
            conf_thres=conf_threshold,
            iou_thres=iou_threshold,
            nc=detect.nc
        )

        # Initialize final output: one list per class
        class_results = [[] for _ in range(num_classes)]

        for nms_results in nms_results_batch:
            if nms_results is None or nms_results.shape[0] == 0:
                continue

            boxes = nms_results[:, 0:4]
            confs = nms_results[:, 4]
            clses = nms_results[:, 5].to(torch.int64)
            logits_out = nms_results[:, 6 + detect.nc:]

            output_rows = torch.cat([boxes, confs.unsqueeze(1), logits_out], dim=1)

            for cls_idx in range(num_classes):
                cls_mask = clses == cls_idx
                if cls_mask.any():
                    class_results[cls_idx].append(output_rows[cls_mask])

        class_results = [
            torch.cat(rows).cpu().numpy().astype(np.float32) if rows else np.zeros((0, 5 + num_classes), dtype=np.float32)
            for rows in class_results
        ]
        all_results.append(class_results)

        features = []
        feat_inputs = [cv2_pre_hooks[i].input[0][batch_idx] for i in range(3)]  # [C, H, W] for P3, P4, P5

        if args.subset in ['train', 'val']:
            gt_boxes = gt_boxes_batch[batch_idx]
            gt_labels = gt_labels_batch[batch_idx]
            for gt_box, gt_label in zip(gt_boxes, gt_labels):
                best_iou = 0
                best_pred = None
                for pred in nms_results:
                    x0, y0, x1, y1, conf, _, *_ = pred
                    pred_box = [x0.item(), y0.item(), x1.item(), y1.item()]
                    iou = compute_iou(pred_box, gt_box)
                    #print(pred_box)
                    #print(gt_box)
                    if iou > 0.5 and conf.item() > conf_threshold and iou > best_iou:
                        best_iou = iou
                        best_pred = pred
                if best_pred is None:
                    continue

                x0, y0, x1, y1 = best_pred[:4]
                box_w, box_h = x1 - x0, y1 - y0
                box_size = max(box_w, box_h)
                scale_idx = 0 if box_size < 64 else 1 if box_size < 128 else 2
                stride = scale_strides[scale_idx]
                feat_map = feat_inputs[scale_idx]
                # TODO: yolo_ops_scale backwards
                cx = (x0 + x1) / 2
                cy = (y0 + y1) / 2
                x_feat = min(max(int(cx / stride), 0), feat_map.shape[2] - 1)
                y_feat = min(max(int(cy / stride), 0), feat_map.shape[1] - 1)
                feature_vec = feat_map[:, y_feat, x_feat]
                C = feature_vec.shape[0]
                if C < C_max:
                    padded = torch.zeros(C_max, device=feature_vec.device)
                    padded[:C] = feature_vec
                    feature_vec = padded
                else:
                    feature_vec = feature_vec[:C_max]
                projected = proj(feature_vec)
                features.append({'feature': projected.detach().cpu().numpy().tolist(), 'label': gt_label})

        elif args.subset == 'testOOD':
            filtered = []
            for i, box in enumerate(nms_results):
                x0, y0, x1, y1, conf, _, *_ = box
                if conf.item() > 0.5:
                    filtered.append((i, [x0.item(), y0.item(), x1.item(), y1.item()], conf.item()))
            keep = []
            for i, (idx_i, box_i, _) in enumerate(filtered):
                redundant = False
                for j, (idx_j, box_j, _) in enumerate(filtered):
                    if i != j and compute_iou(box_i, box_j) > 0.5:
                        redundant = True
                        break
                if not redundant:
                    keep.append(idx_i)
            for idx in keep:
                x0, y0, x1, y1 = nms_results[idx][:4]
                box_w, box_h = x1 - x0, y1 - y0
                box_size = max(box_w, box_h)
                scale_idx = 0 if box_size < 64 else 1 if box_size < 128 else 2
                stride = scale_strides[scale_idx]
                feat_map = feat_inputs[scale_idx]
                # TODO: yolo_ops_scale backwards
                cx = (x0 + x1) / 2
                cy = (y0 + y1) / 2
                x_feat = min(max(int(cx / stride), 0), feat_map.shape[2] - 1)
                y_feat = min(max(int(cy / stride), 0), feat_map.shape[1] - 1)
                feature_vec = feat_map[:, y_feat, x_feat]
                C = feature_vec.shape[0]
                if C < C_max:
                    padded = torch.zeros(C_max, device=feature_vec.device)
                    padded[:C] = feature_vec
                    feature_vec = padded
                else:
                    feature_vec = feature_vec[:C_max]
                projected = proj(feature_vec)
                features.append({'feature': projected.detach().cpu().numpy().tolist()})

        all_features.append(features)

    return all_results, all_features

###################################################################################################
############### Load Dataset ######################################################################
###################################################################################################
print("Building datasets")

batch_size = 8
# Preload images (keep original path and loaded image)
dataset = ImagePathDataset(args.imageset_path)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)

###################################################################################################
############### Build model #######################################################################
###################################################################################################
print("Building model")

# Load model
model, hooks = load_and_prepare_model(args.model_path)

########################################################################################################
########################## TESTING DATA  ###############################################################
########################################################################################################
print(f"Testing {args.imageset_path} data")

score_threshold = 0.2 # only detections with a max softmax above this score are considered valid
iou_threshold = 0.5
allResults = {}
# Mahalanobis++
gt_boxes_batch = None
gt_labels_batch = None
feature_id_train = []
train_labels = []
feature_id_val = []
feature_ood = []
for image_paths_batch, imgs_batch in tqdm.tqdm(dataloader):
    
    # Must match YOLO input size # ---> Mahalanobis++
    if args.subset in ['train', 'val']:
        gt_boxes_batch = [load_voc_labels(p)[0] for p in image_paths_batch]
        gt_labels_batch = [load_voc_labels(p)[1] for p in image_paths_batch]

    # Optional: resize and preprocess here if your model expects fixed size
    results_batch, features_batch = run_predict_batch(
        imgs_batch,
        image_paths_batch,
        model,
        hooks,
        int(args.num_classes),
        conf_threshold=score_threshold,
        iou_threshold=iou_threshold,
        gt_boxes_batch=gt_boxes_batch,
        gt_labels_batch=gt_labels_batch)

    for i, (image_path, result, features) in enumerate(zip(image_paths_batch, results_batch, features_batch)):
        imName = "JPEGImages/" + image_path.name
        allResults[imName] = []
        all_detections = None

        # Save Mahalanobis++ features
        if args.subset == 'train':
            for item in features:
                feature_id_train.append(np.array([item['feature']]))
                train_labels.append(np.array([item['label']]))
        elif args.subset == 'val':
            for item in features:
                feature_id_val.append(np.array([item['feature']]))
        elif args.subset == 'testOOD':
            for item in features:
                feature_ood.append(np.array([item['feature']]))

        # Collect results from each class and concatenate into a list of all the results
        for j in range(np.shape(result)[0]):
            dets = result[j]
            if len(dets) == 0:
                continue

            bboxes = dets[:, :4] # column 0-3
            dists = dets[:, 5:5+int(args.num_classes)]  # Logits from column index 5
            scores = dets[:, 4] # column 4
            scoresT = np.expand_dims(scores, axis=1)
            feats = dets[:, 5+int(args.num_classes):]  # Mahalanobis++

            # Winning class must be class j for this detection to be considered valid
            mask = np.argmax(dists, axis = 1)==j
            if np.sum(mask) == 0:
                continue

            # Check thresholds are above the score cutoff
            imDets = np.concatenate((dists, bboxes, scoresT), 1)[mask]
            scores = scores[mask]
            mask2 = scores >= score_threshold
            if np.sum(mask2) == 0:
                continue

            imDets = imDets[mask2]

            if all_detections is None:
                all_detections = imDets
            else:
                all_detections = np.concatenate((all_detections, imDets))

        if all_detections is not None:
            # Remove doubled-up detections -- this shouldn't really happen
            detections, idxes = np.unique(all_detections, return_index=True, axis=0)
            allResults[imName] = detections.tolist()

# Save raw detection results (existing logic)
if args.subset in ['train', 'val', 'test']:
    jsonRes = json.dumps(allResults)
    raw_save_dir = f'{BASE_RESULTS_FOLDER}/YOLOv8/raw/custom/{args.subset}'
    os.makedirs(raw_save_dir, exist_ok=True)

    with open(f'{raw_save_dir}/{args.saveNm}.json', 'w') as f:
        f.write(jsonRes)

# --- Save Mahalanobis++ features ---
if args.subset in ['train', 'val', 'testOOD']:
    maha_save_dir = f'{BASE_RESULTS_FOLDER}/YOLOv8/mahalanobis/custom/{args.subset}'
    os.makedirs(maha_save_dir, exist_ok=True)

    if args.subset == 'train':   
        print(f"Mahalanobis-Train: {len(feature_id_train)}")
        #print(feature_id_train)
        print(f"Mahalanobis-TrainLabels: {len(train_labels)}")
        #print(train_labels)
        np.save(os.path.join(maha_save_dir, f'{args.saveNm}_feature_id_train.npy'), np.concatenate(feature_id_train, axis=0))
        np.save(os.path.join(maha_save_dir, f'{args.saveNm}_train_labels.npy'), np.concatenate(train_labels, axis=0))
    elif args.subset == 'val':
        print(f"Mahalanobis-Val: {len(feature_id_val)}")
        np.save(os.path.join(maha_save_dir, f'{args.saveNm}_feature_id_val.npy'), np.concatenate(feature_id_val, axis=0))
    elif args.subset == 'testOOD':
        print(f"Mahalanobis-OOD: {len(feature_ood)}")
        np.save(os.path.join(maha_save_dir, f'{args.saveNm}_feature_ood.npy'), np.concatenate(feature_ood, axis=0))
