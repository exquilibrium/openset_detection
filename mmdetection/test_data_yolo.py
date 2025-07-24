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

def run_predict(img, 
                img_path,
                model,
                hooks,
                num_classes,
                conf_threshold=0.2,
                iou_threshold=0.5,
                gt_boxes=None,
                gt_labels=None):
    """
    Run prediction with a YOLO model and apply Non-Maximum Suppression (NMS) to the results.

    Args:
        img_path (str): Path to an image file.
        model (YOLO): YOLO model object.
        hooks (list): List of hooks for the model.
        num_classes (int): Number of os classes.
        conf_threshold (float, optional): Confidence threshold for detection. Default is 0.5.
        iou_threshold (float, optional): Intersection over Union (IoU) threshold for NMS. Default is 0.7.

    Returns:
        list: List of selected bounding box dictionaries after NMS.
    """
    # Unpack hooks from load_and_prepare_model()
    input_hook, detect, detect_hook, cv2_hooks, cv3_hooks, cv2_pre_hooks = hooks
    scale_strides = [8, 16, 32]  # Correspond to P3, P4, P5

    # Run inference. Results are stored by hooks
    with torch.no_grad():
        model(imgs_batch, verbose=False, half=True)

    # Reverse engineer outputs to find logits
    # See Detect.forward(): https://github.com/ultralytics/ultralytics/blob/b638c4ed9a24270a6875cdd47d9eeda99204ef5a/ultralytics/nn/modules/head.py#L22
    shape = detect_hook.input[0][0].shape  # BCHW
    x = []
    for i in range(detect.nl):
        x.append(torch.cat((cv2_hooks[i].output, cv3_hooks[i].output), 1))
    x_cat = torch.cat([xi.view(shape[0], detect.no, -1) for xi in x], 2)
    box, classes = x_cat.split((detect.reg_max * 4, detect.nc), 1)

    # Assume batch size 1 (i.e. just running with one image)
    # Loop here to batch images
    batch_idx = 0
    xywh_sigmoid = detect_hook.output[0][batch_idx]
    all_logits = classes[batch_idx]

    # Get original image shape and model image shape to transform boxes
    img_shape = input_hook.input[0].shape[2:]
    orig_img_shape = model.predictor.batch[1][batch_idx].shape[:2]

    # Transpose data
    xywh_sigmoid = xywh_sigmoid.T      # shape: [6300, 4 + C]
    all_logits = all_logits.T          # shape: [6300, C]

    # Compute predictions
    # Slice tensors
    coords = xywh_sigmoid[:, :4]          # [N, 4] — x0, y0, x1, y1
    activations = xywh_sigmoid[:, 4:]     # [N, C]
    logits = all_logits                   # [N, C]

    # Scale all boxes using vector ops (no loop)
    coords_cpu = coords.detach().cpu().numpy()  # shape [N, 4]
    scaled_coords = np.array([
        yolo_ops.scale_boxes(img_shape, coords_cpu[i], orig_img_shape)
        for i in range(coords_cpu.shape[0])
    ])  # shape [N, 4]

    # Convert to cx, cy, w, h (bbox_xywh)
    scaled_coords = torch.tensor(scaled_coords, dtype=torch.float32)  # [N, 4]
    x0, y0, x1, y1 = scaled_coords[:, 0], scaled_coords[:, 1], scaled_coords[:, 2], scaled_coords[:, 3]
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2
    w = x1 - x0
    h = y1 - y0
    bbox_xywh = torch.stack([cx, cy, w, h], dim=1)  # [N, 4]
    
    # Non-Maximum-Suppresion (Retain only the most relevant boxes based on confidence scores)
    boxes_for_nms = torch.cat([
        bbox_xywh.to(coords.device)  ,       # [N, 4]
        activations.to(coords.device)  ,     # [N, C]
        activations.to(coords.device)  ,     # [N, C] again (YOLO-style input)
        logits.to(coords.device)             # [N, C]
    ], dim=1).T.unsqueeze(0)  # Transpose to final shape: [1, 4+3C, N] -> [1, N, 4+3C]

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

        for b in range(nms_results.shape[0]):
            box = nms_results[b, :]
            x0, y0, x1, y1, conf, cls, *acts_and_logits = box
            logits = acts_and_logits[detect.nc:]
            cls_idx = int(cls.item())
            bbox = [x0.item(), y0.item(), x1.item(), y1.item()]  # xyxy
            score = conf.item()
            logits = [p.item() for p in logits]


            # Pack: [x1, y1, x2, y2, score, logit_0, ..., logit_C]
            row = bbox + [score] + logits
            class_results[cls_idx].append(row)

    # Convert each class list to np.ndarray of shape (k_i, 5+num_classes)
    class_results = [
        np.array(cls_boxes, dtype=np.float32) if cls_boxes else np.zeros((0, 5 + num_classes), dtype=np.float32)
        for cls_boxes in class_results
    ]

    # ---------- Mahalanobis++  Feature Extraction ----------
    features = []
    # train, val
    if args.subset in ['train', 'val']:
        features_labels = []

        # Loop over GT boxes
        for gt_box, gt_label in zip(gt_boxes, gt_labels):
            best_iou = 0
            best_pred = None

            for b in range(nms_results.shape[0]):
                box = nms_results[b, :]
                x0, y0, x1, y1, conf, cls, *acts_and_logits = box
                pred_box = [x0.item(), y0.item(), x1.item(), y1.item()]
                #print(pred_box)
                iou = compute_iou(pred_box, gt_box)

                if iou > 0.5 and conf.item() > conf_threshold and iou > best_iou:
                    best_iou = iou
                    best_pred = box

            if best_pred is None:
                #print("No best match???")
                continue  # No matched prediction

            # Use best_pred to extract feature vector
            x0, y0, x1, y1, conf, cls, *acts_and_logits = best_pred
            # Undo scaling
            x0, y0, x1, y1 = yolo_ops.scale_boxes(orig_img_shape, np.array([x0.cpu(), y0.cpu(), x1.cpu(), y1.cpu()]), img_shape)
            box_w = x1 - x0
            box_h = y1 - y0
            box_size = max(box_w, box_h)

            if box_size < 64:
                scale_idx = 0  # P3
            elif box_size < 128:
                scale_idx = 1  # P4
            else:
                scale_idx = 2  # P5

            stride = scale_strides[scale_idx]
            feat_map = cv2_pre_hooks[scale_idx].input[0]  # [B, C, H, W]
            feat_map = feat_map[0]           # [C, H, W]
            cx = (x0 + x1) / 2
            cy = (y0 + y1) / 2
            x_feat = int(cx.item() / stride)
            y_feat = int(cy.item() / stride)
            H, W = feat_map.shape[1:]
            x_feat = min(max(x_feat, 0), W - 1)
            y_feat = min(max(y_feat, 0), H - 1)

            # Extract feature map to fixed size
            feature_vec = feat_map[:, y_feat, x_feat]  # shape: (C,)
            C = feature_vec.shape[0]

            if C < C_max:
                padded = torch.zeros(C_max, device=feature_vec.device)
                padded[:C] = feature_vec
                feature_vec = padded
            elif C > C_max:
                print("Oversized feature???")
                feature_vec = feature_vec[:C_max]  # should rarely happen

            # Project
            projected_vec = proj(feature_vec)  # shape: (256,)
            feature_vec_np = projected_vec.detach().cpu().numpy().tolist()

            #print(gt_box, gt_label)
            features_labels.append({
                'feature': feature_vec_np,
                'label': gt_label
            })
        features = features_labels
    
    # test_ood.txt
    elif args.subset == 'testOOD':
        features_ood = []

        # Get all boxes above threshold
        filtered = []
        for b in range(nms_results.shape[0]):
            box = nms_results[b, :]
            x0, y0, x1, y1, conf, cls, *acts_and_logits = box
            if conf.item() > 0.5:
                filtered.append((b, [x0.item(), y0.item(), x1.item(), y1.item()], conf.item()))

        # Keep only boxes that do NOT overlap (IoU > 0.5) with any other
        for i, (b_idx, box_i, score_i) in enumerate(filtered):
            is_redundant = False
            for j, (_, box_j, _) in enumerate(filtered):
                if i == j:
                    continue
                iou = compute_iou(box_i, box_j)
                if iou > 0.5:
                    is_redundant = True
                    break

            if is_redundant:
                continue

            # Extract feature for non-redundant box
            x0, y0, x1, y1 = box_i
            # Undo scaling
            x0, y0, x1, y1 = yolo_ops.scale_boxes(orig_img_shape, np.array([x0, y0, x1, y1]), img_shape)
            box_w = x1 - x0
            box_h = y1 - y0
            box_size = max(box_w, box_h)

            if box_size < 64:
                scale_idx = 0
            elif box_size < 128:
                scale_idx = 1
            else:
                scale_idx = 2

            stride = scale_strides[scale_idx]
            feat_map = cv2_pre_hooks[scale_idx].input[0]  # [B, C, H, W]
            feat_map = feat_map[0]           # [C, H, W]
            cx = (x0 + x1) / 2
            cy = (y0 + y1) / 2
            x_feat = int(cx / stride)
            y_feat = int(cy / stride)
            H, W = feat_map.shape[1:]
            x_feat = min(max(x_feat, 0), W - 1)
            y_feat = min(max(y_feat, 0), H - 1)

            # Extract feature map to fixed size
            feature_vec = feat_map[:, y_feat, x_feat]  # shape: (C,)
            C = feature_vec.shape[0]

            if C < C_max:
                padded = torch.zeros(C_max, device=feature_vec.device)
                padded[:C] = feature_vec
                feature_vec = padded
            elif C > C_max:
                print("Oversized feature???")
                feature_vec = feature_vec[:C_max]  # should rarely happen

            # Project
            projected_vec = proj(feature_vec)  # shape: (256,)
            feature_vec_np = projected_vec.detach().cpu().numpy().tolist()

            features_ood.append({
                'feature': feature_vec_np,
            })
        features = features_ood
    
    return class_results, features

###################################################################################################
############### Load Dataset ######################################################################
###################################################################################################
print("Building datasets")

# Batching not implemented
batch_size = 1
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
gt_boxes = None
gt_labels = None
feature_id_train = []
train_labels = []
feature_id_val = []
feature_ood = []
for image_paths_batch, imgs_batch in tqdm.tqdm(dataloader):
    image_path = image_paths_batch[0]
    img = imgs_batch[0]

    imName = image_path.name
    imName = "JPEGImages/" + imName 
    allResults[imName] = []
    all_detections = None
    all_scores = []
    
    # Must match YOLO input size # ---> Mahalanobis++
    if args.subset in ['train', 'val']:
        gt_boxes, gt_labels = load_voc_labels(image_path)

    # Predict logits using image and path
    result, features = run_predict(img,
                        image_path,
                        model,
                        hooks, 
                        int(args.num_classes),
                        conf_threshold=score_threshold,
                        iou_threshold=iou_threshold,
                        gt_boxes=gt_boxes,
                        gt_labels=gt_labels)

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

    #collect results from each class and concatenate into a list of all the results
    for j in range(np.shape(result)[0]):
        dets = result[j]

        if len(dets) == 0:
            continue

        bboxes = dets[:, :4] # column 0-3
        dists = dets[:, 5:5+int(args.num_classes)]  # Logits from column index 5
        scores = dets[:, 4] # column 4
        scoresT = np.expand_dims(scores, axis=1)
        feats = dets[:, 5+int(args.num_classes):]  # Mahalanobis++

        #winning class must be class j for this detection to be considered valid
        mask = np.argmax(dists, axis = 1)==j

        # This check needs to be done after mask with j,because YOLO trains on all classes and uses indices according to that.
        # dists = dists[:, id_classes] # Only keep id classes

        if np.sum(mask) == 0:
            continue

        #check thresholds are above the score cutoff
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

    if all_detections is None:
        continue
    else:
        #remove doubled-up detections -- this shouldn't really happen
        detections, idxes = np.unique(all_detections, return_index = True, axis = 0)

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