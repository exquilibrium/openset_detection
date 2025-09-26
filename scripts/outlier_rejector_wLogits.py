import argparse
import os
import warnings
from pathlib import Path

import cv2
import torch
import torch.nn as nn
from torchvision.ops import roi_align
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from ultralytics import YOLO
from ultralytics.nn.modules.head import Detect
from ultralytics.utils import ops as yolo_ops

import numpy as np
import tqdm
import json
import sys

from typing import Dict
from dataclasses import dataclass
from sklearn.metrics import roc_curve
from sklearn.covariance import EmpiricalCovariance

    
###################################################################################################
############## Setup Model ########################################################################
###################################################################################################
# Import projection weights
C_max = 512
C_out = 256

PROJ_PATH = "/home/chen_le/openset_detection/scripts/projection.pt"
assert os.path.isfile(PROJ_PATH), f"Projection file not found: {PROJ_PATH}"

# create the layer
proj = nn.Linear(C_max, C_out).to(device)
proj.eval()

# load state
ckpt = torch.load(PROJ_PATH, map_location=device)
state = ckpt.get("state_dict", ckpt)
proj.load_state_dict(state, strict=True)

class SaveInputOnly:
    def __init__(self):
        self.input = None

    def __call__(self, module, input):
        self.input = input

class SaveIO:
    """Robust PyTorch forward hook container for saving input/output."""
    def __init__(self):
        self.input = None
        self.output = None
        self.handle = None

    def __call__(self, module, module_in, module_out):
        self.input = module_in
        self.output = module_out

    def register(self, module):
        self.handle = module.register_forward_hook(self)

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

def load_and_prepare_model(model_path):
    # Load YOLO model
    model = YOLO(model_path)
    detect = model.model.model[-1]  # Detect head is always the last module in YOLOv8

    # Hook to full model input
    input_hook = SaveIO()
    input_hook.register(model.model)

    # Register forward hook on the Detect module
    detect_hook = SaveIO()
    detect_hook.register(detect)

    # Hook internal detection conv layers
    cv2_hooks = [SaveIO() for _ in range(detect.nl)]
    cv3_hooks = [SaveIO() for _ in range(detect.nl)]
    cv2_pre_hooks = [SaveInputOnly() for _ in range(detect.nl)]  # assumed compatible

    # Pre-logit features and hooks
    prelogit_features = [None, None, None]
    prelogit_hooks = []

    def make_prelogit_hook(index):
        def hook_fn(module, input, output):
            prelogit_features[index] = output  # shape: [B, C, H, W]
        return hook_fn

    for scale_idx in range(detect.nl):
        # Hook cv2/cv3 layers
        cv2_hooks[scale_idx].register(detect.cv2[scale_idx])
        cv3_hooks[scale_idx].register(detect.cv3[scale_idx])
        detect.cv2[scale_idx].register_forward_pre_hook(cv2_pre_hooks[scale_idx])

        # Hook conv layer before classification
        conv_prelogit = detect.cv3[scale_idx][-1]  # ✅ last conv before classification
        hook = conv_prelogit.register_forward_hook(make_prelogit_hook(scale_idx))
        prelogit_hooks.append(hook)

    # Return all hooks and buffers
    hooks = [
        input_hook,
        detect,
        detect_hook,
        cv2_hooks,
        cv3_hooks,
        cv2_pre_hooks,
        prelogit_features,
        prelogit_hooks
    ]

    return model, hooks

###################################################################################################
############## Get Feature Map ####################################################################
###################################################################################################
def _boxes_iou_xyxy(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    inter_x0, inter_y0 = max(ax0, bx0), max(ay0, by0)
    inter_x1, inter_y1 = min(ax1, bx1), min(ay1, by1)
    inter_w = max(0.0, inter_x1 - inter_x0)
    inter_h = max(0.0, inter_y1 - inter_y0)
    inter = inter_w * inter_h
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter + 1e-9
    return inter / union

def get_feature_map(img, model, hooks, box):
    """
    Returns:
        list[float] | None
        Logits vector for the detection that best matches `box` (xyxy in ORIGINAL image scale).
        None if no sufficiently good match is found.
    """
    # Unpack required hooks (from load_and_prepare_model tuple)
    assert isinstance(hooks, (list, tuple)) and len(hooks) >= 8, \
        "Expected full hooks tuple: (input_hook, detect, detect_hook, cv2_hooks, cv3_hooks, cv2_pre_hooks, prelogit_features, prelogit_hooks)"
    input_hook, detect, detect_hook, cv2_hooks, cv3_hooks, _, _, _ = hooks

    # 1) Forward pass to populate hooks
    with torch.no_grad():
        model(img, verbose=False, half=True)

    # 2) Model (letterboxed) and original image shapes
    try:
        img_shape = tuple(input_hook.input[0].shape[2:])                 # (H_model, W_model)
        orig_img_shape = tuple(model.predictor.batch[1][0].shape[:2])    # (H_orig, W_orig)
    except Exception:
        # Fallback: use the input tensor and raw image
        img_shape = img.shape[-2:] if isinstance(img, torch.Tensor) else img.shape[:2]
        orig_img_shape = img.shape[-2:] if isinstance(img, torch.Tensor) else img.shape[:2]

    # 3) Rebuild Detect outputs to access logits (mirrors your run_predict)
    shape = detect_hook.input[0][0].shape  # BCHW
    x = []
    for i in range(detect.nl):
        x.append(torch.cat((cv2_hooks[i].output, cv3_hooks[i].output), 1))
    x_cat = torch.cat([xi.view(shape[0], detect.no, -1) for xi in x], 2)
    _, classes = x_cat.split((detect.reg_max * 4, detect.nc), 1)

    batch_idx = 0
    xywh_sigmoid = detect_hook.output[0][batch_idx]  # [no, HW]
    all_logits   = classes[batch_idx]                # [no, HW]

    # 4) Transpose to [N, ...]
    xywh_sigmoid = xywh_sigmoid.T  # [N, 4+C]
    all_logits   = all_logits.T    # [N, C]

    # 5) Convert coords (model-input -> original) for IoU matching
    coords = xywh_sigmoid[:, :4]   # [N, 4] in model-input scale
    coords_cpu = coords.detach().cpu().numpy()
    scaled_coords = np.stack([
        yolo_ops.scale_boxes(img_shape, coords_cpu[i], orig_img_shape)
        for i in range(coords_cpu.shape[0])
    ], axis=0)  # [N, 4] xyxy in ORIGINAL scale

    # 6) Build NMS input to get final detections (and keep logits alongside)
    activations = xywh_sigmoid[:, 4:]  # [N, C] (sigmoid probs)
    logits      = all_logits           # [N, C]
    # Convert to cx,cy,w,h for Ultralytics NMS
    sc = torch.tensor(scaled_coords, dtype=torch.float32, device=coords.device)
    x0, y0, x1, y1 = sc[:, 0], sc[:, 1], sc[:, 2], sc[:, 3]
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    w, h   = (x1 - x0).clamp(min=0), (y1 - y0).clamp(min=0)
    bbox_xywh = torch.stack([cx, cy, w, h], dim=1)  # [N, 4]

    boxes_for_nms = torch.cat([
        bbox_xywh,           # [N, 4]
        activations,         # [N, C]
        activations,         # [N, C] again (Ultralytics signature)
        logits               # [N, C]
    ], dim=1).T.unsqueeze(0) # [1, 4+3C, N]

    nms_results_batch = yolo_ops.non_max_suppression(
        boxes_for_nms,
        conf_thres=0.2,
        iou_thres=0.5,
        nc=detect.nc
    )

    # 7) Pick detection that best matches the provided box (original scale)
    target_xyxy = [
        float(box["xmin"]), float(box["ymin"]),
        float(box["xmax"]), float(box["ymax"])
    ]
    best_logits = None
    best_iou = 0.0

    for nms_results in nms_results_batch:
        if nms_results is None or nms_results.shape[0] == 0:
            continue
        for b in range(nms_results.shape[0]):
            det = nms_results[b, :]
            dx0, dy0, dx1, dy1, conf, cls, *acts_and_logits = det
            det_xyxy = [dx0.item(), dy0.item(), dx1.item(), dy1.item()]
            iou = _boxes_iou_xyxy(target_xyxy, det_xyxy)
            if iou > best_iou:
                best_iou = iou
                # acts_and_logits = [probs C, probs C (dup), logits C]
                best_logits = [p.item() for p in acts_and_logits[detect.nc:]]

    # Require a minimal IoU so we don't return unrelated logits
    if best_logits is None or best_iou < 0.3:
        return None
    return best_logits

###################################################################################################
############## Mahalanobis++ Threshold ############################################################
###################################################################################################
# Helpers
def _to_cuda_double(t):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.as_tensor(t, device=dev, dtype=torch.float32)

def _l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)

def _fit_shared_precision_and_means_np(feature_id_train: np.ndarray, train_labels: np.ndarray):
    """
    Mahalanobis++ fit:
      - L2 normalize features
      - per-class means
      - shared precision from centered data
    """
    X = _l2_normalize(feature_id_train)
    y = train_labels.ravel()
    classes = np.unique(y)

    means = []
    centered_blocks = []
    for c in classes:
        fs = X[y == c]
        m = fs.mean(axis=0, keepdims=False)
        means.append(m)
        centered_blocks.append(fs - m)

    centered = np.concatenate(centered_blocks, axis=0).astype(np.float64)
    ec = EmpiricalCovariance(assume_centered=True).fit(centered)

    mean_t = _to_cuda_double(np.vstack(means))   # [C, D]
    prec_t = _to_cuda_double(ec.precision_)      # [D, D]
    return mean_t, prec_t, classes

def _mahal_scores_to_class(f_np: np.ndarray, mu_row: torch.Tensor, prec_t: torch.Tensor) -> np.ndarray:
    """
    Score = negative squared Mahalanobis distance to the class mean (higher => more inlier-like).
    """
    f = _to_cuda_double(f_np)                        # [N, D]
    diff = f - mu_row.unsqueeze(0)                  # [N, D]
    md2 = torch.sum((diff @ prec_t) * diff, dim=1)  # [N]
    return (-md2).detach().cpu().numpy()

def _unpack_paths(paths):
    """
    Accept either:
      - dict with explicit keys:
        {
          'feature_id_train': '...npy',
          'train_labels':     '...npy',
          'feature_id_val':   '...npy',
          'val_labels':       '...npy',
          'feature_test':     '...npy',
          'test_labels':      '...npy',
        }
      - or a list/tuple of 6 paths in the exact order above.
    """
    if isinstance(paths, dict):
        return (paths['feature_id_train'], paths['train_labels'],
                paths['feature_id_val'], paths['val_labels'],
                paths['feature_test'], paths['test_labels'])
    if isinstance(paths, (list, tuple)) and len(paths) == 6:
        return tuple(paths)
    raise ValueError("`paths` must be a dict with required keys or a 6-tuple/list in the documented order.")

# Initialization
@dataclass
class MahaState:
    """
    Precomputed Mahalanobis++ model for fast runtime checks.
    - mean_t, prec_t live on CUDA if available
    - class_ids is a numpy array of class labels (same order as rows in mean_t)
    - thresholds: dict[label -> score threshold] where score<thr => outlier
    """
    mean_t: torch.Tensor
    prec_t: torch.Tensor
    class_ids: np.ndarray
    thresholds: Dict[int, float]
    tpr_target: float

    # Convenience mapping
    def class_row(self, label: int):
        idx = np.where(self.class_ids == label)[0]
        return int(idx[0]) if idx.size else None

def precompute_maha_state(paths, tpr_target: float = 0.95) -> MahaState:
    """
    One-time initialization: loads arrays, fits Mahalanobis++ parameters,
    computes per-class thresholds at the desired TPR, and returns a reusable state.
    """
    (feature_id_train_path, train_labels_path,
     feature_id_val_path,   val_labels_path,
     feature_test_path,     test_labels_path) = _unpack_paths(paths)

    # Load
    feature_id_train = np.load(feature_id_train_path)
    train_labels     = np.load(train_labels_path)
    feature_id_val   = np.load(feature_id_val_path)
    val_labels       = np.load(val_labels_path)
    feature_test     = np.load(feature_test_path)
    test_labels      = np.load(test_labels_path)

    # Normalize once (stay consistent through pipeline)
    feature_id_train = _l2_normalize(feature_id_train)
    feature_id_val   = _l2_normalize(feature_id_val)
    feature_test     = _l2_normalize(feature_test)

    # Fit Mahalanobis++ params
    mean_t, prec_t, classes = _fit_shared_precision_and_means_np(feature_id_train, train_labels)
    class_to_row = {c: i for i, c in enumerate(classes)}

    # Per-class thresholds (score<thr => outlier)
    thresholds = {}
    for c in classes:
        row = class_to_row[c]
        pos = feature_id_val[val_labels.ravel() == c]   # inliers for class c
        neg = feature_test[test_labels.ravel() != c]    # OOD for class c

        if pos.size == 0 or neg.size == 0:
            # If we cannot compute a ROC threshold, be conservative:
            thresholds[c] = -np.inf
            continue

        pos_scores = _mahal_scores_to_class(pos, mean_t[row], prec_t)
        neg_scores = _mahal_scores_to_class(neg, mean_t[row], prec_t)

        scores = np.concatenate([pos_scores, neg_scores])
        labels = np.concatenate([np.ones_like(pos_scores), np.zeros_like(neg_scores)])

        fpr, tpr, thr = roc_curve(labels, scores)
        idx = int(np.argmin(np.abs(tpr - tpr_target)))
        thresholds[c] = thr[idx]

    print("Class Thresholds:")
    print(f'    drone : {thresholds[0]}')
    print(f'    lander: {thresholds[1]}')
    print(f'    lru2  : {thresholds[2]}')
    
    return MahaState(
        mean_t=mean_t,
        prec_t=prec_t,
        class_ids=classes,
        thresholds=thresholds,
        tpr_target=float(tpr_target),
    )

# Save/Load
def save_maha_state(state: MahaState, path: str):
    # Move to CPU for portability, keep dtype
    payload = {
        "mean": state.mean_t.detach().cpu().numpy(),
        "prec": state.prec_t.detach().cpu().numpy(),
        "class_ids": state.class_ids,
        "thresholds": state.thresholds,
        "tpr_target": state.tpr_target,
    }
    np.savez(path, **payload)

def load_maha_state(path: str) -> MahaState:
    z = np.load(path, allow_pickle=True)
    mean_t = _to_cuda_double(z["mean"])
    prec_t = _to_cuda_double(z["prec"])
    class_ids = z["class_ids"]
    thresholds = dict(z["thresholds"].item()) if isinstance(z["thresholds"].item(), dict) else dict(z["thresholds"])
    tpr_target = float(z["tpr_target"])
    return MahaState(mean_t, prec_t, class_ids, thresholds, tpr_target)


# Runtime
def is_maha_outlier(feature_map: np.ndarray, label: int, state: MahaState) -> bool:
    """
    Real-time check using precomputed `state`.
    Returns True if `feature_map` is below the class threshold (i.e., an outlier).
    """
    row = state.class_row(label)

    f = np.asarray(feature_map, dtype=np.float64)
    if f.ndim == 1:
        f = f[None, :]
    f = _l2_normalize(f)

    score = _mahal_scores_to_class(f, state.mean_t[row], state.prec_t)[0]
    return bool(score < state.thresholds[label]), score


# Main function
label_dict = {
    'drone'     : 0,
    'lander'    : 1,
    'lru2'      : 2,
}

def reject_outlier_detections(detections, img, model, hooks, maha_state, image_path):
    filtered_detections = []
    outlier_detections = []
    for box in detections:
        feature_map = get_feature_map(img, model, hooks, box)
        if feature_map is None:
            # No matching detection → skip this bbox
            continue

        label = label_dict[box["obj_name"]]
        b, score = is_maha_outlier(feature_map, label, maha_state)
        if b:
            #print(f"Outlier detected: {box}")
            #print(f"    score: {score}")
            box["path"] = image_path
            outlier_detections.append(box)
        else:
            #print(f"Inlier detected: {box}")
            #print(f"    score: {score}")
            filtered_detections.append(box)
    return filtered_detections, outlier_detections


