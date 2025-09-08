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


def load_and_prepare_model(model_path):
    # Load YOLO model
    model = YOLO(model_path)
    detect = model.model.model[-1]  # YOLOv8 Detect head

    # Pre-logit features and hooks
    prelogit_features =  [None] * detect.nl # [None, None, None]

    def make_prelogit_hook(index):
        def hook_fn(module, input, output):
            prelogit_features[index] = output.detach()  # shape: [B, C, H, W]
        return hook_fn

    for scale_idx in range(detect.nl):
        # Hook conv layer before classification
        conv_prelogit = detect.cv3[scale_idx][-1]  # last conv before classification
        conv_prelogit.register_forward_hook(make_prelogit_hook(scale_idx))
        
    return model, prelogit_features

###################################################################################################
############## Get Feature Map ####################################################################
###################################################################################################
def get_feature_map(img, model, hooks, box):
    # hooks is prelogit_features list from load_and_prepare_model()
    prelogit_features = hooks

    # 1) Run inference to populate hooks
    with torch.no_grad():
        model(img, verbose=False, half=True)

    # 2) Infer model-input (letterboxed) size from P3 feature map + stride
    scale_strides = [8, 16, 32]           # P3, P4, P5
    feat_p3 = prelogit_features[0]        # shape: [B, C, H, W]
    assert feat_p3 is not None, "Prelogit hook not populated (did model(img) run?)"
    H3, W3 = int(feat_p3.shape[2]), int(feat_p3.shape[3])
    img_shape = (H3 * scale_strides[0], W3 * scale_strides[0])  # (model_input_h, model_input_w)
    orig_img_shape = img.shape[:2]        # (orig_h, orig_w)

    # 3) Convert xyxy from original -> model-input coords (accounts for letterbox)
    x0, y0, x1, y1 = box["xmin"], box["ymin"], box["xmax"], box["ymax"]
    xyxy = np.array([x0, y0, x1, y1], dtype=np.float32)
    x0i, y0i, x1i, y1i = yolo_ops.scale_boxes(orig_img_shape, xyxy, img_shape)

    # 4) Choose FPN level from *model-input* box size (match offline)
    box_w = x1i - x0i
    box_h = y1i - y0i
    box_size = max(box_w, box_h)
    if box_size < 64:
        scale_idx, stride = 0, 8
    elif box_size < 128:
        scale_idx, stride = 1, 16
    else:
        scale_idx, stride = 2, 32

    # 5) Grab the right feature map and ROIAlign in model-input coords
    feat_map = prelogit_features[scale_idx][0]     # [C, H, W]
    rois = torch.tensor([[0, x0i, y0i, x1i, y1i]], dtype=torch.float32, device=feat_map.device)
    pooled = roi_align(
        input=feat_map.unsqueeze(0).float(),  # [1, C, H, W]
        boxes=rois,
        output_size=(3, 3),
        spatial_scale=1.0 / stride,
        aligned=True
    )  # [1, C, 3, 3]

    # 6) Pool + pad + project exactly like offline
    feature_vec = pooled.view(pooled.shape[1], -1).mean(dim=1)  # [C]
    C = feature_vec.shape[0]
    if C < C_max:
        padded = torch.zeros(C_max, device=feature_vec.device, dtype=feature_vec.dtype)
        padded[:C] = feature_vec
        feature_vec = padded
    elif C > C_max:
        feature_vec = feature_vec[:C_max]

    projected_vec = proj(feature_vec)                      # (256,)
    return projected_vec.detach().cpu().numpy().tolist()


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


