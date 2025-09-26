#!/usr/bin/env python3
# Fit PCA (512->256) on ROIAligned YOLOv8 features using the TRAIN split (ID only).
# Saves {"mu":[512], "P":[256,512], "n":N} to --out_pca.

import argparse, os, sys
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.ops import roi_align
from ultralytics import YOLO
from ultralytics.utils import ops as yolo_ops

# -------------------------- Config --------------------------
C_MAX = 512
OUT_DIM = 256
SCALE_STRIDES = [8, 16, 32]  # P3,P4,P5

# ID class presets (choose with --id_set)
ID_SETS = {
    'xml' : ['aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','diningtable','dog','horse','motorbike','person'],
    'lru1' : ['drone','lander','lru2'],
    'lru1_drone' : ['lander','lru2'],
    'lru1_lander' : ['drone','lru2'],
    'lru1_lru2' : ['drone','lander'],
    'ardea10' : ['lander','lru1','lru2'],
    'ardea10_lander' : ['lru1','lru2'],
    'ardea10_lru1' : ['lander','lru2'],
    'ardea10_lru2' : ['lander','lru1'],
}

# -------------------------- Data ---------------------------
class ImagePathDataset(Dataset):
    def __init__(self, imageset_path: str):
        p = Path(imageset_path)
        lines = p.read_text().splitlines()
        self.image_paths = [Path(line.strip()) for line in lines if line.strip()]

    def __len__(self): return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = cv2.imread(str(path))  # BGR uint8
        if img is None:
            raise FileNotFoundError(f"cv2.imread failed: {path}")
        return path, img

def collate_fn(batch):
    paths, imgs = zip(*batch)  # batch_size=1 here
    return list(paths), list(imgs)

def load_voc_id_gt(image_path: Path, id_classes):
    boxes, labels = [], []
    annot_path = image_path.parent.parent / "Annotations" / image_path.with_suffix('.xml').name
    if not annot_path.exists():
        return boxes, labels
    root = ET.parse(annot_path).getroot()
    for obj in root.findall("object"):
        name = obj.find("name").text
        if name not in id_classes:
            continue
        cls_id = id_classes.index(name)
        bb = obj.find("bndbox")
        xmin = int(float(bb.find("xmin").text)); ymin = int(float(bb.find("ymin").text))
        xmax = int(float(bb.find("xmax").text)); ymax = int(float(bb.find("ymax").text))
        boxes.append([xmin, ymin, xmax, ymax]); labels.append(cls_id)
    return boxes, labels

def compute_iou_xyxy(a, b):
    xA, yA = max(a[0], b[0]), max(a[1], b[1])
    xB, yB = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter <= 0: return 0.0
    areaA = (a[2]-a[0])*(a[3]-a[1]); areaB = (b[2]-b[0])*(b[3]-b[1])
    return inter / float(areaA + areaB - inter)

# ------------------------- Hooks ---------------------------
class SaveIO:
    def __init__(self): self.input=None; self.output=None; self.h=None
    def __call__(self, m, mi, mo): self.input=mi; self.output=mo
    def register(self, module): self.h = module.register_forward_hook(self)
    def remove(self):
        if self.h is not None: self.h.remove(); self.h=None

def load_model_with_hooks(model_path):
    model = YOLO(model_path)
    detect = model.model.model[-1]  # YOLOv8 Detect head

    input_hook = SaveIO(); input_hook.register(model.model)
    detect_hook = SaveIO(); detect_hook.register(detect)

    cv2_hooks, cv3_hooks = [], []
    prelogit_features = [None, None, None]
    prelogit_handles = []

    def make_prelogit_hook(i):
        def fn(m, inp, out): prelogit_features[i] = out  # [B,C,H,W]
        return fn

    for i in range(detect.nl):
        h2 = SaveIO(); h2.register(detect.cv2[i]); cv2_hooks.append(h2)
        h3 = SaveIO(); h3.register(detect.cv3[i]); cv3_hooks.append(h3)
        conv_prelogit = detect.cv3[i][-1]  # last conv before classification
        prelogit_handles.append(conv_prelogit.register_forward_hook(make_prelogit_hook(i)))

    hooks = (input_hook, detect_hook, cv2_hooks, cv3_hooks, prelogit_features, prelogit_handles)
    return model, detect, hooks

# --------------------- Running PCA (online) ----------------
class RunningPCA:
    """Accumulate first/second moments; then eigendecompose covariance."""
    def __init__(self, C=C_MAX, dtype=torch.float64, device="cpu"):
        self.C = C
        self.dtype = dtype
        self.device = torch.device(device)
        self.n = 0
        self.sum_x  = torch.zeros(C, dtype=dtype, device=self.device)
        self.sum_xx = torch.zeros(C, C, dtype=dtype, device=self.device)

    @torch.no_grad()
    def update(self, x_512: torch.Tensor):
        x = x_512.to(self.dtype).to(self.device)
        self.sum_x += x
        self.sum_xx += torch.outer(x, x)
        self.n += 1

    @torch.no_grad()
    def finalize(self, out_dim=OUT_DIM):
        assert self.n > 1, "Need at least two samples to fit PCA."
        mu  = self.sum_x / self.n                             # [512]
        Exx = self.sum_xx / self.n                            # [512,512]
        cov = Exx - torch.outer(mu, mu)                       # [512,512]
        evals, evecs = torch.linalg.eigh(cov)                 # ascending eigvals
        idx = torch.argsort(evals, descending=True)[:out_dim]
        P = evecs[:, idx].T.contiguous().to(torch.float32)    # [256,512]
        mu = mu.to(torch.float32)
        return {"mu": mu.cpu(), "P": P.cpu(), "n": int(self.n)}

# ------------------- Feature extraction (train) ------------
@torch.no_grad()
def extract_train_features_and_update_pca(img_bgr, image_path, model, detect, hooks, id_classes,
                                          pca_accum: RunningPCA,
                                          conf_thr=0.2, iou_thr=0.5):
    input_hook, detect_hook, cv2_hooks, cv3_hooks, prelogit_features, _ = hooks

    # forward
    model(img_bgr, verbose=False, half=True)

    # Rebuild raw per-anchor predictions (per Detect.forward)
    shape = detect_hook.input[0][0].shape  # BCHW
    x = [torch.cat((cv2_hooks[i].output, cv3_hooks[i].output), 1) for i in range(detect.nl)]
    x_cat = torch.cat([xi.view(shape[0], detect.no, -1) for xi in x], 2)
    # classes = x_cat.split((detect.reg_max * 4, detect.nc), 1)[1]
    box, classes = x_cat.split((detect.reg_max * 4, detect.nc), 1)

    b = 0  # batch idx
    xywh_sigmoid = detect_hook.output[0][b].T      # [N, 4+C]
    activ = xywh_sigmoid[:, 4:]                    # [N, C]
    logits = classes[b].T                          # [N, C]
    coords = xywh_sigmoid[:, :4]                   # [N, 4] in model space

    # scale to original image coords
    img_shape = input_hook.input[0].shape[2:]
    orig_img_shape = model.predictor.batch[1][b].shape[:2]
    coords_cpu = coords.detach().cpu().numpy()
    scaled = np.array([yolo_ops.scale_boxes(img_shape, coords_cpu[i], orig_img_shape)
                       for i in range(coords_cpu.shape[0])], dtype=np.float32)
    x0, y0, x1, y1 = [torch.tensor(scaled[:, k], dtype=torch.float32) for k in range(4)]
    cx, cy = (x0+x1)/2, (y0+y1)/2; w = x1-x0; h = y1-y0
    bbox_xywh = torch.stack([cx, cy, w, h], dim=1)

    # NMS over YOLO packed tensor: [1, N, 4+3C]
    packed = torch.cat([bbox_xywh, activ, activ, logits], dim=1).T.unsqueeze(0)
    nms = yolo_ops.non_max_suppression(packed, conf_thres=conf_thr, iou_thres=iou_thr, nc=detect.nc)[0]
    if nms is None or nms.numel() == 0:
        return 0  # no preds

    # load GT for this image (ID only)
    gt_boxes, gt_labels = load_voc_id_gt(image_path, id_classes)
    if not gt_boxes:
        return 0

    # match GT to best pred (IoU>=0.5), then extract ROIAlign feature and update PCA
    count = 0
    for gt_box in gt_boxes:
        best, best_iou = None, 0.0
        for k in range(nms.shape[0]):
            bb = nms[k, :4].tolist()
            iou = compute_iou_xyxy(bb, gt_box)
            if iou > 0.5 and iou > best_iou:
                best_iou = iou; best = nms[k, :]

        if best is None:
            continue

        # scale the chosen pred back to model input space for ROIAlign
        x0b, y0b, x1b, y1b = best[:4].tolist()
        x0m, y0m, x1m, y1m = yolo_ops.scale_boxes(orig_img_shape, np.array([x0b, y0b, x1b, y1b]), img_shape)

        # choose level by size
        box_w, box_h = x1m - x0m, y1m - y0m
        box_size = max(box_w, box_h)
        if box_size < 64: lvl = 0
        elif box_size < 128: lvl = 1
        else: lvl = 2

        # feature map and roi
        stride = SCALE_STRIDES[lvl]
        fmap = prelogit_features[lvl][0].float()  # [C,H,W], B=1
        rois = torch.tensor([[0, x0m, y0m, x1m, y1m]], dtype=torch.float32, device=fmap.device)
        pooled = roi_align(
            input=fmap.unsqueeze(0),  # [1,C,H,W]
            boxes=rois,
            output_size=(5, 5),
            spatial_scale=1.0/stride,
            aligned=True
        )  # [1,C,5,5]
        vecC = pooled.view(pooled.shape[1], -1).mean(dim=1)  # [C]

        # pad/truncate to 512
        C = vecC.numel()
        if C < C_MAX:
            vecC = nn.functional.pad(vecC, (0, C_MAX - C))
        elif C > C_MAX:
            vecC = vecC[:C_MAX]

        pca_accum.update(vecC.cpu())
        count += 1

    return count

# ----------------------------- Main ------------------------
def parse_args():
    ap = argparse.ArgumentParser("Fit PCA on YOLOv8 ROIAligned TRAIN features (ID only).")
    ap.add_argument("--model", required=True, help="Path to YOLOv8 model (.pt)")
    ap.add_argument("--imageset", required=True, help="Path to TRAIN ImageSets .txt (absolute or relative paths per line).")
    ap.add_argument("--id_set", required=True, choices=ID_SETS.keys(), help="Choose predefined ID class set.")
    ap.add_argument("--out_pca", required=True, help="Output path for PCA (e.g., /path/pca_512to256.pt)")
    ap.add_argument("--conf", type=float, default=0.2, help="Confidence threshold for NMS.")
    ap.add_argument("--iou", type=float, default=0.5, help="IoU threshold for NMS.")
    ap.add_argument("--num_workers", type=int, default=4)
    return ap.parse_args()

def main():
    args = parse_args()
    id_classes = ID_SETS[args.id_set]
    print(f"[INFO] ID classes ({args.id_set}): {id_classes}")

    # data
    ds = ImagePathDataset(args.imageset)
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)
    print(f"[INFO] Train images: {len(ds)}")

    # model + hooks
    model, detect, hooks = load_model_with_hooks(args.model)
    model.eval()

    # PCA accumulator on CPU for stability
    pca = RunningPCA(C=C_MAX, dtype=torch.float64, device="cpu")

    total_feats = 0
    for paths, imgs in dl:
        path = paths[0]; img = imgs[0]
        total_feats += extract_train_features_and_update_pca(
            img_bgr=img, image_path=path, model=model, detect=detect, hooks=hooks,
            id_classes=id_classes, pca_accum=pca, conf_thr=args.conf, iou_thr=args.iou
        )

    if total_feats < 2:
        print("[WARN] Collected <2 features; PCA not saved.")
        return

    ckpt = pca.finalize(out_dim=OUT_DIM)
    os.makedirs(os.path.dirname(args.out_pca), exist_ok=True)
    torch.save(ckpt, args.out_pca)
    print(f"[OK] Saved PCA to {args.out_pca} (samples: {ckpt['n']}, mu: {tuple(ckpt['mu'].shape)}, P: {tuple(ckpt['P'].shape)})")

if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()

"""
python fit_pca_from_train_yolov8.py \
  --model /path/to/yolov8.pt \
  --imageset /path/to/train_images.txt \
  --id_set ardea10 \
  --out_pca /home/chen_le/openset_detection/scripts/pca_512to256.pt \
  --conf 0.2 --iou 0.5

"""