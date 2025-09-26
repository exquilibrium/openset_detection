import argparse
import os
import sys
import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.covariance import EmpiricalCovariance
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from base_dirs import *


def _fit_shared_precision_and_means(feature_id_train, train_labels):
    """Fit shared precision (Mahalanobis++) and class means."""
    # L2-normalize
    X = feature_id_train / np.linalg.norm(feature_id_train, axis=-1, keepdims=True)
    y = train_labels.ravel()
    classes = np.unique(y)

    means = []
    centered = []
    for c in classes:
        fs = X[y == c]
        m = fs.mean(axis=0)
        means.append(m)
        centered.append(fs - m)

    centered = np.concatenate(centered, axis=0).astype(np.float64)
    ec = EmpiricalCovariance(assume_centered=True)
    ec.fit(centered)

    mean = torch.from_numpy(np.vstack(means)).cuda().double()        # [C, D]
    prec = torch.from_numpy(ec.precision_).cuda().double()           # [D, D]
    return mean, prec, classes


def evaluate_Mahalanobis_outlier_per_class(
    feature_id_train, train_labels,
    feature_id_val, val_labels,
    feature_test, test_labels
):
    """
    Class-conditional outlier rejection:
      For each class c:
        - positives/inliers:   val where y==c, scored vs mean(c)
        - negatives/outliers:  test where y!=c, scored vs mean(c)
    Returns:
      per_class: dict[c] = {"pos_scores": np.array, "neg_scores": np.array, "n_pos": int, "n_neg": int}
      agg_pos, agg_neg: concatenated arrays across classes (for macro/weighted metrics)
    """
    # Normalize all features
    feature_id_train = feature_id_train / np.linalg.norm(feature_id_train, axis=-1, keepdims=True)
    feature_id_val   = feature_id_val   / np.linalg.norm(feature_id_val, axis=-1, keepdims=True)
    feature_test     = feature_test     / np.linalg.norm(feature_test, axis=-1, keepdims=True)

    val_labels  = val_labels.ravel()
    test_labels = test_labels.ravel()
    train_labels = train_labels.ravel()

    mean, prec, classes = _fit_shared_precision_and_means(feature_id_train, train_labels)  # [C,D], [D,D], [C]
    class_to_row = {c:i for i,c in enumerate(classes)}

    # Torch copies for fast batched scoring
    prec_t = prec
    mean_t = mean

    def mahal_score_to_class(f_np, class_row):
        # Score: negative Mahalanobis distance squared to mean of that class (higher = more inlier-like)
        f = torch.from_numpy(f_np).cuda().double()       # [N, D]
        mu = mean_t[class_row][None, :]                  # [1, D]
        diff = f - mu                                    # [N, D]
        # (x-mu)^T P (x-mu)
        md2 = torch.sum((diff @ prec_t) * diff, dim=1)   # [N]
        return (-md2).detach().cpu().numpy()

    per_class = {}
    agg_pos = []
    agg_neg = []

    # Build per-class splits and score
    for c in classes:
        c_row = class_to_row[c]
        # positives: val samples with label==c
        pos_mask = (val_labels == c)
        pos_feats = feature_id_val[pos_mask]
        if pos_feats.size > 0:
            pos_scores = mahal_score_to_class(pos_feats, c_row)
        else:
            pos_scores = np.array([])

        # negatives: test samples with label!=c
        neg_mask = (test_labels != c)
        neg_feats = feature_test[neg_mask]
        if neg_feats.size > 0:
            neg_scores = mahal_score_to_class(neg_feats, c_row)
        else:
            neg_scores = np.array([])

        per_class[c] = {
            "pos_scores": pos_scores,
            "neg_scores": neg_scores,
            "n_pos": pos_feats.shape[0],
            "n_neg": neg_feats.shape[0],
        }
        if pos_scores.size:
            agg_pos.append(pos_scores)
        if neg_scores.size:
            agg_neg.append(neg_scores)

    agg_pos = np.concatenate(agg_pos) if len(agg_pos) else np.array([])
    agg_neg = np.concatenate(agg_neg) if len(agg_neg) else np.array([])

    return per_class, agg_pos, agg_neg

def compute_quantile_thresholds(per_class, keep_fraction=0.70):
    """Return per-class quantile thresholds: dict[c -> tau]."""
    thresholds = {}
    q = 1.0 - keep_fraction
    for c, d in per_class.items():
        pos = np.asarray(d["pos_scores"])
        if pos.size == 0:
            thresholds[c] = -np.inf
        else:
            thresholds[c] = float(np.quantile(pos, q))
    return thresholds


def compute_mode_mad_thresholds(per_class, k=2.5, bins=64):
    """Return per-class mode–MAD thresholds: dict[c -> tau]."""
    thresholds = {}
    for c, d in per_class.items():
        pos = np.asarray(d["pos_scores"])
        if pos.size == 0:
            thresholds[c] = -np.inf
            continue

        # mode via histogram
        hist, edges = np.histogram(pos, bins=bins)
        mode_bin = int(np.argmax(hist))
        mode = 0.5 * (edges[mode_bin] + edges[mode_bin+1])

        left = pos[pos <= mode]
        if left.size < 3:
            tau = float(np.quantile(pos, 0.25))
        else:
            mad = float(np.median(np.abs(left - np.median(left))))
            tau = float(mode - k * mad)
        thresholds[c] = tau
    return thresholds


def _compute_auroc(labels, scores):
    fpr, tpr, thr = roc_curve(labels, scores, pos_label=1)
    return fpr, tpr, thr, roc_auc_score(labels, scores)

def _compute_aupr(labels, scores):
    precision, recall, thr = precision_recall_curve(labels, scores, pos_label=1)  # AUPR-In (ID positive)
    return precision, recall, thr, auc(recall, precision)

def post_process_outlier_per_class(
    per_class, agg_pos, agg_neg, save_prefix, title_suffix="",
    thresholds_quant=None, thresholds_mmad=None
):
    """
    - Per-class ROC+AUROC and PR+AUPR
    - Macro and weighted AUROC/AUPR
    - Per-class histograms with: TPR=0.95 thr, and (if given) Quantile + Mode–MAD thresholds
    """
    save_dir = BASE_DIR_FOLDER + '/results_img'
    os.makedirs(save_dir, exist_ok=True)

    class_auc = {}        # AUROC per class
    class_aupr = {}       # AUPR per class
    class_counts = {}     # (n_pos, n_neg) per class

    # Per-class ROC/PR and histogram
    for c, d in per_class.items():
        pos = d["pos_scores"]  # ID (y==c)
        neg = d["neg_scores"]  # OOD wrt class c (y!=c)
        n_pos, n_neg = d["n_pos"], d["n_neg"]
        if pos.size == 0 or neg.size == 0:
            continue

        scores = np.concatenate([pos, neg])
        labels = np.concatenate([np.ones_like(pos), np.zeros_like(neg)])

        # --- ROC / AUROC ---
        fpr, tpr, thr_roc, auroc = _compute_auroc(labels, scores)
        class_auc[c] = auroc
        class_counts[c] = (n_pos, n_neg)

        # --- PR / AUPR (AUPR-In: ID positive) ---
        precision, recall, thr_pr, aupr = _compute_aupr(labels, scores)
        class_aupr[c] = aupr

        # Threshold at TPR=0.95 (if attainable)
        idx = np.argmin(np.abs(tpr - 0.95))
        thr95 = thr_roc[idx]

        # Histogram (clipped for visibility)
        lower = np.percentile(scores, 1)
        upper = np.percentile(scores, 99)
        pos_clip = np.clip(pos, lower, upper)
        neg_clip = np.clip(neg, lower, upper)

        plt.figure(figsize=(8,4))
        plt.hist(pos_clip, bins=50, alpha=0.6, label=f'Inliers(y={c})', color='blue', density=True)
        plt.hist(neg_clip, bins=50, alpha=0.6, label=f'Outliers(y≠{c})', color='red', density=True)
        plt.axvline(thr95, linestyle='--', color='black', label=f'TPR95 τ={thr95:.3f}')
        if thresholds_quant is not None and c in thresholds_quant:
            tq = float(thresholds_quant[c]); plt.axvline(tq, linestyle='--', color='green',  label=f'Quantile τ={tq:.3f}')
        if thresholds_mmad is not None and c in thresholds_mmad:
            tm = float(thresholds_mmad[c]); plt.axvline(tm, linestyle='-.', color='purple', label=f'Mode–MAD τ={tm:.3f}')
        plt.title(f"Class {c} — Mahalanobis++ Scores {title_suffix}")
        plt.xlabel("Score (higher=inlier-like)"); plt.ylabel("Density"); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{save_prefix}_class{c}_hist.png")); plt.close()

        # ROC curve
        plt.figure(figsize=(6,6))
        plt.plot(fpr, tpr, label=f'AUROC={auroc:.4f}')
        plt.plot([0,1],[0,1],'k--')
        plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"Class {c} — ROC {title_suffix}")
        plt.legend(); plt.grid(True); plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{save_prefix}_class{c}_roc.png")); plt.close()

        # PR curve
        plt.figure(figsize=(6,6))
        plt.plot(recall, precision, label=f'AUPR-In={aupr:.4f}')
        plt.xlabel("Recall (ID)"); plt.ylabel("Precision (ID)"); plt.title(f"Class {c} — PR {title_suffix}")
        plt.legend(); plt.grid(True); plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{save_prefix}_class{c}_pr.png")); plt.close()

    # Macro/Weighted AUROC and AUPR
    if class_auc:
        macro_auroc = float(np.mean(list(class_auc.values())))
        macro_aupr  = float(np.mean(list(class_aupr.values()))) if class_aupr else np.nan

        weights = []
        aucs = []
        auprs = []
        for c in class_auc:
            n_pos, n_neg = class_counts[c]
            weights.append(n_pos)         # weight by positive (inlier) count
            aucs.append(class_auc[c])
            if c in class_aupr: auprs.append(class_aupr[c])

        weights = np.asarray(weights, dtype=float)
        weighted_auroc = float(np.average(aucs, weights=weights)) if weights.sum() > 0 else macro_auroc
        weighted_aupr  = float(np.average(auprs, weights=weights)) if (len(auprs)>0 and weights.sum()>0) else macro_aupr

        print(f"[Per-class] Macro AUROC: {macro_auroc:.4f} | Weighted AUROC: {weighted_auroc:.4f}")
        print(f"[Per-class] Macro AUPR : {macro_aupr:.4f} | Weighted AUPR : {weighted_aupr:.4f}")
    else:
        print("[Per-class] Not enough data to compute class AUROCs/AUPRs.")

    # Global aggregate ROC & PR (across all classes)
    if agg_pos.size and agg_neg.size:
        scores_all = np.concatenate([agg_pos, agg_neg])
        labels_all = np.concatenate([np.ones_like(agg_pos), np.zeros_like(agg_neg)])

        fpr, tpr, _ , auroc_all = _compute_auroc(labels_all, scores_all)
        precision, recall, _, aupr_all = _compute_aupr(labels_all, scores_all)
        print(f"[Aggregate across classes] AUROC: {auroc_all:.4f} | AUPR-In: {aupr_all:.4f}")

        # Aggregate ROC
        plt.figure(figsize=(6,6))
        plt.plot(fpr, tpr, label=f'AUROC={auroc_all:.4f}')
        plt.plot([0,1],[0,1],'k--')
        plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"Aggregate ROC {title_suffix}")
        plt.legend(); plt.grid(True); plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{save_prefix}_aggregate_roc.png")); plt.close()

        # Aggregate PR
        plt.figure(figsize=(6,6))
        plt.plot(recall, precision, label=f'AUPR-In={aupr_all:.4f}')
        plt.xlabel("Recall (ID)"); plt.ylabel("Precision (ID)"); plt.title(f"Aggregate PR {title_suffix}")
        plt.legend(); plt.grid(True); plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{save_prefix}_aggregate_pr.png")); plt.close()
    else:
        print("[Aggregate] Not enough data for aggregate ROC/PR.")


parser = argparse.ArgumentParser()
parser.add_argument('--saveNm', type=str, required=True, help='Base filename used when saving features')
parser.add_argument('--dataset', type=str, required=True, help='Dataset split train/val/testOOD')
parser.add_argument('--use_yolo', action='store_true', help='Enable yolo evaluation')
args = parser.parse_args()

model_type = 'FRCNN'
if args.use_yolo:
    model_type = 'YOLOv8'

# ===== Feature Map =====
train_path = os.path.join(BASE_RESULTS_FOLDER, model_type, 'mahalanobis', args.dataset, 'train')
val_path   = os.path.join(BASE_RESULTS_FOLDER, model_type, 'mahalanobis', args.dataset, 'val')
test_path  = os.path.join(BASE_RESULTS_FOLDER, model_type, 'mahalanobis', args.dataset, 'testOOD')  # keep folder name

feature_id_train = np.load(os.path.join(train_path, f'{args.saveNm}_feature_id_train.npy'))
train_labels     = np.load(os.path.join(train_path, f'{args.saveNm}_train_labels.npy'))
feature_id_val   = np.load(os.path.join(val_path,   f'{args.saveNm}_feature_id_val.npy'))
val_labels       = np.load(os.path.join(val_path,   f'{args.saveNm}_val_labels.npy'))  # <-- ensure you saved this
feature_test     = np.load(os.path.join(test_path,  f'{args.saveNm}_feature_ood.npy'))
test_labels      = np.load(os.path.join(test_path,  f'{args.saveNm}_test_labels.npy'))  # <-- FIXED PATH

# --- Outlier rejection (per class) ---
per_class, agg_pos, agg_neg = evaluate_Mahalanobis_outlier_per_class(
    feature_id_train, train_labels,
    feature_id_val,   val_labels,
    feature_test,     test_labels
)
thr_quant = compute_quantile_thresholds(per_class, keep_fraction=0.70)
thr_mmad  = compute_mode_mad_thresholds(per_class, k=2.5)
post_process_outlier_per_class(
    per_class, agg_pos, agg_neg,
    save_prefix=os.path.join(args.saveNm + "_feature_"),
    title_suffix="(Feature Maps)",
    thresholds_quant=thr_quant,
    thresholds_mmad=thr_mmad
)

# ===== Logits =====
feature_id_train = np.load(os.path.join(train_path, f'{args.saveNm}_feature_id_train_logits.npy'))
train_labels     = np.load(os.path.join(train_path, f'{args.saveNm}_train_labels.npy'))
feature_id_val   = np.load(os.path.join(val_path,   f'{args.saveNm}_feature_id_val_logits.npy'))
val_labels       = np.load(os.path.join(val_path,   f'{args.saveNm}_val_labels.npy'))      # ensure exists
feature_test     = np.load(os.path.join(test_path,  f'{args.saveNm}_feature_ood_logits.npy'))
test_labels      = np.load(os.path.join(test_path,  f'{args.saveNm}_test_labels.npy'))     # FIXED PATH

per_class, agg_pos, agg_neg = evaluate_Mahalanobis_outlier_per_class(
    feature_id_train, train_labels,
    feature_id_val,   val_labels,
    feature_test,     test_labels
)
thr_quant = compute_quantile_thresholds(per_class, keep_fraction=0.70)
thr_mmad  = compute_mode_mad_thresholds(per_class, k=2.5)
post_process_outlier_per_class(
    per_class, agg_pos, agg_neg,
    save_prefix=os.path.join(args.saveNm + "_logits_"),
    title_suffix="(Logits)",
    thresholds_quant=thr_quant,
    thresholds_mmad=thr_mmad
)
