import argparse
import os
import sys
import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.covariance import EmpiricalCovariance
from sklearn.metrics import roc_auc_score, roc_curve, auc

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


def post_process_outlier_per_class(per_class, agg_pos, agg_neg, save_prefix, title_suffix=""):
    """
    - Per-class ROC + AUROC
    - Macro and weighted AUROC across classes
    - Per-class histograms with threshold at TPR=0.95 (per class)
    """
    save_dir = BASE_DIR_FOLDER + '/results_img'
    os.makedirs(save_dir, exist_ok=True)

    class_auc = {}
    class_counts = {}

    # Per-class ROC and histogram
    for c, d in per_class.items():
        pos = d["pos_scores"]
        neg = d["neg_scores"]
        n_pos, n_neg = d["n_pos"], d["n_neg"]
        if pos.size == 0 or neg.size == 0:
            continue

        scores = np.concatenate([pos, neg])
        labels = np.concatenate([np.ones_like(pos), np.zeros_like(neg)])
        fpr, tpr, thr = roc_curve(labels, scores)
        auroc = roc_auc_score(labels, scores)
        class_auc[c] = auroc
        class_counts[c] = (n_pos, n_neg)

        # Find threshold at TPR=0.95 (if attainable)
        idx = np.argmin(np.abs(tpr - 0.95))
        thr95 = thr[idx]

        # Histogram (clipped for visibility)
        lower = np.percentile(scores, 1)
        upper = np.percentile(scores, 99)
        pos_clip = np.clip(pos, lower, upper)
        neg_clip = np.clip(neg, lower, upper)

        plt.figure(figsize=(8,4))
        plt.hist(pos_clip, bins=50, alpha=0.6, label=f'Inliers(y={c})', density=True)
        plt.hist(neg_clip, bins=50, alpha=0.6, label=f'Outliers(y≠{c})', density=True)
        plt.axvline(thr95, linestyle='--', label='TPR=0.95 thr')
        plt.title(f"Class {c} — Mahalanobis++ Scores {title_suffix}")
        plt.xlabel("Score (higher=inlier-like)")
        plt.ylabel("Density")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{save_prefix}_class{c}_hist.png"))
        plt.close()

        # ROC
        plt.figure(figsize=(6,6))
        plt.plot(fpr, tpr, label=f'AUROC={auroc:.4f}')
        plt.plot([0,1],[0,1],'k--')
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title(f"Class {c} — ROC {title_suffix}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{save_prefix}_class{c}_roc.png"))
        plt.close()

    # Macro AUROC (unweighted mean)
    if class_auc:
        macro_auroc = np.mean(list(class_auc.values()))
        # Weighted AUROC by positive counts (can choose pos, or pos+neg)
        weights = []
        aucs = []
        for c, auc_c in class_auc.items():
            n_pos, n_neg = class_counts[c]
            weights.append(n_pos)  # weight by positive (inlier) count
            aucs.append(auc_c)
        weights = np.array(weights, dtype=float)
        if weights.sum() > 0:
            weighted_auroc = np.average(aucs, weights=weights)
        else:
            weighted_auroc = macro_auroc

        print(f"[Per-class] Macro AUROC: {macro_auroc:.4f} | Weighted AUROC: {weighted_auroc:.4f}")
    else:
        print("[Per-class] Not enough data to compute class AUROCs.")

    # Global aggregate ROC (across all classes)
    if agg_pos.size and agg_neg.size:
        scores_all = np.concatenate([agg_pos, agg_neg])
        labels_all = np.concatenate([np.ones_like(agg_pos), np.zeros_like(agg_neg)])
        fpr, tpr, _ = roc_curve(labels_all, scores_all)
        auroc_all = roc_auc_score(labels_all, scores_all)
        print(f"[Aggregate across classes] AUROC: {auroc_all:.4f}")

        plt.figure(figsize=(6,6))
        plt.plot(fpr, tpr, label=f'AUROC={auroc_all:.4f}')
        plt.plot([0,1],[0,1],'k--')
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title(f"Aggregate ROC {title_suffix}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{save_prefix}_aggregate_roc.png"))
        plt.close()
    else:
        print("[Aggregate] Not enough data for aggregate ROC.")


# -------- Original OOD (kept in case you still want it) --------
def evaluate_Mahalanobis_norm(feature_id_train, feature_id_val, feature_ood, train_labels, _test_labels_unused=None):
    print(np.shape(feature_id_train))
    print(np.shape(feature_id_val))
    print(np.shape(feature_ood))
    print(np.shape(train_labels))

    feature_id_val = feature_id_val / np.linalg.norm(feature_id_val, axis=-1, keepdims=True)
    feature_ood     = feature_ood     / np.linalg.norm(feature_ood, axis=-1, keepdims=True)
    feature_id_train= feature_id_train/ np.linalg.norm(feature_id_train, axis=-1, keepdims=True)

    classes_ids = np.unique(train_labels.ravel())
    train_means = []
    train_feat_centered = []
    for i in tqdm.tqdm(classes_ids):
        fs = feature_id_train[train_labels.ravel() == i]
        _m = fs.mean(axis=0)
        train_means.append(_m)
        train_feat_centered.extend(fs - _m)
    ec = EmpiricalCovariance(assume_centered=True)
    ec.fit(np.array(train_feat_centered).astype(np.float64))
    mean = torch.from_numpy(np.array(train_means)).cuda().double()
    prec = torch.from_numpy(ec.precision_).cuda().double()

    score_id = -np.array([(((f - mean) @ prec) * (f - mean)).sum(axis=-1).min().cpu().item()
                          for f in tqdm.tqdm(torch.from_numpy(feature_id_val).cuda().double())])
    score_ood = -np.array([(((f - mean) @ prec) * (f - mean)).sum(axis=-1).min().cpu().item()
                           for f in tqdm.tqdm(torch.from_numpy(feature_ood).cuda().double())])
    return score_id, score_ood


def post_process_Mahalanobis_norm(score_id, score_ood, save_prefix, logits=False):
    s = '_logits' if logits else ''

    mean_id = np.mean(score_id); std_id = np.std(score_id)
    score_id_norm  = (score_id - mean_id) / std_id
    score_ood_norm = (score_ood - mean_id) / std_id

    if np.mean(score_ood_norm) > np.mean(score_id_norm):
        print("Inverting score signs (high score must mean more ID-like)...")
        score_id_norm *= -1; score_ood_norm *= -1

    scores = np.concatenate([score_id_norm, score_ood_norm])
    labels = np.concatenate([np.ones_like(score_id_norm), np.zeros_like(score_ood_norm)])
    fpr, tpr, thresholds = roc_curve(labels, scores)
    auroc = roc_auc_score(labels, scores)
    print(f"AUROC: {auroc:.4f}")

    def tpr_at_fpr(fpr, tpr, target_fpr):
        idx = np.argmin(np.abs(fpr - target_fpr))
        return tpr[idx]
    for target in [0.2, 0.1, 0.05]:
        print(f"TPR at FPR={target:.2f}: {tpr_at_fpr(fpr, tpr, target):.4f}")

    # Choose TPR=0.95 threshold for plotting
    idx = np.argmin(np.abs(tpr - 0.95))
    threshold = thresholds[idx]
    label_thresh = 'TPR=0.95 threshold'

    save_dir = BASE_DIR_FOLDER + '/results_img'
    os.makedirs(save_dir, exist_ok=True)
    lower_clip = np.percentile(scores, 1)
    upper_clip = np.percentile(scores, 99)
    score_id_clipped  = np.clip(score_id_norm,  lower_clip, upper_clip)
    score_ood_clipped = np.clip(score_ood_norm, lower_clip, upper_clip)

    plt.figure(figsize=(8,4))
    plt.hist(score_id_clipped, bins=50, alpha=0.6, label='ID (val)', density=True)
    plt.hist(score_ood_clipped, bins=50, alpha=0.6, label='OOD (test)', density=True)
    plt.axvline(threshold, linestyle='--', label=label_thresh)
    plt.title("Mahalanobis++ Normalized Score Distributions")
    plt.xlabel("Normalized Mahalanobis Score")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{save_prefix}mahalanobis_score_distribution{s}.png"))
    plt.close()

    plt.figure(figsize=(6,6))
    plt.plot(fpr, tpr, label=f'AUROC = {auroc:.4f}')
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Mahalanobis++ ROC Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{save_prefix}mahalanobis_roc_curve{s}.png"))
    plt.close()

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
post_process_outlier_per_class(
    per_class, agg_pos, agg_neg,
    save_prefix=os.path.join(args.saveNm + "_feature_"),
    title_suffix="(Feature Maps)"
)

# --- If you still want legacy OOD (min over classes) for comparison ---
score_id, score_ood = evaluate_Mahalanobis_norm(
    feature_id_train,
    feature_id_val,
    feature_test,
    train_labels,
    _test_labels_unused=test_labels
)
post_process_Mahalanobis_norm(score_id, score_ood, save_prefix=args.saveNm + "_feature_", logits=False)

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
post_process_outlier_per_class(
    per_class, agg_pos, agg_neg,
    save_prefix=os.path.join(args.saveNm + "_logits_"),
    title_suffix="(Logits)"
)

# Legacy OOD on logits (optional)
score_id, score_ood = evaluate_Mahalanobis_norm(
    feature_id_train,
    feature_id_val,
    feature_test,
    train_labels
)
post_process_Mahalanobis_norm(score_id, score_ood, save_prefix=args.saveNm + "_logits_", logits=True)
