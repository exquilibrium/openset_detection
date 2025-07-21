import argparse
import os
import sys
import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.covariance import EmpiricalCovariance
from sklearn.metrics import roc_auc_score, roc_curve

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from base_dirs import *

parser = argparse.ArgumentParser()
parser.add_argument('--saveNm', type=str, required=True, help='Base filename used when saving features')
parser.add_argument('--dataset', type=str, required=True, help='Dataset split train/val/testOOD')
parser.add_argument('--use_yolo', action='store_true', help='Enable yolo evaluation')
args = parser.parse_args()

model_type = 'FRCNN'
if args.use_yolo:
    model_type= 'YOLOv8'


# Load training features and labels
train_path = os.path.join(BASE_RESULTS_FOLDER, model_type, 'mahalanobis', args.dataset, 'train')
feature_id_train = np.load(os.path.join(train_path, f'{args.saveNm}_feature_id_train.npy'))
train_labels = np.load(os.path.join(train_path, f'{args.saveNm}_train_labels.npy'))

# Load validation features
val_path = os.path.join(BASE_RESULTS_FOLDER, model_type, 'mahalanobis', args.dataset, 'val')
feature_id_val = np.load(os.path.join(val_path, f'{args.saveNm}_feature_id_val.npy'))

# Load test (OOD) features
test_path = os.path.join(BASE_RESULTS_FOLDER, model_type, 'mahalanobis', args.dataset, 'testOOD')
feature_ood = np.load(os.path.join(test_path, f'{args.saveNm}_feature_ood.npy'))
#feature_ood = np.load(os.path.join(test_path, f'frcnn_GMMDet_Voc_xml_feature_ood.npy'))
#feature_ood = np.load(os.path.join(test_path, f'frcnn_GMMDet_Voc_xml_yolo_feature_ood.npy'))


def evaluate_Mahalanobis_norm(feature_id_train, feature_id_val, feature_ood, train_labels):
    """
    feature_id_train (numpy array): ID train samples, (n_train x d).
    feature_id_val (numpy array): ID val samples, (n_val x d).
    feature_ood (numpy array): OOD samples (n_ood x d)
    train_labels (numpy array): The labels of the in-distribution training samples.
    Returns:
    tuple: The Mahalanobis scores for in-distribution validation and out-of-distribution samples.
    """
    print(np.shape(feature_id_train))
    print(np.shape(feature_id_val))
    print(np.shape(feature_ood))
    print(np.shape(train_labels))

    # normalize features
    feature_id_val = feature_id_val/np.linalg.norm(feature_id_val,axis=-1,keepdims=True)
    feature_ood = feature_ood/np.linalg.norm(feature_ood,axis=-1,keepdims=True)
    feature_id_train = feature_id_train/np.linalg.norm(feature_id_train,axis=-1,keepdims=True)

    # estimate mean and covariance
    classes_ids = np.unique(train_labels)
    #print(classes_ids)
    train_means = []
    train_feat_centered = []
    for i in tqdm.tqdm(classes_ids): # 1000 because imageNet1k
        fs = feature_id_train[train_labels == i]
        _m = fs.mean(axis=0)
        train_means.append(_m)
        train_feat_centered.extend(fs - _m)
    ec = EmpiricalCovariance(assume_centered=True)
    ec.fit(np.array(train_feat_centered).astype(np.float64))
    mean = np.array(train_means)
    prec = (ec.precision_)
    mean = torch.from_numpy(mean).cuda().double()
    prec = torch.from_numpy(prec).cuda().double()

    # compute Scores
    score_id = -np.array([(((f - mean) @ prec) * (f - mean)).sum(axis=-1).min().cpu().item() for f in
                              tqdm.tqdm(torch.from_numpy(feature_id_val).cuda().double())])
    score_ood = -np.array([(((f - mean) @ prec) * (f - mean)).sum(axis=-1).min().cpu().item() for f in
                           tqdm.tqdm(torch.from_numpy(feature_ood).cuda().double())])
    return score_id, score_ood

score_id, score_ood = evaluate_Mahalanobis_norm(
    feature_id_train,
    feature_id_val,
    feature_ood,
    train_labels
)

# --- 1. Normalize scores using ID statistics ---
mean_id = np.mean(score_id)
std_id = np.std(score_id)

score_id_norm = (score_id - mean_id) / std_id
score_ood_norm = (score_ood - mean_id) / std_id  # Normalize OOD using ID stats

# --- 1.5 Optional: Invert sign if OOD scores > ID scores (incorrectly)
if np.mean(score_ood_norm) > np.mean(score_id_norm):
    print("Inverting score signs (high score must mean more ID-like)...")
    score_id_norm *= -1
    score_ood_norm *= -1

# --- 2. Combine scores and labels ---
scores = np.concatenate([score_id_norm, score_ood_norm])
labels = np.concatenate([np.ones_like(score_id_norm), np.zeros_like(score_ood_norm)])

# --- 3. Compute ROC and AUROC ---
fpr, tpr, thresholds = roc_curve(labels, scores)
auroc = roc_auc_score(labels, scores)
print(f"AUROC: {auroc:.4f}")

# --- 4. TPR at specific FPR levels ---
def tpr_at_fpr(fpr, tpr, target_fpr):
    idx = np.argmin(np.abs(fpr - target_fpr))
    return tpr[idx]

for target in [0.2, 0.1, 0.05]:
    tpr_val = tpr_at_fpr(fpr, tpr, target)
    print(f"TPR at FPR={target:.2f}: {tpr_val:.4f}")

# --- 5. FPR for ID-only "OOD" set (false positive check) ---
use_tpr95 = True
if use_tpr95:
    # --- Find threshold at TPR = 95% ---
    target_tpr = 0.95
    idx = np.argmin(np.abs(tpr - target_tpr))
    threshold_tpr95 = thresholds[idx]
    print(f"Threshold at TPR=0.95: {threshold_tpr95:.4f}")
    threshold = threshold_tpr95
    label_thresh = 'TPR=0.95 threshold'
else:
    # Use threshold on normalized scores (5th percentile of ID)
    threshold_id5 = np.percentile(score_id_norm, 5)
    false_positives = np.sum(score_ood_norm < threshold)
    fpr_manual = false_positives / len(score_ood_norm)
    print(f"False positive rate (on ID-only OOD set): {fpr_manual:.4f}")
    threshold = threshold_id5
    label_thresh ='ID 5% threshold'

plot = True
if plot:
    save_dir = "/home/chen/Desktop/"

    # --- 6. Histogram of scores ---
    # Define clip range (e.g., 1st and 99th percentiles)
    lower_clip = np.percentile(np.concatenate([score_id_norm, score_ood_norm]), 1)
    upper_clip = np.percentile(np.concatenate([score_id_norm, score_ood_norm]), 99)

    # Clip scores
    score_id_clipped = np.clip(score_id_norm, lower_clip, upper_clip)
    score_ood_clipped = np.clip(score_ood_norm, lower_clip, upper_clip)

    plt.figure(figsize=(8, 4))
    plt.hist(score_id_clipped, bins=50, alpha=0.6, label='ID (val)', color='blue', density=True)
    plt.hist(score_ood_clipped, bins=50, alpha=0.6, label='OOD (test)', color='red', density=True)
    plt.axvline(threshold, color='black', linestyle='--', label=label_thresh)
    plt.title("Mahalanobis++ Normalized Score Distributions")
    plt.xlabel("Normalized Mahalanobis Score")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    #plt.show()
    plt.savefig(os.path.join(save_dir, "mahalanobis_score_distribution.png"))

    # --- 7. ROC Curve Plot ---
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f'AUROC = {auroc:.4f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Mahalanobis++ ROC Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    #plt.show()
    plt.savefig(os.path.join(save_dir, "mahalanobis_roc_curve.png"))