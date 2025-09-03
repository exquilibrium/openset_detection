import argparse
import os
import sys
import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.covariance import EmpiricalCovariance, OAS, LedoitWolf
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, roc_curve

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from base_dirs import *

# ---------------------------
# Utilities
# ---------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPS = 1e-12

def l2norm(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    return X / (np.linalg.norm(X, axis=-1, keepdims=True) + EPS)

def fit_pca_whiten(Xtr: np.ndarray, n_components: int = 128, seed: int = 0):
    if n_components is None or n_components <= 0 or n_components >= Xtr.shape[1]:
        # no PCA
        class Identity:
            def fit(self, X): return self
            def transform(self, X): return X
        return Identity().fit(Xtr)
    pca = PCA(n_components=n_components, whiten=True, svd_solver='auto', random_state=seed)
    pca.fit(Xtr)
    return pca

def cov_estimator(centered: np.ndarray, kind: str = "oas"):
    centered = centered.astype(np.float64, copy=False)
    if kind.lower() == "oas":
        ec = OAS(assume_centered=True).fit(centered)
    elif kind.lower() in ("lw", "ledoitwolf", "ledoit-wolf"):
        ec = LedoitWolf(assume_centered=True).fit(centered)
    elif kind.lower() == "empirical":
        ec = EmpiricalCovariance(assume_centered=True).fit(centered)
    else:
        raise ValueError(f"Unknown covariance estimator '{kind}'")
    return ec.precision_

def per_class_dists_np(X: np.ndarray, means: np.ndarray, prec: np.ndarray) -> np.ndarray:
    """Mahalanobis distances (N,C) using torch for speed/stability."""
    X_t = torch.from_numpy(X).to(device=device, dtype=torch.float64)               # (N,d)
    M_t = torch.from_numpy(means).to(device=device, dtype=torch.float64)          # (C,d)
    P_t = torch.from_numpy(prec).to(device=device, dtype=torch.float64)           # (d,d)
    Xc = X_t[:, None, :] - M_t[None, :, :]                                        # (N,C,d)
    D = ((Xc @ P_t) * Xc).sum(dim=-1)                                             # (N,C)
    return D.detach().cpu().numpy()

def train_own_class_dists(Xtr: np.ndarray, ytr: np.ndarray,
                          means: np.ndarray, prec: np.ndarray, class_ids: np.ndarray):
    """Distances of each train sample to its OWN class mean; dict: k -> (n_k,)"""
    M_t = torch.from_numpy(means).to(device=device, dtype=torch.float64)          # (C,d)
    P_t = torch.from_numpy(prec).to(device=device, dtype=torch.float64)
    d_train = {}
    for i, k in enumerate(class_ids):
        fs = Xtr[ytr == k]
        if fs.size == 0:
            d_train[k] = np.array([], dtype=np.float64)
            continue
        Xt = torch.from_numpy(fs).to(device=device, dtype=torch.float64)
        Xc = Xt - M_t[i:i+1, :]
        d_k = ((Xc @ P_t) * Xc).sum(dim=-1).detach().cpu().numpy()
        d_train[k] = d_k
    return d_train

def per_class_taus(d_train: dict, class_ids: np.ndarray,
                   method: str = "mad", alpha: float = 0.05):
    """
    Returns taus (C,) and diagnostics per class.
    method:
      - 'mad': tau_k = median + c * (1.4826 * MAD), with c≈2.4 ~95% one-sided
      - 'quantile': tau_k = Quantile_{1-alpha}(d_k)
    """
    taus, stats = [], {}
    if method.lower() == "mad":
        c = 2.4  # ~95% one-sided under near-normal tails
        for k in class_ids:
            d_k = np.asarray(d_train[k], dtype=np.float64)
            m = np.median(d_k) if d_k.size else 0.0
            mad = np.median(np.abs(d_k - m)) if d_k.size else 0.0
            tau = m + c * (1.4826 * mad)
            taus.append(tau)
            stats[k] = dict(median=float(m), mad=float(mad), tau=float(tau), method="mad")
    elif method.lower() == "quantile":
        q = 1.0 - alpha
        for k in class_ids:
            d_k = np.asarray(d_train[k], dtype=np.float64)
            tau = np.quantile(d_k, q) if d_k.size else 0.0
            taus.append(tau)
            stats[k] = dict(q=float(q), tau=float(tau), method="quantile")
    else:
        raise ValueError(f"Unknown per-class tau method '{method}'")
    return np.asarray(taus, dtype=np.float64), stats

def youden_threshold(fpr, tpr, thr):
    j = tpr - fpr
    return thr[np.argmax(j)]

# ---------------------------
# Core pipeline (robust Maha++)
# ---------------------------
def robust_mahalanobis_eval(
    feature_id_train: np.ndarray,
    feature_id_val: np.ndarray,
    feature_ood: np.ndarray,
    train_labels: np.ndarray,
    *,
    pca_dim: int = 128,
    cov_kind: str = "oas",
    mean_shrink_lambda: float = 200.0,
    cap_per_class_for_cov: int = 800,
    tau_method: str = "mad",          # 'mad' or 'quantile'
    alpha: float = 0.05,              # per-class tail for quantile
    operating_point: str = "s>=0",    # 's>=0' (default), 'tpr95', 'id5', 'youden'
    plot_title: str = "Mahalanobis++ (robust, per-class)",
    save_prefix: str = "maha_robust"
):
    """
    Returns: dict with AUROC, chosen threshold, per-class taus, paths, etc.
    """

    # 0) Normalize features (row-wise) once
    Xtr = l2norm(feature_id_train)
    Xva = l2norm(feature_id_val)
    Xoo = l2norm(feature_ood)
    ytr = np.asarray(train_labels)

    # 1) PCA-whiten (stabilizes Σ in low-n regimes)
    pca = fit_pca_whiten(Xtr, n_components=pca_dim, seed=0)
    Xtr_p = pca.transform(Xtr)
    Xva_p = pca.transform(Xva)
    Xoo_p = pca.transform(Xoo)

    # 2) Class means with shrinkage toward global mean (helps small classes)
    class_ids = np.unique(ytr)
    mu_global = Xtr_p.mean(axis=0)
    means, for_cov = [], []
    rng = np.random.default_rng(0)

    for k in class_ids:
        fs = Xtr_p[ytr == k]
        mu_k = fs.mean(axis=0) if fs.size else mu_global.copy()

        # mean shrinkage (larger lambda => stronger shrinkage for small n_k)
        w = len(fs) / (len(fs) + mean_shrink_lambda) if len(fs) > 0 else 0.0
        mu_k = w * mu_k + (1.0 - w) * mu_global
        means.append(mu_k)

        # class-balanced residuals for Σ (cap per class to avoid domination)
        if len(fs) > 0:
            if cap_per_class_for_cov and len(fs) > cap_per_class_for_cov:
                idx = rng.choice(len(fs), cap_per_class_for_cov, replace=False)
                fs_cov = fs[idx]
            else:
                fs_cov = fs
            for_cov.append(fs_cov - mu_k)

    means = np.stack(means, axis=0) if len(means) else np.zeros((0, Xtr_p.shape[1]), dtype=np.float64)
    centered = np.concatenate(for_cov, axis=0) if len(for_cov) else np.zeros((0, Xtr_p.shape[1]), dtype=np.float64)

    # 3) Covariance (precision) with shrinkage
    prec = cov_estimator(centered, kind=cov_kind) if centered.size else np.eye(Xtr_p.shape[1], dtype=np.float64)

    # 4) Per-class distances
    D_id  = per_class_dists_np(Xva_p, means, prec)    # (n_val, C)
    D_ood = per_class_dists_np(Xoo_p, means, prec)    # (n_ood, C)

    # 5) Train own-class distances -> per-class taus
    d_train = train_own_class_dists(Xtr_p, ytr, means, prec, class_ids)
    taus, tau_stats = per_class_taus(d_train, class_ids, method=tau_method, alpha=alpha)

    # 6) Aggregate to a single score (higher = more ID)
    #    s = max_k (tau_k - D_k).  s>=0 means accepted by at least one class at its tau_k.
    s_id_raw  = np.max(taus[None, :] - D_id,  axis=1)
    s_ood_raw = np.max(taus[None, :] - D_ood, axis=1)

    # Normalize by ID stats for plotting/ROC
    mu, sd = s_id_raw.mean(), (s_id_raw.std() + EPS)
    s_id = (s_id_raw - mu) / sd
    s_ood = (s_ood_raw - mu) / sd

    # 7) ROC / AUROC
    scores = np.concatenate([s_id, s_ood])
    labels = np.concatenate([np.ones_like(s_id), np.zeros_like(s_ood)])
    fpr, tpr, thr = roc_curve(labels, scores)
    auroc = roc_auc_score(labels, scores)

    # 8) Choose operating threshold
    # Default: 's>=0' i.e., threshold on raw score at 0 -> convert to normalized for plotting
    if operating_point.lower() == "s>=0":
        thr_raw = 0.0
        thr_norm = (thr_raw - mu) / sd
        label_thr = "Per-class α region (s ≥ 0)"
    elif operating_point.lower() == "tpr95":
        idx = np.argmin(np.abs(tpr - 0.95))
        thr_norm = thr[idx]
        label_thr = "TPR=0.95 threshold"
    elif operating_point.lower() == "id5":
        thr_norm = np.percentile(s_id, 5)
        label_thr = "ID 5% threshold"
    elif operating_point.lower() == "youden":
        thr_norm = youden_threshold(fpr, tpr, thr)
        label_thr = "Youden J threshold"
    else:
        raise ValueError(f"Unknown operating_point '{operating_point}'")

    # Manual FPR at chosen threshold
    fpr_manual = (s_ood >= thr_norm).mean()

    # 9) Single panel: hist + ROC
    save_dir = os.path.join(BASE_DIR_FOLDER, "results_img")
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"{save_prefix}_panel.png")

    all_scores = np.concatenate([s_id, s_ood])
    lo, hi = np.percentile(all_scores, 1), np.percentile(all_scores, 99)
    s_id_c  = np.clip(s_id,  lo, hi)
    s_ood_c = np.clip(s_ood, lo, hi)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.hist(s_id_c,  bins=50, alpha=0.6, density=True, label='ID (val)')
    ax.hist(s_ood_c, bins=50, alpha=0.6, density=True, label='OOD (test)')
    ax.axvline(thr_norm, linestyle='--', label=label_thr)
    ax.set_title(plot_title + "\nAggregated per-class score  s = max_k(τ_k - D_k)")
    ax.set_xlabel("Normalized score (higher = more ID)")
    ax.set_ylabel("Density")
    ax.legend()

    ax = axes[1]
    ax.plot(fpr, tpr, label=f"AUROC = {auroc:.4f}")
    ax.plot([0,1], [0,1], 'k--')
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC")
    ax.grid(True)
    ax.legend()

    fig.tight_layout()
    plt.savefig(out_path)
    print(f"[Saved] {out_path}")

    # Metrics printout
    print(f"AUROC: {auroc:.4f}")
    for target in [0.20, 0.10, 0.05]:
        j = np.argmin(np.abs(fpr - target))
        print(f"TPR at FPR={target:.2f}: {tpr[j]:.4f}")
    print(f"FPR at chosen threshold: {fpr_manual:.4f}")

    return {
        "auroc": float(auroc),
        "op_threshold_norm": float(thr_norm),
        "per_class_taus": dict(zip(class_ids.tolist(), taus.tolist())),
        "tau_stats": tau_stats,
        "panel_path": out_path,
        "fpr_curve": fpr, "tpr_curve": tpr
    }

# ---------------------------
# CLI / Script entry
# ---------------------------
def run_block(saveNm: str, dataset: str, model_type: str,
              split_suffix: str,
              feature_train_name: str, feature_val_name: str, feature_ood_name: str,
              args):
    # Load features
    train_path = os.path.join(BASE_RESULTS_FOLDER, model_type, 'mahalanobis', dataset, 'train')
    val_path   = os.path.join(BASE_RESULTS_FOLDER, model_type, 'mahalanobis', dataset, 'val')
    test_path  = os.path.join(BASE_RESULTS_FOLDER, model_type, 'mahalanobis', dataset, 'testOOD')

    feature_id_train = np.load(os.path.join(train_path, f'{saveNm}_{feature_train_name}.npy'))
    train_labels     = np.load(os.path.join(train_path, f'{saveNm}_train_labels.npy'))

    feature_id_val   = np.load(os.path.join(val_path,   f'{saveNm}_{feature_val_name}.npy'))
    feature_ood      = np.load(os.path.join(test_path,  f'{saveNm}_{feature_ood_name}.npy'))

    save_prefix = f"{saveNm}_{split_suffix}"

    return robust_mahalanobis_eval(
        feature_id_train=feature_id_train,
        feature_id_val=feature_id_val,
        feature_ood=feature_ood,
        train_labels=train_labels,
        pca_dim=args.pca_dim,
        cov_kind=args.cov_kind,
        mean_shrink_lambda=args.mean_shrink_lambda,
        cap_per_class_for_cov=args.cap_per_class_for_cov,
        tau_method=args.tau_method,
        alpha=args.alpha,
        operating_point=args.operating_point,
        plot_title=f"Mahalanobis++ ({split_suffix})",
        save_prefix=save_prefix
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--saveNm', type=str, required=True, help='Base filename used when saving features')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset split train/val/testOOD')
    parser.add_argument('--use_yolo', action='store_true', help='Enable yolo evaluation')

    # NEW robust options (sensible defaults)
    parser.add_argument('--pca_dim', type=int, default=128, help='PCA whitened dim (0 or >=d to disable)')
    parser.add_argument('--cov_kind', type=str, default='oas', choices=['oas', 'lw', 'ledoitwolf', 'empirical'])
    parser.add_argument('--mean_shrink_lambda', type=float, default=200.0, help='Mean shrinkage strength')
    parser.add_argument('--cap_per_class_for_cov', type=int, default=800, help='Cap residuals per class for Σ')
    parser.add_argument('--tau_method', type=str, default='mad', choices=['mad', 'quantile'],
                        help='Per-class τ_k method')
    parser.add_argument('--alpha', type=float, default=0.05, help='Per-class tail prob (for quantile)')
    parser.add_argument('--operating_point', type=str, default='s>=0',
                        choices=['s>=0', 'tpr95', 'id5', 'youden'],
                        help='Threshold rule for final decision on aggregated score')

    args = parser.parse_args()
    model_type = 'FRCNN'
    if args.use_yolo:
        model_type = 'YOLOv8'

    # --- Feature Maps block ---
    _ = run_block(
        saveNm=args.saveNm, dataset=args.dataset, model_type=model_type,
        split_suffix="features",
        feature_train_name="feature_id_train",
        feature_val_name="feature_id_val",
        feature_ood_name="feature_ood",
        args=args
    )

    # --- Logits block ---
    _ = run_block(
        saveNm=args.saveNm, dataset=args.dataset, model_type=model_type,
        split_suffix="logits",
        feature_train_name="feature_id_train_logits",
        feature_val_name="feature_id_val_logits",
        feature_ood_name="feature_ood_logits",
        args=args
    )
