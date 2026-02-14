from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
import logging
import os
from typing import List, Dict
from collections import defaultdict
from sklearn import metrics as M

# -------------------------
# Basics: Logging
# -------------------------
def setup_logger(log_path: str):
    for h in logging.root.handlers[:]:
        logging.root.removeHandler(h)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
        force=True
    )
    return logging.getLogger(__name__)

def save_confusion_matrix(cm, save_path, filename="confusion_matrix.txt"):
    """
    Save confusion matrix to file
    
    Args:
        cm: Confusion matrix (numpy array or list)
        save_path: Save directory
        filename: Filename
    """
    _ensure_dir(save_path)
    filepath = os.path.join(save_path, filename)
    
    cm_array = np.array(cm)
    np.savetxt(filepath, cm_array, fmt='%d', delimiter='\t')
    
    # Also save in readable format
    with open(filepath.replace('.txt', '_readable.txt'), 'w') as f:
        f.write("Confusion Matrix:\n")
        f.write("=" * 30 + "\n")
        for i, row in enumerate(cm_array):
            f.write(f"Class {i}: {row}\n")
        f.write("=" * 30 + "\n")
        f.write(f"Total samples: {cm_array.sum()}\n")


# -------------------------
# Training schedule: alignment loss warmup
# -------------------------
class AlignLambdaScheduler:
    """warm-up phase=0, then linearly ramp up to target in ramp"""
    def __init__(self, warmup_epochs_single: int, target: float, ramp_epochs: int):
        self.warmup = int(warmup_epochs_single)
        self.target = float(target)
        self.ramp = int(ramp_epochs)

    def __call__(self, epoch: int) -> float:
        if epoch < self.warmup:
            return 0.0
        if self.ramp <= 0:
            return self.target
        t = min(1.0, (epoch - self.warmup + 1) / self.ramp)
        return self.target * t

# -------------------------
# Data: modality determination/bucketing
# -------------------------
def infer_mode_from_row(row):
    """Unified missing value judgment logic with MultiModalDatasetCSV"""
    def _has(v):
        if v is None:
            return False
        if not isinstance(v, str):
            v = str(v)
        v = v.strip().lower()
        return not (v in {"", "nan", "none", "null"})

    has_fmri = _has(getattr(row, "fmri_dst", None))
    has_eeg  = _has(getattr(row, "eeg_dst", None))

    if has_fmri and has_eeg:
        return "both"
    elif has_fmri:
        return "fmri_only"
    elif has_eeg:
        return "eeg_only"
    else:
        return "none"

def bucket_by_mode(sub_info, indices):
    buckets = {"fmri_only": [], "eeg_only": [], "both": []}
    for i in indices:
        m = infer_mode_from_row(sub_info.iloc[i])
        if m in buckets:
            buckets[m].append(i)
    return {k: np.array(v, dtype=int) for k, v in buckets.items()}

# -------------------------
# A/E multi-scale packing (index slicing)
# -------------------------
def build_A_list(fmri_data, scales, idx=None, device="cpu"):
    if not fmri_data:
        return [None] * len(scales)
    out = []
    for s in scales:
        if s in fmri_data:
            t = fmri_data[s]
            t = t[idx] if (idx is not None and len(idx) > 0) else t
            out.append(t.to(device))
        else:
            out.append(None)
    return out

def build_E_list(eeg_data, scales, idx=None, device="cpu"):
    if not eeg_data:
        return [None] * len(scales)
    out = []
    for s in scales:
        if s in eeg_data:
            band_dict = {}
            for band, t in eeg_data[s].items():
                t2 = t[idx] if (idx is not None and len(idx) > 0) else t
                band_dict[band] = t2.to(device)
            out.append(band_dict)
        else:
            out.append(None)
    return out

# -------------------------
# Evaluation: overall/separate/OOF balanced
# -------------------------
def summarize_val(y_true, y_pred, probs, num_class):
    # Returns: accuracy / bal_acc / macro_f1 / auc / sensitivity / specificity

    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    probs  = np.asarray(probs) if probs is not None and len(probs) else np.zeros((0, num_class))
    accuracy = (y_true == y_pred).mean() if len(y_true) else float('nan')
    bal_acc  = balanced_accuracy_score(y_true, y_pred) if len(y_true) else float('nan')
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0) if len(y_true) else float('nan')
    sens     = M.recall_score(y_true, y_pred, average='macro', zero_division=0) if len(y_true) else float('nan')
    spec     = sens
    if len(np.unique(y_true)) > 1 and probs.shape[0] > 0:
        if probs.ndim == 1:
            mask = np.isfinite(probs)
        else:
            mask = np.all(np.isfinite(probs), axis=1)
        mask = mask & np.isfinite(y_true)
        drop = (~mask).sum()
        if drop > 0:
            bad_idx = np.flatnonzero(~mask)[:10].tolist()
            print(f"[WARN][summarize_val] Drop {drop} rows with NaN in probs/y_true. idx(first10)={bad_idx}")
        y_true_m = y_true[mask]; probs_m = probs[mask]
        if y_true_m.size > 0:
            auc = roc_auc_score(y_true_m, probs_m[:,1] if probs_m.ndim==2 and probs_m.shape[1]==2 else probs_m, 
                                **({} if probs_m.ndim==1 or probs_m.shape[1]==2 else dict(multi_class='ovr', average='macro')))
        else:
            auc = np.nan
    else:
        auc = np.nan
    return dict(accuracy=accuracy, bal_acc=bal_acc, macro_f1=macro_f1, auc=auc, sens=sens, spec=spec)

def log_split_summary(cv, name, idxs, subInfo, labels_np, ds_cfg, logger):
    """
    Uniformly count before augmentation for a subset (train/test):
      - Total N
      - Overall class distribution
      - N for each modality (fmri_only / eeg_only / both) and class distribution within modality
    name: "TRAIN" or "TEST"
    """
    idxs = np.asarray(idxs, dtype=int)
    N = len(idxs)
    if N == 0:
        logger.info(f"[FOLD-{cv}] [{name}] EMPTY split.")
        return

    # ---- Overall class distribution ----
    vals, cnts = np.unique(labels_np[idxs], return_counts=True)
    cls_dist = {int(k): int(v) for k, v in zip(vals.tolist(), cnts.tolist())}

    logger.info(f"[FOLD-{cv}] [{name}] base_N={N} | per-class={cls_dist}")

    # ---- Count by modality × class ----
    mode_cls_counts = defaultdict(lambda: np.zeros(ds_cfg.num_class, dtype=int))
    for i in idxs:
        m = infer_mode_from_row(subInfo.iloc[int(i)])
        if m == "none":
            continue
        y = int(labels_np[int(i)])
        mode_cls_counts[m][y] += 1

    # Fixed order for stable output
    for m in ["fmri_only", "eeg_only", "both"]:
        if m in mode_cls_counts:
            arr = mode_cls_counts[m]
            total_m = int(arr.sum())
            # arr.tolist() is like [nc_count, ad_count] (if num_class=2)
            logger.info(
                f"[FOLD-{cv}] [{name}] mode={m:9s} | total={total_m:3d} | per-class={arr.tolist()}"
            )

    # If there are other rare modalities (e.g., future modes), print them as well
    other_modes = [m for m in mode_cls_counts.keys() if m not in {"fmri_only", "eeg_only", "both"}]
    for m in other_modes:
        arr = mode_cls_counts[m]
        total_m = int(arr.sum())
        logger.info(
            f"[FOLD-{cv}] [{name}] mode={m:9s} | total={total_m:3d} | per-class={arr.tolist()}"
        )

def _ensure_dir(path_str):
    Path(path_str).mkdir(parents=True, exist_ok=True)

# ==============================
# Adversarial noise training: noise injection functions
# ==============================
def inject_fmri_noise(fmri_scales: List[torch.Tensor], 
                     corruption_rate: float = 0.3,
                     roi_dropout_rate: float = 0.3,
                     block_mask_size: int = 10,
                     block_mask_prob: float = 0.5,
                     generator=None) -> List[torch.Tensor]:
    """
    Inject noise/masking into fMRI data
    
    Args:
        fmri_scales: List[torch.Tensor], each element is [B, N, N] fMRI connectivity matrix
        corruption_rate: Corruption sample ratio (0-1)
        roi_dropout_rate: ROI dropout ratio (0-1)
        block_mask_size: Block mask size
        block_mask_prob: Block mask probability
        generator: Random number generator
    
    Returns:
        corrupted_fmri_scales: Corrupted fMRI data list
        corruption_mask: [B] Boolean tensor, True indicates the sample is corrupted
    """
    if len(fmri_scales) == 0 or fmri_scales[0] is None:
        return fmri_scales, torch.zeros((), dtype=torch.bool, device='cpu')
    
    B = fmri_scales[0].shape[0]
    device = fmri_scales[0].device
    
    # Randomly select samples to corrupt
    rng = np.random.default_rng()
    
    n_corrupt = int(B * corruption_rate)
    corrupt_indices = rng.choice(B, size=n_corrupt, replace=False)
    corruption_mask = torch.zeros(B, dtype=torch.bool, device=device)
    corruption_mask[corrupt_indices] = True
    
    corrupted_scales = []
    for scale_mat in fmri_scales:
        if scale_mat is None:
            corrupted_scales.append(None)
            continue
        
        corrupted = scale_mat.clone()
        
        # Apply noise to corrupted samples
        for b_idx in corrupt_indices:
            mat = corrupted[b_idx]  # [N, N]
            N = mat.shape[0]
            
            # Strategy 1: ROI dropout (randomly zero out rows and columns of some ROIs)
            if roi_dropout_rate > 0:
                n_drop = max(1, int(N * roi_dropout_rate))
                roi_indices = rng.choice(N, size=n_drop, replace=False)
                mat[roi_indices, :] = 0
                mat[:, roi_indices] = 0
            
            # Strategy 2: Block masking (randomly mask square regions)
            if block_mask_prob > 0 and rng.random() < block_mask_prob:
                block_size = min(block_mask_size, N // 2)
                if block_size > 0:
                    start_i = rng.integers(0, max(1, N - block_size))
                    start_j = rng.integers(0, max(1, N - block_size))
                    end_i = min(start_i + block_size, N)
                    end_j = min(start_j + block_size, N)
                    mat[start_i:end_i, start_j:end_j] = 0
        
        corrupted_scales.append(corrupted)
    
    return corrupted_scales, corruption_mask


def inject_eeg_noise(eeg_scales: List[Dict[str, torch.Tensor]],
                     corruption_rate: float = 0.3,
                     channel_dropout_rate: float = 0.3,
                     noise_scale: float = 2.0,
                     generator=None) -> List[Dict[str, torch.Tensor]]:
    """
    Inject noise/masking into EEG data
    
    Args:
        eeg_scales: List[Dict[str, torch.Tensor]], each element is {band: [B, N, N]} dictionary
        corruption_rate: Corruption sample ratio (0-1)
        channel_dropout_rate: Channel dropout ratio (0-1)
        noise_scale: 1/f noise strength
        generator: Random number generator
    
    Returns:
        corrupted_eeg_scales: Corrupted EEG data list
        corruption_mask: [B] Boolean tensor, True indicates the sample is corrupted
    """
    if len(eeg_scales) == 0 or eeg_scales[0] is None:
        return eeg_scales, torch.zeros((), dtype=torch.bool, device='cpu')
    
    # Get B and device from first non-None scale
    B, device = None, None
    for scale_dict in eeg_scales:
        if scale_dict is not None:
            for band_tensor in scale_dict.values():
                if band_tensor is not None:
                    B = band_tensor.shape[0]
                    device = band_tensor.device
                    break
            if B is not None:
                break
    
    if B is None:
        return eeg_scales, torch.zeros((), dtype=torch.bool, device='cpu')
    
    # Randomly select samples to corrupt
    rng = np.random.default_rng()
    
    n_corrupt = int(B * corruption_rate)
    corrupt_indices = rng.choice(B, size=n_corrupt, replace=False)
    corruption_mask = torch.zeros(B, dtype=torch.bool, device=device)
    corruption_mask[corrupt_indices] = True
    
    corrupted_scales = []
    for scale_dict in eeg_scales:
        if scale_dict is None:
            corrupted_scales.append(None)
            continue
        
        corrupted_dict = {}
        for band, band_tensor in scale_dict.items():
            if band_tensor is None:
                corrupted_dict[band] = None
                continue
            
            corrupted = band_tensor.clone()  # [B, N, N]
            N = corrupted.shape[1]
            
            # Apply noise to corrupted samples
            for b_idx in corrupt_indices:
                mat = corrupted[b_idx]  # [N, N]
                
                # Strategy 1: Channel masking (randomly zero out rows and columns of some channels)
                if channel_dropout_rate > 0:
                    n_drop = max(1, int(N * channel_dropout_rate))
                    ch_indices = rng.choice(N, size=n_drop, replace=False)
                    mat[ch_indices, :] = 0
                    mat[:, ch_indices] = 0
                
                # Strategy 2: Strong 1/f noise (add low-frequency noise)
                if noise_scale > 0:
                    # Generate 1/f noise (simplified: use low-frequency random noise)
                    noise = torch.randn_like(mat) * noise_scale
                    # Low-pass filter the noise (simplified: simulate with average pooling)
                    kernel_size = max(3, N // 10)
                    if kernel_size > 1 and N > kernel_size:
                        noise_2d = noise.unsqueeze(0).unsqueeze(0)  # [1, 1, N, N]
                        noise_pooled = F.avg_pool2d(
                            noise_2d,
                            kernel_size=kernel_size,
                            stride=1,
                            padding=kernel_size // 2
                        ).squeeze(0).squeeze(0)  # [N', N']
                        # Crop or pad to original size
                        if noise_pooled.shape[0] != N:
                            if noise_pooled.shape[0] > N:
                                noise_pooled = noise_pooled[:N, :N]
                            else:
                                # Pad (simple edge replication)
                                pad_h = N - noise_pooled.shape[0]
                                pad_w = N - noise_pooled.shape[1]
                                noise_pooled = F.pad(noise_pooled, (0, pad_w, 0, pad_h), mode='replicate')
                        noise = noise_pooled
                    mat = mat + noise
            
            corrupted_dict[band] = corrupted
        
        corrupted_scales.append(corrupted_dict)
    
    return corrupted_scales, corruption_mask


def compute_entropy_regularization(msre_distribution: torch.Tensor,
                                   corruption_mask: torch.Tensor,
                                   clean_entropy_weight: float = 1.0,
                                   corrupt_entropy_weight: float = 1.0) -> torch.Tensor:
    """
    Compute entropy regularization: minimize entropy for clean samples (encourage high confidence),
    maximize entropy for corrupted samples (encourage low confidence)
    
    Args:
        msre_distribution: [B, num_class] MSRE distribution (probability distribution after softmax)
        corruption_mask: [B] Boolean tensor, True indicates the sample is corrupted
        clean_entropy_weight: Entropy weight for clean samples (minimize entropy)
        corrupt_entropy_weight: Entropy weight for corrupted samples (maximize entropy)
    
    Returns:
        entropy_reg: Scalar, entropy regularization loss
    """
    # Compute entropy for each sample: H(p) = -sum(p * log(p))
    eps = 1e-8
    entropy = -(msre_distribution * torch.log(msre_distribution + eps)).sum(dim=1)  # [B]
    
    # Separate clean and corrupted samples
    clean_mask = ~corruption_mask
    corrupt_mask = corruption_mask
    
    entropy_reg = torch.zeros((), device=msre_distribution.device, dtype=torch.float32)
    
    # Clean samples: minimize entropy (encourage high confidence)
    if clean_mask.any():
        clean_entropy = entropy[clean_mask].mean()
        entropy_reg = entropy_reg + clean_entropy_weight * clean_entropy
    
    # Corrupted samples: maximize entropy (encourage low confidence) = minimize negative entropy
    if corrupt_mask.any():
        corrupt_entropy = entropy[corrupt_mask].mean()
        # Maximum entropy = log(num_class), we encourage approaching maximum entropy
        max_entropy = np.log(msre_distribution.shape[1])
        # Directly minimize negative entropy (i.e., maximize entropy): entropy_reg += -corrupt_entropy_weight * corrupt_entropy
        # This way when corrupt_entropy is small, gradient will be large, effectively pushing entropy to increase
        entropy_reg = entropy_reg - corrupt_entropy_weight * corrupt_entropy
    
    return entropy_reg


def compute_msre_sensitive_metrics(msre_distribution: torch.Tensor,
                                   corruption_mask: torch.Tensor = None) -> Dict[str, torch.Tensor]:
    """
    Compute MSRE sensitive metrics (entropy, margin, logit gap, etc.) for more accurate observation of MSRE changes
    
    Args:
        msre_distribution: [B, num_class] MSRE distribution (probability distribution after softmax)
        corruption_mask: [B] Boolean tensor, True indicates the sample is corrupted (optional)
    
    Returns:
        metrics: Dictionary containing:
            - entropy: [B] Entropy for each sample
            - entropy_mean: Scalar tensor, average entropy
            - entropy_clean: Scalar tensor, average entropy for clean samples (if mask provided)
            - entropy_corrupt: Scalar tensor, average entropy for corrupted samples (if mask provided)
            - max_prob: [B] Maximum probability for each sample
            - max_prob_mean: Scalar tensor, average maximum probability
            - margin: [B] Margin for each sample (max probability - second max probability)
            - margin_mean: Scalar tensor, average margin
    """
    eps = 1e-8
    B, num_class = msre_distribution.shape
    
    # Compute entropy
    entropy = -(msre_distribution * torch.log(msre_distribution + eps)).sum(dim=1)  # [B]
    entropy_mean = entropy.mean()
    
    # Compute maximum probability
    max_prob, _ = msre_distribution.max(dim=1)  # [B]
    max_prob_mean = max_prob.mean()
    
    # Compute margin (max probability - second max probability)
    sorted_probs, _ = torch.sort(msre_distribution, dim=1, descending=True)
    margin = sorted_probs[:, 0] - sorted_probs[:, 1]  # [B]
    margin_mean = margin.mean()
    
    metrics = {
        "entropy": entropy,
        "entropy_mean": entropy_mean,
        "max_prob": max_prob,
        "max_prob_mean": max_prob_mean,
        "margin": margin,
        "margin_mean": margin_mean,
    }
    
    # If corruption_mask provided, compute metrics separately for clean and corrupted samples
    if corruption_mask is not None:
        clean_mask = ~corruption_mask
        corrupt_mask = corruption_mask
        
        if clean_mask.any():
            metrics["entropy_clean"] = entropy[clean_mask].mean()
            metrics["max_prob_clean"] = max_prob[clean_mask].mean()
            metrics["margin_clean"] = margin[clean_mask].mean()
        
        if corrupt_mask.any():
            metrics["entropy_corrupt"] = entropy[corrupt_mask].mean()
            metrics["max_prob_corrupt"] = max_prob[corrupt_mask].mean()
            metrics["margin_corrupt"] = margin[corrupt_mask].mean()
    
    return metrics
