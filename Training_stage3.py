"""
Stage3: Post-Fusion Training Script
Handles post-fusion training stage
Loads complete fusion model weights from Stage2
Supports 5-fold cross-validation
"""
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Sampler
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import MyModels
from datasets.multimodal_dataset_by_path import MultiModalDatasetCSV as MultiModalDataset
from datasets.multimodal_dataset_by_path import multimodal_collate_flex, make_collate_with_normalizer
from datasets.opec_fmri_normalizer import CopulaNormalizer
import argparse
import datetime
import yaml
from torch.utils.tensorboard import SummaryWriter
from types import SimpleNamespace
from tqdm import tqdm
from utils import (
    setup_logger,
    AlignLambdaScheduler, infer_mode_from_row, bucket_by_mode, build_A_list, build_E_list,
    summarize_val, log_split_summary, _ensure_dir, save_confusion_matrix,
    inject_fmri_noise, inject_eeg_noise, compute_entropy_regularization,
    compute_msre_sensitive_metrics
)
# Optional imports: Contrastive Learning and KD (only used when enabled in config)
try:
    from utils_contrastive import ContrastiveLoss, TripletContrastiveLoss
    HAS_CONTRASTIVE = True
except ImportError:
    HAS_CONTRASTIVE = False
    ContrastiveLoss = None
    TripletContrastiveLoss = None

try:
    from utils_kd import KnowledgeDistillationLoss
    HAS_KD = True
except ImportError:
    HAS_KD = False
    KnowledgeDistillationLoss = None

import warnings
from sklearn.metrics import confusion_matrix

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
normalizer = CopulaNormalizer(alpha=0.9, apply_knn=False)


class BalancedTriModalSampler(Sampler):
    """
    Balanced sampler: ensures 1:1:1 ratio of fmri_only, eeg_only, both in each batch
    
    Note: Input indices are base_index (original CSV row indices), need to map to dataset's expanded_index
    """
    def __init__(self, dataset, fmri_only_indices, eeg_only_indices, both_indices, 
                 batch_size):
        self.dataset = dataset
        self.fmri_only_base_indices = np.array(fmri_only_indices, dtype=int)
        self.eeg_only_base_indices = np.array(eeg_only_indices, dtype=int)
        self.both_base_indices = np.array(both_indices, dtype=int)
        self.batch_size = batch_size
        
        # Calculate number of each type per batch (try to maintain 1:1:1)
        samples_per_type = batch_size // 3
        self.fmri_per_batch = samples_per_type
        self.eeg_per_batch = samples_per_type
        self.both_per_batch = batch_size - 2 * samples_per_type  # Remaining goes to both
        
        # Create index list (map base_index to expanded_index positions)
        self._build_index_list()
    
    def _build_index_list(self):
        """Build balanced index list, map base_index to positions in expanded_index"""
        # Get dataset's expanded_index
        expanded_index = self.dataset.index  # This is expanded_index (numpy array)
        base_index = self.dataset.base_index  # This is base_index (numpy array), corresponding to original CSV row indices
        
        # Create base_index -> expanded_index position mapping
        # expanded_index[i] is a value from base_index
        # We need to find all positions in expanded_index where value equals a base_index
        base_to_expanded_positions = {}
        for pos, base_idx in enumerate(expanded_index):
            if base_idx not in base_to_expanded_positions:
                base_to_expanded_positions[base_idx] = []
            base_to_expanded_positions[base_idx].append(pos)
        
        # Convert base_index to positions in expanded_index
        def base_to_expanded(base_indices):
            expanded_positions = []
            for base_idx in base_indices:
                if base_idx in base_to_expanded_positions:
                    expanded_positions.extend(base_to_expanded_positions[base_idx])
            return np.array(expanded_positions, dtype=int)
        
        fmri_expanded = base_to_expanded(self.fmri_only_base_indices)
        eeg_expanded = base_to_expanded(self.eeg_only_base_indices)
        both_expanded = base_to_expanded(self.both_base_indices)
        
        indices = []
        
        # Calculate how many complete batches are needed
        n_fmri = len(fmri_expanded)
        n_eeg = len(eeg_expanded)
        n_both = len(both_expanded)
        
        # Calculate how many batches each type can generate
        n_batches_fmri = n_fmri // self.fmri_per_batch if self.fmri_per_batch > 0 else 0
        n_batches_eeg = n_eeg // self.eeg_per_batch if self.eeg_per_batch > 0 else 0
        n_batches_both = n_both // self.both_per_batch if self.both_per_batch > 0 else 0
        
        # Take minimum to ensure all three types can provide enough samples
        n_batches = min(n_batches_fmri, n_batches_eeg, n_batches_both)
        
        if n_batches == 0:
            # If cannot form complete batches, add all samples in order
            indices = list(fmri_expanded) + list(eeg_expanded) + list(both_expanded)
        else:
            # Use numpy random generator
            rng = np.random.default_rng()
            
            # Shuffle indices (positions in expanded_index)
            fmri_shuffled = rng.permutation(fmri_expanded)
            eeg_shuffled = rng.permutation(eeg_expanded)
            both_shuffled = rng.permutation(both_expanded)
            
            # Organize by batch
            for i in range(n_batches):
                # Each batch: fmri + eeg + both
                start_fmri = i * self.fmri_per_batch
                end_fmri = start_fmri + self.fmri_per_batch
                start_eeg = i * self.eeg_per_batch
                end_eeg = start_eeg + self.eeg_per_batch
                start_both = i * self.both_per_batch
                end_both = start_both + self.both_per_batch
                
                indices.extend(fmri_shuffled[start_fmri:end_fmri].tolist())
                indices.extend(eeg_shuffled[start_eeg:end_eeg].tolist())
                indices.extend(both_shuffled[start_both:end_both].tolist())
            
            # Add remaining samples
            remaining_fmri = fmri_shuffled[n_batches * self.fmri_per_batch:]
            remaining_eeg = eeg_shuffled[n_batches * self.eeg_per_batch:]
            remaining_both = both_shuffled[n_batches * self.both_per_batch:]
            indices.extend(remaining_fmri.tolist())
            indices.extend(remaining_eeg.tolist())
            indices.extend(remaining_both.tolist())
        
        self.indices = indices
    
    def __iter__(self):
        return iter(self.indices)
    
    def __len__(self):
        return len(self.indices)

def load_stage1_ckpt(model, stage1_fmri_ckpt_path, stage1_eeg_ckpt_path, logger):
    """
    Load fMRI and EEG branch weights from Stage1 checkpoint
    
    Args:
        model: Current model
        stage1_fmri_ckpt_path: Stage1 fMRI best checkpoint path (complete model state_dict, contains fmri_net.*)
        stage1_eeg_ckpt_path: Stage1 EEG best checkpoint path (complete model state_dict, contains eeg_net.*)
        logger: Logger
    """
    # Load fMRI branch
    if stage1_fmri_ckpt_path and os.path.exists(stage1_fmri_ckpt_path):
        ckpt_fmri = torch.load(stage1_fmri_ckpt_path, map_location=device)
        # Stage1 saves complete model state_dict, need to extract fmri_net part
        fmri_state = {k: v for k, v in ckpt_fmri.items() if k.startswith("fmri_net.")}
        if len(fmri_state) > 0:
            model.load_state_dict(fmri_state, strict=False)
            logger.info(f"Loaded fMRI branch from: {stage1_fmri_ckpt_path}")
            logger.info(f"  Loaded {len(fmri_state)} parameters")
        else:
            logger.warning(f"No fmri_net.* keys found in: {stage1_fmri_ckpt_path}")
    else:
        logger.warning(f"Stage1 fMRI ckpt not found: {stage1_fmri_ckpt_path}")
    
    # Load EEG branch
    if stage1_eeg_ckpt_path and os.path.exists(stage1_eeg_ckpt_path):
        ckpt_eeg = torch.load(stage1_eeg_ckpt_path, map_location=device)
        # Stage1 saves complete model state_dict, need to extract eeg_net part
        eeg_state = {k: v for k, v in ckpt_eeg.items() if k.startswith("eeg_net.")}
        if len(eeg_state) > 0:
            model.load_state_dict(eeg_state, strict=False)
            logger.info(f"Loaded EEG branch from: {stage1_eeg_ckpt_path}")
            logger.info(f"  Loaded {len(eeg_state)} parameters")
        else:
            logger.warning(f"No eeg_net.* keys found in: {stage1_eeg_ckpt_path}")
    else:
        logger.warning(f"Stage1 EEG ckpt not found: {stage1_eeg_ckpt_path}")

def run_once(config, save_path, fold, writer=None):
    ds_cfg = SimpleNamespace(**config["dataset"])
    train_cfg = SimpleNamespace(**config["training"])
    
    _ensure_dir(save_path)
    _ensure_dir(os.path.join(save_path, "log"))
    _ensure_dir(os.path.join(save_path, "tb"))
    _ensure_dir(os.path.join(save_path, "bestmodel"))
    
    logger = setup_logger(os.path.join(save_path, f"log/fold{fold}_stage3_post.log"))
    logger.info(f"========== STAGE3: Post-Fusion (Fold {fold+1}/5) ==========")
    
    tb_logdir = os.path.join(save_path, f"tb/tb_fold{fold}")
    os.makedirs(tb_logdir, exist_ok=True)
    tb = SummaryWriter(log_dir=tb_logdir) if writer is None else writer
    
    kfold_dir = ds_cfg.kfold_base_dir
    fold_dir = os.path.join(kfold_dir, f"fold{fold}")
    
    train_all_csv = os.path.join(fold_dir, "train_all.csv")
    val_all_csv = os.path.join(fold_dir, "val_all.csv")
    
    subInfo_train = pd.read_csv(train_all_csv)
    subInfo_val_all = pd.read_csv(val_all_csv)
    logger.info(f"Using validation set from: {val_all_csv}")
    
    labels_train = subInfo_train["diagnosis_code"].to_numpy(dtype=np.int64)
    train_index_base = np.arange(len(subInfo_train), dtype=int)
    
    train_modes_raw = [infer_mode_from_row(subInfo_train.iloc[i]) for i in train_index_base]
    keep_train = [i for i, m in zip(train_index_base, train_modes_raw) if m != "none"]
    train_index_base = np.array(keep_train, dtype=int)
    
    val_modes_all = [infer_mode_from_row(subInfo_val_all.iloc[i]) for i in range(len(subInfo_val_all))]
    keep_val_all = [i for i, m in enumerate(val_modes_all) if m != "none"]
    val_index_all_filtered = np.array(keep_val_all, dtype=int)
    labels_val_all = subInfo_val_all["diagnosis_code"].to_numpy(dtype=np.int64)
    
    logger.info("Stage3 Data Summary:")
    log_split_summary(1, "TRAIN", train_index_base, subInfo_train, labels_train, ds_cfg, logger)
    log_split_summary(1, "VAL",   val_index_all_filtered,  subInfo_val_all,   labels_val_all,   ds_cfg, logger)
    
    # Pre-bucket
    mode_buckets = bucket_by_mode(subInfo_train, train_index_base)
    after_allow_modes = set(getattr(train_cfg, "after_allow_modes", ["fmri_only", "eeg_only", "both"]))
    logger.info(f"Mode buckets: {[(k, len(v)) for k, v in mode_buckets.items()]}")
    
    # Model initialization
    use_msre = bool(getattr(train_cfg, "use_msre", False))
    logger.info(f"[MODEL] MSRE module is {'ENABLED' if use_msre else 'DISABLED'}")
    
    use_corr_align = bool(getattr(train_cfg, "use_corr_align", True))
    corr_align_dim = int(getattr(train_cfg, "corr_align_dim", 16))
    logger.info(f"[MODEL] Correlation Alignment is {'ENABLED' if use_corr_align else 'DISABLED'}")
    
    # Adversarial noise training configuration
    use_adversarial_noise = bool(getattr(train_cfg, "use_adversarial_noise", False))
    corruption_rate = float(getattr(train_cfg, "corruption_rate", 0.4))  # 30-50% samples corrupted
    entropy_reg_lambda = float(getattr(train_cfg, "entropy_reg_lambda", 0.5))  # Entropy regularization weight
    clean_entropy_weight = float(getattr(train_cfg, "clean_entropy_weight", 1.0))
    corrupt_entropy_weight = float(getattr(train_cfg, "corrupt_entropy_weight", 1.0))
    logger.info(f"[ADVERSARIAL NOISE] {'ENABLED' if use_adversarial_noise else 'DISABLED'}")
    if use_adversarial_noise:
        logger.info(f"  corruption_rate={corruption_rate}, entropy_reg_lambda={entropy_reg_lambda}")
        logger.info(f"  clean_entropy_weight={clean_entropy_weight}, corrupt_entropy_weight={corrupt_entropy_weight}")
    
    use_contrastive = bool(getattr(train_cfg, "use_contrastive", False))
    use_kd = bool(getattr(train_cfg, "use_kd", False))
    
    if use_contrastive and not HAS_CONTRASTIVE:
        logger.warning("use_contrastive=True but utils_contrastive not available. Disabling contrastive learning.")
        use_contrastive = False
    
    if use_kd and not HAS_KD:
        logger.warning("use_kd=True but utils_kd not available. Disabling knowledge distillation.")
        use_kd = False
    
    if use_contrastive:
        logger.info("[MODEL] Contrastive Learning is ENABLED (replacing align_loss)")
        contrastive_loss_fn = TripletContrastiveLoss(
            temperature=float(getattr(train_cfg, "contrastive_temperature", 0.07)),
            margin=float(getattr(train_cfg, "contrastive_margin", 1.0))
        )
    else:
        contrastive_loss_fn = None
    
    if use_kd:
        logger.info("[MODEL] Knowledge Distillation is ENABLED (replacing align_loss)")
        # Teacher model will be loaded in training loop
        teacher_model = None
        kd_loss_fn = KnowledgeDistillationLoss(
            alpha=float(getattr(train_cfg, "kd_alpha", 0.5)),
            temperature=float(getattr(train_cfg, "kd_temperature", 4.0))
        )
    else:
        teacher_model = None
        kd_loss_fn = None
    
    model = MyModels.MAHGCNNET(
        ROInum=ds_cfg.scales[0],
        layer=int(ds_cfg.layers),
        num_class=ds_cfg.num_class,
        num_scales=len(ds_cfg.scales),
        fusion_mode=train_cfg.fusion_mode,
        use_msre=use_msre,
        use_corr_align=use_corr_align,
        corr_align_dim=corr_align_dim,
    ).to(device)
    
    # ========= If KD is enabled, load Teacher model =========
    if use_kd:
        stage2_cfg = getattr(train_cfg, "stage2", {})
        teacher_ckpt_path = getattr(stage2_cfg, "best_fusion_ckpt", "") if isinstance(stage2_cfg, SimpleNamespace) else stage2_cfg.get("best_fusion_ckpt", "")
        if teacher_ckpt_path and os.path.exists(teacher_ckpt_path):
            teacher_model = MyModels.MAHGCNNET(
                ROInum=ds_cfg.scales[0],
                layer=int(ds_cfg.layers),
                num_class=ds_cfg.num_class,
                num_scales=len(ds_cfg.scales),
                fusion_mode=train_cfg.fusion_mode,
                use_msre=use_msre,
                use_corr_align=use_corr_align,  # Use same config as student
                corr_align_dim=corr_align_dim,
            ).to(device)

            teacher_state_dict = torch.load(teacher_ckpt_path, map_location=device)
            missing_keys, unexpected_keys = teacher_model.load_state_dict(teacher_state_dict, strict=False)
            if missing_keys:
                logger.warning(f"Teacher model missing keys (will be initialized randomly): {missing_keys[:5]}..." if len(missing_keys) > 5 else f"Teacher model missing keys: {missing_keys}")
            if unexpected_keys:
                logger.warning(f"Teacher model unexpected keys (ignored): {unexpected_keys[:5]}..." if len(unexpected_keys) > 5 else f"Teacher model unexpected keys: {unexpected_keys}")
            teacher_model.eval()  # Freeze Teacher model
            for param in teacher_model.parameters():
                param.requires_grad = False
            logger.info(f"Loaded Teacher model (frozen) from: {teacher_ckpt_path}")
        else:
            logger.warning(f"use_kd=True but Teacher checkpoint not found: {teacher_ckpt_path}. Disabling KD.")
            use_kd = False
            kd_loss_fn = None
    
    # ========= Load complete model weights from Stage2 (priority) =========
    stage2_cfg = getattr(train_cfg, "stage2", {})
    stage2_result_dir = stage2_cfg.get("stage2_result_dir", "results/Stage2")
    stage2_fusion_ckpt_name = stage2_cfg.get("best_fusion_ckpt_name", "bestmodel/best_model_fusion_fold{fold}.pth")
    
    # Resolve checkpoint path
    stage2_fusion_ckpt = os.path.join(stage2_result_dir, f"fold{fold}", stage2_fusion_ckpt_name.format(fold=fold))
    
    stage2_loaded = False
    if stage2_fusion_ckpt and os.path.exists(stage2_fusion_ckpt):
        logger.info(f"Loading Stage2 fusion checkpoint from: {stage2_fusion_ckpt}")
        ckpt = torch.load(stage2_fusion_ckpt, map_location=device)
        
        # Check if Stage2 checkpoint contains complete model weights
        ckpt_keys = set(ckpt.keys())
        has_fmri = any(k.startswith("fmri_net.") for k in ckpt_keys)
        has_eeg = any(k.startswith("eeg_net.") for k in ckpt_keys)
        has_fuser = any(k.startswith("fuser.") for k in ckpt_keys)
        
        logger.info(f"Stage2 checkpoint contains: fmri_net={has_fmri}, eeg_net={has_eeg}, fuser={has_fuser}")
        
        model.load_state_dict(ckpt, strict=False)
        logger.info(f"Loaded {len(ckpt)} parameters from Stage2")
        stage2_loaded = True
    else:
        logger.warning(f"Stage2 fusion ckpt not found: {stage2_fusion_ckpt}")
    
    # ========= Load single-modal branch weights from Stage1 (as backup or supplement) =========
    stage1_cfg = getattr(train_cfg, "stage1", {})
    stage1_result_dir = stage1_cfg.get("stage1_result_dir", "results/Stage1")
    stage1_fmri_ckpt_name = stage1_cfg.get("best_fmri_ckpt_name", "bestmodel/best_model_fmri_fold{fold}.pth")
    stage1_eeg_ckpt_name = stage1_cfg.get("best_eeg_ckpt_name", "bestmodel/best_model_eeg_fold{fold}.pth")
    
    # Resolve checkpoint paths
    stage1_fmri_ckpt = os.path.join(stage1_result_dir, f"fold{fold}", stage1_fmri_ckpt_name.format(fold=fold))
    stage1_eeg_ckpt = os.path.join(stage1_result_dir, f"fold{fold}", stage1_eeg_ckpt_name.format(fold=fold))
    
    if stage1_fmri_ckpt or stage1_eeg_ckpt:
        if stage2_loaded:
            logger.info("Stage1 checkpoints provided but Stage2 already loaded. Stage1 will only fill missing parameters.")
        else:
            logger.info("Loading Stage1 checkpoints (Stage2 not available)...")
        load_stage1_ckpt(model, stage1_fmri_ckpt, stage1_eeg_ckpt, logger)
    else:
        if not stage2_loaded:
            logger.warning("=" * 60)
            logger.warning("WARNING: No Stage1 or Stage2 checkpoints provided!")
            logger.warning("Single-modal branches (fmri_net, eeg_net) will use RANDOM INITIALIZATION.")
            logger.warning("This is NOT recommended for Stage3 training.")
            logger.warning("=" * 60)
        else:
            logger.info("No Stage1 checkpoints provided, relying on Stage2 checkpoint only.")
    
    # Post stage: freeze fusion module and fusion classification head, only train single-modal branches
    freeze_fusion = bool(getattr(train_cfg, "freeze_fusion", True))
    if freeze_fusion:
        logger.info("Freezing fusion module and fusion classification head...")
        for name, p in model.named_parameters():
            # Freeze fusion-related parameters
            if (name.startswith("fuser.bcmf") or name.startswith("fuser.dualcross") or  # Support both old and new names 
                name.startswith("fuser.fmri_proj") or 
                name.startswith("fuser.eeg_proj") or 
                name.startswith("fuser.readout_fused") or
                name.startswith("fuser.cls_fused") or
                name.startswith("fuser.fmri_align_proj") or
                name.startswith("fuser.eeg_align_proj")):
                p.requires_grad = False
            else:
                p.requires_grad = True
        logger.info("Fusion module and fusion classification head are frozen.")
    else:
        logger.info("All parameters are trainable (fusion not frozen).")
        for name, p in model.named_parameters():
            p.requires_grad = True
    
    # Loss function
    eff_train_cnt_base = np.bincount(labels_train[train_index_base], minlength=ds_cfg.num_class)
    if train_cfg.imbalance_loss:
        cw = 1.0 / (eff_train_cnt_base + 1e-8)
        cw = cw / cw.mean()
        cw = np.minimum(np.sqrt(cw), 5.0)
        loss_func = nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float32, device=device))
    else:
        loss_func = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    
    # Alignment loss scheduler (η = align loss weight, paper symbol)
    eta_target = float(getattr(train_cfg, "eta", getattr(train_cfg, "align_lambda", 0.2)))
    eta_ramp = int(getattr(train_cfg, "eta_ramp_epochs", getattr(train_cfg, "align_ramp_epochs", 10)))
    align_sched = AlignLambdaScheduler(
        warmup_epochs_single=0,  # post stage no longer has warmup
        target=eta_target,
        ramp_epochs=eta_ramp
    )
    
    # Best model path (overall)
    best_model_path_post = None
    best_post_metric = -float("inf")
    best_post_epoch = -1
    
    # Best model path (fmri-only)
    best_model_path_post_fmri = None
    best_post_fmri_metric = -float("inf")
    best_post_fmri_epoch = -1
    
    # Best model path (eeg-only)
    best_model_path_post_eeg = None
    best_post_eeg_metric = -float("inf")
    best_post_eeg_epoch = -1
    
    # Training parameters
    epochs = int(getattr(train_cfg, "post_epochs", 40))
    eval_interval = int(getattr(train_cfg, "eval_interval", 5))
    min_save_epoch = int(getattr(train_cfg, "min_epochs_before_saving", 0))
    select_metric = getattr(train_cfg, "select_metric", "auc")
    
    # Validation DataLoader (only use val_csv, no oversampling for validation)
    test_dataset_all = MultiModalDataset(
        subInfo_val_all, val_index_all_filtered, ds_cfg.scales, mode="both",
        labels_np=labels_val_all, num_class=ds_cfg.num_class,
        normalizer=normalizer
    )
    
    val_loader_to_use = DataLoader(
        test_dataset_all,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        num_workers=train_cfg.num_workers,
        collate_fn=make_collate_with_normalizer(
            normalizer, base_collate=multimodal_collate_flex, phase="transform",
            apply_random_mask=False,
        ),
        pin_memory=True
    )
    
    use_tqdm_train = bool(getattr(train_cfg, "use_tqdm", True))
    global_step = 0
    
    # Early stopping
    early_stopped = False
    drop_streak = 0
    last_val_acc = None
    patience = int(getattr(train_cfg, "early_stop_patience", 10))
    use_early_stop = bool(getattr(train_cfg, "early_stop_on_drop", True))
    logger.info("Early stopping is " + ("ON" if use_early_stop else "OFF"))
    
    # ======= Epoch Loop =======
    for epoch in tqdm(range(epochs), desc=f"Stage3-fold{fold+1} epochs", disable=not use_tqdm_train):
        train_loss_sum = 0.0
        train_loss_n   = 0
        fmri_only_loss_sum = 0.0
        fmri_only_cnt      = 0
        eeg_only_loss_sum  = 0.0
        eeg_only_cnt       = 0
        train_correct = 0
        train_total   = 0
        train_correct_both = 0
        train_total_both = 0
        train_correct_f = 0
        train_total_f = 0
        train_correct_e = 0
        train_total_e = 0
        
        # For accumulating MSRE values during training (only both modality)
        train_msre_a_list = []
        train_msre_b_list = []
        train_corruption_mask_a_list = []  # Only for msre_a (when fMRI is corrupted)
        train_corruption_mask_b_list = []  # Only for msre_b (when EEG is corrupted)
        
        eta_now = float(align_sched(epoch))
        model.train()
        
        # Select training data (post stage allows all modalities)
        allow = after_allow_modes
        selected = [mode_buckets[m] for m in allow if m in mode_buckets and mode_buckets[m].size > 0]
        if len(selected) == 0:
            selected = [
                mode_buckets.get("fmri_only", np.array([], dtype=int)),
                mode_buckets.get("eeg_only",  np.array([], dtype=int)),
                mode_buckets.get("both",      np.array([], dtype=int))
            ]
        if any(len(x) > 0 for x in selected):
            train_index_epoch = np.unique(np.concatenate([x for x in selected if x.size > 0]))
        else:
            train_index_epoch = np.array([], dtype=int)
        
        if train_index_epoch.size == 0:
            logger.warning(f"[Epoch {epoch+1}] No samples. Skip.")
            continue
        
        # Separate three types of indices
        fmri_only_indices = mode_buckets.get("fmri_only", np.array([], dtype=int))
        eeg_only_indices = mode_buckets.get("eeg_only", np.array([], dtype=int))
        both_indices = mode_buckets.get("both", np.array([], dtype=int))
        
        # Filter: only keep indices in train_index_epoch
        fmri_only_indices = np.intersect1d(fmri_only_indices, train_index_epoch)
        eeg_only_indices = np.intersect1d(eeg_only_indices, train_index_epoch)
        both_indices = np.intersect1d(both_indices, train_index_epoch)
        
        # Check if there is enough data
        if len(fmri_only_indices) == 0 and len(eeg_only_indices) == 0 and len(both_indices) == 0:
            logger.warning(f"[Epoch {epoch+1}] No samples after filtering. Skip.")
            continue
        
        # Use balanced sampling
        use_balanced_sampling = bool(getattr(train_cfg, "use_balanced_sampling", True))
        
        train_dataset = MultiModalDataset(
            subInfo_train, train_index_epoch, ds_cfg.scales, mode="both",
            labels_np=labels_train, num_class=ds_cfg.num_class,
            normalizer=normalizer, pair_expand_by_label={0: 50, 1: 10}
        )
        
        if use_balanced_sampling and len(fmri_only_indices) > 0 and len(eeg_only_indices) > 0 and len(both_indices) > 0:
            # Use balanced sampler
            # Note: sampler returns indices should be base_index (original CSV row indices)
            # Dataset will automatically expand indices based on expand_map internally
            balanced_sampler = BalancedTriModalSampler(
                train_dataset,
                fmri_only_indices.tolist(),
                eeg_only_indices.tolist(),
                both_indices.tolist(),
                train_cfg.batch_size
            )
            train_loader = DataLoader(
                train_dataset,
                batch_size=train_cfg.batch_size,
                sampler=balanced_sampler,
                num_workers=train_cfg.num_workers,
                collate_fn=make_collate_with_normalizer(
                    normalizer, base_collate=multimodal_collate_flex, phase="transform",
                    apply_random_mask=True,
                    mask_threshold=0.1,
                    mask_prob=0.5,
                    mask_modalities=("fmri", "eeg")
                ),
                pin_memory=True,
                drop_last=False
            )
            if epoch == 0:
                logger.info(f"[Balanced Sampling] Base indices - fmri_only={len(fmri_only_indices)}, eeg_only={len(eeg_only_indices)}, both={len(both_indices)}")
                logger.info(f"[Balanced Sampling] Each batch: ~{balanced_sampler.fmri_per_batch} fmri + ~{balanced_sampler.eeg_per_batch} eeg + ~{balanced_sampler.both_per_batch} both")
                logger.info(f"[Balanced Sampling] Total expanded samples in sampler: {len(balanced_sampler)}")
        else:
            # Use normal random sampling
            train_loader = DataLoader(
                train_dataset,
                batch_size=train_cfg.batch_size,
                shuffle=True,
                num_workers=train_cfg.num_workers,
                collate_fn=make_collate_with_normalizer(
                    normalizer, base_collate=multimodal_collate_flex, phase="transform",
                    apply_random_mask=True,
                    mask_threshold=0.1,
                    mask_prob=0.5,
                    mask_modalities=("fmri", "eeg")
                ),
                pin_memory=True,
                drop_last=False
            )
            if epoch == 0:
                logger.info(f"[Random Sampling] Total samples={len(train_index_epoch)}")
        
        # Training loop
        for step, batch in enumerate(
            tqdm(train_loader, desc=f"Train E{epoch+1}", leave=False, disable=not use_tqdm_train),
            1
        ):
            fmri_data, eeg_data, b_y, fmri_idx, eeg_idx, both_f_idx, both_e_idx = batch
            b_y = b_y.view(-1).long().to(device)
            
            both_has = (len(both_f_idx) > 0) and (len(both_e_idx) > 0)
            y_both = b_y[fmri_idx[both_f_idx]] if both_has else None
            
            # fmri_only / eeg_only indices
            if len(fmri_idx) > 0:
                mask_fmri_only = torch.ones(len(fmri_idx), dtype=torch.bool)
                if len(both_f_idx) > 0:
                    mask_fmri_only[both_f_idx] = False
                fmri_only_idx_in_f = torch.nonzero(mask_fmri_only, as_tuple=False).view(-1)
                y_fmri_only = b_y[fmri_idx[fmri_only_idx_in_f]] if len(fmri_only_idx_in_f) > 0 else None
            else:
                fmri_only_idx_in_f, y_fmri_only = torch.empty(0, dtype=torch.long), None
            
            if len(eeg_idx) > 0:
                mask_eeg_only = torch.ones(len(eeg_idx), dtype=torch.bool)
                if len(both_e_idx) > 0:
                    mask_eeg_only[both_e_idx] = False
                eeg_only_idx_in_e = torch.nonzero(mask_eeg_only, as_tuple=False).view(-1)
                y_eeg_only = b_y[eeg_idx[eeg_only_idx_in_e]] if len(eeg_only_idx_in_e) > 0 else None
            else:
                eeg_only_idx_in_e, y_eeg_only = torch.empty(0, dtype=torch.long), None
            
            loss_parts, cnt_parts = [], []
            msre_loss = torch.zeros((), device=device, dtype=torch.float32)
            
            # (A) both
            if both_has:
                A_list_both = build_A_list(fmri_data, ds_cfg.scales, idx=both_f_idx, device=device)
                E_list_both = build_E_list(eeg_data, ds_cfg.scales, idx=both_e_idx, device=device)
                
                # ========== Adversarial noise training: randomly select samples and inject noise ==========
                corruption_mask = None
                corrupt_modality = None
                if use_adversarial_noise and use_msre:
                    # Randomly select modality to corrupt (EEG or fMRI)
                    corrupt_modality = np.random.choice(["fmri", "eeg"])
                    
                    if corrupt_modality == "fmri":
                        # Inject noise into fMRI
                        A_list_both, corruption_mask = inject_fmri_noise(
                            A_list_both,
                            corruption_rate=corruption_rate,
                            roi_dropout_rate=0.3,
                            block_mask_size=10,
                            block_mask_prob=0.5
                        )
                    else:  # eeg
                        # Inject noise into EEG
                        E_list_both, corruption_mask = inject_eeg_noise(
                            E_list_both,
                            corruption_rate=corruption_rate,
                            channel_dropout_rate=0.3,
                            noise_scale=2.0
                        )
                
                model_output = model(fmri_scales=A_list_both, eeg_scales=E_list_both)
                # Unpack model output: now includes g_layers_* for scale-wise alignment
                if use_msre and len(model_output) >= 12:
                    out_fmri, out_eeg, out_fused, align_loss, fused_cons, fmri_feat, eeg_feat, msre_info, corr_align_loss, g_layers_fmri, g_layers_eeg, g_layers_fuse = model_output
                elif len(model_output) >= 12:
                    out_fmri, out_eeg, out_fused, align_loss, fused_cons, fmri_feat, eeg_feat, msre_info, corr_align_loss, g_layers_fmri, g_layers_eeg, g_layers_fuse = model_output
                    msre_info = None
                elif len(model_output) >= 9:
                    # Backward compatibility: old format without g_layers_*
                    out_fmri, out_eeg, out_fused, align_loss, fused_cons, fmri_feat, eeg_feat, msre_info, corr_align_loss = model_output
                    g_layers_fmri, g_layers_eeg, g_layers_fuse = None, None, None
                    msre_info = None if msre_info is None else msre_info
                else:
                    out_fmri, out_eeg, out_fused, align_loss, *_ = model_output
                    fused_cons, fmri_feat, eeg_feat = None, None, None
                    msre_info = None
                    corr_align_loss = torch.zeros((), device=device, dtype=torch.float32)
                    g_layers_fmri, g_layers_eeg, g_layers_fuse = None, None, None
                
                # Post stage: CE(single-modal) + align_loss / contrastive_loss / kd_loss
                ce_parts = []
                if out_fmri is not None:
                    ce_parts.append(loss_func(out_fmri, y_both))
                if out_eeg is not None:
                    ce_parts.append(loss_func(out_eeg, y_both))
                cls_loss = torch.stack(ce_parts).mean() if len(ce_parts) > 0 else torch.zeros((), device=device)
                
                # Select which alignment loss to use based on config
                if use_kd and teacher_model is not None:
                    # Knowledge Distillation: Student (EEG) learns from Teacher's output distribution
                    with torch.no_grad():
                        teacher_output = teacher_model(fmri_scales=A_list_both, eeg_scales=E_list_both)
                        if len(teacher_output) >= 3:
                            teacher_logits = teacher_output[2]  # out_fused
                        else:
                            teacher_logits = None
                    
                    if out_eeg is not None and teacher_logits is not None:
                        kd_loss, kd_ce_loss, kd_kl_loss = kd_loss_fn(out_eeg, teacher_logits, y_both)
                        loss_both = kd_loss
                        if tb is not None:
                            tb.add_scalar("loss/kd_ce", float(kd_ce_loss.item()), global_step)
                            tb.add_scalar("loss/kd_kl", float(kd_kl_loss.item()), global_step)
                    else:
                        loss_both = cls_loss
                
                elif use_contrastive and contrastive_loss_fn is not None:
                    # Contrastive Learning: use TripletContrastiveLoss
                    if fmri_feat is not None and eeg_feat is not None and fused_cons is not None:
                        contrastive_loss = contrastive_loss_fn(fmri_feat, eeg_feat, fused_cons, y_both)
                        loss_both = cls_loss + eta_now * contrastive_loss
                        if tb is not None:
                            tb.add_scalar("loss/contrastive", float(contrastive_loss.item()), global_step)
                    else:
                        # If features are not available, fall back to original align_loss
                        if train_cfg.fusion_mode in ("align_loss", "bcmf", "dual_cross") and (align_loss is not None):  # Support both old and new names
                            loss_both = cls_loss + eta_now * align_loss
                        else:
                            loss_both = cls_loss
                
                else:
                    # Default: use scale-wise alignment loss (paper: L_align^m = Σ_s || g^m_s - g^{fuse}_s ||_2^2)
                    # Only compute on paired batch (both modalities available)
                    scale_wise_align_loss = torch.zeros((), device=device, dtype=torch.float32)
                    if g_layers_fmri is not None and g_layers_eeg is not None and g_layers_fuse is not None:
                        # Paper: L_align^m = Σ_s || g^m_s - g^{fuse}_s ||_2^2 for m ∈ {f, e}
                        # g_layers_fmri: list of [B, hidden_dim] - graph-level embeddings (g^f_s)
                        # g_layers_eeg: list of [B, hidden_dim] - graph-level embeddings (g^e_s)
                        # g_layers_fuse: list of [B, hidden_dim] - graph-level embeddings (g^{fuse}_s)
                        align_parts = []
                        for s in range(len(g_layers_fuse)):
                            g_f_s = g_layers_fmri[s]  # [B, hidden_dim]
                            g_e_s = g_layers_eeg[s]   # [B, hidden_dim]
                            g_fuse_s = g_layers_fuse[s].detach()  # [B, hidden_dim] - teacher (stop gradient)
                            
                            # L2 alignment: || g^f_s - g^{fuse}_s ||_2^2 + || g^e_s - g^{fuse}_s ||_2^2
                            align_f = F.mse_loss(g_f_s, g_fuse_s)  # Mean over batch and features
                            align_e = F.mse_loss(g_e_s, g_fuse_s)  # Mean over batch and features
                            align_parts.append(align_f + align_e)
                        
                        if len(align_parts) > 0:
                            scale_wise_align_loss = torch.stack(align_parts).sum()  # Sum over scales
                        
                        if tb is not None:
                            tb.add_scalar("loss/scale_wise_align", float(scale_wise_align_loss.item()), global_step)
                    
                    # Use scale-wise alignment if available, otherwise fall back to cat-level
                    if scale_wise_align_loss.item() > 0:
                        loss_both = cls_loss + eta_now * scale_wise_align_loss
                    elif train_cfg.fusion_mode in ("align_loss", "dual_cross") and (align_loss is not None):
                        loss_both = cls_loss + eta_now * align_loss
                    else:
                        loss_both = cls_loss
                
                # Correlation alignment loss (if enabled)
                corr_align_lambda = float(getattr(train_cfg, "corr_align_lambda", 0.1))
                corr_align_loss_weighted = corr_align_lambda * corr_align_loss if corr_align_loss is not None else torch.zeros((), device=device)
                
                # ========== Entropy regularization term (adversarial noise training) ==========
                entropy_reg_loss = torch.zeros((), device=device, dtype=torch.float32)
                if use_adversarial_noise and use_msre and msre_info is not None and corruption_mask is not None and corrupt_modality is not None:
                    if isinstance(msre_info, list) and len(msre_info) > 0:
                        msre_info_last = msre_info[-1]
                    elif isinstance(msre_info, dict):
                        msre_info_last = msre_info
                    else:
                        msre_info_last = None
                    
                    if msre_info_last is not None:
                        # Select corresponding MSRE distribution based on corrupted modality
                        if corrupt_modality == "fmri":
                            msre_dist = msre_info_last.get("msre_a", None)  # fMRI's MSRE (fMRI is corrupted, should decrease)
                        else:  # eeg
                            msre_dist = msre_info_last.get("msre_b", None)  # EEG's MSRE (EEG is corrupted, should decrease)
                        
                        if msre_dist is not None:
                            # Ensure msre_dist is a probability distribution (softmax)
                            if msre_dist.dim() == 2:  # [B, num_class]
                                # If already a probability distribution (sum=1), use directly; otherwise softmax
                                msre_sum = msre_dist.sum(dim=1, keepdim=True)
                                if not torch.allclose(msre_sum, torch.ones_like(msre_sum), atol=1e-3):
                                    msre_dist = F.softmax(msre_dist, dim=1)
                                
                                # Compute entropy regularization term
                                entropy_reg_loss = compute_entropy_regularization(
                                    msre_dist,
                                    corruption_mask,
                                    clean_entropy_weight=clean_entropy_weight,
                                    corrupt_entropy_weight=corrupt_entropy_weight
                                )
                                
                                if tb is not None:
                                    tb.add_scalar("loss/entropy_reg", float(entropy_reg_loss.item()), global_step)
                                    # Record weighted entropy regularization term
                                    tb.add_scalar("loss/entropy_reg_weighted", float(entropy_reg_lambda * entropy_reg_loss.item()), global_step)
                
                # MSRE loss
                if use_msre and msre_info is not None:
                    if isinstance(msre_info, list) and len(msre_info) > 0:
                        msre_info_last = msre_info[-1]
                    elif isinstance(msre_info, dict):
                        msre_info_last = msre_info
                    else:
                        msre_info_last = None
                    
                    if msre_info_last is not None:
                        logits_f_a = msre_info_last.get("logits_f_a", None)
                        logits_f_b = msre_info_last.get("logits_f_b", None)
                        
                        msre_ce_parts = []
                        if logits_f_a is not None:
                            msre_ce_parts.append(loss_func(logits_f_a, y_both))
                        if logits_f_b is not None:
                            msre_ce_parts.append(loss_func(logits_f_b, y_both))
                        
                        if len(msre_ce_parts) > 0:
                            msre_ce_loss = torch.stack(msre_ce_parts).mean()
                            msre_lambda = float(getattr(train_cfg, "msre_lambda", 1.0))
                            msre_reg_lambda = float(getattr(train_cfg, "msre_reg_lambda", 0.1))
                            
                            # MSRE regularization term
                            msre_a = msre_info_last.get("msre_a", None)
                            msre_b = msre_info_last.get("msre_b", None)
                            msre_reg_parts = []
                            if msre_a is not None:
                                entropy_a = -(msre_a * torch.log(msre_a + 1e-8)).sum(dim=1).mean()
                                msre_reg_parts.append(F.relu(0.1 - entropy_a))
                            if msre_b is not None:
                                entropy_b = -(msre_b * torch.log(msre_b + 1e-8)).sum(dim=1).mean()
                                msre_reg_parts.append(F.relu(0.1 - entropy_b))
                            
                            msre_reg_loss = torch.stack(msre_reg_parts).mean() if len(msre_reg_parts) > 0 else torch.zeros((), device=device)
                            msre_loss = msre_lambda * msre_ce_loss + msre_reg_lambda * msre_reg_loss
                            
                            # Record MSRE values
                            msre_weight_a = msre_info_last.get("msre_weight_a", None)
                            msre_weight_b = msre_info_last.get("msre_weight_b", None)
                            
                            # Accumulate MSRE values during training (for statistics at end of epoch, only both modality)
                            # Note: corruption_mask is only valid for the corrupted modality
                            msre_a = msre_info_last.get("msre_a", None)
                            msre_b = msre_info_last.get("msre_b", None)
                            
                            if msre_a is not None:
                                train_msre_a_list.append(msre_a.detach().cpu())
                                # Only use corruption_mask for msre_a when fMRI is corrupted
                                if corruption_mask is not None and corrupt_modality == "fmri":
                                    train_corruption_mask_a_list.append(corruption_mask.detach().cpu())
                                else:
                                    # If fMRI is not corrupted, add all-False mask
                                    batch_size = msre_a.shape[0]
                                    train_corruption_mask_a_list.append(torch.zeros(batch_size, dtype=torch.bool))
                            
                            if msre_b is not None:
                                train_msre_b_list.append(msre_b.detach().cpu())
                                # Only use corruption_mask for msre_b when EEG is corrupted
                                if corruption_mask is not None and corrupt_modality == "eeg":
                                    train_corruption_mask_b_list.append(corruption_mask.detach().cpu())
                                else:
                                    # If EEG is not corrupted, add all-False mask
                                    batch_size = msre_b.shape[0]
                                    train_corruption_mask_b_list.append(torch.zeros(batch_size, dtype=torch.bool))
                            
                            if tb is not None:
                                if msre_a is not None:  # Modality: fmri
                                    # Record sensitive metrics (entropy, margin, etc.)
                                    metrics_a = compute_msre_sensitive_metrics(msre_a, corruption_mask)
                                    tb.add_scalar("msre/msre_a_max_prob", float(metrics_a['max_prob_mean'].item()), global_step)
                                    tb.add_scalar("msre/msre_a_entropy", float(metrics_a['entropy_mean'].item()), global_step)
                                    tb.add_scalar("msre/msre_a_margin", float(metrics_a['margin_mean'].item()), global_step)
                                    
                                    # If corruption_mask is provided, separately record metrics for clean and corrupted samples
                                    if corruption_mask is not None:
                                        if "entropy_clean" in metrics_a:
                                            tb.add_scalar("msre/msre_a_entropy_clean", float(metrics_a['entropy_clean'].item()), global_step)
                                            tb.add_scalar("msre/msre_a_max_prob_clean", float(metrics_a['max_prob_clean'].item()), global_step)
                                            tb.add_scalar("msre/msre_a_margin_clean", float(metrics_a['margin_clean'].item()), global_step)
                                        if "entropy_corrupt" in metrics_a:
                                            tb.add_scalar("msre/msre_a_entropy_corrupt", float(metrics_a['entropy_corrupt'].item()), global_step)
                                            tb.add_scalar("msre/msre_a_max_prob_corrupt", float(metrics_a['max_prob_corrupt'].item()), global_step)
                                            tb.add_scalar("msre/msre_a_margin_corrupt", float(metrics_a['margin_corrupt'].item()), global_step)
                                
                                if msre_b is not None:  # Modality: eeg
                                    # Record sensitive metrics (entropy, margin, etc.)
                                    metrics_b = compute_msre_sensitive_metrics(msre_b, corruption_mask)
                                    tb.add_scalar("msre/msre_b_max_prob", float(metrics_b['max_prob_mean'].item()), global_step)
                                    tb.add_scalar("msre/msre_b_entropy", float(metrics_b['entropy_mean'].item()), global_step)
                                    tb.add_scalar("msre/msre_b_margin", float(metrics_b['margin_mean'].item()), global_step)
                                    
                                    # If corruption_mask is provided, separately record metrics for clean and corrupted samples
                                    if corruption_mask is not None:
                                        if "entropy_clean" in metrics_b:
                                            tb.add_scalar("msre/msre_b_entropy_clean", float(metrics_b['entropy_clean'].item()), global_step)
                                            tb.add_scalar("msre/msre_b_max_prob_clean", float(metrics_b['max_prob_clean'].item()), global_step)
                                            tb.add_scalar("msre/msre_b_margin_clean", float(metrics_b['margin_clean'].item()), global_step)
                                        if "entropy_corrupt" in metrics_b:
                                            tb.add_scalar("msre/msre_b_entropy_corrupt", float(metrics_b['entropy_corrupt'].item()), global_step)
                                            tb.add_scalar("msre/msre_b_max_prob_corrupt", float(metrics_b['max_prob_corrupt'].item()), global_step)
                                            tb.add_scalar("msre/msre_b_margin_corrupt", float(metrics_b['margin_corrupt'].item()), global_step)
                                
                                if msre_weight_a is not None:
                                    tb.add_scalar("msre/msre_weight_a_mean", float(msre_weight_a.mean().item()), global_step)
                                if msre_weight_b is not None:
                                    tb.add_scalar("msre/msre_weight_b_mean", float(msre_weight_b.mean().item()), global_step)
                            
                            # Record MSRE values for all layers (if there are multiple layers)
                            if isinstance(msre_info, list) and len(msre_info) > 1:
                                for layer_idx, layer_msre in enumerate(msre_info):
                                    layer_msre_a = layer_msre.get("msre_a", None)
                                    layer_msre_b = layer_msre.get("msre_b", None)
                                    if tb is not None:
                                        if layer_msre_a is not None:
                                            layer_msre_a_max = layer_msre_a.max(dim=1)[0].mean()
                                            tb.add_scalar(f"msre/layer{layer_idx}_msre_a_max", float(layer_msre_a_max.item()), global_step)
                                        if layer_msre_b is not None:
                                            layer_msre_b_max = layer_msre_b.max(dim=1)[0].mean()
                                            tb.add_scalar(f"msre/layer{layer_idx}_msre_b_max", float(layer_msre_b_max.item()), global_step)
                
                # Statistics
                with torch.no_grad():
                    if out_fused is not None and train_cfg.fusion_mode in ("align_loss", "bcmf", "dual_cross"):  # Support both old and new names
                        out_main = out_fused
                    else:
                        out_main = out_fmri if out_fmri is not None else out_eeg
                    
                    pred_main = torch.argmax(out_main, dim=1)
                    correct_main = (pred_main == y_both).sum().item()
                    n_both_now = int(y_both.shape[0])
                    train_correct += correct_main
                    train_total += n_both_now
                    train_correct_both += correct_main
                    train_total_both += n_both_now
                
                loss_parts.append(loss_both + msre_loss + corr_align_loss_weighted + entropy_reg_lambda * entropy_reg_loss)
                cnt_parts.append(int(y_both.shape[0]))
            
            # (B) fmri_only
            if len(fmri_only_idx_in_f) > 0:
                A_list_f = build_A_list(fmri_data, ds_cfg.scales, idx=fmri_only_idx_in_f, device=device)
                out_fmri, *_ = model(fmri_scales=A_list_f, fmri_only=True, eeg_only=False)
                
                with torch.no_grad():
                    pred_f = torch.argmax(out_fmri, dim=1)
                    correct_f = (pred_f == y_fmri_only).sum().item()
                    n_f_now = int(y_fmri_only.shape[0])
                    train_correct += correct_f
                    train_total += n_f_now
                    train_correct_f += correct_f
                    train_total_f += n_f_now
                
                loss_f = loss_func(out_fmri, y_fmri_only)
                loss_parts.append(loss_f * train_cfg.fmri_only_loss_weight)
                cnt_parts.append(int(y_fmri_only.shape[0]))
                
                fmri_only_loss_sum += float(loss_f.item()) * int(y_fmri_only.shape[0])
                fmri_only_cnt      += int(y_fmri_only.shape[0])
            
            # (C) eeg_only
            if len(eeg_only_idx_in_e) > 0:
                E_list_e = build_E_list(eeg_data, ds_cfg.scales, idx=eeg_only_idx_in_e, device=device)
                _, out_eeg, *_ = model(eeg_scales=E_list_e, eeg_only=True, fmri_only=False)
                
                with torch.no_grad():
                    pred_e = torch.argmax(out_eeg, dim=1)
                    correct_e = (pred_e == y_eeg_only).sum().item()
                    n_e_now = int(y_eeg_only.shape[0])
                    train_correct += correct_e
                    train_total += n_e_now
                    train_correct_e += correct_e
                    train_total_e += n_e_now
                
                loss_e = loss_func(out_eeg, y_eeg_only)
                loss_parts.append(loss_e * train_cfg.eeg_only_loss_weight)
                cnt_parts.append(int(y_eeg_only.shape[0]))
                
                eeg_only_loss_sum += float(loss_e.item()) * int(y_eeg_only.shape[0])
                eeg_only_cnt      += int(y_eeg_only.shape[0])
            
            if len(cnt_parts) == 0:
                continue
            
            w = torch.tensor(cnt_parts, dtype=torch.float32, device=device)
            w = w / w.sum()
            loss_total = torch.stack(loss_parts).mul(w).sum()
            
            optimizer.zero_grad(set_to_none=True)
            loss_total.backward()
            optimizer.step()
            
            train_loss_sum += float(loss_total.item())
            train_loss_n   += 1
            
            if tb is not None:
                global_step += 1
                tb.add_scalar("iter/train_loss", float(loss_total.item()), global_step)
                if use_msre and msre_loss.item() > 0:
                    tb.add_scalar("iter/msre_loss", float(msre_loss.item()), global_step)
        
        # Training loss logging
        if tb is not None and train_loss_n > 0:
            tb.add_scalar("loss/train", train_loss_sum / max(1, train_loss_n), epoch + 1)
        if tb is not None and train_total > 0:
            tb.add_scalar("metrics/train_acc", float(train_correct) / float(train_total), epoch + 1)
        
        # ================= Training phase MSRE statistics (only both modality) =================
        if use_msre and len(train_msre_a_list) > 0:
            train_msre_a_all = torch.cat(train_msre_a_list, dim=0)  # (N, num_class)
            train_corruption_mask_a_all = None
            if len(train_corruption_mask_a_list) > 0:
                train_corruption_mask_a_all = torch.cat(train_corruption_mask_a_list, dim=0)
            
            metrics_a_train = compute_msre_sensitive_metrics(train_msre_a_all, train_corruption_mask_a_all)
            
            logger.info(f"[Epoch {epoch+1} MSRE-TRAIN] msre_a: max_prob={metrics_a_train['max_prob_mean']:.4f}, "
                      f"entropy={metrics_a_train['entropy_mean']:.4f}, margin={metrics_a_train['margin_mean']:.4f}")
            
            if train_corruption_mask_a_all is not None:
                if "entropy_clean" in metrics_a_train:
                    logger.info(f"  msre_a clean: entropy={metrics_a_train['entropy_clean']:.4f}, "
                              f"max_prob={metrics_a_train['max_prob_clean']:.4f}, "
                              f"margin={metrics_a_train['margin_clean']:.4f}")
                if "entropy_corrupt" in metrics_a_train:
                    logger.info(f"  msre_a corrupt: entropy={metrics_a_train['entropy_corrupt']:.4f}, "
                              f"max_prob={metrics_a_train['max_prob_corrupt']:.4f}, "
                              f"margin={metrics_a_train['margin_corrupt']:.4f}")
            
            if tb is not None:
                tb.add_scalar("msre_train/msre_a_max_prob", float(metrics_a_train['max_prob_mean'].item()), epoch + 1)
                tb.add_scalar("msre_train/msre_a_entropy", float(metrics_a_train['entropy_mean'].item()), epoch + 1)
                tb.add_scalar("msre_train/msre_a_margin", float(metrics_a_train['margin_mean'].item()), epoch + 1)
                if train_corruption_mask_a_all is not None:
                    if "entropy_clean" in metrics_a_train:
                        tb.add_scalar("msre_train/msre_a_entropy_clean", float(metrics_a_train['entropy_clean'].item()), epoch + 1)
                        tb.add_scalar("msre_train/msre_a_max_prob_clean", float(metrics_a_train['max_prob_clean'].item()), epoch + 1)
                        tb.add_scalar("msre_train/msre_a_margin_clean", float(metrics_a_train['margin_clean'].item()), epoch + 1)
                    if "entropy_corrupt" in metrics_a_train:
                        tb.add_scalar("msre_train/msre_a_entropy_corrupt", float(metrics_a_train['entropy_corrupt'].item()), epoch + 1)
                        tb.add_scalar("msre_train/msre_a_max_prob_corrupt", float(metrics_a_train['max_prob_corrupt'].item()), epoch + 1)
                        tb.add_scalar("msre_train/msre_a_margin_corrupt", float(metrics_a_train['margin_corrupt'].item()), epoch + 1)
        
        if use_msre and len(train_msre_b_list) > 0:
            train_msre_b_all = torch.cat(train_msre_b_list, dim=0)
            train_corruption_mask_b_all = None
            if len(train_corruption_mask_b_list) > 0:
                train_corruption_mask_b_all = torch.cat(train_corruption_mask_b_list, dim=0)
            
            metrics_b_train = compute_msre_sensitive_metrics(train_msre_b_all, train_corruption_mask_b_all)
            
            logger.info(f"[Epoch {epoch+1} MSRE-TRAIN] msre_b: max_prob={metrics_b_train['max_prob_mean']:.4f}, "
                      f"entropy={metrics_b_train['entropy_mean']:.4f}, margin={metrics_b_train['margin_mean']:.4f}")
            
            if train_corruption_mask_b_all is not None:
                if "entropy_clean" in metrics_b_train:
                    logger.info(f"  msre_b clean: entropy={metrics_b_train['entropy_clean']:.4f}, "
                              f"max_prob={metrics_b_train['max_prob_clean']:.4f}, "
                              f"margin={metrics_b_train['margin_clean']:.4f}")
                if "entropy_corrupt" in metrics_b_train:
                    logger.info(f"  msre_b corrupt: entropy={metrics_b_train['entropy_corrupt']:.4f}, "
                              f"max_prob={metrics_b_train['max_prob_corrupt']:.4f}, "
                              f"margin={metrics_b_train['margin_corrupt']:.4f}")
            
            if tb is not None:
                tb.add_scalar("msre_train/msre_b_max_prob", float(metrics_b_train['max_prob_mean'].item()), epoch + 1)
                tb.add_scalar("msre_train/msre_b_entropy", float(metrics_b_train['entropy_mean'].item()), epoch + 1)
                tb.add_scalar("msre_train/msre_b_margin", float(metrics_b_train['margin_mean'].item()), epoch + 1)
                if train_corruption_mask_b_all is not None:
                    if "entropy_clean" in metrics_b_train:
                        tb.add_scalar("msre_train/msre_b_entropy_clean", float(metrics_b_train['entropy_clean'].item()), epoch + 1)
                        tb.add_scalar("msre_train/msre_b_max_prob_clean", float(metrics_b_train['max_prob_clean'].item()), epoch + 1)
                        tb.add_scalar("msre_train/msre_b_margin_clean", float(metrics_b_train['margin_clean'].item()), epoch + 1)
                    if "entropy_corrupt" in metrics_b_train:
                        tb.add_scalar("msre_train/msre_b_entropy_corrupt", float(metrics_b_train['entropy_corrupt'].item()), epoch + 1)
                        tb.add_scalar("msre_train/msre_b_max_prob_corrupt", float(metrics_b_train['max_prob_corrupt'].item()), epoch + 1)
                        tb.add_scalar("msre_train/msre_b_margin_corrupt", float(metrics_b_train['margin_corrupt'].item()), epoch + 1)
        
        # ================= Validation =================
        if (epoch % eval_interval == 0) or (epoch == epochs - 1):
            model.eval()
            
            predicted_all, test_y_all, prob_all = [], [], []
            fmri_only_pred, fmri_only_y, fmri_only_probs = [], [], []
            eeg_only_pred,  eeg_only_y,  eeg_only_probs  = [], [], []
            paired_y = []
            paired_fused_pred, paired_fused_probs = [], []
            
            # For accumulating MSRE values during validation
            val_msre_a_list = []
            val_msre_b_list = []
            val_msre_weight_a_list = []
            val_msre_weight_b_list = []
            
            with torch.no_grad():
                for batch in tqdm(
                    val_loader_to_use,
                    desc=f"Val E{epoch+1}",
                    leave=False,
                    disable=not use_tqdm_train
                ):
                    fmri_data, eeg_data, y_all, fmri_idx, eeg_idx, both_f_idx, both_e_idx = batch
                    y_all = y_all.view(-1).long().to(device)
                    
                    # both
                    if (len(both_f_idx) > 0) and (len(both_e_idx) > 0):
                        A_list_both = build_A_list(fmri_data, ds_cfg.scales, idx=both_f_idx, device=device)
                        E_list_both = build_E_list(eeg_data, ds_cfg.scales, idx=both_e_idx, device=device)
                        y_both = y_all[fmri_idx[both_f_idx]]
                        
                        model_output = model(fmri_scales=A_list_both, eeg_scales=E_list_both)
                        if use_msre and len(model_output) >= 9:
                            out_fmri, out_eeg, out_fused, align_loss, fused_cons, fmri_feat, eeg_feat, msre_info, corr_align_loss = model_output
                        elif len(model_output) >= 9:
                            out_fmri, out_eeg, out_fused, align_loss, fused_cons, fmri_feat, eeg_feat, msre_info, corr_align_loss = model_output
                            msre_info = None
                        else:
                            out_fmri, out_eeg, out_fused, align_loss, *_ = model_output
                            fused_cons, fmri_feat, eeg_feat = None, None, None
                            msre_info = None
                            corr_align_loss = torch.zeros((), device=device, dtype=torch.float32)
                        
                        # Collect MSRE values during validation
                        if use_msre and msre_info is not None:
                            if isinstance(msre_info, list) and len(msre_info) > 0:
                                msre_info_last = msre_info[-1]
                            elif isinstance(msre_info, dict):
                                msre_info_last = msre_info
                            else:
                                msre_info_last = None
                            
                            if msre_info_last is not None:
                                msre_a = msre_info_last.get("msre_a", None)
                                msre_b = msre_info_last.get("msre_b", None)
                                msre_weight_a = msre_info_last.get("msre_weight_a", None)
                                msre_weight_b = msre_info_last.get("msre_weight_b", None)
                                
                                if msre_a is not None:
                                    val_msre_a_list.append(msre_a.cpu())
                                if msre_b is not None:
                                    val_msre_b_list.append(msre_b.cpu())
                                if msre_weight_a is not None:
                                    val_msre_weight_a_list.append(msre_weight_a.cpu())
                                if msre_weight_b is not None:
                                    val_msre_weight_b_list.append(msre_weight_b.cpu())
                        
                        if out_fused is not None and train_cfg.fusion_mode in ("align_loss", "bcmf", "dual_cross"):  # Support both old and new names
                            out_main = out_fused
                        else:
                            out_main = out_fmri if out_fmri is not None else out_eeg
                        
                        p_main = torch.softmax(out_main, dim=1).cpu().numpy()
                        prob_all.append(p_main)
                        predicted_all += torch.argmax(out_main, 1).cpu().tolist()
                        test_y_all    += y_both.cpu().tolist()
                        
                        y_b_np = y_both.detach().cpu().tolist()
                        paired_y += y_b_np
                        if out_fused is not None:
                            p_fused = torch.softmax(out_fused, dim=1).cpu().numpy()
                            paired_fused_probs.append(p_fused)
                            paired_fused_pred += torch.argmax(out_fused, 1).cpu().tolist()
                    
                    # fmri_only
                    if len(fmri_idx) > 0:
                        mask_fmri_only = torch.ones(len(fmri_idx), dtype=torch.bool)
                        if len(both_f_idx) > 0:
                            mask_fmri_only[both_f_idx] = False
                        fmri_only_idx_in_f = torch.nonzero(mask_fmri_only, as_tuple=False).view(-1)
                        if len(fmri_only_idx_in_f) > 0:
                            A_list_f = build_A_list(fmri_data, ds_cfg.scales, idx=fmri_only_idx_in_f, device=device)
                            y_f = y_all[fmri_idx[fmri_only_idx_in_f]]
                            out_fmri, *_ = model(fmri_scales=A_list_f, fmri_only=True, eeg_only=False)
                            p_f = torch.softmax(out_fmri, dim=1).cpu().numpy()
                            prob_all.append(p_f)
                            predicted_all += torch.argmax(out_fmri, 1).cpu().tolist()
                            test_y_all    += y_f.cpu().tolist()
                            fmri_only_probs.append(p_f)
                            fmri_only_pred += torch.argmax(out_fmri, 1).cpu().tolist()
                            fmri_only_y    += y_f.cpu().tolist()
                    
                    # eeg_only
                    if len(eeg_idx) > 0:
                        mask_eeg_only = torch.ones(len(eeg_idx), dtype=torch.bool)
                        if len(both_e_idx) > 0:
                            mask_eeg_only[both_e_idx] = False
                        eeg_only_idx_in_e = torch.nonzero(mask_eeg_only, as_tuple=False).view(-1)
                        if len(eeg_only_idx_in_e) > 0:
                            E_list_e = build_E_list(eeg_data, ds_cfg.scales, idx=eeg_only_idx_in_e, device=device)
                            y_e = y_all[eeg_idx[eeg_only_idx_in_e]]
                            _, out_eeg, *_ = model(eeg_scales=E_list_e, eeg_only=True, fmri_only=False)
                            p_e = torch.softmax(out_eeg, dim=1).cpu().numpy()
                            prob_all.append(p_e)
                            predicted_all += torch.argmax(out_eeg, 1).cpu().tolist()
                            test_y_all    += y_e.cpu().tolist()
                            eeg_only_probs.append(p_e)
                            eeg_only_pred += torch.argmax(out_eeg, 1).cpu().tolist()
                            eeg_only_y    += y_e.cpu().tolist()
            
            if len(test_y_all) > 0:
                # Overall metrics (keep existing logic)
                y_true = np.array(test_y_all)
                y_pred = np.array(predicted_all)
                probs_mc_full = np.vstack(prob_all) if len(prob_all) else np.zeros((0, ds_cfg.num_class))
                m_main = summarize_val(y_true, y_pred, probs_mc_full, ds_cfg.num_class)
                
                logger.info(
                    f"[Epoch {epoch+1} EVAL] "
                    f"|acc: {m_main['accuracy']:.4f} |bal_acc: {m_main['bal_acc']:.4f} "
                    f"|macro_f1: {m_main['macro_f1']:.4f} |auc: {m_main['auc']:.4f}"
                )
                
                if tb is not None:
                    tb.add_scalar("metrics/acc", float(m_main["accuracy"]), epoch + 1)
                    tb.add_scalar("metrics/bal_acc", float(m_main["bal_acc"]), epoch + 1)
                    tb.add_scalar("metrics/macro_f1", float(m_main["macro_f1"]), epoch + 1)
                    if np.isfinite(m_main["auc"]):
                        tb.add_scalar("metrics/auc", float(m_main["auc"]), epoch + 1)
                
                # Record MSRE values during validation (using sensitive metrics)
                if use_msre and len(val_msre_a_list) > 0:
                    val_msre_a_all = torch.cat(val_msre_a_list, dim=0)  # (N, num_class)
                    metrics_a_val = compute_msre_sensitive_metrics(val_msre_a_all, corruption_mask=None)
                    tb.add_scalar("msre_val/msre_a_max_prob", float(metrics_a_val['max_prob_mean'].item()), epoch + 1)
                    tb.add_scalar("msre_val/msre_a_entropy", float(metrics_a_val['entropy_mean'].item()), epoch + 1)
                    tb.add_scalar("msre_val/msre_a_margin", float(metrics_a_val['margin_mean'].item()), epoch + 1)
                    logger.info(f"[Epoch {epoch+1} MSRE-VAL] msre_a: max_prob={metrics_a_val['max_prob_mean']:.4f}, "
                              f"entropy={metrics_a_val['entropy_mean']:.4f}, margin={metrics_a_val['margin_mean']:.4f}")
                
                if use_msre and len(val_msre_b_list) > 0:
                    val_msre_b_all = torch.cat(val_msre_b_list, dim=0)
                    metrics_b_val = compute_msre_sensitive_metrics(val_msre_b_all, corruption_mask=None)
                    tb.add_scalar("msre_val/msre_b_max_prob", float(metrics_b_val['max_prob_mean'].item()), epoch + 1)
                    tb.add_scalar("msre_val/msre_b_entropy", float(metrics_b_val['entropy_mean'].item()), epoch + 1)
                    tb.add_scalar("msre_val/msre_b_margin", float(metrics_b_val['margin_mean'].item()), epoch + 1)
                    logger.info(f"[Epoch {epoch+1} MSRE-VAL] msre_b: max_prob={metrics_b_val['max_prob_mean']:.4f}, "
                              f"entropy={metrics_b_val['entropy_mean']:.4f}, margin={metrics_b_val['margin_mean']:.4f}")
                
                if use_msre and len(val_msre_weight_a_list) > 0:
                    val_msre_weight_a_all = torch.cat(val_msre_weight_a_list, dim=0)
                    tb.add_scalar("msre_val/msre_weight_a_mean", float(val_msre_weight_a_all.mean().item()), epoch + 1)
                
                if use_msre and len(val_msre_weight_b_list) > 0:
                    val_msre_weight_b_all = torch.cat(val_msre_weight_b_list, dim=0)
                    tb.add_scalar("msre_val/msre_weight_b_mean", float(val_msre_weight_b_all.mean().item()), epoch + 1)
                
                # Save checkpoint at each eval (overall / fused, format consistent with Stage1/Stage2)
                acc_str = f"{m_main['accuracy']:.4f}".replace('.', 'p')
                auc_str = f"{m_main['auc']:.4f}".replace('.', 'p') if np.isfinite(m_main["auc"]) else "nan"
                
                # Add identifier based on experiment type
                exp_suffix = ""
                if use_contrastive:
                    exp_suffix = "_contrastive"
                elif use_kd:
                    exp_suffix = "_kd"
                
                checkpoint_path_post = os.path.join(
                    save_path,
                    f"bestmodel/checkpoint_stage3_post{exp_suffix}_epoch{epoch+1:03d}_acc{acc_str}_auc{auc_str}_fold{fold}.pth"
                )
                torch.save(model.state_dict(), checkpoint_path_post)
                logger.info(f"  Checkpoint saved (overall): {checkpoint_path_post}")
                
                # Save confusion matrix
                from sklearn.metrics import confusion_matrix
                cm_main = confusion_matrix(y_true, y_pred, labels=list(range(ds_cfg.num_class)))
                save_confusion_matrix(cm_main, save_path, f"confusion_matrix_overall_epoch{epoch+1:03d}_fold{fold}.txt")
                
                # Subset metrics calculation
                m_paired = None
                m_fmri_only = None
                m_eeg_only = None
                
                # (1) PAIRED subset metrics
                if len(paired_y) > 0:
                    if len(paired_fused_pred) > 0 and len(paired_fused_probs) > 0:
                        y_paired = np.array(paired_y)
                        y_pred_paired = np.array(paired_fused_pred)
                        probs_paired = np.vstack(paired_fused_probs) if len(paired_fused_probs) else np.zeros((0, ds_cfg.num_class))
                        m_paired = summarize_val(y_paired, y_pred_paired, probs_paired, ds_cfg.num_class)
                        logger.info(
                            f"[Epoch {epoch+1} EVAL] PAIRED    "
                            f"|acc: {m_paired['accuracy']:.4f} |bal_acc: {m_paired['bal_acc']:.4f} "
                            f"|macro_f1: {m_paired['macro_f1']:.4f} |auc: {m_paired['auc']:.4f}"
                        )
                        # Save paired confusion matrix
                        cm_paired = confusion_matrix(y_paired, y_pred_paired, labels=list(range(ds_cfg.num_class)))
                        save_confusion_matrix(cm_paired, save_path, f"confusion_matrix_paired_epoch{epoch+1:03d}_fold{fold}.txt")
                    else:
                        logger.info(f"[Epoch {epoch+1} EVAL] PAIRED    |no samples (no fused output)")
                else:
                    logger.info(f"[Epoch {epoch+1} EVAL] PAIRED    |no samples")
                
                # (2) FMRI-ONLY subset metrics
                if len(fmri_only_y) > 0:
                    y_fmri = np.array(fmri_only_y)
                    y_pred_fmri = np.array(fmri_only_pred)
                    probs_fmri = np.vstack(fmri_only_probs) if len(fmri_only_probs) else np.zeros((0, ds_cfg.num_class))
                    m_fmri_only = summarize_val(y_fmri, y_pred_fmri, probs_fmri, ds_cfg.num_class)
                    logger.info(
                        f"[Epoch {epoch+1} EVAL] FMRI-ONLY "
                        f"|acc: {m_fmri_only['accuracy']:.4f} |bal_acc: {m_fmri_only['bal_acc']:.4f} "
                        f"|macro_f1: {m_fmri_only['macro_f1']:.4f} |auc: {m_fmri_only['auc']:.4f}"
                    )
                    
                    # Save checkpoint at each eval (fmri-only, format consistent with Stage1/Stage2)
                    acc_str_fmri = f"{m_fmri_only['accuracy']:.4f}".replace('.', 'p')
                    auc_str_fmri = f"{m_fmri_only['auc']:.4f}".replace('.', 'p') if np.isfinite(m_fmri_only["auc"]) else "nan"
                    
                    # Add identifier based on experiment type
                    exp_suffix = ""
                    if use_contrastive:
                        exp_suffix = "_contrastive"
                    elif use_kd:
                        exp_suffix = "_kd"
                    
                    checkpoint_path_fmri = os.path.join(
                        save_path,
                        f"bestmodel/checkpoint_stage3_post_fmri{exp_suffix}_epoch{epoch+1:03d}_acc{acc_str_fmri}_auc{auc_str_fmri}_fold{fold}.pth"
                    )
                    torch.save(model.state_dict(), checkpoint_path_fmri)
                    logger.info(f"  Checkpoint saved (fmri-only): {checkpoint_path_fmri}")
                    
                    # Save fmri-only confusion matrix
                    cm_fmri = confusion_matrix(y_fmri, y_pred_fmri, labels=list(range(ds_cfg.num_class)))
                    save_confusion_matrix(cm_fmri, save_path, f"confusion_matrix_fmri_only_epoch{epoch+1:03d}_fold{fold}.txt")
                else:
                    logger.info(f"[Epoch {epoch+1} EVAL] FMRI-ONLY |no samples")
                
                # (3) EEG-ONLY subset metrics
                if len(eeg_only_y) > 0:
                    y_eeg = np.array(eeg_only_y)
                    y_pred_eeg = np.array(eeg_only_pred)
                    probs_eeg = np.vstack(eeg_only_probs) if len(eeg_only_probs) else np.zeros((0, ds_cfg.num_class))
                    m_eeg_only = summarize_val(y_eeg, y_pred_eeg, probs_eeg, ds_cfg.num_class)
                    logger.info(
                        f"[Epoch {epoch+1} EVAL] EEG-ONLY  "
                        f"|acc: {m_eeg_only['accuracy']:.4f} |bal_acc: {m_eeg_only['bal_acc']:.4f} "
                        f"|macro_f1: {m_eeg_only['macro_f1']:.4f} |auc: {m_eeg_only['auc']:.4f}"
                    )
                    
                    # Save checkpoint at each eval (eeg-only, format consistent with Stage1/Stage2)
                    acc_str_eeg = f"{m_eeg_only['accuracy']:.4f}".replace('.', 'p')
                    auc_str_eeg = f"{m_eeg_only['auc']:.4f}".replace('.', 'p') if np.isfinite(m_eeg_only["auc"]) else "nan"
                    
                    # Add identifier based on experiment type
                    exp_suffix = ""
                    if use_contrastive:
                        exp_suffix = "_contrastive"
                    elif use_kd:
                        exp_suffix = "_kd"
                    
                    checkpoint_path_eeg = os.path.join(
                        save_path,
                        f"bestmodel/checkpoint_stage3_post_eeg{exp_suffix}_epoch{epoch+1:03d}_acc{acc_str_eeg}_auc{auc_str_eeg}_fold{fold}.pth"
                    )
                    torch.save(model.state_dict(), checkpoint_path_eeg)
                    logger.info(f"  Checkpoint saved (eeg-only): {checkpoint_path_eeg}")
                    
                    # Save eeg-only confusion matrix
                    cm_eeg = confusion_matrix(y_eeg, y_pred_eeg, labels=list(range(ds_cfg.num_class)))
                    save_confusion_matrix(cm_eeg, save_path, f"confusion_matrix_eeg_only_epoch{epoch+1:03d}_fold{fold}.txt")
                else:
                    logger.info(f"[Epoch {epoch+1} EVAL] EEG-ONLY  |no samples")
                
                # Save best models (overall / fmri-only / eeg-only)
                if epoch >= min_save_epoch:
                    # (1) Overall best model
                    metric_val = m_main.get(select_metric, m_main.get("accuracy", np.nan))
                    if np.isfinite(metric_val) and metric_val > best_post_metric:
                        best_post_metric = float(metric_val)
                        best_post_epoch = epoch + 1
                        
                        # Add identifier based on experiment type
                        exp_suffix = ""
                        if use_contrastive:
                            exp_suffix = "_contrastive"
                        elif use_kd:
                            exp_suffix = "_kd"
                        
                        best_model_path_post = os.path.join(
                            save_path,
                            f"bestmodel/best_model_post{exp_suffix}_fold{fold}.pth"
                        )
                        torch.save(model.state_dict(), best_model_path_post)
                        logger.info(
                            f"[Fold {fold+1}] New best POST (overall) "
                            f"{select_metric.upper()}={best_post_metric:.4f} @epoch {best_post_epoch} "
                            f"-> {best_model_path_post}"
                        )
                        
                        # Save confusion matrix of best model
                        cm_best = confusion_matrix(y_true, y_pred, labels=list(range(ds_cfg.num_class)))
                        save_confusion_matrix(cm_best, save_path, f"confusion_matrix_best_overall_fold{fold}.txt")
                    
                    # (2) FMRI-ONLY best model
                    if m_fmri_only is not None:
                        metric_val_fmri = m_fmri_only.get(select_metric, m_fmri_only.get("accuracy", np.nan))
                        if np.isfinite(metric_val_fmri) and metric_val_fmri > best_post_fmri_metric:
                            best_post_fmri_metric = float(metric_val_fmri)
                            best_post_fmri_epoch = epoch + 1
                            
                            # Add identifier based on experiment type
                            exp_suffix = ""
                            if use_contrastive:
                                exp_suffix = "_contrastive"
                            elif use_kd:
                                exp_suffix = "_kd"
                            
                            best_model_path_post_fmri = os.path.join(
                                save_path,
                                f"bestmodel/best_model_post_fmri{exp_suffix}_fold{fold}.pth"
                            )
                            torch.save(model.state_dict(), best_model_path_post_fmri)
                            logger.info(
                                f"[Fold {fold+1}] New best POST (fmri-only) "
                                f"{select_metric.upper()}={best_post_fmri_metric:.4f} @epoch {best_post_fmri_epoch} "
                                f"-> {best_model_path_post_fmri}"
                            )
                            
                            # Save confusion matrix of best fmri-only model
                            cm_best_fmri = confusion_matrix(y_fmri, y_pred_fmri, labels=list(range(ds_cfg.num_class)))
                            save_confusion_matrix(cm_best_fmri, save_path, f"confusion_matrix_best_fmri_only_fold{fold}.txt")
                    
                    # (3) EEG-ONLY best model
                    if m_eeg_only is not None:
                        metric_val_eeg = m_eeg_only.get(select_metric, m_eeg_only.get("accuracy", np.nan))
                        if np.isfinite(metric_val_eeg) and metric_val_eeg > best_post_eeg_metric:
                            best_post_eeg_metric = float(metric_val_eeg)
                            best_post_eeg_epoch = epoch + 1
                            
                            # Add identifier based on experiment type
                            exp_suffix = ""
                            if use_contrastive:
                                exp_suffix = "_contrastive"
                            elif use_kd:
                                exp_suffix = "_kd"
                            
                            best_model_path_post_eeg = os.path.join(
                                save_path,
                                f"bestmodel/best_model_post_eeg{exp_suffix}_fold{fold}.pth"
                            )
                            torch.save(model.state_dict(), best_model_path_post_eeg)
                            logger.info(
                                f"[Fold {fold+1}] New best POST (eeg-only) "
                                f"{select_metric.upper()}={best_post_eeg_metric:.4f} @epoch {best_post_eeg_epoch} "
                                f"-> {best_model_path_post_eeg}"
                            )
                            
                            # Save confusion matrix of best eeg-only model
                            cm_best_eeg = confusion_matrix(y_eeg, y_pred_eeg, labels=list(range(ds_cfg.num_class)))
                            save_confusion_matrix(cm_best_eeg, save_path, f"confusion_matrix_best_eeg_only_fold{fold}.txt")
                
                # Early stopping
                if use_early_stop:
                    acc_now = m_main["accuracy"]
                    if np.isnan(acc_now):
                        logger.info(f"[EarlyStop] No valid samples in this evaluation, skipping early stop count")
                    else:
                        if (last_val_acc is None) or (acc_now > last_val_acc + 1e-6):
                            last_val_acc = acc_now
                            drop_streak = 0
                        else:
                            drop_streak += 1
                            logger.info(
                                f"[EarlyStop] acc not improved {drop_streak}/{patience} times "
                                f"(current {acc_now:.4f})"
                            )
                        if drop_streak >= patience:
                            logger.info(
                                f"[EarlyStop] No improvement for {drop_streak} consecutive times, triggering early stop at epoch={epoch+1}"
                            )
                            early_stopped = True
                            break
    
    # Close TensorBoard
    if tb is not None:
        tb.flush()
        tb.close()
    
    result = {
        "fold": int(fold),
        "best_post_metric": float(best_post_metric) if np.isfinite(best_post_metric) else float("nan"),
        "best_post_epoch": int(best_post_epoch),
        "best_model_path_post": best_model_path_post or "",
        "best_post_fmri_metric": float(best_post_fmri_metric) if np.isfinite(best_post_fmri_metric) else float("nan"),
        "best_post_fmri_epoch": int(best_post_fmri_epoch),
        "best_model_path_post_fmri": best_model_path_post_fmri or "",
        "best_post_eeg_metric": float(best_post_eeg_metric) if np.isfinite(best_post_eeg_metric) else float("nan"),
        "best_post_eeg_epoch": int(best_post_eeg_epoch),
        "best_model_path_post_eeg": best_model_path_post_eeg or "",
        "early_stopped": early_stopped,
    }
    
    logger.info(
        f"[Fold {fold+1} Summary] "
        f"POST (overall): {select_metric}={result['best_post_metric']:.4f} @epoch {best_post_epoch} | "
        f"POST (fmri-only): {select_metric}={result['best_post_fmri_metric']:.4f} @epoch {best_post_fmri_epoch} | "
        f"POST (eeg-only): {select_metric}={result['best_post_eeg_metric']:.4f} @epoch {best_post_eeg_epoch}"
    )
    
    return result


def main(config, save_path):
    """
    Stage3 main training function with 5-fold cross-validation
    
    Args:
        config: Configuration dictionary
        save_path: Main save path
    """
    train_cfg = SimpleNamespace(**config["training"])
    
    # Create main save directory
    _ensure_dir(save_path)
    _ensure_dir(os.path.join(save_path, "log"))
    
    logger = setup_logger(os.path.join(save_path, "log/stage3_main.log"))
    logger.info(f"========== STAGE3: 5-Fold Cross-Validation Training ==========")
    
    # 5-fold cross-validation
    n_folds = 5
    all_results = []
    
    for fold in range(n_folds):
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting Fold {fold+1}/{n_folds}")
        logger.info(f"{'='*60}")
        
        fold_save_path = os.path.join(save_path, f"fold{fold}")
        _ensure_dir(fold_save_path)
        
        try:
            result = run_once(config, fold_save_path, fold, writer=None)
            all_results.append(result)
            logger.info(f"Fold {fold+1} completed successfully")
        except Exception as e:
            logger.error(f"Fold {fold+1} failed with error: {e}", exc_info=True)
            all_results.append({
                "fold": int(fold),
                "best_post_metric": float("nan"),
                "best_post_epoch": -1,
                "best_model_path_post": "",
                "best_post_fmri_metric": float("nan"),
                "best_post_fmri_epoch": -1,
                "best_model_path_post_fmri": "",
                "best_post_eeg_metric": float("nan"),
                "best_post_eeg_epoch": -1,
                "best_model_path_post_eeg": "",
                "early_stopped": False,
            })
    
    # Summary results
    df_summary = pd.DataFrame(all_results)
    summary_path = os.path.join(save_path, "log/stage3_post_summary.csv")
    df_summary.to_csv(summary_path, index=False)
    logger.info(f"\nSummary saved to: {summary_path}")
    
    # Print fold-wise results
    select_metric = getattr(train_cfg, "select_metric", "auc")
    logger.info("\n" + "="*60)
    logger.info(f"Stage3 Fold-wise Results (by {select_metric}):")
    logger.info("="*60)
    for _, row in df_summary.iterrows():
        logger.info(
            f"Fold {int(row['fold'])}: "
            f"{select_metric}={row['best_post_metric']:.4f} @epoch {int(row['best_post_epoch']):3d} | "
            f"early_stopped={row['early_stopped']}"
        )
    
    # Calculate mean across folds
    if len(df_summary) > 0:
        mean_metric = df_summary['best_post_metric'].mean()
        logger.info("\n" + "="*60)
        logger.info(f"Mean across folds: {select_metric}={mean_metric:.4f}")
        logger.info("="*60)
    
    logger.info("\n" + "="*60)
    logger.info("Stage3 Training Complete!")
    logger.info("="*60)
    
    return df_summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/config_stage3.yaml')
    parser.add_argument('--save_path', default=None, help='Override save path (optional)')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    print("Config loaded:", args.config)
    
    # Create result save directory
    if args.save_path:
        save_path = args.save_path
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = config["logging"]["result_dir"]
        save_path = os.path.join(result_dir, f"stage3_post-{timestamp}")
    
    os.makedirs(save_path, exist_ok=True)
    
    # Save configuration copy
    config_copy_path = os.path.join(save_path, "config_copy.yaml")
    with open(config_copy_path, "w") as f:
        yaml.dump(config, f, sort_keys=False)
    print(f"Config saved to: {config_copy_path}")
    
    # Start training
    main(config, save_path)

