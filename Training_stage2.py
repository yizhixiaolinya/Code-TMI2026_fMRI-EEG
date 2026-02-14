"""
Stage2: Fusion Warmup Training Script
Handles fusion warmup training stage
Loads single-modal branch best checkpoint from Stage1
Supports 5-fold cross-validation
"""
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
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
    infer_mode_from_row, bucket_by_mode, build_A_list, build_E_list,
    summarize_val, log_split_summary, _ensure_dir,
    inject_fmri_noise, inject_eeg_noise, compute_entropy_regularization,
    compute_msre_sensitive_metrics
)
import warnings

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
normalizer = CopulaNormalizer(alpha=0.9, apply_knn=False)


def load_stage1_ckpt(model, stage1_fmri_ckpt_path, stage1_eeg_ckpt_path, logger):
    if stage1_fmri_ckpt_path and os.path.exists(stage1_fmri_ckpt_path):
        ckpt_fmri = torch.load(stage1_fmri_ckpt_path, map_location=device)
        fmri_state = {k: v for k, v in ckpt_fmri.items() if k.startswith("fmri_net.")}
        if len(fmri_state) > 0:
            model.load_state_dict(fmri_state, strict=False)
            logger.info(f"Loaded fMRI branch from: {stage1_fmri_ckpt_path}")
            logger.info(f"  Loaded {len(fmri_state)} parameters")
        else:
            logger.warning(f"No fmri_net.* keys found in: {stage1_fmri_ckpt_path}")
    else:
        logger.warning(f"Stage1 fMRI ckpt not found: {stage1_fmri_ckpt_path}")
    
    if stage1_eeg_ckpt_path and os.path.exists(stage1_eeg_ckpt_path):
        ckpt_eeg = torch.load(stage1_eeg_ckpt_path, map_location=device)
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
    
    # Create subdirectories
    _ensure_dir(save_path)
    _ensure_dir(os.path.join(save_path, "log"))
    _ensure_dir(os.path.join(save_path, "tb"))
    _ensure_dir(os.path.join(save_path, "bestmodel"))
    
    logger = setup_logger(os.path.join(save_path, f"log/fold{fold}_stage2_fusion.log"))
    logger.info(f"========== STAGE2: Fusion Warmup (Fold {fold+1}/5) ==========")
    
    # TensorBoard writer
    tb_logdir = os.path.join(save_path, f"tb/tb_fold{fold}")
    os.makedirs(tb_logdir, exist_ok=True)
    tb = SummaryWriter(log_dir=tb_logdir) if writer is None else writer
    
    # ========= Load CSV files for current fold =========
    kfold_dir = ds_cfg.kfold_base_dir
    fold_dir = os.path.join(kfold_dir, f"fold{fold}")
    
    train_pair_csv = os.path.join(fold_dir, "train_pair.csv")
    val_pair_csv = os.path.join(fold_dir, "val_pair.csv")
    
    subInfo_train = pd.read_csv(train_pair_csv)
    subInfo_val   = pd.read_csv(val_pair_csv)
    
    labels_train = subInfo_train["diagnosis_code"].to_numpy(dtype=np.int64)
    labels_val   = subInfo_val["diagnosis_code"].to_numpy(dtype=np.int64)
    
    train_index_base = np.arange(len(subInfo_train), dtype=int)
    test_index_base  = np.arange(len(subInfo_val),   dtype=int)
    
    # Remove none samples
    train_modes_raw = [infer_mode_from_row(subInfo_train.iloc[i]) for i in train_index_base]
    val_modes_raw   = [infer_mode_from_row(subInfo_val.iloc[i])   for i in test_index_base]
    
    keep_train = [i for i, m in zip(train_index_base, train_modes_raw) if m != "none"]
    keep_val   = [i for i, m in zip(test_index_base,  val_modes_raw)   if m != "none"]
    
    train_index_base = np.array(keep_train, dtype=int)
    test_index_base  = np.array(keep_val,   dtype=int)
    
    # Print statistics
    logger.info("Stage2 Data Summary:")
    log_split_summary(1, "TRAIN", train_index_base, subInfo_train, labels_train, ds_cfg, logger)
    log_split_summary(1, "VAL",   test_index_base,  subInfo_val,   labels_val,   ds_cfg, logger)
    
    # Pre-bucket (for training data selection)
    mode_buckets = bucket_by_mode(subInfo_train, train_index_base)
    logger.info(f"Mode buckets: both={len(mode_buckets.get('both', []))}")
    
    # Model initialization
    use_msre = bool(getattr(train_cfg, "use_msre", False))
    use_corr_align = bool(getattr(train_cfg, "use_corr_align", True))
    corr_align_dim = int(getattr(train_cfg, "corr_align_dim", 16))
    logger.info(f"[MODEL] TCP module is {'ENABLED' if use_msre else 'DISABLED'}")
    logger.info(f"[MODEL] Correlation alignment is {'ENABLED' if use_corr_align else 'DISABLED'} (dim={corr_align_dim})")
    
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
    
    # ========= Load single-modal branch weights from Stage1 =========
    stage1_cfg = getattr(train_cfg, "stage1", {})
    stage1_result_dir = stage1_cfg.get("stage1_result_dir", "results/Stage1")
    stage1_fmri_ckpt_name = stage1_cfg.get("best_fmri_ckpt_name", "bestmodel/best_model_fmri_fold{fold}.pth")
    stage1_eeg_ckpt_name = stage1_cfg.get("best_eeg_ckpt_name", "bestmodel/best_model_eeg_fold{fold}.pth")
    
    # Resolve checkpoint paths
    stage1_fmri_ckpt = os.path.join(stage1_result_dir, f"fold{fold}", stage1_fmri_ckpt_name.format(fold=fold))
    stage1_eeg_ckpt = os.path.join(stage1_result_dir, f"fold{fold}", stage1_eeg_ckpt_name.format(fold=fold))
    
    if stage1_fmri_ckpt or stage1_eeg_ckpt:
        logger.info("Loading Stage1 checkpoints...")
        load_stage1_ckpt(model, stage1_fmri_ckpt, stage1_eeg_ckpt, logger)
    else:
        logger.warning("No Stage1 checkpoints provided, using random initialization")
    
    # Train all modules (encoders + fusion), not frozen
    trainable_params = []
    for name, p in model.named_parameters():
        p.requires_grad = True
        trainable_params.append(name)
    
    logger.info(f"All parameters are trainable: {len(trainable_params)}")
    logger.info("Stage2: Training encoders (fmri_net, eeg_net) and fusion module (fuser) together")
    
    # Loss function
    eff_train_cnt_base = np.bincount(labels_train[train_index_base], minlength=ds_cfg.num_class)
    if train_cfg.imbalance_loss:
        cw = 1.0 / (eff_train_cnt_base + 1e-8)
        cw = cw / cw.mean()
        cw = np.minimum(np.sqrt(cw), 5.0)
        loss_func = nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float32, device=device))
    else:
        loss_func = nn.CrossEntropyLoss()
    
    # Create optimizer, include all trainable parameters
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    
    # Best model path
    best_model_path_fusion = None
    best_fusion_metric = -float("inf")
    best_fusion_epoch = -1
    
    # Training parameters
    epochs = int(getattr(train_cfg, "fusion_warmup_epochs", 10))
    eval_interval = int(getattr(train_cfg, "eval_interval_fusion_warmup", 1))
    min_save_epoch = int(getattr(train_cfg, "min_epochs_before_saving", 0))
    select_metric = getattr(train_cfg, "select_metric", "auc")
    
    # Verify DataLoader (only use paired data, no oversampling for validation)
    test_dataset = MultiModalDataset(
        subInfo_val, test_index_base, ds_cfg.scales, mode="both",
        labels_np=labels_val, num_class=ds_cfg.num_class,
        normalizer=normalizer
    )
    
    test_loader = DataLoader(
        test_dataset,
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
    
    # ======= Epoch Loop =======
    for epoch in tqdm(range(epochs), desc=f"Stage2-fold{fold+1} epochs", disable=not use_tqdm_train):
        train_loss_sum = 0.0
        train_loss_n   = 0
        train_correct = 0
        train_total   = 0
        
        # For accumulating TCP values during training
        train_tcp_a_list = []
        train_tcp_b_list = []
        train_corruption_mask_a_list = []  # Only for tcp_a (when fMRI is corrupted)
        train_corruption_mask_b_list = []  # Only for tcp_b (when EEG is corrupted)
        
        model.train()
        
        # Only use both modalities data
        allow = {"both"}
        selected = [mode_buckets[m] for m in allow if m in mode_buckets and mode_buckets[m].size > 0]
        if any(len(x) > 0 for x in selected):
            train_index_epoch = np.unique(np.concatenate([x for x in selected if x.size > 0]))
        else:
            train_index_epoch = np.array([], dtype=int)
        
        if train_index_epoch.size == 0:
            logger.warning(f"[Epoch {epoch+1}] No paired samples. Skip.")
            continue
        
        train_dataset = MultiModalDataset(
            subInfo_train, train_index_epoch, ds_cfg.scales, mode="both",
            labels_np=labels_train, num_class=ds_cfg.num_class,
            normalizer=normalizer, pair_expand_by_label={0: 20, 1: 20}
        )
        
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
        
        # Training loop
        for step, batch in enumerate(
            tqdm(train_loader, desc=f"Train E{epoch+1}", leave=False, disable=not use_tqdm_train),
            1
        ):
            fmri_data, eeg_data, b_y, fmri_idx, eeg_idx, both_f_idx, both_e_idx = batch
            b_y = b_y.view(-1).long().to(device)
            
            both_has = (len(both_f_idx) > 0) and (len(both_e_idx) > 0)
            if not both_has:
                continue
            
            y_both = b_y[fmri_idx[both_f_idx]]
            
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
                        block_mask_prob=0.5,
                        generator=g_train
                    )
                else:  # eeg
                    # Inject noise into EEG
                    E_list_both, corruption_mask = inject_eeg_noise(
                        E_list_both,
                        corruption_rate=corruption_rate,
                        channel_dropout_rate=0.3,
                        noise_scale=2.0,
                        generator=g_train
                    )
            
            # Model forward
            model_output = model(fmri_scales=A_list_both, eeg_scales=E_list_both)
            if use_msre and len(model_output) >= 9:
                out_fmri, out_eeg, out_fused, align_loss, _, _, _, tcp_info, corr_align_loss = model_output
            elif len(model_output) >= 9:
                out_fmri, out_eeg, out_fused, align_loss, _, _, _, _, corr_align_loss = model_output
                tcp_info = None
            else:
                out_fmri, out_eeg, out_fused, align_loss, *_ = model_output
                tcp_info = None
                corr_align_loss = torch.zeros((), device=device, dtype=torch.float32)
            
            # Fusion warmup: only train fused CE
            if out_fused is not None and train_cfg.fusion_mode in ("align_loss", "bcmf", "dual_cross"):  # Support both old and new names
                loss_both = loss_func(out_fused, y_both)
            else:
                if out_fmri is not None:
                    loss_both = loss_func(out_fmri, y_both)
                else:
                    loss_both = loss_func(out_eeg, y_both)
            
            # TCP loss (if enabled)
            tcp_loss = torch.zeros((), device=device, dtype=torch.float32)
            tcp_ce_loss_value = 0.0
            if use_msre and tcp_info is not None:
                if isinstance(tcp_info, list) and len(tcp_info) > 0:
                    tcp_info_last = tcp_info[-1]
                elif isinstance(tcp_info, dict):
                    tcp_info_last = tcp_info
                else:
                    tcp_info_last = None
                
                if tcp_info_last is not None:
                    logits_f_a = tcp_info_last.get("logits_f_a", None)
                    logits_f_b = tcp_info_last.get("logits_f_b", None)
                    
                    tcp_ce_parts = []
                    if logits_f_a is not None:
                        tcp_ce_parts.append(loss_func(logits_f_a, y_both))
                    if logits_f_b is not None:
                        tcp_ce_parts.append(loss_func(logits_f_b, y_both))
                    
                    if len(tcp_ce_parts) > 0:
                        tcp_ce_loss = torch.stack(tcp_ce_parts).mean()
                        tcp_ce_loss_value = float(tcp_ce_loss.item())
                        tcp_lambda = float(getattr(train_cfg, "tcp_lambda", 1.0))
                        tcp_loss = tcp_lambda * tcp_ce_loss
                        
                    
                    # Record TCP values
                    tcp_a = tcp_info_last.get("tcp_a", None)
                    tcp_b = tcp_info_last.get("tcp_b", None)
                    tcp_weight_a = tcp_info_last.get("tcp_weight_a", None)
                    tcp_weight_b = tcp_info_last.get("tcp_weight_b", None)
                    
                    # Accumulate TCP values during training (for statistics at end of epoch)
                    # Note: corruption_mask is only valid for the corrupted modality
                    if tcp_a is not None:
                        train_tcp_a_list.append(tcp_a.detach().cpu())
                        # Only use corruption_mask for tcp_a when fMRI is corrupted
                        if corruption_mask is not None and corrupt_modality == "fmri":
                            train_corruption_mask_a_list.append(corruption_mask.detach().cpu())
                        else:
                            # If fMRI is not corrupted, add all-False mask
                            batch_size = tcp_a.shape[0]
                            train_corruption_mask_a_list.append(torch.zeros(batch_size, dtype=torch.bool))
                    
                    if tcp_b is not None:
                        train_tcp_b_list.append(tcp_b.detach().cpu())
                        # Only use corruption_mask for tcp_b when EEG is corrupted
                        if corruption_mask is not None and corrupt_modality == "eeg":
                            train_corruption_mask_b_list.append(corruption_mask.detach().cpu())
                        else:
                            # If EEG is not corrupted, add all-False mask
                            batch_size = tcp_b.shape[0]
                            train_corruption_mask_b_list.append(torch.zeros(batch_size, dtype=torch.bool))
                    
                    if tb is not None:
                        if tcp_a is not None:
                            # Record sensitive metrics (entropy, margin, etc.)
                            metrics_a = compute_msre_sensitive_metrics(tcp_a, corruption_mask)
                            tb.add_scalar("tcp/tcp_a_max_prob", float(metrics_a['max_prob_mean'].item()), global_step)
                            tb.add_scalar("tcp/tcp_a_entropy", float(metrics_a['entropy_mean'].item()), global_step)
                            tb.add_scalar("tcp/tcp_a_margin", float(metrics_a['margin_mean'].item()), global_step)
                            
                            # If corruption_mask is provided, separately record metrics for clean and corrupted samples
                            if corruption_mask is not None:
                                if "entropy_clean" in metrics_a:
                                    tb.add_scalar("tcp/tcp_a_entropy_clean", float(metrics_a['entropy_clean'].item()), global_step)
                                    tb.add_scalar("tcp/tcp_a_max_prob_clean", float(metrics_a['max_prob_clean'].item()), global_step)
                                    tb.add_scalar("tcp/tcp_a_margin_clean", float(metrics_a['margin_clean'].item()), global_step)
                                if "entropy_corrupt" in metrics_a:
                                    tb.add_scalar("tcp/tcp_a_entropy_corrupt", float(metrics_a['entropy_corrupt'].item()), global_step)
                                    tb.add_scalar("tcp/tcp_a_max_prob_corrupt", float(metrics_a['max_prob_corrupt'].item()), global_step)
                                    tb.add_scalar("tcp/tcp_a_margin_corrupt", float(metrics_a['margin_corrupt'].item()), global_step)
                        
                        if tcp_b is not None:
                            # Record sensitive metrics (entropy, margin, etc.)
                            metrics_b = compute_msre_sensitive_metrics(tcp_b, corruption_mask)
                            tb.add_scalar("tcp/tcp_b_max_prob", float(metrics_b['max_prob_mean'].item()), global_step)
                            tb.add_scalar("tcp/tcp_b_entropy", float(metrics_b['entropy_mean'].item()), global_step)
                            tb.add_scalar("tcp/tcp_b_margin", float(metrics_b['margin_mean'].item()), global_step)
                            
                            # If corruption_mask is provided, separately record metrics for clean and corrupted samples
                            if corruption_mask is not None:
                                if "entropy_clean" in metrics_b:
                                    tb.add_scalar("tcp/tcp_b_entropy_clean", float(metrics_b['entropy_clean'].item()), global_step)
                                    tb.add_scalar("tcp/tcp_b_max_prob_clean", float(metrics_b['max_prob_clean'].item()), global_step)
                                    tb.add_scalar("tcp/tcp_b_margin_clean", float(metrics_b['margin_clean'].item()), global_step)
                                if "entropy_corrupt" in metrics_b:
                                    tb.add_scalar("tcp/tcp_b_entropy_corrupt", float(metrics_b['entropy_corrupt'].item()), global_step)
                                    tb.add_scalar("tcp/tcp_b_max_prob_corrupt", float(metrics_b['max_prob_corrupt'].item()), global_step)
                                    tb.add_scalar("tcp/tcp_b_margin_corrupt", float(metrics_b['margin_corrupt'].item()), global_step)
                        
                        if tcp_weight_a is not None:
                            tb.add_scalar("tcp/tcp_weight_a_mean", float(tcp_weight_a.mean().item()), global_step)
                        if tcp_weight_b is not None:
                            tb.add_scalar("tcp/tcp_weight_b_mean", float(tcp_weight_b.mean().item()), global_step)
                    
                    # Record TCP values for all layers (if there are multiple layers)
                    if isinstance(tcp_info, list) and len(tcp_info) > 1:
                        for layer_idx, layer_tcp in enumerate(tcp_info):
                            layer_tcp_a = layer_tcp.get("tcp_a", None)
                            layer_tcp_b = layer_tcp.get("tcp_b", None)
                            if tb is not None:
                                if layer_tcp_a is not None:
                                    layer_tcp_a_max = layer_tcp_a.max(dim=1)[0].mean()
                                    tb.add_scalar(f"tcp/layer{layer_idx}_tcp_a_max", float(layer_tcp_a_max.item()), global_step)
                                if layer_tcp_b is not None:
                                    layer_tcp_b_max = layer_tcp_b.max(dim=1)[0].mean()
                                    tb.add_scalar(f"tcp/layer{layer_idx}_tcp_b_max", float(layer_tcp_b_max.item()), global_step)
            
            # Correlation alignment loss (if enabled)
            corr_align_lambda = float(getattr(train_cfg, "corr_align_lambda", 0.1))
            corr_align_loss_weighted = corr_align_lambda * corr_align_loss if corr_align_loss is not None else torch.zeros((), device=device)
            
            # ========== Entropy regularization term (adversarial noise training) ==========
            entropy_reg_loss = torch.zeros((), device=device, dtype=torch.float32)
            if use_adversarial_noise and use_msre and tcp_info is not None and corruption_mask is not None and corrupt_modality is not None:
                if isinstance(tcp_info, list) and len(tcp_info) > 0:
                    tcp_info_last = tcp_info[-1]
                elif isinstance(tcp_info, dict):
                    tcp_info_last = tcp_info
                else:
                    tcp_info_last = None
                
                if tcp_info_last is not None:
                    # Select corresponding TCP distribution based on corrupted modality
                    # Note: if fMRI is corrupted, we expect EEG's TCP to decrease (because fMRI is unreliable)
                    # If EEG is corrupted, we expect fMRI's TCP to decrease (because EEG is unreliable)
                    # But actually, we should look at the corrupted modality's TCP and let it decrease
                    if corrupt_modality == "fmri":
                        tcp_dist = tcp_info_last.get("tcp_a", None)  # fMRI's TCP (fMRI is corrupted, should decrease)
                    else:  # eeg
                        tcp_dist = tcp_info_last.get("tcp_b", None)  # EEG's TCP (EEG is corrupted, should decrease)
                    
                    if tcp_dist is not None:
                        # Ensure tcp_dist is a probability distribution (softmax)
                        if tcp_dist.dim() == 2:  # [B, num_class]
                            # If already a probability distribution (sum=1), use directly; otherwise softmax
                            tcp_sum = tcp_dist.sum(dim=1, keepdim=True)
                            if not torch.allclose(tcp_sum, torch.ones_like(tcp_sum), atol=1e-3):
                                tcp_dist = F.softmax(tcp_dist, dim=1)
                            
                            # Compute entropy regularization term
                            entropy_reg_loss = compute_entropy_regularization(
                                tcp_dist,
                                corruption_mask,
                                clean_entropy_weight=clean_entropy_weight,
                                corrupt_entropy_weight=corrupt_entropy_weight
                            )
                            
                            if tb is not None:
                                tb.add_scalar("loss/entropy_reg", float(entropy_reg_loss.item()), global_step)
                                # Record weighted entropy regularization term
                                tb.add_scalar("loss/entropy_reg_weighted", float(entropy_reg_lambda * entropy_reg_loss.item()), global_step)
            
            loss_total = loss_both + tcp_loss + corr_align_loss_weighted + entropy_reg_lambda * entropy_reg_loss
            
            optimizer.zero_grad(set_to_none=True)
            loss_total.backward()
            optimizer.step()
            
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
            
            train_loss_sum += float(loss_total.item())
            train_loss_n   += 1
            
            if tb is not None:
                global_step += 1
                tb.add_scalar("iter/train_loss", float(loss_total.item()), global_step)
                if use_msre and tcp_loss.item() > 0:
                    tb.add_scalar("iter/tcp_loss", float(tcp_loss.item()), global_step)
            
        if tb is not None and train_loss_n > 0:
            tb.add_scalar("loss/train", train_loss_sum / max(1, train_loss_n), epoch + 1)
        if tb is not None and train_total > 0:
            tb.add_scalar("metrics/train_acc", float(train_correct) / float(train_total), epoch + 1)
        
        # ================= TCP  =================
        if use_msre and len(train_tcp_a_list) > 0:
            train_tcp_a_all = torch.cat(train_tcp_a_list, dim=0)  # (N, num_class)
            train_corruption_mask_a_all = None
            if len(train_corruption_mask_a_list) > 0:
                train_corruption_mask_a_all = torch.cat(train_corruption_mask_a_list, dim=0)
            
            metrics_a_train = compute_msre_sensitive_metrics(train_tcp_a_all, train_corruption_mask_a_all)
            
            logger.info(f"[Epoch {epoch+1} TCP-TRAIN] tcp_a: max_prob={metrics_a_train['max_prob_mean']:.4f}, "
                      f"entropy={metrics_a_train['entropy_mean']:.4f}, margin={metrics_a_train['margin_mean']:.4f}")
            
            if train_corruption_mask_a_all is not None:
                if "entropy_clean" in metrics_a_train:
                    logger.info(f"  tcp_a clean: entropy={metrics_a_train['entropy_clean']:.4f}, "
                              f"max_prob={metrics_a_train['max_prob_clean']:.4f}, "
                              f"margin={metrics_a_train['margin_clean']:.4f}")
                if "entropy_corrupt" in metrics_a_train:
                    logger.info(f"  tcp_a corrupt: entropy={metrics_a_train['entropy_corrupt']:.4f}, "
                              f"max_prob={metrics_a_train['max_prob_corrupt']:.4f}, "
                              f"margin={metrics_a_train['margin_corrupt']:.4f}")
            
            if tb is not None:
                tb.add_scalar("tcp_train/tcp_a_max_prob", float(metrics_a_train['max_prob_mean'].item()), epoch + 1)
                tb.add_scalar("tcp_train/tcp_a_entropy", float(metrics_a_train['entropy_mean'].item()), epoch + 1)
                tb.add_scalar("tcp_train/tcp_a_margin", float(metrics_a_train['margin_mean'].item()), epoch + 1)
                if train_corruption_mask_a_all is not None:
                    if "entropy_clean" in metrics_a_train:
                        tb.add_scalar("tcp_train/tcp_a_entropy_clean", float(metrics_a_train['entropy_clean'].item()), epoch + 1)
                        tb.add_scalar("tcp_train/tcp_a_max_prob_clean", float(metrics_a_train['max_prob_clean'].item()), epoch + 1)
                        tb.add_scalar("tcp_train/tcp_a_margin_clean", float(metrics_a_train['margin_clean'].item()), epoch + 1)
                    if "entropy_corrupt" in metrics_a_train:
                        tb.add_scalar("tcp_train/tcp_a_entropy_corrupt", float(metrics_a_train['entropy_corrupt'].item()), epoch + 1)
                        tb.add_scalar("tcp_train/tcp_a_max_prob_corrupt", float(metrics_a_train['max_prob_corrupt'].item()), epoch + 1)
                        tb.add_scalar("tcp_train/tcp_a_margin_corrupt", float(metrics_a_train['margin_corrupt'].item()), epoch + 1)
        
        if use_msre and len(train_tcp_b_list) > 0:
            train_tcp_b_all = torch.cat(train_tcp_b_list, dim=0)
            train_corruption_mask_b_all = None
            if len(train_corruption_mask_b_list) > 0:
                train_corruption_mask_b_all = torch.cat(train_corruption_mask_b_list, dim=0)
            
            metrics_b_train = compute_tcp_sensitive_metrics(train_tcp_b_all, train_corruption_mask_b_all)
            
            logger.info(f"[Epoch {epoch+1} TCP-TRAIN] tcp_b: max_prob={metrics_b_train['max_prob_mean']:.4f}, "
                      f"entropy={metrics_b_train['entropy_mean']:.4f}, margin={metrics_b_train['margin_mean']:.4f}")
            
            if train_corruption_mask_b_all is not None:
                if "entropy_clean" in metrics_b_train:
                    logger.info(f"  tcp_b clean: entropy={metrics_b_train['entropy_clean']:.4f}, "
                              f"max_prob={metrics_b_train['max_prob_clean']:.4f}, "
                              f"margin={metrics_b_train['margin_clean']:.4f}")
                if "entropy_corrupt" in metrics_b_train:
                    logger.info(f"  tcp_b corrupt: entropy={metrics_b_train['entropy_corrupt']:.4f}, "
                              f"max_prob={metrics_b_train['max_prob_corrupt']:.4f}, "
                              f"margin={metrics_b_train['margin_corrupt']:.4f}")
            
            if tb is not None:
                tb.add_scalar("tcp_train/tcp_b_max_prob", float(metrics_b_train['max_prob_mean'].item()), epoch + 1)
                tb.add_scalar("tcp_train/tcp_b_entropy", float(metrics_b_train['entropy_mean'].item()), epoch + 1)
                tb.add_scalar("tcp_train/tcp_b_margin", float(metrics_b_train['margin_mean'].item()), epoch + 1)
                if train_corruption_mask_b_all is not None:
                    if "entropy_clean" in metrics_b_train:
                        tb.add_scalar("tcp_train/tcp_b_entropy_clean", float(metrics_b_train['entropy_clean'].item()), epoch + 1)
                        tb.add_scalar("tcp_train/tcp_b_max_prob_clean", float(metrics_b_train['max_prob_clean'].item()), epoch + 1)
                        tb.add_scalar("tcp_train/tcp_b_margin_clean", float(metrics_b_train['margin_clean'].item()), epoch + 1)
                    if "entropy_corrupt" in metrics_b_train:
                        tb.add_scalar("tcp_train/tcp_b_entropy_corrupt", float(metrics_b_train['entropy_corrupt'].item()), epoch + 1)
                        tb.add_scalar("tcp_train/tcp_b_max_prob_corrupt", float(metrics_b_train['max_prob_corrupt'].item()), epoch + 1)
                        tb.add_scalar("tcp_train/tcp_b_margin_corrupt", float(metrics_b_train['margin_corrupt'].item()), epoch + 1)
        
        # ================= val =================
        if (epoch % eval_interval == 0) or (epoch == epochs - 1):
            model.eval()
            
            predicted_all, test_y_all, prob_all = [], [], []
            
            val_tcp_a_list = []
            val_tcp_b_list = []
            val_tcp_weight_a_list = []
            val_tcp_weight_b_list = []
            
            with torch.no_grad():
                for batch in tqdm(
                    test_loader,
                    desc=f"Val E{epoch+1}",
                    leave=False,
                    disable=not use_tqdm_train
                ):
                    fmri_data, eeg_data, y_all, fmri_idx, eeg_idx, both_f_idx, both_e_idx = batch
                    y_all = y_all.view(-1).long().to(device)
                    
                    # Only process paired data
                    if (len(both_f_idx) > 0) and (len(both_e_idx) > 0):
                        A_list_both = build_A_list(fmri_data, ds_cfg.scales, idx=both_f_idx, device=device)
                        E_list_both = build_E_list(eeg_data, ds_cfg.scales, idx=both_e_idx, device=device)
                        y_both = y_all[fmri_idx[both_f_idx]]
                        
                        model_output = model(fmri_scales=A_list_both, eeg_scales=E_list_both)
                        if use_msre and len(model_output) >= 9:
                            out_fmri, out_eeg, out_fused, align_loss, _, _, _, tcp_info, corr_align_loss = model_output
                        elif len(model_output) >= 9:
                            out_fmri, out_eeg, out_fused, align_loss, _, _, _, _, corr_align_loss = model_output
                            tcp_info = None
                        else:
                            out_fmri, out_eeg, out_fused, align_loss, *_ = model_output
                            tcp_info = None
                            corr_align_loss = torch.zeros((), device=device, dtype=torch.float32)
                        
                        if use_msre and tcp_info is not None:
                            if isinstance(tcp_info, list) and len(tcp_info) > 0:
                                tcp_info_last = tcp_info[-1]
                            elif isinstance(tcp_info, dict):
                                tcp_info_last = tcp_info
                            else:
                                tcp_info_last = None
                            
                            if tcp_info_last is not None:
                                tcp_a = tcp_info_last.get("tcp_a", None)
                                tcp_b = tcp_info_last.get("tcp_b", None)
                                tcp_weight_a = tcp_info_last.get("tcp_weight_a", None)
                                tcp_weight_b = tcp_info_last.get("tcp_weight_b", None)
                                
                                if tcp_a is not None:
                                    val_tcp_a_list.append(tcp_a.cpu())
                                if tcp_b is not None:
                                    val_tcp_b_list.append(tcp_b.cpu())
                                if tcp_weight_a is not None:
                                    val_tcp_weight_a_list.append(tcp_weight_a.cpu())
                                if tcp_weight_b is not None:
                                    val_tcp_weight_b_list.append(tcp_weight_b.cpu())
                        
                        if out_fused is not None and train_cfg.fusion_mode in ("align_loss", "bcmf", "dual_cross"):  # Support both old and new names
                            out_main = out_fused
                        else:
                            out_main = out_fmri if out_fmri is not None else out_eeg
                        
                        p_main = torch.softmax(out_main, dim=1).cpu().numpy()
                        prob_all.append(p_main)
                        predicted_all += torch.argmax(out_main, 1).cpu().tolist()
                        test_y_all    += y_both.cpu().tolist()
            
            if len(test_y_all) > 0:
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
                
                if use_msre and len(val_tcp_a_list) > 0:
                    val_tcp_a_all = torch.cat(val_tcp_a_list, dim=0)  # (N, num_class)
                    metrics_a_val = compute_msre_sensitive_metrics(val_tcp_a_all, corruption_mask=None)
                    
                    tb.add_scalar("tcp_val/tcp_a_max_prob", float(metrics_a_val['max_prob_mean'].item()), epoch + 1)
                    tb.add_scalar("tcp_val/tcp_a_entropy", float(metrics_a_val['entropy_mean'].item()), epoch + 1)
                    tb.add_scalar("tcp_val/tcp_a_margin", float(metrics_a_val['margin_mean'].item()), epoch + 1)
                    logger.info(f"[Epoch {epoch+1} TCP-VAL] tcp_a: max_prob={metrics_a_val['max_prob_mean']:.4f}, "
                              f"entropy={metrics_a_val['entropy_mean']:.4f}, margin={metrics_a_val['margin_mean']:.4f}")
                
                if use_msre and len(val_tcp_b_list) > 0:
                    val_tcp_b_all = torch.cat(val_tcp_b_list, dim=0)
                    metrics_b_val = compute_msre_sensitive_metrics(val_tcp_b_all, corruption_mask=None)
                    tb.add_scalar("tcp_val/tcp_b_max_prob", float(metrics_b_val['max_prob_mean'].item()), epoch + 1)
                    tb.add_scalar("tcp_val/tcp_b_entropy", float(metrics_b_val['entropy_mean'].item()), epoch + 1)
                    tb.add_scalar("tcp_val/tcp_b_margin", float(metrics_b_val['margin_mean'].item()), epoch + 1)
                    logger.info(f"[Epoch {epoch+1} TCP-VAL] tcp_b: max_prob={metrics_b_val['max_prob_mean']:.4f}, "
                              f"entropy={metrics_b_val['entropy_mean']:.4f}, margin={metrics_b_val['margin_mean']:.4f}")
                
                if use_msre and len(val_tcp_weight_a_list) > 0:
                    val_tcp_weight_a_all = torch.cat(val_tcp_weight_a_list, dim=0)
                    tb.add_scalar("tcp_val/tcp_weight_a_mean", float(val_tcp_weight_a_all.mean().item()), epoch + 1)
                
                if use_msre and len(val_tcp_weight_b_list) > 0:
                    val_tcp_weight_b_all = torch.cat(val_tcp_weight_b_list, dim=0)
                    tb.add_scalar("tcp_val/tcp_weight_b_mean", float(val_tcp_weight_b_all.mean().item()), epoch + 1)
                
                # Save checkpoint at each eval (format consistent with Stage1)
                acc_str = f"{m_main['accuracy']:.4f}".replace('.', 'p')
                auc_str = f"{m_main['auc']:.4f}".replace('.', 'p') if np.isfinite(m_main["auc"]) else "nan"
                checkpoint_path_fusion = os.path.join(
                    save_path,
                    f"bestmodel/checkpoint_stage2_fusion_epoch{epoch+1:03d}_acc{acc_str}_auc{auc_str}_fold{fold}.pth"
                )
                torch.save(model.state_dict(), checkpoint_path_fusion)
                logger.info(f"  Checkpoint saved: {checkpoint_path_fusion}")

                # Save best model
                if epoch >= min_save_epoch:
                    metric_val = m_main.get(select_metric, m_main.get("accuracy", np.nan))
                    if np.isfinite(metric_val) and metric_val > best_fusion_metric:
                        best_fusion_metric = float(metric_val)
                        best_fusion_epoch = epoch + 1
                        best_model_path_fusion = os.path.join(
                            save_path,
                            f"bestmodel/best_model_fusion_fold{fold}.pth"
                        )
                        torch.save(model.state_dict(), best_model_path_fusion)
                        logger.info(
                            f"[Fold {fold+1}] New best FUSION "
                            f"{select_metric.upper()}={best_fusion_metric:.4f} @epoch {best_fusion_epoch} "
                            f"-> {best_model_path_fusion}"
                        )
    
    
    # Close TensorBoard
    if tb is not None:
        tb.flush()
        tb.close()
    
    result = {
        "fold": int(fold),
        "best_fusion_metric": float(best_fusion_metric) if np.isfinite(best_fusion_metric) else float("nan"),
        "best_fusion_epoch": int(best_fusion_epoch),
        "best_model_path_fusion": best_model_path_fusion or "",
    }
    
    logger.info(
        f"[Fold {fold+1} Summary] "
        f"FUSION: {select_metric}={result['best_fusion_metric']:.4f} @epoch {best_fusion_epoch}"
    )
    
    return result


def main(config, save_path):
    """
    Stage2 main training function with 5-fold cross-validation
    
    Args:
        config: Configuration dictionary
        save_path: Main save path
    """
    train_cfg = SimpleNamespace(**config["training"])
    
    # Create main save directory
    _ensure_dir(save_path)
    _ensure_dir(os.path.join(save_path, "log"))
    
    logger = setup_logger(os.path.join(save_path, "log/stage2_main.log"))
    logger.info(f"========== STAGE2: 5-Fold Cross-Validation Training ==========")
    
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
                "best_fusion_metric": float("nan"),
                "best_fusion_epoch": -1,
                "best_model_path_fusion": "",
            })
    
    # Summary results
    df_summary = pd.DataFrame(all_results)
    summary_path = os.path.join(save_path, "log/stage2_fusion_summary.csv")
    df_summary.to_csv(summary_path, index=False)
    logger.info(f"\nSummary saved to: {summary_path}")
    
    # Print fold-wise results
    select_metric = getattr(train_cfg, "select_metric", "auc")
    logger.info("\n" + "="*60)
    logger.info(f"Stage2 Fold-wise Results (by {select_metric}):")
    logger.info("="*60)
    for _, row in df_summary.iterrows():
        logger.info(
            f"Fold {int(row['fold'])}: "
            f"{select_metric}={row['best_fusion_metric']:.4f} @epoch {int(row['best_fusion_epoch']):3d}"
        )
    
    # Calculate mean across folds
    if len(df_summary) > 0:
        mean_metric = df_summary['best_fusion_metric'].mean()
        logger.info("\n" + "="*60)
        logger.info(f"Mean across folds: {select_metric}={mean_metric:.4f}")
        logger.info("="*60)
    
    logger.info("\n" + "="*60)
    logger.info("Stage2 Training Complete!")
    logger.info("="*60)
    
    return df_summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/config_stage2.yaml')
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
        save_path = os.path.join(result_dir, f"stage2_fusion-{timestamp}")
    
    os.makedirs(save_path, exist_ok=True)
    
    # Save configuration copy
    config_copy_path = os.path.join(save_path, "config_copy.yaml")
    with open(config_copy_path, "w") as f:
        yaml.dump(config, f, sort_keys=False)
    print(f"Config saved to: {config_copy_path}")
    
    # Start training
    main(config, save_path)

