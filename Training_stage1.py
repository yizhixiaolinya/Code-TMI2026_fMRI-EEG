"""
Stage1: Single-Modal Warmup Training Script
Handles single-modal warmup training stage (EEG-only + fMRI-only)
Supports 5-fold cross-validation
"""
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
import os
import torch
import torch.nn as nn
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
    infer_mode_from_row, build_A_list, build_E_list,
    summarize_val, log_split_summary, _ensure_dir
)
from sklearn.metrics import (
    accuracy_score, recall_score, roc_auc_score,
    confusion_matrix
)
import warnings

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
normalizer = CopulaNormalizer(alpha=0.9, apply_knn=False)


def run_once(config, save_path, fold, writer=None):
    ds_cfg = SimpleNamespace(**config["dataset"])
    train_cfg = SimpleNamespace(**config["training"])
    
    # Create subdirectories
    _ensure_dir(save_path)
    _ensure_dir(os.path.join(save_path, "log"))
    _ensure_dir(os.path.join(save_path, "tb"))
    _ensure_dir(os.path.join(save_path, "bestmodel"))
    
    logger = setup_logger(os.path.join(save_path, f"log/fold{fold}_stage1_warmup.log"))
    logger.info(f"========== STAGE1: Single-Modal Warmup (Fold {fold+1}/5) ==========")
    
    # TensorBoard writer
    tb_logdir = os.path.join(save_path, f"tb/tb_fold{fold}")
    os.makedirs(tb_logdir, exist_ok=True)
    tb = SummaryWriter(log_dir=tb_logdir) if writer is None else writer
    
    # ========= Load CSV files for current fold =========
    kfold_dir = ds_cfg.kfold_base_dir
    fold_dir = os.path.join(kfold_dir, f"fold{fold}")
    
    train_eeg_csv = os.path.join(fold_dir, "train_single_eeg.csv")
    train_fmri_csv = os.path.join(fold_dir, "train_single_fmri.csv")
    val_eeg_csv = os.path.join(fold_dir, "val_single_eeg.csv")
    val_fmri_csv = os.path.join(fold_dir, "val_single_fmri.csv")
    
    subInfo_train_eeg  = pd.read_csv(train_eeg_csv)
    subInfo_train_fmri = pd.read_csv(train_fmri_csv)
    
    labels_train_eeg  = subInfo_train_eeg["diagnosis_code"].to_numpy(dtype=np.int64)
    labels_train_fmri = subInfo_train_fmri["diagnosis_code"].to_numpy(dtype=np.int64)
    
    train_index_eeg_only  = np.arange(len(subInfo_train_eeg),  dtype=int)
    train_index_fmri_only = np.arange(len(subInfo_train_fmri), dtype=int)
    
    # ========= Load validation CSV for warmup (EEG-only / fMRI-only) =========
    subInfo_val_eeg    = pd.read_csv(val_eeg_csv)
    subInfo_val_fmri   = pd.read_csv(val_fmri_csv)
    
    labels_val_eeg     = subInfo_val_eeg["diagnosis_code"].to_numpy(dtype=np.int64)
    labels_val_fmri    = subInfo_val_fmri["diagnosis_code"].to_numpy(dtype=np.int64)
    
    val_index_eeg_only  = np.arange(len(subInfo_val_eeg),  dtype=int)
    val_index_fmri_only = np.arange(len(subInfo_val_fmri), dtype=int)
    
    # Print statistics
    logger.info("Stage1 Data Summary:")
    logger.info(f"  Train: fmri_only={len(train_index_fmri_only)}, eeg_only={len(train_index_eeg_only)}")
    logger.info(f"  Val:   fmri_only={len(val_index_fmri_only)}, eeg_only={len(val_index_eeg_only)}")
    
    # Model initialization
    use_msre = bool(getattr(train_cfg, "use_msre", False))
    logger.info(f"[MODEL] MSRE module is {'ENABLED' if use_msre else 'DISABLED'}")
    
    model = MyModels.MAHGCNNET(
        ROInum=ds_cfg.scales[0],
        layer=int(ds_cfg.layers),
        num_class=ds_cfg.num_class,
        num_scales=len(ds_cfg.scales),
        fusion_mode=train_cfg.fusion_mode,
        use_msre=use_msre,
    ).to(device)
    
    for name, p in model.named_parameters():
        if name.startswith("fuser."):
            p.requires_grad = False
        else:
            p.requires_grad = True
    
    eff_train_cnt_base = np.bincount(labels_train_fmri, minlength=ds_cfg.num_class)
    if train_cfg.imbalance_loss:
        cw = 1.0 / (eff_train_cnt_base + 1e-8)
        cw = cw / cw.mean()
        cw = np.minimum(np.sqrt(cw), 5.0)
        loss_func = nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float32, device=device))
    else:
        loss_func = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    
    best_model_path_fmri = None
    best_model_path_eeg  = None
    best_acc_fmri = -float("inf")
    best_epoch_fmri = -1
    best_acc_eeg = -float("inf")
    best_epoch_eeg = -1
    
    eval_interval = int(getattr(train_cfg, "eval_interval", 5))
    epochs = int(getattr(train_cfg, "epochs", 100))
    min_save_epoch = int(getattr(train_cfg, "min_epochs_before_saving", 20))
    
    val_fmri_only_loader = None
    val_eeg_only_loader  = None
    
    if len(val_index_fmri_only) > 0:
        val_fmri_only_dataset = MultiModalDataset(
            subInfo_val_fmri, val_index_fmri_only, ds_cfg.scales,
            mode="fmri",
            labels_np=labels_val_fmri,
            num_class=ds_cfg.num_class,
            normalizer=normalizer
        )
        val_fmri_only_loader = DataLoader(
            val_fmri_only_dataset,
            batch_size=train_cfg.batch_size,
            shuffle=False,
            num_workers=train_cfg.num_workers,
            collate_fn=make_collate_with_normalizer(
                normalizer, base_collate=multimodal_collate_flex, phase="transform",
                apply_random_mask=False,
            ),
            pin_memory=True
        )
    
    if len(val_index_eeg_only) > 0:
        val_eeg_only_dataset = MultiModalDataset(
            subInfo_val_eeg, val_index_eeg_only, ds_cfg.scales,
            mode="eeg",
            labels_np=labels_val_eeg,
            num_class=ds_cfg.num_class,
            normalizer=normalizer
        )
        val_eeg_only_loader = DataLoader(
            val_eeg_only_dataset,
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
    for epoch in tqdm(range(epochs), desc=f"Stage1-fold{fold+1} epochs", disable=not use_tqdm_train):
        train_loss_sum = 0.0
        train_loss_n   = 0
        fmri_only_loss_sum = 0.0
        fmri_only_cnt      = 0
        eeg_only_loss_sum  = 0.0
        eeg_only_cnt       = 0
        train_correct = 0
        train_total   = 0
        train_correct_f = 0
        train_total_f = 0
        train_correct_e = 0
        train_total_e = 0
        
        model.train()
        
        fmri_train_loader = None
        if len(train_index_fmri_only) > 0:
            fmri_train_dataset = MultiModalDataset(
                subInfo_train_fmri, train_index_fmri_only,
                ds_cfg.scales, mode="fmri",
                labels_np=labels_train_fmri,
                num_class=ds_cfg.num_class,
                normalizer=normalizer,
                fmri_only_expand_by_label={0: 1, 1: 3}
            )
            fmri_train_loader = DataLoader(
                fmri_train_dataset,
                batch_size=train_cfg.batch_size,
                shuffle=True,
                num_workers=train_cfg.num_workers,
                collate_fn=make_collate_with_normalizer(
                    normalizer,
                    base_collate=multimodal_collate_flex,
                    phase="transform",
                    apply_random_mask=True,
                    mask_threshold=0.1,
                    mask_prob=0.5,
                    mask_modalities=("fmri", "eeg")
                ),
                pin_memory=True,
                drop_last=False
            )
        
        eeg_train_loader = None
        if len(train_index_eeg_only) > 0:
            eeg_train_dataset = MultiModalDataset(
                subInfo_train_eeg, train_index_eeg_only,
                ds_cfg.scales, mode="eeg",
                labels_np=labels_train_eeg,
                num_class=ds_cfg.num_class,
                normalizer=normalizer
            )
            eeg_train_loader = DataLoader(
                eeg_train_dataset,
                batch_size=train_cfg.batch_size,
                shuffle=True,
                num_workers=train_cfg.num_workers,
                collate_fn=make_collate_with_normalizer(
                    normalizer,
                    base_collate=multimodal_collate_flex,
                    phase="transform",
                    apply_random_mask=True,
                    mask_threshold=0.1,
                    mask_prob=0.5,
                    mask_modalities=("fmri", "eeg")
                ),
                pin_memory=True,
                drop_last=False
            )
        
        if fmri_train_loader is not None:
            for step, batch in enumerate(
                tqdm(fmri_train_loader, desc=f"Train FMRI-only E{epoch+1}", leave=False, disable=not use_tqdm_train),
                1
            ):
                fmri_data, eeg_data, b_y, fmri_idx, eeg_idx, both_f_idx, both_e_idx = batch
                b_y = b_y.view(-1).long().to(device)
                
                if len(fmri_idx) == 0:
                    continue
                fmri_only_idx_in_f = torch.arange(len(fmri_idx), device=fmri_idx.device)
                y_fmri_only = b_y[fmri_idx[fmri_only_idx_in_f]]
                
                A_list_f = build_A_list(fmri_data, ds_cfg.scales, idx=fmri_only_idx_in_f, device=device)
                out_fmri, *_ = model(fmri_scales=A_list_f, fmri_only=True, eeg_only=False)
                loss_f = loss_func(out_fmri, y_fmri_only)
                
                optimizer.zero_grad(set_to_none=True)
                loss_f.backward()
                optimizer.step()
                
                # Statistics
                with torch.no_grad():
                    pred_f = torch.argmax(out_fmri, dim=1)
                    correct_f = (pred_f == y_fmri_only).sum().item()
                    n_f_now = int(y_fmri_only.shape[0])
                    train_correct += correct_f
                    train_total += n_f_now
                    train_correct_f += correct_f
                    train_total_f += n_f_now
                
                n_f = int(y_fmri_only.shape[0])
                fmri_only_loss_sum += float(loss_f.item()) * n_f
                fmri_only_cnt      += n_f
                train_loss_sum     += float(loss_f.item())
                train_loss_n       += 1
                
                if tb is not None:
                    global_step += 1
                    tb.add_scalar("iter/train_loss", float(loss_f.item()), global_step)
                    tb.add_scalar("iter/fmri_only_loss", float(loss_f.item()), global_step)
        
        if eeg_train_loader is not None:
            for step, batch in enumerate(
                tqdm(eeg_train_loader, desc=f"Train EEG-only E{epoch+1}", leave=False, disable=not use_tqdm_train),
                1
            ):
                fmri_data, eeg_data, b_y, fmri_idx, eeg_idx, both_f_idx, both_e_idx = batch
                b_y = b_y.view(-1).long().to(device)
                
                if len(eeg_idx) == 0:
                    continue
                eeg_only_idx_in_e = torch.arange(len(eeg_idx), device=eeg_idx.device)
                y_eeg_only = b_y[eeg_idx[eeg_only_idx_in_e]]
                
                E_list_e = build_E_list(eeg_data, ds_cfg.scales, idx=eeg_only_idx_in_e, device=device)
                _, out_eeg, *_ = model(eeg_scales=E_list_e, eeg_only=True, fmri_only=False)
                loss_e = loss_func(out_eeg, y_eeg_only)
                
                optimizer.zero_grad(set_to_none=True)
                loss_e.backward()
                optimizer.step()
                
                # Statistics
                with torch.no_grad():
                    pred_e = torch.argmax(out_eeg, dim=1)
                    correct_e = (pred_e == y_eeg_only).sum().item()
                    n_e_now = int(y_eeg_only.shape[0])
                    train_correct += correct_e
                    train_total += n_e_now
                    train_correct_e += correct_e
                    train_total_e += n_e_now
                
                n_e = int(y_eeg_only.shape[0])
                eeg_only_loss_sum += float(loss_e.item()) * n_e
                eeg_only_cnt      += n_e
                train_loss_sum    += float(loss_e.item())
                train_loss_n      += 1
                
                if tb is not None:
                    global_step += 1
                    tb.add_scalar("iter/train_loss", float(loss_e.item()), global_step)
                    tb.add_scalar("iter/eeg_only_loss", float(loss_e.item()), global_step)
        
        # Training loss logging
        if tb is not None and train_loss_n > 0:
            tb.add_scalar("loss/train", train_loss_sum / max(1, train_loss_n), epoch + 1)
        if tb is not None and fmri_only_cnt > 0:
            tb.add_scalar("loss/fmri_only", fmri_only_loss_sum / fmri_only_cnt, epoch + 1)
        if tb is not None and eeg_only_cnt > 0:
            tb.add_scalar("loss/eeg_only", eeg_only_loss_sum / eeg_only_cnt, epoch + 1)
        if tb is not None and train_total > 0:
            tb.add_scalar("metrics/train_acc", float(train_correct) / float(train_total), epoch + 1)
        
        # ================= Validation =================
        if (epoch % eval_interval == 0) or (epoch == epochs - 1):
            model.eval()
            m_fmri_only = None
            m_eeg_only  = None
            
            # ===== fMRI-only val =====
            if val_fmri_only_loader is not None:
                # Collect all batch predictions completely
                y_f_all, yhat_f_all, probs_f_all = [], [], []
                with torch.no_grad():
                    for batch in tqdm(
                        val_fmri_only_loader,
                        desc=f"Val FMRI-only E{epoch+1}",
                        leave=False,
                        disable=not use_tqdm_train
                    ):
                        fmri_data, eeg_data, y_all, fmri_idx, eeg_idx, both_f_idx, both_e_idx = batch
                        y_all = y_all.view(-1).long().to(device)
                        
                        if len(fmri_idx) == 0:
                            continue
                        fmri_only_idx_in_f = torch.arange(len(fmri_idx), device=fmri_idx.device)
                        y_f = y_all[fmri_idx[fmri_only_idx_in_f]]
                        
                        A_list_f = build_A_list(fmri_data, ds_cfg.scales, idx=fmri_only_idx_in_f, device=device)
                        out_fmri, *_ = model(fmri_scales=A_list_f, fmri_only=True, eeg_only=False)
                        p_f = torch.softmax(out_fmri, dim=1).cpu().numpy()
                        
                        # Accumulate: true labels, predicted labels, prediction probabilities
                        y_f_all.extend(y_f.cpu().tolist())
                        yhat_f_all.extend(torch.argmax(out_fmri, 1).cpu().tolist())
                        probs_f_all.append(p_f)
                
                if len(y_f_all) > 0:
                    # Unified evaluation: concatenate all results
                    y_f_np = np.array(y_f_all)
                    yhat_f_np = np.array(yhat_f_all)
                    probs_f_np = np.vstack(probs_f_all)
                    
                    # Calculate binary classification metrics (0=NC, 1=AD)
                    acc = accuracy_score(y_f_np, yhat_f_np)
                    # sen: Sensitivity/recall for AD class (class 1)
                    sen = recall_score(y_f_np, yhat_f_np, pos_label=1, zero_division=0)
                    # spe: Specificity for NC class (class 0) = TN / (TN + FP)
                    cm = confusion_matrix(y_f_np, yhat_f_np, labels=[0, 1])
                    TN, FP, FN, TP = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
                    spe = TN / (TN + FP) if (TN + FP) > 0 else 0.0
                    # auc: Use probability of positive class (AD)
                    if len(np.unique(y_f_np)) > 1:
                        auc = roc_auc_score(y_f_np, probs_f_np[:, 1])
                    else:
                        auc = np.nan
                    
                    # Print detailed metrics and confusion matrix
                    logger.info("="*80)
                    logger.info(f"Stage1 | Epoch {epoch+1} | VAL-FMRI-ONLY")
                    logger.info(f"  N={len(y_f_all):>4d}")
                    logger.info(f"  acc={acc:.4f} | sen={sen:.4f} | spe={spe:.4f} | auc={auc:.4f}")
                    logger.info(f"  Confusion Matrix (2x2):")
                    logger.info(f"              Predicted")
                    logger.info(f"              NC    AD")
                    logger.info(f"  Actual NC   {TN:4d}  {FP:4d}")
                    logger.info(f"        AD   {FN:4d}  {TP:4d}")
                    logger.info("="*80)
                    
                    # Keep original summarize_val for compatibility
                    m_fmri_only = summarize_val(y_f_np, yhat_f_np, probs_f_np, ds_cfg.num_class)
                    m_fmri_only['acc'] = acc
                    m_fmri_only['sen'] = sen
                    m_fmri_only['spe'] = spe
                    m_fmri_only['auc'] = auc
                    m_fmri_only['cm'] = cm
                    
                    if tb is not None:
                        tb.add_scalar("single/fmri_only/acc", float(acc), epoch + 1)
                        tb.add_scalar("single/fmri_only/sen", float(sen), epoch + 1)
                        tb.add_scalar("single/fmri_only/spe", float(spe), epoch + 1)
                        tb.add_scalar("single/fmri_only/bal_acc", float(m_fmri_only["bal_acc"]), epoch + 1)
                        tb.add_scalar("single/fmri_only/macro_f1", float(m_fmri_only["macro_f1"]), epoch + 1)
                        if np.isfinite(auc):
                            tb.add_scalar("single/fmri_only/auc", float(auc), epoch + 1)
                    
                    # Save checkpoint at each eval
                    acc_str = f"{acc:.4f}".replace('.', 'p')
                    auc_str = f"{auc:.4f}".replace('.', 'p') if np.isfinite(auc) else "nan"
                    checkpoint_path_fmri = os.path.join(
                        save_path,
                        f"bestmodel/checkpoint_stage1_fmri_epoch{epoch+1:03d}_acc{acc_str}_auc{auc_str}_fold{fold}.pth"
                    )
                    torch.save(model.state_dict(), checkpoint_path_fmri)
                    logger.info(f"  Checkpoint saved: {checkpoint_path_fmri}")
            
            # ===== EEG-only val =====
            if val_eeg_only_loader is not None:
                # Collect all batch predictions completely
                y_e_all, yhat_e_all, probs_e_all = [], [], []
                with torch.no_grad():
                    for batch in tqdm(
                        val_eeg_only_loader,
                        desc=f"Val EEG-only E{epoch+1}",
                        leave=False,
                        disable=not use_tqdm_train
                    ):
                        fmri_data, eeg_data, y_all, fmri_idx, eeg_idx, both_f_idx, both_e_idx = batch
                        y_all = y_all.view(-1).long().to(device)
                        
                        if len(eeg_idx) == 0:
                            continue
                        eeg_only_idx_in_e = torch.arange(len(eeg_idx), device=eeg_idx.device)
                        y_e = y_all[eeg_idx[eeg_only_idx_in_e]]
                        
                        E_list_e = build_E_list(eeg_data, ds_cfg.scales, idx=eeg_only_idx_in_e, device=device)
                        _, out_eeg, *_ = model(eeg_scales=E_list_e, eeg_only=True, fmri_only=False)
                        p_e = torch.softmax(out_eeg, dim=1).cpu().numpy()
                        
                        # Accumulate: true labels, predicted labels, prediction probabilities
                        y_e_all.extend(y_e.cpu().tolist())
                        yhat_e_all.extend(torch.argmax(out_eeg, 1).cpu().tolist())
                        probs_e_all.append(p_e)
                
                if len(y_e_all) > 0:
                    # Unified evaluation: concatenate all results
                    y_e_np = np.array(y_e_all)
                    yhat_e_np = np.array(yhat_e_all)
                    probs_e_np = np.vstack(probs_e_all)
                    
                    # Calculate binary classification metrics (0=NC, 1=AD)
                    acc = accuracy_score(y_e_np, yhat_e_np)
                    # sen: Sensitivity/recall for AD class (class 1)
                    sen = recall_score(y_e_np, yhat_e_np, pos_label=1, zero_division=0)
                    # spe: Specificity for NC class (class 0) = TN / (TN + FP)
                    cm = confusion_matrix(y_e_np, yhat_e_np, labels=[0, 1])
                    TN, FP, FN, TP = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
                    spe = TN / (TN + FP) if (TN + FP) > 0 else 0.0
                    # auc: Use probability of positive class (AD)
                    if len(np.unique(y_e_np)) > 1:
                        auc = roc_auc_score(y_e_np, probs_e_np[:, 1])
                    else:
                        auc = np.nan
                    
                    # Print detailed metrics and confusion matrix
                    logger.info("="*80)
                    logger.info(f"Stage1 | Epoch {epoch+1} | VAL-EEG-ONLY")
                    logger.info(f"  N={len(y_e_all):>4d}")
                    logger.info(f"  acc={acc:.4f} | sen={sen:.4f} | spe={spe:.4f} | auc={auc:.4f}")
                    logger.info(f"  Confusion Matrix (2x2):")
                    logger.info(f"              Predicted")
                    logger.info(f"              NC    AD")
                    logger.info(f"  Actual NC   {TN:4d}  {FP:4d}")
                    logger.info(f"        AD   {FN:4d}  {TP:4d}")
                    logger.info("="*80)
                    
                    # Keep original summarize_val for compatibility
                    m_eeg_only = summarize_val(y_e_np, yhat_e_np, probs_e_np, ds_cfg.num_class)
                    m_eeg_only['acc'] = acc
                    m_eeg_only['sen'] = sen
                    m_eeg_only['spe'] = spe
                    m_eeg_only['auc'] = auc
                    m_eeg_only['cm'] = cm
                    
                    if tb is not None:
                        tb.add_scalar("single/eeg_only/acc", float(acc), epoch + 1)
                        tb.add_scalar("single/eeg_only/sen", float(sen), epoch + 1)
                        tb.add_scalar("single/eeg_only/spe", float(spe), epoch + 1)
                        tb.add_scalar("single/eeg_only/bal_acc", float(m_eeg_only["bal_acc"]), epoch + 1)
                        tb.add_scalar("single/eeg_only/macro_f1", float(m_eeg_only["macro_f1"]), epoch + 1)
                        if np.isfinite(auc):
                            tb.add_scalar("single/eeg_only/auc", float(auc), epoch + 1)
                    
                    # Save checkpoint at each eval
                    acc_str = f"{acc:.4f}".replace('.', 'p')
                    auc_str = f"{auc:.4f}".replace('.', 'p') if np.isfinite(auc) else "nan"
                    checkpoint_path_eeg = os.path.join(
                        save_path,
                        f"bestmodel/checkpoint_stage1_eeg_epoch{epoch+1:03d}_acc{acc_str}_auc{auc_str}_fold{fold}.pth"
                    )
                    torch.save(model.state_dict(), checkpoint_path_eeg)
                    logger.info(f"  Checkpoint saved: {checkpoint_path_eeg}")
            
            # ====== Save best model (keep original logic) ======
            if epoch >= min_save_epoch:
                if m_fmri_only is not None:
                    acc_f = m_fmri_only.get("acc", m_fmri_only.get("accuracy", np.nan))
                    if np.isfinite(acc_f) and acc_f > best_acc_fmri:
                        best_acc_fmri  = float(acc_f)
                        best_epoch_fmri = epoch + 1
                        best_model_path_fmri = os.path.join(
                            save_path,
                            f"bestmodel/best_model_fmri_fold{fold}.pth"
                        )
                        torch.save(model.state_dict(), best_model_path_fmri)
                        logger.info(
                            f"[Fold {fold+1}] New best FMRI "
                            f"ACC={best_acc_fmri:.4f} @epoch {best_epoch_fmri} "
                            f"-> {best_model_path_fmri}"
                        )
                
                if m_eeg_only is not None:
                    acc_e = m_eeg_only.get("acc", m_eeg_only.get("accuracy", np.nan))
                    if np.isfinite(acc_e) and acc_e > best_acc_eeg:
                        best_acc_eeg   = float(acc_e)
                        best_epoch_eeg = epoch + 1
                        best_model_path_eeg = os.path.join(
                            save_path,
                            f"bestmodel/best_model_eeg_fold{fold}.pth"
                        )
                        torch.save(model.state_dict(), best_model_path_eeg)
                        logger.info(
                            f"[Fold {fold+1}] New best EEG "
                            f"ACC={best_acc_eeg:.4f} @epoch {best_epoch_eeg} "
                            f"-> {best_model_path_eeg}"
                        )
    
    # Close TensorBoard
    if tb is not None:
        tb.flush()
        tb.close()
    
    # Calculate combined metric
    combined_metric = -float("inf")
    if np.isfinite(best_acc_fmri) and np.isfinite(best_acc_eeg):
        combined_metric = (best_acc_fmri + best_acc_eeg) / 2.0
    elif np.isfinite(best_acc_fmri):
        combined_metric = best_acc_fmri
    elif np.isfinite(best_acc_eeg):
        combined_metric = best_acc_eeg
    
    result = {
        "fold": int(fold),
        "best_fmri_metric": float(best_acc_fmri) if np.isfinite(best_acc_fmri) else float("nan"),
        "best_fmri_epoch": int(best_epoch_fmri),
        "best_eeg_metric": float(best_acc_eeg) if np.isfinite(best_acc_eeg) else float("nan"),
        "best_eeg_epoch": int(best_epoch_eeg),
        "combined_metric": float(combined_metric) if np.isfinite(combined_metric) else float("nan"),
        "best_model_path_fmri": best_model_path_fmri or "",
        "best_model_path_eeg": best_model_path_eeg or "",
    }
    
    logger.info(
        f"[Fold {fold+1} Summary] "
        f"FMRI: acc={result['best_fmri_metric']:.4f} @epoch {best_epoch_fmri} | "
        f"EEG: acc={result['best_eeg_metric']:.4f} @epoch {best_epoch_eeg} | "
        f"Combined: {result['combined_metric']:.4f}"
    )
    
    return result


def main(config, save_path):
    """
    Stage1 main training function with 5-fold cross-validation
    
    Args:
        config: Configuration dictionary
        save_path: Main save path
    """
    train_cfg = SimpleNamespace(**config["training"])
    
    # Create main save directory
    _ensure_dir(save_path)
    _ensure_dir(os.path.join(save_path, "log"))
    
    logger = setup_logger(os.path.join(save_path, "log/stage1_main.log"))
    logger.info(f"========== STAGE1: 5-Fold Cross-Validation Training ==========")
    
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
                "best_fmri_metric": float("nan"),
                "best_fmri_epoch": -1,
                "best_eeg_metric": float("nan"),
                "best_eeg_epoch": -1,
                "combined_metric": float("nan"),
                "best_model_path_fmri": "",
                "best_model_path_eeg": "",
            })
    
    # Summary results
    df_summary = pd.DataFrame(all_results)
    summary_path = os.path.join(save_path, "log/stage1_warmup_summary.csv")
    df_summary.to_csv(summary_path, index=False)
    logger.info(f"\nSummary saved to: {summary_path}")
    
    # Print fold-wise results
    logger.info("\n" + "="*60)
    logger.info("Stage1 Fold-wise Results:")
    logger.info("="*60)
    for _, row in df_summary.iterrows():
        logger.info(
            f"Fold {int(row['fold'])}: "
            f"combined={row['combined_metric']:.4f} | "
            f"fmri_acc={row['best_fmri_metric']:.4f} @epoch {int(row['best_fmri_epoch']):3d} | "
            f"eeg_acc={row['best_eeg_metric']:.4f} @epoch {int(row['best_eeg_epoch']):3d}"
        )
    
    # Calculate mean across folds
    if len(df_summary) > 0:
        mean_combined = df_summary['combined_metric'].mean()
        mean_fmri = df_summary['best_fmri_metric'].mean()
        mean_eeg = df_summary['best_eeg_metric'].mean()
        logger.info("\n" + "="*60)
        logger.info(f"Mean across folds: combined={mean_combined:.4f}, fmri={mean_fmri:.4f}, eeg={mean_eeg:.4f}")
        logger.info("="*60)
    
    logger.info("\n" + "="*60)
    logger.info("Stage1 Training Complete!")
    logger.info("="*60)
    
    return df_summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/config_stage1.yaml')
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
        save_path = os.path.join(result_dir, f"stage1_warmup-{timestamp}")
    
    os.makedirs(save_path, exist_ok=True)
    
    # Save configuration copy
    config_copy_path = os.path.join(save_path, "config_copy.yaml")
    with open(config_copy_path, "w") as f:
        yaml.dump(config, f, sort_keys=False)
    print(f"Config saved to: {config_copy_path}")
    
    # Start training
    main(config, save_path)

