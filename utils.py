"""Shared data, training, and evaluation helpers for FE-KTNet."""
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn import metrics as M
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader


def setup_logger(log_path: str):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
        force=True
    )
    return logging.getLogger(__name__)


def build_A_list(fmri_data, scales, idx=None, device="cpu"):
    out = []
    for s in scales:
        t = fmri_data[s]
        out.append((t if idx is None else t[idx]).to(device))
    return out


def build_E_list(eeg_data, scales, idx=None, device="cpu"):
    out = []
    for s in scales:
        out.append({band: (t if idx is None else t[idx]).to(device)
                    for band, t in eeg_data[s].items()})
    return out


def training_arguments(stage):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=f"configs/config_stage{stage}.yaml")
    parser.add_argument("--save_path")
    parser.add_argument("--task", choices=("AD-NC", "MCI-NC", "AD-MCI"))
    parser.add_argument("--fold", type=int, choices=range(5))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--smoke-steps", type=int, default=0,
                        help="Bounded verification only; uses a small subject subset and one epoch.")
    parser.add_argument("--eval-only", action="store_true",
                        help="Evaluate saved best checkpoints without training.")
    args = parser.parse_args()
    with open(args.config) as handle:
        config = yaml.safe_load(handle)
    if args.task is not None:
        config["dataset"]["task"] = args.task
    if args.fold is not None:
        config["dataset"]["fold"] = args.fold
    config["runtime"] = dict(device=args.device, smoke_steps=args.smoke_steps,
                             eval_only=args.eval_only)
    return config, args.save_path or config["logging"]["result_dir"]


def selected_folds(config):
    fold = config["dataset"]["fold"]
    return range(5) if fold is None else [int(fold)]


def run_directory(root, task, fold):
    return Path(root) / task / f"fold{fold}"


def best_checkpoint(root, task, fold, modality):
    return run_directory(root, task, fold) / "bestmodel" / f"best_model_{modality}_fold{fold}.pth"


def read_split(config, fold, split, mode):
    directory = Path(config["dataset"]["kfold_base_dir"]) / config["dataset"]["task"] / f"fold{fold}"
    frame = pd.read_csv(directory / f"{split}.csv", keep_default_na=False)
    if mode in ("fmri", "both"):
        frame = frame.loc[frame.fmri_dst.ne("")]
    if mode in ("eeg", "both"):
        frame = frame.loc[frame.eeg_dst.ne("")]
    return frame.reset_index(drop=True)


def smoke_subset(frame):
    frame = frame.copy()
    frame["_availability"] = frame.fmri_dst.ne("").astype(int) + 2 * frame.eeg_dst.ne("").astype(int)
    subjects = []
    for _, group in frame.groupby(["diagnosis_code", "_availability"], sort=True):
        subjects.extend(group.component_key.drop_duplicates().sort_values().head(2))
    return frame.loc[frame.component_key.isin(subjects)].drop(columns="_availability").reset_index(drop=True)


def make_loader(frame, config, mode, training=False, dfc=False):
    from datasets.multimodal_dataset_by_path import (
        MultiModalDatasetCSV, HierarchicalSampler, multimodal_collate_flex,
    )

    ds = config["dataset"]
    tr = config["training"]
    dataset = MultiModalDatasetCSV(frame, scales=ds["scales"],
        mode="both" if mode == "all" else mode, use_paired_windows=dfc,
        dfc_inventory=ds.get("dfc_inventory"),
        dfc_cache_dir=ds.get("dfc_cache_dir", "Data/dFC44"),
        dfc_max_windows=1 if config["runtime"]["smoke_steps"] else None)
    steps = config["runtime"]["smoke_steps"]
    if steps:
        dataset.df = smoke_subset(dataset.df.iloc[dataset.index])
        dataset.index = np.arange(len(dataset.df))
    frame = dataset.df.iloc[dataset.index].reset_index(drop=True)
    batch_size = min(tr["batch_size"], 4) if steps else tr["batch_size"]
    sampler = None
    if training:
        sampler = HierarchicalSampler(frame,
            num_samples=batch_size * steps if steps else None)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, shuffle=False,
                      num_workers=0 if steps else tr["num_workers"],
                      collate_fn=multimodal_collate_flex)


def binary_metrics(frame):
    labels = frame.diagnosis_code.to_numpy()
    probs = frame.probability.to_numpy()
    if not np.isfinite(probs).all():
        raise ValueError("Non-finite subject prediction.")
    pred = (probs >= .5).astype(int)
    tn, fp, fn, tp = M.confusion_matrix(labels, pred, labels=[0, 1]).ravel()
    return dict(accuracy=float((pred == labels).mean()),
                auc=float(roc_auc_score(labels, probs)) if np.unique(labels).size == 2 else None,
                sensitivity=float(tp / (tp + fn)) if tp + fn else None,
                specificity=float(tn / (tn + fp)) if tn + fp else None,
                macro_f1=float(f1_score(labels, pred, average="macro", zero_division=0)),
                subjects=len(frame))


@torch.no_grad()
def evaluate(model, loader, mode, scales, device):
    model.eval()
    rows = []
    offset = 0
    for f_data, e_data, labels, f_idx, e_idx, _, _ in loader:
        if mode == "fmri":
            logits = model(build_A_list(f_data, scales, device=device))["logits"]
        elif mode == "eeg":
            logits = model(build_E_list(e_data, scales, device=device))["logits"]
        else:
            logits = model(build_A_list(f_data, scales, device=device),
                           build_E_list(e_data, scales, device=device))["fused_logits"]
        probs = logits.softmax(-1)[:, 1].cpu().numpy()
        meta = loader.dataset.df.iloc[loader.dataset.index[offset:offset + len(labels)]]
        for row, probability in zip(meta.itertuples(), probs):
            rows.append(dict(component_key=row.component_key, dataset=row.dataset,
                             diagnosis_code=int(row.diagnosis_code), probability=float(probability)))
        offset += len(labels)
    records = pd.DataFrame(rows)
    if records.empty:
        raise ValueError(f"No {mode} evaluation records.")
    if (records.groupby("component_key").diagnosis_code.nunique() != 1).any():
        raise ValueError("Conflicting labels within an evaluation subject.")
    subjects = records.groupby(["component_key", "dataset", "diagnosis_code"], as_index=False).probability.mean()
    return binary_metrics(subjects), subjects


def load_weights(model, checkpoint, device):
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(saved["model_state_dict"], strict=True)
    return saved


def fit(model, train_loader, valid_loaders, models_to_save, loss_fn, config, root, fold, stage):
    """One shared epoch loop; stages define their own objective and saved branches."""
    task = config["dataset"]["task"]
    directory = run_directory(root, task, fold)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "bestmodel").mkdir(exist_ok=True)
    with open(directory / "config.yaml", "w") as handle:
        yaml.safe_dump({**config, "dataset": {**config["dataset"], "fold": fold}},
                       handle, sort_keys=False)
    logger = setup_logger(str(directory / "train.log"))
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad),
        lr=config["training"]["lr"], weight_decay=config["training"]["weight_decay"])
    epochs = 1 if config["runtime"]["smoke_steps"] else config["training"]["epochs"]
    best = {name: -float("inf") for name in models_to_save}
    if train_loader.dataset.skipped_records:
        pd.DataFrame(train_loader.dataset.skipped_records).to_csv(
            directory / "skipped_dfc_records.csv", index=False)
        logger.info("Skipped %d zero-variance dFC records", len(train_loader.dataset.skipped_records))
    history = []
    run_name = "_".join(models_to_save)
    logger.info("Stage=%s branches=%s train_subjects=%d train_records=%d smoke_steps=%d",
                stage, run_name, train_loader.dataset.df.component_key.nunique(),
                len(train_loader.dataset), config["runtime"]["smoke_steps"])
    for epoch in range(epochs):
        model.train()
        train_loader.sampler.set_epoch(epoch)
        total_loss = 0.
        paired_count = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss, paired = loss_fn(batch)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Stage {stage} epoch {epoch}: non-finite loss.")
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach())
            paired_count += paired
        record = dict(stage=stage, branches=run_name, epoch=epoch + 1, train_loss=total_loss / len(train_loader),
                      paired_samples=paired_count)
        for name, branch in models_to_save.items():
            values, _ = evaluate(branch, valid_loaders[name], name,
                                 config["dataset"]["scales"], config["runtime"]["device"])
            auc = values["auc"]
            if auc is None:
                raise ValueError(f"Validation AUC requires both classes: fold={fold}, modality={name}")
            record.update({f"val_{name}_{key}": value for key, value in values.items()})
            if auc > best[name]:
                best[name] = auc
                torch.save(dict(model_state_dict=branch.state_dict(), epoch=epoch + 1,
                    optimizer_state_dict=optimizer.state_dict(), val_auc=auc,
                    task=task, fold=fold, stage=stage, modality=name),
                    best_checkpoint(root, task, fold, name))
        history.append(record)
        pd.DataFrame(history).to_csv(directory / f"metrics_{run_name}.csv", index=False)
        logger.info(json.dumps(record))
    return history


def final_evaluation(models, config, root, fold):
    """Evaluate each best checkpoint once on internal test and external cohorts."""
    directory = run_directory(root, config["dataset"]["task"], fold)
    directory.mkdir(parents=True, exist_ok=True)
    summary = []
    for mode, model in models.items():
        load_weights(model, best_checkpoint(root, config["dataset"]["task"], fold, mode),
                     config["runtime"]["device"])
        roles = ["test_all"] if mode == "fusion" else [
            "test_all", "external_Diabete-dementia", "external_P-ADIC"]
        for role in roles:
            frame = read_split(config, fold, role, "both" if mode == "fusion" else mode)
            if frame.empty or frame.diagnosis_code.nunique() != 2:
                continue
            loader = make_loader(frame, config, "both" if mode == "fusion" else mode)
            values, subjects = evaluate(model, loader, mode, config["dataset"]["scales"],
                                         config["runtime"]["device"])
            subjects.to_csv(directory / f"predictions_{mode}_{role}.csv", index=False)
            summary.append(dict(fold=fold, modality=mode, split=role, **values))
    pd.DataFrame(summary).to_csv(directory / "evaluation.csv", index=False)
    return summary


def summarize_folds(root, task, folds):
    frames = [pd.read_csv(run_directory(root, task, f) / "evaluation.csv") for f in folds]
    records = pd.concat(frames, ignore_index=True)
    records.to_csv(Path(root) / task / "evaluation_by_fold.csv", index=False)
    metrics = ["accuracy", "auc", "sensitivity", "specificity", "macro_f1"]
    summary = records.groupby(["modality", "split"])[metrics].agg(["mean", "std"])
    summary.to_csv(Path(root) / task / "evaluation_summary.csv")
    print(summary.to_string())
