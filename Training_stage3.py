"""Stage 3: Stage-1 single-modal students distilled from a frozen Stage-2 teacher."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from MyModels import MAHGCNNET, MAHGCNNET_fMRI, MAHGCNNET_EEG
from utils import (training_arguments, selected_folds, read_split,
    make_loader, build_A_list, build_E_list, best_checkpoint, load_weights, fit,
    final_evaluation, summarize_folds)


def _kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    teacher_probs = F.softmax(teacher_logits.detach() / tau, dim=1)
    student_log_probs = F.log_softmax(student_logits / tau, dim=1)
    return tau**2 * F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")


def mixed_stage3_loss(students, teacher, batch, scales, device, eta=.5, kappa=.5, tau=2.):
    """CE over all available samples per modality; KD over the paired subset only."""
    fmri, eeg, labels, f_idx, e_idx, paired_f, paired_e = batch
    labels = labels.to(device)
    outputs = {}
    total = torch.zeros((), device=device)
    for mode, data, indices, builder in (
        ("fmri", fmri, f_idx, build_A_list), ("eeg", eeg, e_idx, build_E_list)
    ):
        if len(indices):
            outputs[mode] = students[mode](builder(data, scales, device=device))
            total = total + F.cross_entropy(outputs[mode]["logits"], labels[indices.to(device)])
    if len(paired_f):
        teacher.eval()
        with torch.no_grad():
            target = teacher(build_A_list(fmri, scales, idx=paired_f, device=device),
                             build_E_list(eeg, scales, idx=paired_e, device=device))
        for mode, indices in (("fmri", paired_f), ("eeg", paired_e)):
            indices = indices.to(device)
            source = outputs[mode]
            feature_loss = sum(
                (feature[indices] - fused.detach()).square().sum(-1).mean()
                for feature, fused in zip(source["feat_layers"], target["fused_features"])
            )
            total = total + eta * feature_loss + kappa * _kd_loss(
                source["logits"][indices], target["fused_logits"], tau)
    return total, len(paired_f)


def run_once(config, save_path, fold):
    ds, tr = config["dataset"], config["training"]
    device = config["runtime"]["device"]
    students = nn.ModuleDict({
        "fmri": MAHGCNNET_fMRI(min(ds["scales"]), ds["layers"], ds["num_class"]),
        "eeg": MAHGCNNET_EEG(min(ds["scales"]), ds["layers"], ds["num_class"]),
    }).to(device)
    if not config["runtime"]["eval_only"]:
        for mode, student in students.items():
            load_weights(student, best_checkpoint(tr["stage1"]["stage1_result_dir"],
                         ds["task"], fold, mode), device)
        teacher = MAHGCNNET(min(ds["scales"]), ds["layers"], ds["num_class"]).to(device)
        load_weights(teacher, best_checkpoint(tr["stage2"]["stage2_result_dir"],
                     ds["task"], fold, "fusion"), device)
        teacher.requires_grad_(False)
        teacher.eval()
        train = make_loader(read_split(config, fold, "train_all", "all"), config,
                            "all", training=True, dfc=True)
        val = {mode: make_loader(read_split(config, fold, "val_all", mode), config, mode)
               for mode in students}

        def loss_fn(batch):
            return mixed_stage3_loss(students, teacher, batch, ds["scales"], device,
                                     tr["eta"], tr["kappa"], tr["tau"])

        fit(students, train, val, dict(students.items()), loss_fn,
            config, save_path, fold, stage=3)
    return final_evaluation(dict(students.items()), config, save_path, fold)


def main(config, save_path):
    folds = list(selected_folds(config))
    for fold in folds:
        run_once(config, save_path, fold)
    summarize_folds(save_path, config["dataset"]["task"], folds)


if __name__ == "__main__":
    main(*training_arguments(3))
