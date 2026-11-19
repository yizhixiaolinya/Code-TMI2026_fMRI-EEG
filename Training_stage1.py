"""Stage 1: independent HGAA encoders on all available training subjects."""
import torch.nn.functional as F
from MyModels import MAHGCNNET_fMRI, MAHGCNNET_EEG
from utils import (training_arguments, selected_folds, read_split,
    make_loader, build_A_list, build_E_list, fit, final_evaluation, summarize_folds)


def run_once(config, save_path, fold):
    ds = config["dataset"]
    device = config["runtime"]["device"]
    models = {
        "fmri": MAHGCNNET_fMRI(min(ds["scales"]), ds["layers"], ds["num_class"]).to(device),
        "eeg": MAHGCNNET_EEG(min(ds["scales"]), ds["layers"], ds["num_class"]).to(device),
    }
    if not config["runtime"]["eval_only"]:
        for mode, model in models.items():
            train = make_loader(read_split(config, fold, "train_all", mode), config,
                                mode, training=True)
            val = make_loader(read_split(config, fold, "val_all", mode), config, mode)

            def loss_fn(batch):
                fmri, eeg, labels, _, _, _, _ = batch
                inputs = (build_A_list(fmri, ds["scales"], device=device) if mode == "fmri"
                          else build_E_list(eeg, ds["scales"], device=device))
                return F.cross_entropy(model(inputs)["logits"], labels.to(device)), 0

            fit(model, train, {mode: val}, {mode: model}, loss_fn, config,
                save_path, fold, stage=1)
    return final_evaluation(models, config, save_path, fold)


def main(config, save_path):
    folds = list(selected_folds(config))
    for fold in folds:
        run_once(config, save_path, fold)
    summarize_folds(save_path, config["dataset"]["task"], folds)


if __name__ == "__main__":
    main(*training_arguments(1))
