"""Stage 2: scale-wise BCMF trained on paired training dFC, with fused CE only."""
import torch.nn.functional as F
import torch
from MyModels import MAHGCNNET
from utils import (training_arguments, selected_folds, read_split,
    make_loader, build_A_list, build_E_list, best_checkpoint, fit,
    final_evaluation, summarize_folds)


def run_once(config, save_path, fold):
    ds, tr = config["dataset"], config["training"]
    device = config["runtime"]["device"]
    model = MAHGCNNET(min(ds["scales"]), ds["layers"], ds["num_class"]).to(device)
    if not config["runtime"]["eval_only"]:
        stage1 = tr["stage1"]["stage1_result_dir"]
        states = [torch.load(best_checkpoint(stage1, ds["task"], fold, mode),
                            map_location=device, weights_only=False)["model_state_dict"]
                  for mode in ("fmri", "eeg")]
        model.load_stage1(*states)
        train = make_loader(read_split(config, fold, "train_all", "both"),
                            config, "both", training=True, dfc=True)
        val = make_loader(read_split(config, fold, "val_all", "both"), config, "both")

        def loss_fn(batch):
            fmri, eeg, labels, _, _, _, _ = batch
            output = model(build_A_list(fmri, ds["scales"], device=device),
                           build_E_list(eeg, ds["scales"], device=device))
            return F.cross_entropy(output["fused_logits"], labels.to(device)), len(labels)

        fit(model, train, {"fusion": val}, {"fusion": model}, loss_fn,
            config, save_path, fold, stage=2)
    return final_evaluation({"fusion": model}, config, save_path, fold)


def main(config, save_path):
    folds = list(selected_folds(config))
    for fold in folds:
        run_once(config, save_path, fold)
    summarize_folds(save_path, config["dataset"]["task"], folds)


if __name__ == "__main__":
    main(*training_arguments(2))
