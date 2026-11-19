# FE-KTNet: fMRI-EEG Knowledge Transfer Network

FE-KTNet learns from paired fMRI-EEG data and transfers multimodal knowledge to models that use a single modality at inference.

![Framework overview](Code_framework.png)

## Training and evaluation

Run from the repository root after supplying the data loader and prepared inputs:

```bash
python Training_stage1.py --config configs/config_stage1.yaml --task AD-NC --fold 0
python Training_stage2.py --config configs/config_stage2.yaml --task AD-NC --fold 0
python Training_stage3.py --config configs/config_stage3.yaml --task AD-NC --fold 0
```

Omit `--fold` to run all five folds. Use `--task MCI-NC` or `--task AD-MCI` for the other tasks. 

Outputs under `results/StageN/{task}/fold{fold}/` include the configuration, logs, best validation-AUC checkpoints, and subject predictions. 

Use `--eval-only` to evaluate saved checkpoints or `--device cpu` for CPU execution. For a short training check, use `--smoke-steps 2` with a separate `--save_path`. 
