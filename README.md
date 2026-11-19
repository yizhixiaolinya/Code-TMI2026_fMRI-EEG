# FE-KTNet: fMRI-EEG Knowledge Transfer Network

![TMI2026](https://img.shields.io/badge/TMI-2026-blue)
![MICCAI2026](https://img.shields.io/badge/MICCAI-2026-green)

This repository hosts the shared implementation for our TMI 2026 and MICCAI 2026 fMRI-EEG studies. All code updates are maintained here.

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
