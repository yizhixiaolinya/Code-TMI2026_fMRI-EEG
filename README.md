# FUSE-AD: Multi-modal Fusion Framework for Alzheimer's Disease Diagnosis

FUSE-AD is a hierarchical multi-modal fusion framework for Alzheimer's Disease (AD) diagnosis that combines functional Magnetic Resonance Imaging (fMRI) and Electroencephalography (EEG) data.

## Key Features

- **Hierarchical Graph Attention Architecture (HGAA)**: Processes brain connectivity networks at multiple ROI scales (100-500 ROIs) with fine-to-coarse pooling
- **Bidirectional Cross-Modal Fusion (BCMF)**: Scale-wise bidirectional cross-attention mechanism for effective cross-modal integration
- **Three-Stage Progressive Training**:
  - **Stage 1**: Single-modal (independent fMRI/EEG encoder training)
  - **Stage 2**: Fusion (train fusion module with frozen encoders, paired data only)
  - **Stage 3**: Multiscale modality-specific alignment (teacher-student distillation with scale-wise L2 alignment)
- **Scale-wise Graph Embeddings**: Fixed-dimension graph-level representations `[B, d]` for each scale, derived from node-level features via global average pooling
- **Missing Modality Handling**: Seamlessly handles fMRI-only, EEG-only, and paired samples
- **5-Fold Cross-Validation**: Subject-level strict separation for reliable evaluation

## Framework Overview

![Code Framework](Code_framework.png)

## Architecture

### Hierarchical Graph Attention Architecture (HGAA)
- **Input**: Multi-scale adjacency matrices `[B, R_s, R_s]` where `R_s` varies by scale (100, 200, 300, 400, 500)
- **Processing**: Fine-to-coarse pooling with GCN layers and optional self-attention
- **Output**: 
  - Node-level features: `[B, R_s, d]` where `d=64` (fixed hidden dimension)
  - Graph-level embeddings: `[B, d]` per scale via global average pooling

### Bidirectional Cross-Modal Fusion (BCMF)
- **Input**: Node-level features from fMRI and EEG encoders `[B, R_s, d]`
- **Process**: Shared similarity matrix computation, bidirectional cross-attention, and feature refinement
- **Output**: Fused graph embeddings `g^{fuse}_s \in \mathbb{R}^{B \times d}` per scale
- **Mixing**: Learnable scale-specific coefficients `λ_s` for weighted fusion in `learnable_mix` mode

### Stage 3 Alignment
- **Objective**: Align single-modal graph embeddings `g^m_s` with fused teacher embeddings `g^{fuse}_s`
- **Loss**: `L_{align}^m = \sum_s \|g^m_s - g^{fuse}_s\|_2^2` where `m \in \{f, e\}`
- **Scope**: Only computed on paired batches

## Installation

```bash
pip install torch numpy pandas scipy scikit-learn pyyaml tensorboard tqdm
```

## Data Preparation

### CSV Format
Required columns: `sub_id`, `eeg_dst`, `fmri_dst`, `diagnosis_cat`, `diagnosis_code`

### Directory Structure
```
Data/Data_all_AD/New-split/Splits_cv5/
├── fold0/
│   ├── train_eeg_only.csv      # EEG-only training samples
│   ├── train_fmri_only.csv     # fMRI-only training samples
│   ├── train_pair.csv          # Paired training samples (Stage 2)
│   ├── train_all.csv           # All training samples (Stage 3)
│   ├── val_eeg_only.csv
│   ├── val_fmri_only.csv
│   ├── val_pair.csv
│   └── val_all.csv
└── fold1-4/  # Similar structure
```

## Training

### Stage 1: Single-Modal Warmup
Train fMRI and EEG encoders independently:
```bash
python Training_stage1.py --config configs/config_stage1.yaml --save_path results/Stage1
```

### Stage 2: Fusion Warmup
Load Stage 1 checkpoints and train fusion module (frozen encoders, paired data only):
```bash
python Training_stage2.py --config configs/config_stage2.yaml --save_path results/Stage2
```

### Stage 3: Multiscale Alignment
Load Stage 2 checkpoint as teacher, train student encoders with scale-wise alignment:
```bash
python Training_stage3.py --config configs/config_stage3.yaml --save_path results/Stage3
```

## Configuration

Key parameters in `configs/config_*.yaml`:
- `embed_dim`: Hidden dimension `d` (default: 64, must match across all stages)
- `use_msre`: Enable Modality-Specific Reliability Estimator (optional)
- `use_corr_align`: Enable correlation alignment before fusion (optional)
- `fusion_mode`: `"bcmf"`, `"learnable_mix"`, or `"concat_only"`
- `freeze_fusion`: Freeze fusion module in Stage 3 (default: true)

## Results

Results are saved in `results/Stage*/` with:
- Best models per fold: `bestmodel/best_model_*_fold{fold}.pth`
- Training logs and metrics
- TensorBoard summaries

### Metrics
- Accuracy (ACC)
- Balanced Accuracy
- Macro F1-Score
- AUC-ROC
- Sensitivity (SEN)
- Specificity (SPE)
- Confusion matrices

### Visualization
```bash
tensorboard --logdir results/Stage1/tb
```
