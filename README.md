# BIP — Birch Vitality Prediction

A deep-learning pipeline for predicting the **vitality score** (1–5) of birch trees from field photographs. Trees are modelled as *bags of images* using **Multiple Instance Learning (MIL)**: every photo of a tree is encoded individually and the per-image features are aggregated into a single tree-level representation before regression.

---

## Table of Contents

- [Problem](#problem)
- [Data](#data)
- [Project Structure](#project-structure)
- [Model Architecture](#model-architecture)
- [Configuration](#configuration)
- [Setup](#setup)
- [Usage](#usage)
- [Outputs](#outputs)

---

## Problem

Urban birch trees in Bratislava were surveyed and assigned a vitality score on a **1–5 scale** (5 = excellent health). Each tree was photographed from multiple angles. The goal is to train a model that, given a set of photos of a previously unseen tree, predicts its vitality score.

---

## Data

| Item | Details |
|---|---|
| `data/birch_trees_bratislava.csv` | Metadata for all surveyed trees — tree ID, GPS coordinates (N, E), trunk circumference (cm), vitality score (1–5), fungal infection score (1–3) |
| `data/<tree_id>/` | One folder per tree containing the field photographs (`.jpg`) |

The dataset contains **~95 trees**, of which ~80 have associated images and are used for modelling. A 90/10 stratified train-test split is applied, with vitality binned into 4 groups to ensure balanced splits.

---

## Project Structure

```
bip/
├── data/
│   ├── birch_trees_bratislava.csv   # Tree metadata
│   └── <tree_id>/                  # One folder per tree with photos
├── outputs/
│   ├── <backbone>_<aggregator>/
│   │   ├── checkpoints/            # Saved model weights
│   │   └── logs/                   # CV results & per-epoch history CSVs
│   ├── test_set.csv                # Held-out test split
│   └── test_results.csv            # Per-tree predictions on test set
├── src/
│   ├── main.py                     # Training entry point
│   ├── predict.py                  # Inference entry point
│   ├── config/
│   │   └── config.py               # All hyperparameters & paths
│   ├── dataset/
│   │   └── dataset.py              # BirchDataset, DataLoader, augmentations
│   ├── model/
│   │   └── model.py                # BirchVitalityModel (backbone + aggregator + head)
│   ├── train/
│   │   └── train.py                # train_epoch, val_epoch, EarlyStopping, checkpointing
│   ├── evaluate/
│   │   ├── evaluate.py             # K-fold cross-validation runner
│   │   └── evaluate_test_set.py    # Final evaluation on held-out test set
│   └── notebooks/
│       └── 01_eda.ipynb            # Exploratory data analysis
└── requirements.txt
```

---

## Model Architecture

```
Photos (N × 3 × 224 × 224)
        │
        ▼
  CNN/ViT Backbone          ← pretrained on ImageNet via timm
        │  (N × feature_dim)
        ▼
    Aggregator              ← mean / max / attention
        │  (feature_dim,)
        ▼
  Regression Head           ← Linear → ReLU → Dropout → Linear
        │  (scalar)
        ▼
  Vitality score  (1–5)
```

### Backbones (via `timm`)

| Config preset | Backbone | Aggregator |
|---|---|---|
| `Config` (default) | `vit_base_patch16_224` | attention |
| `ViTAttention` | `vit_small_patch16_224` | attention |
| `EfficientNetAttention` | `efficientnet_b2` | attention |
| `ResNet50Mean` | `resnet50` | mean |

### Aggregators

| Name | Description |
|---|---|
| `mean` | Simple average of all image features |
| `max` | Element-wise max over image features |
| `attention` | Learnable softmax attention — weights each image by its informativeness |

### Training details

- **Loss**: MSE
- **Optimiser**: Adam with weight decay
- **Scheduler**: ReduceLROnPlateau (factor 0.5, patience 5)
- **Early stopping**: patience 10 epochs (saves best weights per fold)
- **Cross-validation**: stratified K-fold (default 3 folds)
- **Resume**: training can be resumed from the last saved checkpoint automatically

---

## Configuration

All settings live in `src/config/config.py` and are grouped into four dataclasses:

```python
ModelConfig   # backbone, aggregator, hidden_dim, dropout, pretrained, freeze_backbone
TrainConfig   # epochs, batch_size, lr, weight_decay, patience, cv_folds, resume
DataConfig    # image_size, num_workers, augmentation ("light" | "heavy")
PathConfig    # data_dir, image_dir, csv_path, output_dir, checkpoint_dir, log_dir
```

To run with a different preset, change the `config` passed to `main()`:

```python
from config.config import ViTAttention, EfficientNetAttention, ResNet50Mean
config = EfficientNetAttention()
```

---

## Setup

**1. Create and activate a virtual environment**

```bash
python -m venv venv
venv\Scripts\activate
```

**2. Install PyTorch (GPU — CUDA 12.4)**

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

For CPU-only:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**3. Install remaining dependencies**

```bash
pip install -r requirements.txt
```

---

## Usage

### Train

Runs stratified K-fold CV, saves per-fold checkpoints, promotes the best fold's weights to `best_model.pt`, and saves the held-out test split.

```bash
python src/main.py
```

### Evaluate on the held-out test set

```bash
python src/evaluate/evaluate_test_set.py
```

Prints MAE, RMSE, and R² on the test set and saves per-tree results to `outputs/test_results.csv`.

### Predict vitality for a new tree

```bash
python src/predict.py --images path/to/tree/photos/
```

Or with a specific checkpoint:

```bash
python src/predict.py \
  --images path/to/tree/photos/ \
  --checkpoint outputs/vit_base_patch16_224_attention/checkpoints/best_model.pt
```

The script loads all images from the folder, runs the MIL forward pass, and prints the raw, clipped (1.0–5.0), and rounded (0.5-step) vitality prediction.

---

## Outputs

After training, the following files are produced:

| Path | Contents |
|---|---|
| `outputs/<model>/checkpoints/fold_<n>_best.pt` | Best weights for fold `n` |
| `outputs/<model>/checkpoints/best_model.pt` | Best weights across all folds |
| `outputs/<model>/logs/cv_results.csv` | Per-fold validation MAE / R² / loss |
| `outputs/<model>/logs/fold_<n>_history.csv` | Epoch-level training curves |
| `outputs/test_set.csv` | Held-out test trees (generated by `main.py`) |
| `outputs/test_results.csv` | Per-tree predictions on the test set |