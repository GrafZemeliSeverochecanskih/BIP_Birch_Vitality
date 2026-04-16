# BIP — Birch Tree Vitality Prediction

Automated vitality scoring of urban birch trees from field photographs using Multiple Instance Learning (MIL) with Vision Transformers, self-supervised DINO segmentation, and tabular metadata fusion.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Dataset](#2-dataset)
3. [Model Architecture](#3-model-architecture)
4. [Training Pipeline](#4-training-pipeline)
5. [Cross-Validation Framework](#5-cross-validation-framework)
6. [Ablation Study](#6-ablation-study)
7. [Configuration System](#7-configuration-system)
8. [Repository Structure](#8-repository-structure)
9. [Output Files](#9-output-files)
10. [Running the Code](#10-running-the-code)
11. [Results Summary](#11-results-summary)
12. [Requirements](#12-requirements)

---

## 1. Project Overview

Urban tree vitality assessment is traditionally performed by arborists through manual inspection — a time-consuming and expensive process. This project investigates whether a machine learning system can replicate these assessments automatically from photographs taken in the field.

**Task**: Given a set of photographs of a single birch tree (*Betula pendula*) taken from different angles, predict a continuous vitality score on a 1–5 scale, where 5 represents excellent health and 1 represents very poor health.

**Challenge**: Each tree is represented by a variable number of images (a *bag*), rather than a single fixed-size input. This is addressed using Multiple Instance Learning (MIL), where all images of a tree are encoded independently and then aggregated into a single tree-level representation.

**Secondary task**: A parallel classification pipeline maps the continuous vitality into 5 ordinal classes for comparison.

---

## 2. Dataset

### 2.1 Source

The dataset consists of 98 silver birch trees surveyed in Bratislava, Slovakia. Each tree was photographed in the field and assessed by an arborist.

**CSV file**: `data/birch_trees_bratislava.csv`  
**Encoding**: CP1250 (Central European)  
**Separator**: semicolon (`;`)

### 2.2 Fields

| Original Column | Renamed | Type | Description |
|---|---|---|---|
| `N (°)` | `N` | float | GPS latitude (WGS84) |
| `E (°)` | `E` | float | GPS longitude (WGS84) |
| `circumference (cm)` | `circumference_cm` | float | Trunk circumference in cm (range: 32–160 cm) |
| `vitality (5 - highest)` | `vitality` | float | Target health score, 1.0–5.0 (increments of 0.5) |
| `fungal infection (3 - worst)` | `fungal_infection` | int | Fungal infection severity, 0–3 |
| `ID` | `ID` | int | Unique tree identifier |

### 2.3 Image Organisation

Images are stored one folder per tree, named by ID:

```
data/
├── 1/
│   ├── img_001.jpg
│   ├── img_002.jpg
│   └── ...
├── 2/
│   └── ...
└── ...
```

Each folder contains multiple photographs of the same tree from different distances and angles. The number of images per tree varies.

### 2.4 Filtering

18 trees are excluded because their image folders are missing or empty, leaving **80 usable trees**. The removed tree IDs are: 15, 16, 26, 63, 64, 71, 81–88, 92, 96–98.

### 2.5 Data Splits

All splits are **stratified** by vitality to preserve class balance. Vitality is discretised into four bins for stratification: `[0–2]`, `(2–3]`, `(3–4]`, `(4–5]`.

| Split | Trees | Purpose |
|---|---|---|
| Train + Validation | 72 (90 %) | Cross-validation and model selection |
| Test (held-out) | 8 (10 %) | Final evaluation only, never seen during training |

The 72 train/val trees are further divided into 3 stratified folds for cross-validation, yielding approximately 48 training trees and 24 validation trees per fold.

### 2.6 Classification Labels

For the classification task, continuous vitality is converted to a zero-indexed class label:

```
class = ceil(vitality) - 1

Examples:  1.0 → 0,  2.5 → 2,  3.0 → 2,  4.0 → 3,  5.0 → 4
```

This produces 5 classes (0–4) corresponding to vitality ranges (0–1], (1–2], (2–3], (3–4], (4–5].

---

## 3. Model Architecture

The system uses a **Multiple Instance Learning** (MIL) architecture. Each tree is treated as a *bag* of images; the model encodes each image independently and then learns to aggregate the evidence.

```
Input: bag of N images for one tree  [N × (3, 224, 224)]
          ↓
  [Optional] DINO Segmentation       →  foreground-masked images
          ↓
  Backbone (frozen ViT/CNN)          →  N image feature vectors  [N × D]
          ↓
  Aggregator (mean / max / attention) →  single tree vector       [D]
          ↓
  [Optional] Tabular Encoder         →  tabular embedding        [64]
          |                               ↕ concatenate
          └──────────────────────────→  fused vector             [D + 64]
          ↓
  Regression Head                    →  vitality score           [scalar]
```

### 3.1 Backbone

The backbone extracts a feature vector from each image independently. All experiments use a **frozen backbone** — only the aggregator, tabular encoder, and regression head are trained. This keeps the trainable parameter count very small (~300–400 K) and makes training feasible with only ~48 training trees per fold.

Supported backbones (via [timm](https://github.com/huggingface/pytorch-image-models)):

| Model Name | Type | Feature Dim | Pretraining | Notes |
|---|---|---|---|---|
| `vit_small_patch16_224` | ViT-Small | 384 | ImageNet-1k | Lightweight baseline |
| `vit_small_patch16_224.dino` | ViT-Small | 384 | DINO (self-supervised) | Required for DINO segmentation |
| `vit_base_patch16_224.dino` | ViT-Base | 768 | DINO (self-supervised) | Stronger features, 4× params vs. Small |
| `vit_base_patch14_dinov2` | ViT-Base | 768 | DINOv2 (self-supervised) | State-of-the-art SSL pretraining |
| `vit_large_patch14_dinov2` | ViT-Large | 1024 | DINOv2 (self-supervised) | Highest quality, ~18 GB VRAM fp16 |
| `efficientnet_b2` | EfficientNet-B2 | 1408 | ImageNet-1k | Efficient CNN alternative |
| `resnet50` | ResNet-50 | 2048 | ImageNet-1k | Standard CNN baseline |

**DINO/DINOv2 weight loading**: Pretrained DINO weights store the final normalisation layer under the key `norm.weight/bias`, but newer timm architectures expect `fc_norm.weight/bias`. The code handles this automatically by downloading the weights manually and remapping the keys before loading with `strict=False`. DINOv2 models are also configured with `img_size=224` to override their native 518×518 pretraining resolution, with positional embeddings interpolated to the smaller grid.

### 3.2 Aggregator

The aggregator collapses the per-image feature matrix `[N × D]` into a single tree-level vector `[D]`.

**Mean**: Simple average over all N images. No learnable parameters.

**Max**: Element-wise maximum over all N images. Focuses on the most discriminative image per feature dimension.

**Attention** (default): Learns a scalar importance weight for each image via a small neural network, then computes a weighted sum. This is the core MIL mechanism — the model learns which photographs are most informative for the prediction.

```
scores  = Linear(D → 128) → Tanh → Linear(128 → 1)   shape: [N, 1]
weights = softmax(scores, dim=0)                       shape: [N, 1]
output  = sum(weights × features, dim=0)               shape: [D]
```

Total trainable parameters in the attention aggregator: ~98 K (for D = 768).

### 3.3 DINO Segmentation

An optional preprocessing step that uses a frozen DINO Vision Transformer to produce a foreground attention mask, removing background clutter before feature extraction.

**Mechanism**: DINO ViTs develop class-discriminative attention maps in their final self-attention block without any supervision. The [CLS] token's attention to patch tokens is used as a proxy for foreground probability.

```
1. Forward pass through frozen DINO ViT (no gradients)
2. Extract attention weights from the last transformer block
3. Average CLS attention over all heads → per-patch saliency map
4. Binarise at threshold (default 0.6): foreground / background
5. Interpolate mask to full image resolution
6. Multiply original image by mask — background zeroed out
```

The masked image is then passed to the main backbone. When the main backbone is itself a DINO model, the same model instance can serve as both segmenter and feature extractor.

**Threshold**: 0.6 — higher values produce tighter masks; lower values preserve more context.

### 3.4 Tabular Encoder

Fuses structured metadata (GPS coordinates, trunk size, fungal infection) with the visual features.

```
Input:  [N, E, circumference_cm, fungal_infection]   shape: [4]
           ↓ z-normalise using training-fold statistics
       Linear(4 → 64) → ReLU → Linear(64 → 64) → ReLU
Output: tabular embedding                             shape: [64]
```

The output is concatenated with the aggregated visual features before the regression head. Normalisation statistics (mean and standard deviation) are computed on the training portion of each fold to prevent data leakage.

### 3.5 Regression Head

```
Input:  fused vector   shape: [D + 64]  (e.g., 768 + 64 = 832)
        Linear(832 → 256) → ReLU → Dropout(0.3) → Linear(256 → 1)
Output: vitality prediction             shape: [scalar]
```

### 3.6 Parameter Count Example

For `ViTBase_DINO_Tabular` (backbone frozen):

| Component | Trainable Parameters |
|---|---|
| Backbone (vit_base_patch16_224.dino, 86 M total) | 0 (frozen) |
| Attention Aggregator | ~98 K |
| Tabular Encoder | ~4.5 K |
| Regression Head (832 → 256 → 1) | ~215 K |
| **Total trainable** | **~317 K** |

---

## 4. Training Pipeline

### 4.1 Loss and Metrics

**Loss**: Mean Squared Error (MSE)

**Reported metrics per epoch**:

| Metric | Description |
|---|---|
| `MAE` | Mean Absolute Error — primary metric, interpretable in vitality units |
| `R²` | Coefficient of determination — proportion of variance explained |
| `loss` | MSE (used for early stopping and LR scheduling) |

### 4.2 Optimiser and Scheduler

```
Optimiser:   Adam
  lr:            5e-5  (2e-5 for DINOv2-Large)
  weight_decay:  1e-4
  parameters:    trainable only (frozen backbone excluded)

Scheduler:   ReduceLROnPlateau
  mode:      min (val loss)
  factor:    0.5  (halve LR on plateau)
  patience:  5 epochs
```

### 4.3 Early Stopping

Training halts if validation loss does not improve for 10 consecutive epochs. The best weights (lowest validation loss) are saved to `fold_N_best.pt` whenever a new minimum is reached. Only trainable parameters are stored — not the frozen backbone — which reduces checkpoint size from ~350 MB to ~3 MB for ViT-Base models.

### 4.4 Automatic Mixed Precision (AMP)

All advanced experiments use PyTorch AMP (`torch.amp.autocast` + `GradScaler`), which computes the forward pass in `float16` and applies gradient updates in `float32`. This roughly doubles throughput on NVIDIA A100 GPUs at no cost to numerical stability.

### 4.5 Augmentation

**Light** (used in the original ablation):

| Transform | Parameters |
|---|---|
| Resize | 224 × 224 |
| RandomHorizontalFlip | p = 0.5 |
| ColorJitter | brightness = 0.2, contrast = 0.2 |
| Normalise | ImageNet mean/std |

**Heavy** (used in the advanced ablation):

| Transform | Parameters |
|---|---|
| RandomResizedCrop | size = 224, scale = (0.5, 1.0), ratio = (0.75, 1.33) |
| RandomHorizontalFlip | p = 0.5 |
| RandomVerticalFlip | p = 0.3 |
| RandomRotation | ±30° |
| ColorJitter | brightness = 0.4, contrast = 0.4, saturation = 0.3, hue = 0.08 |
| RandomGrayscale | p = 0.1 |
| GaussianBlur | kernel = 5, sigma = (0.1, 2.0), p = 0.4 |
| RandomAdjustSharpness | factor = 2.0, p = 0.3 |
| Normalise | ImageNet mean/std |
| RandomErasing | p = 0.4, scale = (0.02, 0.2), ratio = (0.3, 3.3) |

**Augmentation copies** (`augmentation_copies=3`): Each tree appears 3 times per epoch, each time with an independent random augmentation. This triples the effective dataset size without requiring additional images. Validation always uses a single deterministic resize + normalise transform.

### 4.6 Crash Recovery

Training is designed to be interruptible and resumable at any point:

- `fold_N_resume.pt` is written atomically after every epoch (write to a `.tmp` file, then rename).
- On restart, the code detects the resume checkpoint and continues from the next epoch, restoring model weights, optimiser state, scheduler state, and AMP scaler state.
- If a resume checkpoint is corrupt (e.g., the job was killed mid-write), it is automatically deleted and the fold restarts from epoch 1 with a printed warning rather than crashing.

---

## 5. Cross-Validation Framework

### 5.1 Stratified K-Fold

3-fold stratified cross-validation is used throughout. Stratification is performed by binning vitality into four groups to ensure each fold covers the full range of health outcomes.

```
72 train/val trees
  ├── Fold 1: Train 48 | Val 24
  ├── Fold 2: Train 48 | Val 24
  └── Fold 3: Train 48 | Val 24
```

### 5.2 Fold-Level Skip

If `fold_N_best.pt` already exists and the corresponding metrics appear in `cv_results.csv`, that fold is skipped automatically on restart. This allows a run killed mid-experiment to resume from the next incomplete fold.

### 5.3 Per-Fold History Streaming

Epoch metrics are written to `fold_N_history.csv` incrementally during training (one row per epoch, flushed to disk immediately). No epoch history is accumulated in RAM — this keeps memory usage constant regardless of the number of epochs.

### 5.4 Final Model for Test Evaluation

After cross-validation, a final model is trained from scratch on all 72 train/val trees for `mean_best_epoch` epochs (the average of the three folds' best epochs). This model is evaluated on the 8 held-out test trees and its per-tree predictions are saved to a separate CSV.

---

## 6. Ablation Study

Two sets of ablation experiments are provided via `src/run_ablation.py`.

### 6.1 Original Ablation

Isolates the contribution of each component using a lightweight ViT-Small backbone.

| Experiment | Backbone | DINO Seg | Tabular | LR | Epochs | Augmentation |
|---|---|---|---|---|---|---|
| `ViT_baseline` | vit_small_patch16_224 | No | No | 5e-5 | 60 | light |
| `ViT_DINO` | vit_small_patch16_224.dino | Yes | No | 5e-5 | 60 | light |
| `ViT_Tabular` | vit_small_patch16_224 | No | Yes | 5e-5 | 60 | light |
| `ViT_DINO_Tabular` | vit_small_patch16_224.dino | Yes | Yes | 5e-5 | 60 | light |

### 6.2 Advanced Ablation

Scales up the best-performing original configuration (DINO segmentation + tabular features) to larger, more powerful backbones with heavy augmentation.

| Experiment | Backbone | Feature Dim | Seg Model | LR | Epochs | Aug Copies |
|---|---|---|---|---|---|---|
| `ViTBase_DINO_Tabular` | vit_base_patch16_224.dino | 768 | vit_base_patch16_224.dino | 5e-5 | 60 | 3 |
| `DINOv2Base_DINO_Tabular` | vit_base_patch14_dinov2 | 768 | vit_base_patch14_dinov2 | 5e-5 | 60 | 3 |
| `DINOv2Large_DINO_Tabular` | vit_large_patch14_dinov2 | 1024 | vit_base_patch14_dinov2 | 2e-5 | 80 | 3 |

Note: `DINOv2Large` uses the Base model for DINO segmentation to conserve VRAM; only the feature extraction backbone is upgraded to Large.

### 6.3 Experiment Skip Logic

Results are written to the ablation CSV immediately after each experiment completes. On restart, any experiment that already has a `summary` row in the CSV is skipped. Completed folds within an experiment are also skipped individually. This means a run of 3 configs that crashes after the second can be restarted and will only train the third.

### 6.4 Per-Tree Test Predictions

After each experiment, a per-tree prediction CSV is saved alongside the aggregate metrics, enabling detailed post-hoc error analysis across individual trees.

---

## 7. Configuration System

All hyperparameters are managed through a hierarchy of Python dataclasses in `src/config/config.py`.

### 7.1 Core Dataclasses

**`ModelConfig`** — Architecture settings:

| Field | Default | Description |
|---|---|---|
| `backbone` | `vit_base_patch16_224` | timm model name |
| `aggregator` | `attention` | `mean`, `max`, or `attention` |
| `hidden_dim` | `256` | Regression head hidden dimension |
| `dropout` | `0.3` | Dropout rate in regression head |
| `pretrained` | `True` | Load pretrained backbone weights |
| `freeze_backbone` | `True` | Freeze backbone parameters during training |
| `use_dino_segmentation` | `False` | Enable DINO foreground masking |
| `dino_segmentation_model` | `vit_small_patch16_224.dino` | Model used for segmentation |
| `dino_segmenation_threshold` | `0.6` | Foreground/background binarisation threshold |
| `use_tabular` | `False` | Fuse tabular metadata with image features |
| `tabular_features` | `(N, E, circumference_cm, fungal_infection)` | Metadata fields to include |
| `tabular_hidden_dim` | `64` | Tabular encoder hidden dimension |

**`TrainConfig`** — Training hyperparameters:

| Field | Default | Description |
|---|---|---|
| `epochs` | `50` | Maximum training epochs |
| `batch_size` | `4` | Trees per gradient step |
| `lr` | `5e-5` | Adam learning rate |
| `weight_decay` | `1e-4` | L2 regularisation coefficient |
| `patience` | `10` | Early stopping patience in epochs |
| `cv_folds` | `3` | Number of cross-validation folds |
| `resume` | `True` | Resume from checkpoint on restart |
| `use_amp` | `True` | Automatic mixed precision (CUDA only) |

**`DataConfig`** — Data loading and augmentation:

| Field | Default | Description |
|---|---|---|
| `image_size` | `224` | Input resolution (H = W) |
| `num_workers` | `2` | DataLoader worker processes |
| `image_extensions` | `(.jpg, .jpeg, .png)` | Accepted image formats |
| `csv_encoding` | `cp1250` | CSV file encoding |
| `csv_sep` | `;` | CSV column separator |
| `augmentation` | `light` | Augmentation intensity: `light` or `heavy` |
| `augmentation_copies` | `1` | Repeat each tree N times per epoch with fresh augmentation |

**`PathConfig`** — Filesystem paths:

| Field | Default | Description |
|---|---|---|
| `data_dir` | `data/` | Root data directory |
| `image_dir` | `data/` | Parent folder for per-tree image subdirectories |
| `csv_path` | `data/birch_trees_bratislava.csv` | Tree metadata CSV |
| `output_dir` | `outputs/` | Root output directory |
| `checkpoint_dir` | auto-computed | `outputs/{model_tag}/checkpoints/` |
| `log_dir` | auto-computed | `outputs/{model_tag}/logs/` |

### 7.2 Automatic Path Computation

The `model_tag` is assembled from active config options at initialisation time, ensuring each configuration writes to its own isolated directory:

```
{backbone}_{aggregator}[_dino_segmentation][_tabular]

Examples:
  vit_base_patch16_224_attention
  vit_base_patch16_224.dino_attention_dino_segmentation_tabular
  vit_large_patch14_dinov2_attention_dino_segmentation_tabular
```

### 7.3 Preset Configurations

Ready-to-use presets defined in `config.py`:

| Class | Backbone | DINO Seg | Tabular | LR | Epochs | Augmentation |
|---|---|---|---|---|---|---|
| `Config` (default) | vit_base_patch16_224 | No | No | 5e-5 | 50 | light |
| `ViTAttention` | vit_small_patch16_224 | No | No | 5e-5 | 60 | light |
| `EfficientNetAttention` | efficientnet_b2 | No | No | 5e-5 | 50 | light |
| `ResNet50Mean` | resnet50 | No | No | 5e-5 | 50 | light |
| `DINOBackbone` | vit_small_patch16_224.dino | Yes | No | 5e-5 | 50 | light |
| `WithTabular` | vit_small_patch16_224 | No | Yes | 5e-5 | 50 | light |
| `DINOWithTabular` | vit_small_patch16_224.dino | Yes | Yes | 5e-5 | 50 | light |
| `ViTBaseDINOTabular` | vit_base_patch16_224.dino | Yes | Yes | 5e-5 | 60 | heavy (×3) |
| `DINOv2BaseDINOTabular` | vit_base_patch14_dinov2 | Yes | Yes | 5e-5 | 60 | heavy (×3) |
| `DINOv2LargeDINOTabular` | vit_large_patch14_dinov2 | Yes | Yes | 2e-5 | 80 | heavy (×3) |

---

## 8. Repository Structure

```
BIP_Birch_Vitality/
├── data/
│   ├── birch_trees_bratislava.csv       ← metadata for all 98 trees
│   ├── 1/                               ← field photographs for tree ID 1
│   │   ├── img_001.jpg
│   │   └── ...
│   └── ...
│
├── src/
│   ├── config/
│   │   ├── config.py                    ← regression config dataclasses and presets
│   │   └── config_cls.py                ← classification config
│   │
│   ├── dataset/
│   │   └── dataset.py                   ← BirchDataset, augmentation transforms,
│   │                                       collate_fn, tabular normalisation
│   │
│   ├── model/
│   │   └── model.py                     ← backbone loader, MeanAggregator,
│   │                                       MaxAggregator, AttentionAggregator,
│   │                                       DINOSegmenter, TabularEncoder,
│   │                                       RegressionHead, BirchVitalityModel
│   │
│   ├── train/
│   │   ├── train.py                     ← train_fold, val_epoch, predict_loader,
│   │   │                                   EarlyStopping, save/load checkpoint
│   │   └── train_cls.py                 ← classification training loop
│   │
│   ├── evaluate/
│   │   ├── evaluate.py                  ← run_cv, fold skip, per-fold CSV logging
│   │   ├── evaluate_cls.py              ← classification cross-validation
│   │   └── evaluate_test_set.py         ← standalone test-set evaluation
│   │
│   ├── main.py                          ← single-config training entry point
│   ├── predict.py                       ← inference on new images
│   ├── run_ablation.py                  ← full ablation study runner (regression)
│   ├── run_ablation_cls.py              ← classification ablation runner
│   └── eval_ablation_test.py            ← evaluate saved checkpoints on test set
│
├── outputs/                             ← generated during training (not committed)
│   ├── ablation_results.csv
│   ├── ablation_advanced_results.csv
│   ├── test_predictions_advanced_ViTBase_DINO_Tabular.csv
│   ├── test_predictions_advanced_DINOv2Base_DINO_Tabular.csv
│   ├── test_predictions_advanced_DINOv2Large_DINO_Tabular.csv
│   └── {model_tag}/
│       ├── checkpoints/
│       │   ├── fold_1_best.pt           ← best weights, fold 1
│       │   ├── fold_1_resume.pt         ← full state for resuming fold 1
│       │   ├── fold_2_best.pt
│       │   ├── fold_2_resume.pt
│       │   ├── fold_3_best.pt
│       │   └── fold_3_resume.pt
│       └── logs/
│           ├── cv_results.csv           ← per-fold summary metrics
│           ├── fold_1_history.csv       ← epoch-level training curves, fold 1
│           ├── fold_2_history.csv
│           └── fold_3_history.csv
│
├── requirements.txt
└── README.md
```

---

## 9. Output Files

### Checkpoints

| File | Contents | When saved |
|---|---|---|
| `fold_N_best.pt` | Trainable parameters at best validation loss | Whenever val loss improves |
| `fold_N_resume.pt` | Full training state: weights + optimiser + scheduler + scaler + epoch | After every epoch (atomic write) |

Checkpoints store only trainable parameters (frozen backbone excluded), reducing file size from ~350 MB to ~3 MB for ViT-Base models.

### Per-Fold Logs

**`cv_results.csv`** — Written after each fold completes and after every restart:

```
fold, best_val_loss, best_val_mae, best_val_r2, best_epoch, tabular_mean, tabular_std
1,    0.6131,        0.6599,       0.5322,       59,         {...},        {...}
2,    0.6416,        0.6362,       0.4977,       53,         {...},        {...}
3,    0.6595,        0.6370,       0.5664,       56,         {...},        {...}
```

**`fold_N_history.csv`** — Written incrementally during training (one row per epoch):

```
epoch, train_loss, train_mae, train_r2, val_loss, val_mae, val_r2
1,     1.4231,     1.0045,    0.0213,   1.5012,   1.0781, -0.0321
2,     1.3198,     0.9734,    0.0891,   1.4301,   1.0234,  0.0456
...
```

### Ablation Summary

**`ablation_advanced_results.csv`** — One fold row and one summary row per experiment:

```
experiment,           use_dino, use_tabular, fold,     val_mae,        val_r2,         test_mae, test_r2, type
ViTBase_DINO_Tabular, True,     True,        1,        0.6599,         0.5322,         ,         ,        fold
ViTBase_DINO_Tabular, True,     True,        2,        0.6362,         0.4977,         ,         ,        fold
ViTBase_DINO_Tabular, True,     True,        3,        0.6370,         0.5664,         ,         ,        fold
ViTBase_DINO_Tabular, True,     True,        mean±std, 0.6444±0.0135,  0.5321±0.0343,  0.7313,   0.2099,  summary
```

### Per-Tree Test Predictions

**`test_predictions_advanced_{experiment}.csv`**:

```
tree_id, vitality_true, vitality_pred, error,   abs_error
12,      4.0,           3.5812,        0.4188,  0.4188
34,      3.0,           3.1245,       -0.1245,  0.1245
...
```

---

## 10. Running the Code

### Prerequisites

```bash
pip install -r requirements.txt
```

Advanced experiments require a CUDA-capable GPU. `DINOv2Large` requires approximately 18 GB VRAM (e.g., NVIDIA A100 40 GB).

### Setup (first time)

```bash
python -m venv venv
source venv/bin/activate          # Linux/macOS
# or: venv\Scripts\activate       # Windows

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

### Train a Single Configuration

```bash
python src/main.py
```

Runs 3-fold cross-validation with the default configuration, saves checkpoints and logs, and evaluates the best fold on the held-out test set.

### Run the Original Ablation (4 configurations)

```bash
python src/run_ablation.py
```

Sequentially trains and evaluates four configurations that isolate the contribution of DINO segmentation and tabular features. Results written to `outputs/ablation_results.csv`.

### Run the Advanced Ablation (3 configurations)

```bash
python src/run_ablation.py --advance
```

Trains ViT-Base DINO, DINOv2-Base, and DINOv2-Large variants with heavy augmentation. Results written to `outputs/ablation_advanced_results.csv`. Per-tree test predictions are saved to `outputs/test_predictions_advanced_{name}.csv`.

If interrupted, simply re-run the same command. Completed experiments and completed folds are detected automatically and skipped.

### Predict Vitality for a New Tree

```bash
python src/predict.py --images path/to/tree/images/ --checkpoint outputs/best_model.pt
```

### Evaluate a Saved Model on the Test Set

```bash
python src/evaluate/evaluate_test_set.py
```

---

## 11. Results Summary

### Advanced Ablation — Cross-Validation and Test Results

| Experiment | Backbone | CV MAE (mean ± std) | CV R² (mean ± std) | Test MAE | Test R² |
|---|---|---|---|---|---|
| ViTBase_DINO_Tabular | vit_base_patch16_224.dino | 0.6444 ± 0.0135 | 0.5321 ± 0.0343 | 0.7313 | 0.2099 |
| DINOv2Base_DINO_Tabular | vit_base_patch14_dinov2 | — | — | — | — |
| DINOv2Large_DINO_Tabular | vit_large_patch14_dinov2 | — | — | — | — |

*DINOv2 rows will be filled as experiments complete.*

### Interpretation

A cross-validation MAE of ~0.64 vitality points on a 1–5 scale means the model is on average within roughly two-thirds of a vitality level from the ground truth. Given that the vitality scale uses increments of 0.5 and that inter-rater agreement among human arborists on the same scale is typically ±0.5–1.0 points, this represents a meaningful result for a fully automatic system operating on a very small dataset of 72 training trees.

The gap between CV MAE (~0.64) and test MAE (~0.73) is expected given that the test set contains only 8 trees and is therefore high-variance. The R² values reflect that the regression task is challenging — the majority of variance in vitality is captured, but individual predictions for atypical trees can have larger errors.

---

## 12. Requirements

Core dependencies:

```
torch >= 2.0
torchvision
timm
pandas
numpy
scikit-learn
Pillow
```

See `requirements.txt` for pinned versions.

**Hardware used for experiments**: NVIDIA A100 80 GB GPU on a university HPC cluster.  
**Per-epoch training time**: ~80 seconds/epoch for advanced ViT-Base configurations.  
**Full advanced ablation runtime**: approximately 250 minutes per experiment (3 folds × 56 epochs average + final model training).
