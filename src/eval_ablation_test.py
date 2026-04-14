"""
eval_ablation_test.py

Checks whether ablation fold checkpoints already exist and, if so,
evaluates each trained ablation on the held-out test set without retraining.

For each ablation it looks for:
  outputs/<model_tag>/checkpoints/fold_1_best.pt
  outputs/<model_tag>/checkpoints/fold_2_best.pt
  outputs/<model_tag>/checkpoints/fold_3_best.pt

If all fold weights are found it picks the best fold by val_loss (from the
saved CV log CSV) and runs inference on the test split.  Ablations with
missing checkpoints are reported and skipped.

Usage:
    cd D:/bip
    python src/eval_ablation_test.py           # regression
    python src/eval_ablation_test.py --cls     # classification
    python src/eval_ablation_test.py --both    # both
"""

import ast
import sys
import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).parent))

from config.config import Config, ModelConfig, TrainConfig, DataConfig, PathConfig
from config.config_cls import ClassificationConfig
from dataset.dataset import (
    BirchDataset,
    BirchClassificationDataset,
    collate_fn,
    collate_cls_fn,
    compute_tabular_stats,
    filter_trees_with_images,
    vitality_to_class,
)
from evaluate.evaluate import make_vitality_bins
from evaluate.evaluate_cls import make_vitality_classes
from model.model import BirchVitalityModel, BirchVitalityClassifier
from train.train import val_epoch, mean_absolute_error, r2_score
from train.train_cls import val_cls_epoch

COLUMN_NAME = {
    "N (°)": "N",
    "E (°)": "E",
    "circumference (cm)": "circumference_cm",
    "vitality (5 - highest)": "vitality",
    "fungal infection (3 - worst)": "fungal_infection",
}

ABLATION_CONFIGS = [
    {"name": "ViT_baseline",      "use_dino_segmentation": False, "use_tabular": False},
    {"name": "ViT_DINO",          "use_dino_segmentation": True,  "use_tabular": False},
    {"name": "ViT_Tabular",       "use_dino_segmentation": False, "use_tabular": True},
    {"name": "ViT_DINO_Tabular",  "use_dino_segmentation": True,  "use_tabular": True},
]


# ── Config builders ───────────────────────────────────────────────────────────

def build_reg_config(ablation: dict) -> Config:
    use_dino = ablation["use_dino_segmentation"]
    use_tab  = ablation["use_tabular"]
    backbone = "vit_small_patch16_224.dino" if use_dino else "vit_small_patch16_224"
    return Config(
        model=ModelConfig(
            backbone=backbone,
            aggregator="attention",
            freeze_backbone=True,
            use_dino_segmentation=use_dino,
            dino_segmentation_model="vit_small_patch16_224.dino",
            dino_segmenation_threshold=0.6,
            use_tabular=use_tab,
            tabular_features=("N", "E", "circumference_cm", "fungal_infection") if use_tab else (),
            tabular_hidden_dim=64,
        ),
        training=TrainConfig(lr=5e-5, epochs=60),
        data=DataConfig(),
        paths=PathConfig(),
    )


def build_cls_config(ablation: dict) -> ClassificationConfig:
    use_dino = ablation["use_dino_segmentation"]
    use_tab  = ablation["use_tabular"]
    backbone = "vit_small_patch16_224.dino" if use_dino else "vit_small_patch16_224"
    return ClassificationConfig(
        model=ModelConfig(
            backbone=backbone,
            aggregator="attention",
            freeze_backbone=True,
            use_dino_segmentation=use_dino,
            dino_segmentation_model="vit_small_patch16_224.dino",
            dino_segmenation_threshold=0.6,
            use_tabular=use_tab,
            tabular_features=("N", "E", "circumference_cm", "fungal_infection") if use_tab else (),
            tabular_hidden_dim=64,
        ),
        training=TrainConfig(lr=5e-5, epochs=60),
        data=DataConfig(),
        paths=PathConfig(),
        num_classes=5,
    )


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def check_checkpoints(checkpoint_dir: Path, n_folds: int):
    """Return (found: list[int], missing: list[int]) fold indices (1-based)."""
    found, missing = [], []
    for k in range(1, n_folds + 1):
        p = checkpoint_dir / f"fold_{k}_best.pt"
        (found if p.exists() else missing).append(k)
    return found, missing


def pick_best_fold(checkpoint_dir: Path, cv_log: Path) -> tuple[int, dict]:
    """
    Read the CV log CSV and return (best_fold_index, row_dict) for the fold
    with lowest val_loss.  Falls back to fold 1 if the log is missing.
    """
    if cv_log.exists():
        df = pd.read_csv(cv_log)
        best_row = df.loc[df["best_val_loss"].idxmin()]
        return int(best_row["fold"]), best_row.to_dict()
    return 1, {}


def load_tabular_stats(cv_log: Path, best_fold: int, fallback_df, tabular_features):
    """
    Try to recover tabular stats from the CV log for the chosen fold.
    Falls back to recomputing from the full df if the log is unavailable.
    """
    if cv_log.exists():
        df = pd.read_csv(cv_log)
        row = df[df["fold"] == best_fold]
        if not row.empty:
            try:
                mean = ast.literal_eval(row.iloc[0]["tabular_mean"])
                std  = ast.literal_eval(row.iloc[0]["tabular_std"])
                if mean:  # non-empty dict means tabular was used
                    return mean, std
            except Exception:
                pass
    # Fallback: compute from the full train_val set
    return compute_tabular_stats(fallback_df, tabular_features)


# ── Regression test evaluation ────────────────────────────────────────────────

def eval_reg_ablation(ablation, df_train_val, df_test, device):
    name = ablation["name"]
    cfg  = build_reg_config(ablation)
    n_folds = cfg.training.cv_folds

    found, missing = check_checkpoints(cfg.paths.checkpoint_dir, n_folds)
    print(f"\n{'─'*60}")
    print(f"[REG] {name}")
    print(f"  Checkpoint dir : {cfg.paths.checkpoint_dir}")
    print(f"  Folds found    : {found}")
    if missing:
        print(f"  Folds MISSING  : {missing}  ← skipping test evaluation")
        return None

    cv_log = cfg.paths.log_dir / "cv_results.csv"
    best_fold, cv_row = pick_best_fold(cfg.paths.checkpoint_dir, cv_log)
    checkpoint = cfg.paths.checkpoint_dir / f"fold_{best_fold}_best.pt"
    print(f"  Best fold      : {best_fold}  (val_loss={cv_row.get('best_val_loss', '?')})")
    print(f"  Checkpoint     : {checkpoint}")

    # Tabular stats
    tabular_mean, tabular_std = {}, {}
    if cfg.model.use_tabular and cfg.model.tabular_features:
        tabular_mean, tabular_std = load_tabular_stats(
            cv_log, best_fold, df_train_val, cfg.model.tabular_features
        )

    # Test loader
    test_ds = BirchDataset(
        df_test,
        image_dir=cfg.paths.image_dir,
        mode="val",
        image_size=cfg.data.image_size,
        image_extension=cfg.data.image_extensions,
        tabular_features=cfg.model.tabular_features,
        tabular_mean=tabular_mean,
        tabular_std=tabular_std,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=cfg.data.num_workers,
    )

    # Model
    n_tab = len(cfg.model.tabular_features) if cfg.model.use_tabular else 0
    model = BirchVitalityModel(
        backbone_name=cfg.model.backbone,
        aggregator_name=cfg.model.aggregator,
        hidden_dim=cfg.model.hidden_dim,
        dropout=cfg.model.dropout,
        pretrained=False,
        freeze_backbone=cfg.model.freeze_backbone,
        use_dino=cfg.model.use_dino_segmentation,
        dino_seg_threshold=cfg.model.dino_segmenation_threshold,
        dino_seg_model=cfg.model.dino_segmentation_model,
        use_tabular=cfg.model.use_tabular,
        n_tabular_features=n_tab,
        tabular_hidden_dim=cfg.model.tabular_hidden_dim,
    )
    model.load_state_dict(torch.load(checkpoint, weights_only=True))
    model = model.to(device)
    model.eval()

    import torch.nn as nn
    criterion = nn.MSELoss()
    test_m = val_epoch(model, test_loader, criterion, device)

    print(f"  Test MAE = {test_m['mae']:.4f}  |  Test R² = {test_m['r2']:.4f}")
    return {"experiment": name, "test_mae": test_m["mae"], "test_r2": test_m["r2"],
            "best_fold": best_fold, "checkpoint": str(checkpoint)}


# ── Classification test evaluation ───────────────────────────────────────────

def eval_cls_ablation(ablation, df_train_val, df_test, device):
    name = ablation["name"]
    cfg  = build_cls_config(ablation)
    n_folds = cfg.training.cv_folds

    found, missing = check_checkpoints(cfg.paths.checkpoint_dir, n_folds)
    print(f"\n{'─'*60}")
    print(f"[CLS] {name}")
    print(f"  Checkpoint dir : {cfg.paths.checkpoint_dir}")
    print(f"  Folds found    : {found}")
    if missing:
        print(f"  Folds MISSING  : {missing}  ← skipping test evaluation")
        return None

    cv_log = cfg.paths.log_dir / "cv_cls_results.csv"
    best_fold, cv_row = pick_best_fold(cfg.paths.checkpoint_dir, cv_log)
    checkpoint = cfg.paths.checkpoint_dir / f"fold_{best_fold}_best.pt"
    print(f"  Best fold      : {best_fold}  (val_loss={cv_row.get('best_val_loss', '?')})")
    print(f"  Checkpoint     : {checkpoint}")

    # Tabular stats
    tabular_mean, tabular_std = {}, {}
    if cfg.model.use_tabular and cfg.model.tabular_features:
        tabular_mean, tabular_std = load_tabular_stats(
            cv_log, best_fold, df_train_val, cfg.model.tabular_features
        )

    # Test loader
    test_ds = BirchClassificationDataset(
        df_test,
        image_dir=cfg.paths.image_dir,
        mode="val",
        image_size=cfg.data.image_size,
        image_extension=cfg.data.image_extensions,
        tabular_features=cfg.model.tabular_features,
        tabular_mean=tabular_mean,
        tabular_std=tabular_std,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        collate_fn=collate_cls_fn,
        num_workers=cfg.data.num_workers,
    )

    # Model
    n_tab = len(cfg.model.tabular_features) if cfg.model.use_tabular else 0
    model = BirchVitalityClassifier(
        num_classes=cfg.num_classes,
        backbone_name=cfg.model.backbone,
        aggregator_name=cfg.model.aggregator,
        hidden_dim=cfg.model.hidden_dim,
        dropout=cfg.model.dropout,
        pretrained=False,
        freeze_backbone=cfg.model.freeze_backbone,
        use_dino=cfg.model.use_dino_segmentation,
        dino_seg_threshold=cfg.model.dino_segmenation_threshold,
        dino_seg_model=cfg.model.dino_segmentation_model,
        use_tabular=cfg.model.use_tabular,
        n_tabular_features=n_tab,
        tabular_hidden_dim=cfg.model.tabular_hidden_dim,
    )
    model.load_state_dict(torch.load(checkpoint, weights_only=True))
    model = model.to(device)
    model.eval()

    import torch.nn as nn
    criterion = nn.CrossEntropyLoss()
    test_m = val_cls_epoch(model, test_loader, criterion, device, cfg.num_classes)

    print(f"  Test Acc = {test_m['accuracy']:.4f}  |  Test F1 = {test_m['f1']:.4f}")
    return {"experiment": name, "test_accuracy": test_m["accuracy"], "test_f1": test_m["f1"],
            "best_fold": best_fold, "checkpoint": str(checkpoint)}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(cfg):
    df = pd.read_csv(cfg.paths.csv_path, encoding=cfg.data.csv_encoding, sep=cfg.data.csv_sep)
    df = df.rename(columns=COLUMN_NAME)
    df = filter_trees_with_images(df, cfg.paths.image_dir, image_extensions=cfg.data.image_extensions)
    return df


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--cls",  action="store_true", help="Classification ablations only")
    group.add_argument("--both", action="store_true", help="Both regression and classification")
    args = parser.parse_args()

    run_reg = not args.cls
    run_cls = args.cls or args.both

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load data and reproduce the same 90/10 split ──────────────────────────
    base_cfg = Config()
    df = load_data(base_cfg)

    # Regression split (stratified on vitality bins)
    df_train_val_reg, df_test_reg = train_test_split(
        df, test_size=0.1, random_state=42,
        stratify=make_vitality_bins(df["vitality"]),
    )

    # Classification split (stratified on class labels)
    df_train_val_cls, df_test_cls = train_test_split(
        df, test_size=0.1, random_state=42,
        stratify=make_vitality_classes(df["vitality"]),
    )

    print(f"\nTrain/val: {len(df_train_val_reg)} trees | Test: {len(df_test_reg)} trees")
    print("=" * 60)

    reg_rows, cls_rows = [], []

    if run_reg:
        print("\n" + "█" * 60)
        print("REGRESSION ABLATIONS — test-set evaluation")
        print("█" * 60)
        for ablation in ABLATION_CONFIGS:
            row = eval_reg_ablation(ablation, df_train_val_reg, df_test_reg, device)
            if row:
                reg_rows.append(row)

        if reg_rows:
            out = Path("outputs") / "ablation_test_results.csv"
            out.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(reg_rows).to_csv(out, index=False)
            print(f"\nRegression test results saved → {out.resolve()}")
            print(pd.DataFrame(reg_rows)[["experiment", "test_mae", "test_r2"]].to_string(index=False))

    if run_cls:
        print("\n" + "█" * 60)
        print("CLASSIFICATION ABLATIONS — test-set evaluation")
        print("█" * 60)
        for ablation in ABLATION_CONFIGS:
            row = eval_cls_ablation(ablation, df_train_val_cls, df_test_cls, device)
            if row:
                cls_rows.append(row)

        if cls_rows:
            out = Path("outputs") / "ablation_cls_test_results.csv"
            out.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(cls_rows).to_csv(out, index=False)
            print(f"\nClassification test results saved → {out.resolve()}")
            print(pd.DataFrame(cls_rows)[["experiment", "test_accuracy", "test_f1"]].to_string(index=False))

    if not reg_rows and not cls_rows:
        print("\nNo complete ablations found. Train first with run_ablation.py / run_ablation_cls.py")


if __name__ == "__main__":
    main()
