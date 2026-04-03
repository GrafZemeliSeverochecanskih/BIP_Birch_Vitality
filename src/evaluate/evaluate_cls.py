"""
Classification cross-validation runner — mirrors evaluate.py
but builds BirchVitalityClassifier and uses classification metrics.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold

import sys
sys.path.append(str(Path(__file__).parent.parent))

from config.config import Config
from dataset.dataset import (
    build_cls_dataloaders,
    compute_tabular_stats,
    filter_trees_with_images,
    vitality_to_class,
)
from model.model import BirchVitalityClassifier
from train.train_cls import compute_class_weights, train_cls_fold


NUM_CLASSES = 5


def make_vitality_classes(vitality_series):
    """Map continuous vitality to integer class labels for stratification."""
    return vitality_series.apply(lambda v: vitality_to_class(float(v)))


def run_cls_cv(df, config, num_classes=NUM_CLASSES):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Running {config.training.cv_folds}-fold classification CV on {len(df)} trees")

    # Stratify on integer class labels
    cls_labels = make_vitality_classes(df["vitality"])

    skf = StratifiedKFold(
        n_splits=config.training.cv_folds,
        shuffle=True,
        random_state=42,
    )

    fold_results = []
    all_histories = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, cls_labels), start=1):
        print("=" * 50)
        print(f"FOLD {fold} / {config.training.cv_folds}")
        print(f"Train: {len(train_idx)} trees | Val: {len(val_idx)} trees")
        print("=" * 50)

        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)

        # Tabular stats (computed on training fold only)
        tabular_mean, tabular_std = {}, {}
        if config.model.use_tabular and config.model.tabular_features:
            tabular_mean, tabular_std = compute_tabular_stats(
                train_df, config.model.tabular_features
            )
            print(f"Tabular stats: {tabular_mean}")

        train_loader, val_loader = build_cls_dataloaders(
            train_df,
            val_df,
            image_dir=config.paths.image_dir,
            batch_size=config.training.batch_size,
            num_workers=config.data.num_workers,
            image_size=config.data.image_size,
            image_extensions=config.data.image_extensions,
            augmentation=config.data.augmentation,
            tabular_mean=tabular_mean,
            tabular_std=tabular_std,
            tabular_features=config.model.tabular_features,
        )

        # Compute class weights from training fold
        train_labels = [vitality_to_class(float(v)) for v in train_df["vitality"]]
        class_weights = compute_class_weights(train_labels, num_classes)
        print(f"Class weights: {class_weights.tolist()}")

        n_tab = len(config.model.tabular_features) if config.model.use_tabular else 0

        model = BirchVitalityClassifier(
            num_classes=num_classes,
            backbone_name=config.model.backbone,
            aggregator_name=config.model.aggregator,
            hidden_dim=config.model.hidden_dim,
            dropout=config.model.dropout,
            pretrained=config.model.pretrained,
            freeze_backbone=config.model.freeze_backbone,
            use_dino=config.model.use_dino_segmentation,
            dino_seg_threshold=config.model.dino_segmenation_threshold,
            dino_seg_model=config.model.dino_segmentation_model,
            use_tabular=config.model.use_tabular,
            n_tabular_features=n_tab,
            tabular_hidden_dim=config.model.tabular_hidden_dim,
        )

        result = train_cls_fold(
            model, train_loader, val_loader, config, fold, device,
            num_classes=num_classes, class_weights=class_weights,
        )

        fold_results.append({
            "fold": fold,
            "best_val_loss": result["best_val_loss"],
            "best_val_acc": result["best_val_acc"],
            "best_val_f1": result["best_val_f1"],
            "best_epoch": result["best_epoch"],
            "tabular_mean": str(tabular_mean),
            "tabular_std": str(tabular_std),
        })
        all_histories.append(result["history"])

        # Print fold statistics
        train_cls_dist = pd.Series(train_labels).value_counts().sort_index()
        print(f"Train class distribution: {train_cls_dist.to_dict()}")

        print(
            f"Fold {fold} best -> "
            f"loss: {result['best_val_loss']:.4f} | "
            f"acc: {result['best_val_acc']:.4f} | "
            f"F1: {result['best_val_f1']:.4f} | "
            f"(epoch: {result['best_epoch']})"
        )

    results_df = pd.DataFrame(fold_results)
    _save_cls_logs(results_df, all_histories, config)
    summary = _print_cls_summary(results_df, config)

    return {
        "fold_results": fold_results,
        "summary": summary,
        "histories": all_histories,
    }


def _save_cls_logs(results_df, histories, config):
    log_dir = config.paths.log_dir
    results_df.to_csv(log_dir / "cv_cls_results.csv", index=False)
    print("Saved classification fold results")

    for fold_idx, history in enumerate(histories, start=1):
        n_epochs = len(history["train_loss"])
        history_df = pd.DataFrame({
            "epoch": list(range(1, n_epochs + 1)),
            "train_loss": history["train_loss"],
            "train_acc": history["train_acc"],
            "train_f1": history["train_f1"],
            "val_loss": history["val_loss"],
            "val_acc": history["val_acc"],
            "val_f1": history["val_f1"],
        })
        history_df.to_csv(log_dir / f"fold_{fold_idx}_cls_history.csv", index=False)
    print(f"Saved classification epoch histories -> {log_dir}/fold_*_cls_history.csv")


def _print_cls_summary(results_df, config):
    print("=" * 60)
    print("Classification Cross-validation Summary")
    print(f"Backbone: {config.model.backbone}")
    print(f"Aggregator: {config.model.aggregator}")
    print("=" * 60)
    print(f"{'Fold':>5} | {'Val Loss':>9} | {'Val Acc':>8} | {'Val F1':>7} | {'Epoch':>5}")
    print("=" * 60)

    for _, row in results_df.iterrows():
        print(
            f"{int(row['fold']):>5} |"
            f"{row['best_val_loss']:>9.4f} |"
            f"{row['best_val_acc']:>8.4f} |"
            f"{row['best_val_f1']:>7.4f} |"
            f"{int(row['best_epoch']):>5} |"
        )
    print("=" * 60)

    summary = {
        "mean_val_loss": results_df["best_val_loss"].mean(),
        "std_val_loss": results_df["best_val_loss"].std(),
        "mean_val_acc": results_df["best_val_acc"].mean(),
        "std_val_acc": results_df["best_val_acc"].std(),
        "mean_val_f1": results_df["best_val_f1"].mean(),
        "std_val_f1": results_df["best_val_f1"].std(),
    }

    print(
        f"{'Mean':>5} | "
        f"{summary['mean_val_loss']:>9.4f} |"
        f"{summary['mean_val_acc']:>8.4f} |"
        f"{summary['mean_val_f1']:>7.4f} |"
    )
    print(
        f"{'Std':>5} | "
        f"{summary['std_val_loss']:>9.4f} |"
        f"{summary['std_val_acc']:>8.4f} |"
        f"{summary['std_val_f1']:>7.4f} |"
    )
    print("=" * 60)
    print(f"Final: Acc = {summary['mean_val_acc']:.4f} ± {summary['std_val_acc']:.4f}")
    print(f"       F1  = {summary['mean_val_f1']:.4f} ± {summary['std_val_f1']:.4f}")
    return summary
