"""
3-class classification ablation.

Classes:
  0 — low    (vitality <= 1)
  1 — medium (1 < vitality <= 3)
  2 — high   (vitality > 3)
"""
import sys
import argparse
import time
from pathlib import Path
from datetime import datetime

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).parent))

from config.config import ModelConfig, TrainConfig, DataConfig, PathConfig
from config.config_cls import ClassificationConfig
from dataset.dataset import (
    BirchClassificationDataset,
    collate_cls_fn,
    compute_tabular_stats,
    filter_trees_with_images,
    vitality_to_class3,
)
from evaluate.evaluate_cls import run_cls_cv, make_vitality_classes3
from model.model import BirchVitalityClassifier
from train.train_cls import compute_class_weights, train_cls_epoch, predict_cls_loader
from sklearn.model_selection import train_test_split

NUM_CLASSES = 3

COLUMN_NAME = {
    "N (°)": "N",
    "E (°)": "E",
    "circumference (cm)": "circumference_cm",
    "vitality (5 - highest)": "vitality",
    "fungal infection (3 - worst)": "fungal_infection",
}

# ── Original ablation configs ──────────────────────────────────────────────────
ABLATION_CONFIGS = [
    {
        "name": "ViT_baseline",
        "use_dino_segmentation": False,
        "use_tabular": False,
    },
    {
        "name": "ViT_DINO",
        "use_dino_segmentation": True,
        "use_tabular": False,
    },
    {
        "name": "ViT_Tabular",
        "use_dino_segmentation": False,
        "use_tabular": True,
    },
    {
        "name": "ViT_DINO_Tabular",
        "use_dino_segmentation": True,
        "use_tabular": True,
    },
]

# ── Advanced model configs ─────────────────────────────────────────────────────
ADVANCED_ABLATION_CONFIGS = [
    {
        "name": "ViTBase_DINO_Tabular",
        "backbone": "vit_base_patch16_224.dino",
        "seg_model": "vit_base_patch16_224.dino",
        "use_dino_segmentation": True,
        "use_tabular": True,
        "lr": 5e-5,
        "epochs": 60,
        "augmentation": "heavy",
        "augmentation_copies": 3,
    },
    {
        "name": "DINOv2Base_DINO_Tabular",
        "backbone": "vit_base_patch14_dinov2",
        "seg_model": "vit_base_patch14_dinov2",
        "use_dino_segmentation": True,
        "use_tabular": True,
        "lr": 5e-5,
        "epochs": 60,
        "augmentation": "heavy",
        "augmentation_copies": 3,
    },
    {
        "name": "DINOv2Large_DINO_Tabular",
        "backbone": "vit_large_patch14_dinov2",
        "seg_model": "vit_base_patch14_dinov2",
        "use_dino_segmentation": True,
        "use_tabular": True,
        "lr": 2e-5,
        "epochs": 80,
        "augmentation": "heavy",
        "augmentation_copies": 3,
    },
]

CLASS_LABELS = ["low (≤1)", "medium (1-3)", "high (>3)"]


def _make_cls3_paths(model_cfg: ModelConfig) -> PathConfig:
    """Return a PathConfig whose checkpoint/log dirs have a _3cls suffix to avoid
    colliding with the 5-class classification checkpoints."""
    tag = f"{model_cfg.backbone}_{model_cfg.aggregator}_cls"
    if model_cfg.use_dino_segmentation:
        tag += "_dino_segmentation"
    if model_cfg.use_tabular:
        tag += "_tabular"
    tag += "_3cls"
    paths = PathConfig(
        checkpoint_dir=Path(f"outputs/{tag}/checkpoints"),
        log_dir=Path(f"outputs/{tag}/logs"),
    )
    return paths


def build_cls3_ablation_config(ablation: dict) -> ClassificationConfig:
    use_dino = ablation["use_dino_segmentation"]
    use_tab = ablation["use_tabular"]
    backbone = "vit_small_patch16_224.dino" if use_dino else "vit_small_patch16_224"
    model_cfg = ModelConfig(
        backbone=backbone,
        aggregator="attention",
        freeze_backbone=True,
        use_dino_segmentation=use_dino,
        dino_segmentation_model="vit_small_patch16_224.dino",
        dino_segmenation_threshold=0.6,
        use_tabular=use_tab,
        tabular_features=("N", "E", "circumference_cm", "fungal_infection") if use_tab else (),
        tabular_hidden_dim=64,
    )
    return ClassificationConfig(
        model=model_cfg,
        training=TrainConfig(lr=5e-5, epochs=60),
        data=DataConfig(),
        paths=_make_cls3_paths(model_cfg),
        num_classes=NUM_CLASSES,
    )


def build_advanced_cls3_ablation_config(ablation: dict) -> ClassificationConfig:
    model_cfg = ModelConfig(
        backbone=ablation["backbone"],
        aggregator="attention",
        freeze_backbone=True,
        use_dino_segmentation=ablation["use_dino_segmentation"],
        dino_segmentation_model=ablation["seg_model"],
        dino_segmenation_threshold=0.6,
        use_tabular=ablation["use_tabular"],
        tabular_features=("N", "E", "circumference_cm", "fungal_infection") if ablation["use_tabular"] else (),
        tabular_hidden_dim=64,
    )
    return ClassificationConfig(
        model=model_cfg,
        training=TrainConfig(
            lr=ablation.get("lr", 5e-5),
            epochs=ablation.get("epochs", 60),
            use_amp=True,
        ),
        data=DataConfig(
            augmentation=ablation.get("augmentation", "heavy"),
            augmentation_copies=ablation.get("augmentation_copies", 3),
        ),
        paths=_make_cls3_paths(model_cfg),
        num_classes=NUM_CLASSES,
    )


def _train_and_eval_test_cls3(df_train_val, df_test, cfg, fold_results, device, predictions_path=None):
    """Train a final 3-class classifier on all train_val data and evaluate on the held-out test set."""
    if predictions_path is not None and Path(predictions_path).exists():
        print(f"  Test predictions already exist at {predictions_path} — loading from file.")
        pred_df = pd.read_csv(predictions_path)
        acc = pred_df["correct"].mean()
        f1s = []
        for c in range(NUM_CLASSES):
            tp = ((pred_df["class_pred"] == c) & (pred_df["class_true"] == c)).sum()
            fp = ((pred_df["class_pred"] == c) & (pred_df["class_true"] != c)).sum()
            fn = ((pred_df["class_pred"] != c) & (pred_df["class_true"] == c)).sum()
            prec = tp / (tp + fp + 1e-8)
            rec = tp / (tp + fn + 1e-8)
            f1s.append(2 * prec * rec / (prec + rec + 1e-8))
        return {"accuracy": acc, "f1": float(sum(f1s) / len(f1s))}

    tabular_mean, tabular_std = {}, {}
    if cfg.model.use_tabular and cfg.model.tabular_features:
        tabular_mean, tabular_std = compute_tabular_stats(df_train_val, cfg.model.tabular_features)

    shared = dict(
        image_dir=cfg.paths.image_dir,
        image_size=cfg.data.image_size,
        tabular_features=cfg.model.tabular_features,
        tabular_mean=tabular_mean,
        tabular_std=tabular_std,
        image_extension=cfg.data.image_extensions,
        class_fn=vitality_to_class3,
    )
    train_loader = DataLoader(
        BirchClassificationDataset(df_train_val, **shared, mode="train"),
        batch_size=cfg.training.batch_size,
        shuffle=True,
        collate_fn=collate_cls_fn,
        num_workers=cfg.data.num_workers,
    )
    test_loader = DataLoader(
        BirchClassificationDataset(df_test, **shared, mode="val"),
        batch_size=cfg.training.batch_size,
        shuffle=False,
        collate_fn=collate_cls_fn,
        num_workers=cfg.data.num_workers,
    )

    mean_best_epoch = int(round(sum(fr["best_epoch"] for fr in fold_results) / len(fold_results)))
    print(f"  Final model: training for {mean_best_epoch} epochs on all {len(df_train_val)} train_val trees...")

    train_labels = [vitality_to_class3(float(v)) for v in df_train_val["vitality"]]
    class_weights = compute_class_weights(train_labels, NUM_CLASSES)

    n_tab = len(cfg.model.tabular_features) if cfg.model.use_tabular else 0
    model = BirchVitalityClassifier(
        num_classes=NUM_CLASSES,
        backbone_name=cfg.model.backbone,
        aggregator_name=cfg.model.aggregator,
        hidden_dim=cfg.model.hidden_dim,
        dropout=cfg.model.dropout,
        pretrained=cfg.model.pretrained,
        freeze_backbone=cfg.model.freeze_backbone,
        use_dino=cfg.model.use_dino_segmentation,
        dino_seg_threshold=cfg.model.dino_segmenation_threshold,
        dino_seg_model=cfg.model.dino_segmentation_model,
        use_tabular=cfg.model.use_tabular,
        n_tabular_features=n_tab,
        tabular_hidden_dim=cfg.model.tabular_hidden_dim,
    ).to(device)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    use_amp = getattr(cfg.training, "use_amp", False) and device.type == "cuda"
    from torch.amp import GradScaler
    scaler = GradScaler(device="cuda") if use_amp else None

    for epoch in range(1, mean_best_epoch + 1):
        m = train_cls_epoch(model, train_loader, optimizer, criterion, device, NUM_CLASSES, scaler=scaler)
        scheduler.step(m["loss"])
        if epoch % 10 == 0 or epoch == mean_best_epoch:
            print(f"    Epoch {epoch}/{mean_best_epoch} | loss={m['loss']:.4f} | acc={m['accuracy']:.4f}")

    tree_ids, pred_classes, true_classes = predict_cls_loader(model, test_loader, device)
    correct = [int(p == t) for p, t in zip(pred_classes, true_classes)]

    test_acc = sum(correct) / len(correct)
    f1s = []
    for c in range(NUM_CLASSES):
        tp = sum((p == c and t == c) for p, t in zip(pred_classes, true_classes))
        fp = sum((p == c and t != c) for p, t in zip(pred_classes, true_classes))
        fn = sum((p != c and t == c) for p, t in zip(pred_classes, true_classes))
        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        f1s.append(2 * prec * rec / (prec + rec + 1e-8))
    test_f1 = float(sum(f1s) / len(f1s))

    print(f"  Test Acc={test_acc:.4f} | Test F1={test_f1:.4f}")
    print(f"  Class distribution (true): { {CLASS_LABELS[c]: true_classes.count(c) for c in range(NUM_CLASSES)} }")

    if predictions_path is not None:
        pred_df = pd.DataFrame({
            "tree_id": tree_ids,
            "class_true": true_classes,
            "class_pred": pred_classes,
            "correct": correct,
        })
        Path(predictions_path).parent.mkdir(parents=True, exist_ok=True)
        pred_df.to_csv(predictions_path, index=False)
        print(f"  Per-tree predictions saved -> {predictions_path}")

    del model, train_loader, test_loader
    torch.cuda.empty_cache()

    return {"accuracy": test_acc, "f1": test_f1}


def run_cls3_ablation(advanced: bool = False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    mode_tag = "ADVANCED" if advanced else "ORIGINAL"
    print(f"Starting {mode_tag} 3-CLASS CLASSIFICATION ablation at {datetime.now().isoformat()}")
    print(f"Classes: {CLASS_LABELS}")
    print("=" * 60)

    base_cfg = ClassificationConfig()
    df = pd.read_csv(
        base_cfg.paths.csv_path,
        encoding=base_cfg.data.csv_encoding,
        sep=base_cfg.data.csv_sep,
    )
    df = df.rename(columns=COLUMN_NAME)

    df_filtered = filter_trees_with_images(
        df,
        base_cfg.paths.image_dir,
        image_extensions=base_cfg.data.image_extensions,
    )

    cls3_labels = make_vitality_classes3(df_filtered["vitality"])
    df_train_val, df_test = train_test_split(
        df_filtered,
        test_size=0.1,
        random_state=42,
        stratify=cls3_labels,
    )
    print(f"Train/val: {len(df_train_val)} trees | Test hold-out: {len(df_test)} trees")

    # Print class distribution
    for c, label in enumerate(CLASS_LABELS):
        n = (cls3_labels == c).sum()
        print(f"  Class {c} ({label}): {n} trees")
    print("=" * 60)

    fname = "ablation_cls3_advanced_results.csv" if advanced else "ablation_cls3_results.csv"
    output_path = Path("outputs") / fname
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        existing_df = pd.read_csv(output_path)
        all_rows = existing_df.to_dict("records")
        done_experiments = set(existing_df[existing_df["type"] == "summary"]["experiment"].tolist())
        print(f"Found existing results in {output_path}")
        print(f"Already completed: {sorted(done_experiments)}")
    else:
        all_rows = []
        done_experiments = set()

    configs_to_run = ADVANCED_ABLATION_CONFIGS if advanced else ABLATION_CONFIGS

    for ablation in configs_to_run:
        name = ablation["name"]

        if name in done_experiments:
            print(f"\n{'─' * 60}")
            print(f"SKIPPING {name} — already in {output_path.name}")
            print(f"{'─' * 60}")
            continue

        print("\n" + "█" * 60)
        print(f"ABLATION (CLS-3): {name}")
        print(f"DINO segmentation : {ablation['use_dino_segmentation']}")
        print(f"Tabular features: {ablation['use_tabular']}")
        print("█" * 60 + "\n")

        cfg = build_advanced_cls3_ablation_config(ablation) if advanced else build_cls3_ablation_config(ablation)
        cfg.display()

        t0 = time.time()
        results = run_cls_cv(df_train_val, cfg, num_classes=NUM_CLASSES, class_fn=vitality_to_class3)
        elapsed = time.time() - t0

        summary = results["summary"]
        new_rows = []

        for fr in results["fold_results"]:
            new_rows.append({
                "experiment": name,
                "use_dino": ablation["use_dino_segmentation"],
                "use_tabular": ablation["use_tabular"],
                "fold": fr["fold"],
                "val_loss": round(fr["best_val_loss"], 4),
                "val_accuracy": round(fr["best_val_acc"], 4),
                "val_f1": round(fr["best_val_f1"], 4),
                "best_epoch": fr["best_epoch"],
                "test_accuracy": "",
                "test_f1": "",
                "type": "fold",
            })

        pred_fname = "test_predictions_cls3_advanced" if advanced else "test_predictions_cls3"
        predictions_path = Path("outputs") / f"{pred_fname}_{name}.csv"

        print(f"\nEvaluating {name} on held-out test set ({len(df_test)} trees)...")
        test_m = _train_and_eval_test_cls3(
            df_train_val, df_test, cfg, results["fold_results"], device,
            predictions_path=predictions_path,
        )

        new_rows.append({
            "experiment": name,
            "use_dino": ablation["use_dino_segmentation"],
            "use_tabular": ablation["use_tabular"],
            "fold": "mean±std",
            "val_loss": f"{summary['mean_val_loss']:.4f}±{summary['std_val_loss']:.4f}",
            "val_accuracy": f"{summary['mean_val_acc']:.4f}±{summary['std_val_acc']:.4f}",
            "val_f1": f"{summary['mean_val_f1']:.4f}±{summary['std_val_f1']:.4f}",
            "best_epoch": "",
            "test_accuracy": round(test_m["accuracy"], 4),
            "test_f1": round(test_m["f1"], 4),
            "type": "summary",
        })

        all_rows.extend(new_rows)
        done_experiments.add(name)

        # Save immediately after each experiment — crash-safe
        pd.DataFrame(all_rows).to_csv(output_path, index=False)

        print(f"\n {name} completed in {elapsed/60:.1f} min")
        print(f"CV  Accuracy = {summary['mean_val_acc']:.4f} ± {summary['std_val_acc']:.4f}")
        print(f"CV  F1 = {summary['mean_val_f1']:.4f} ± {summary['std_val_f1']:.4f}")
        print(f"Test Accuracy = {test_m['accuracy']:.4f} | Test F1 = {test_m['f1']:.4f}")
        print(f"Results saved -> {output_path}")

    results_df = pd.DataFrame(all_rows)
    print("\n" + "=" * 60)
    print("3-CLASS CLASSIFICATION ABLATION COMPLETE")
    print("=" * 60)
    print(f"Results saved to {output_path.resolve()}")

    summary_df = results_df[results_df["type"] == "summary"][
        ["experiment", "use_dino", "use_tabular", "val_accuracy", "val_f1", "test_accuracy", "test_f1"]
    ]
    print("\n" + summary_df.to_string(index=False))
    print("=" * 60)

    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--advanced",
        action="store_true",
        help="Run advanced model ablations (ViT-Base, DINOv2-Base, DINOv2-Large).",
    )
    args = parser.parse_args()
    run_cls3_ablation(advanced=args.advanced)
