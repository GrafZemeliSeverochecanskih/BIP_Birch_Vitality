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

from config.config import Config, ModelConfig, TrainConfig, DataConfig, PathConfig
from dataset.dataset import BirchDataset, collate_fn, compute_tabular_stats, filter_trees_with_images
from evaluate.evaluate import run_cv, make_vitality_bins
from model.model import BirchVitalityModel
from train.train import train_epoch, val_epoch, predict_loader
from sklearn.model_selection import train_test_split

COLUMN_NAME = {
    "N (°)": "N",
    "E (°)": "E",
    "circumference (cm)": "circumference_cm",
    "vitality (5 - highest)": "vitality",
    "fungal infection (3 - worst)": "fungal_infection",
}

# ── Original ablation configs (unchanged — checkpoints preserved) ──────────────
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

# ── Advanced model configs (larger backbones, DINOv2, heavy augmentation) ──────
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
        "seg_model": "vit_base_patch14_dinov2",   # seg stays at base — saves VRAM
        "use_dino_segmentation": True,
        "use_tabular": True,
        "lr": 2e-5,
        "epochs": 80,
        "augmentation": "heavy",
        "augmentation_copies": 3,
    },
]


def build_ablation_config(ablation: dict) -> Config:
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

    cfg = Config(
        model=model_cfg,
        training=TrainConfig(lr=5e-5, epochs=60),
        data=DataConfig(),
        paths=PathConfig(),
    )
    return cfg


def build_advanced_ablation_config(ablation: dict) -> Config:
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
    cfg = Config(
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
        paths=PathConfig(),
    )
    return cfg


def _train_and_eval_test(df_train_val, df_test, cfg, fold_results, device, predictions_path=None):
    """Train a final model on all train_val data and evaluate on the held-out test set.

    If predictions_path is given, saves per-tree predictions to that CSV.
    Skips training entirely if predictions_path already exists.
    """
    if predictions_path is not None and Path(predictions_path).exists():
        print(f"  Test predictions already exist at {predictions_path} — loading from file.")
        pred_df = pd.read_csv(predictions_path)
        mae = pred_df["abs_error"].mean()
        r2_num = ((pred_df["vitality_true"] - pred_df["vitality_pred"]) ** 2).sum()
        r2_den = ((pred_df["vitality_true"] - pred_df["vitality_true"].mean()) ** 2).sum()
        r2 = float(1 - r2_num / r2_den) if r2_den != 0 else 0.0
        return {"mae": mae, "r2": r2}

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
    )
    train_loader = DataLoader(
        BirchDataset(df_train_val, **shared, mode="train"),
        batch_size=cfg.training.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=cfg.data.num_workers,
    )
    test_loader = DataLoader(
        BirchDataset(df_test, **shared, mode="val"),
        batch_size=cfg.training.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=cfg.data.num_workers,
    )

    mean_best_epoch = int(round(sum(fr["best_epoch"] for fr in fold_results) / len(fold_results)))
    print(f"  Final model: training for {mean_best_epoch} epochs on all {len(df_train_val)} train_val trees...")

    n_tab = len(cfg.model.tabular_features) if cfg.model.use_tabular else 0
    model = BirchVitalityModel(
        backbone_name=cfg.model.backbone,
        aggregator_name=cfg.model.aggregator,
        hidden_dim=cfg.model.hidden_dim,
        dropout=cfg.model.dropout,
        pretrained=cfg.model.pretrained,
        freeze_backbone=cfg.model.freeze_backbone,
        img_size=cfg.data.image_size,
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
    criterion = nn.MSELoss()

    use_amp = getattr(cfg.training, "use_amp", False) and device.type == "cuda"
    from torch.amp import GradScaler
    scaler = GradScaler(device="cuda") if use_amp else None

    for epoch in range(1, mean_best_epoch + 1):
        m = train_epoch(model, train_loader, optimizer, criterion, device, scaler=scaler)
        scheduler.step(m["loss"])
        if epoch % 10 == 0 or epoch == mean_best_epoch:
            print(f"    Epoch {epoch}/{mean_best_epoch} | loss={m['loss']:.4f} | MAE={m['mae']:.4f}")

    # Collect per-tree predictions
    tree_ids, preds, targets = predict_loader(model, test_loader, device)
    errors = [p - t for p, t in zip(preds, targets)]
    abs_errors = [abs(e) for e in errors]

    test_mae = sum(abs_errors) / len(abs_errors)
    ss_res = sum((t - p) ** 2 for p, t in zip(preds, targets))
    t_mean = sum(targets) / len(targets)
    ss_tot = sum((t - t_mean) ** 2 for t in targets)
    test_r2 = float(1 - ss_res / ss_tot) if ss_tot != 0 else 0.0

    print(f"  Test MAE={test_mae:.4f} | Test R²={test_r2:.4f}")

    if predictions_path is not None:
        pred_df = pd.DataFrame({
            "tree_id": tree_ids,
            "vitality_true": targets,
            "vitality_pred": [round(p, 4) for p in preds],
            "error": [round(e, 4) for e in errors],
            "abs_error": [round(e, 4) for e in abs_errors],
        })
        Path(predictions_path).parent.mkdir(parents=True, exist_ok=True)
        pred_df.to_csv(predictions_path, index=False)
        print(f"  Per-tree predictions saved -> {predictions_path}")

    del model, train_loader, test_loader
    torch.cuda.empty_cache()

    return {"mae": test_mae, "r2": test_r2}


def run_ablation(advanced: bool = False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    mode_tag = "ADVANCED" if advanced else "ORIGINAL"
    print(f"Starting {mode_tag} ablation at {datetime.now().isoformat()}")
    print("=" * 60)

    base_cfg = Config()
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

    df_train_val, df_test = train_test_split(
        df_filtered,
        test_size=0.1,
        random_state=42,
        stratify=make_vitality_bins(df_filtered["vitality"]),
    )
    print(f"Train/val: {len(df_train_val)} trees | Test hold-out: {len(df_test)} trees")
    print("=" * 60)

    fname = "ablation_advanced_results.csv" if advanced else "ablation_results.csv"
    output_path = Path("outputs") / fname
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load any previously saved results so we can skip completed experiments
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
        print(f"ABLATION: {name}")
        print(f"DINO segmentation : {ablation['use_dino_segmentation']}")
        print(f"Tabular features: {ablation['use_tabular']}")
        print("█" * 60 + "\n")

        cfg = build_advanced_ablation_config(ablation) if advanced else build_ablation_config(ablation)
        cfg.display()

        t0 = time.time()
        results = run_cv(df_train_val, cfg)
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
                "val_mae": round(fr["best_val_mae"], 4),
                "val_r2": round(fr["best_val_r2"], 4),
                "best_epoch": fr["best_epoch"],
                "test_mae": "",
                "test_r2": "",
                "type": "fold",
            })

        pred_fname = "test_predictions_advanced" if advanced else "test_predictions"
        predictions_path = Path("outputs") / f"{pred_fname}_{name}.csv"

        print(f"\nEvaluating {name} on held-out test set ({len(df_test)} trees)...")
        test_m = _train_and_eval_test(
            df_train_val, df_test, cfg, results["fold_results"], device,
            predictions_path=predictions_path,
        )

        new_rows.append({
            "experiment": name,
            "use_dino": ablation["use_dino_segmentation"],
            "use_tabular": ablation["use_tabular"],
            "fold": "mean±std",
            "val_loss": f"{summary['mean_val_loss']:.4f}±{summary['std_val_loss']:.4f}",
            "val_mae": f"{summary['mean_val_mae']:.4f}±{summary['std_val_mae']:.4f}",
            "val_r2": f"{summary['mean_val_r2']:.4f}±{summary['std_val_r2']:.4f}",
            "best_epoch": "",
            "test_mae": round(test_m["mae"], 4),
            "test_r2": round(test_m["r2"], 4),
            "type": "summary",
        })

        all_rows.extend(new_rows)
        done_experiments.add(name)

        # Save immediately after each experiment — crash-safe
        pd.DataFrame(all_rows).to_csv(output_path, index=False)

        print(f"\n {name} completed in {elapsed/60:.1f} min")
        print(f"CV  MAE = {summary['mean_val_mae']:.4f} ± {summary['std_val_mae']:.4f}")
        print(f"CV  R^2 = {summary['mean_val_r2']:.4f} ± {summary['std_val_r2']:.4f}")
        print(f"Test MAE = {test_m['mae']:.4f} | Test R^2 = {test_m['r2']:.4f}")
        print(f"Results saved -> {output_path}")

    results_df = pd.DataFrame(all_rows)
    print("\n" + "=" * 60)
    print("ABLATION COMPLETE")
    print("=" * 60)
    print(f"Results saved to {output_path.resolve()}")

    summary_df = results_df[results_df["type"] == "summary"][
        ["experiment", "use_dino", "use_tabular", "val_mae", "val_r2", "test_mae", "test_r2"]
    ]
    print("\n" + summary_df.to_string(index=False))
    print("=" * 60)

    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--advanced",
        action="store_true",
        help="Run advanced model ablations (ViT-Base, DINOv2-Base, DINOv2-Large) "
             "instead of the original small-model ablations.",
    )
    args = parser.parse_args()
    run_ablation(advanced=args.advanced)
