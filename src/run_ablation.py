import sys
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
from train.train import train_epoch, val_epoch
from sklearn.model_selection import train_test_split

COLUMN_NAME = {
    "N (°)": "N",
    "E (°)": "E",
    "circumference (cm)": "circumference_cm",
    "vitality (5 - highest)": "vitality",
    "fungal infection (3 - worst)": "fungal_infection",
}

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


def _train_and_eval_test(df_train_val, df_test, cfg, fold_results, device):
    """Train a final model on all train_val data and evaluate on the held-out test set."""
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

    for epoch in range(1, mean_best_epoch + 1):
        m = train_epoch(model, train_loader, optimizer, criterion, device)
        scheduler.step(m["loss"])
        if epoch % 10 == 0 or epoch == mean_best_epoch:
            print(f"    Epoch {epoch}/{mean_best_epoch} | loss={m['loss']:.4f} | MAE={m['mae']:.4f}")

    test_m = val_epoch(model, test_loader, criterion, device)
    print(f"  Test MAE={test_m['mae']:.4f} | Test R²={test_m['r2']:.4f}")
    return test_m


def run_ablation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Starting ablation at {datetime.now().isoformat()}")
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

    all_rows = []

    for ablation in ABLATION_CONFIGS:
        name = ablation["name"]
        print("\n" + "█" * 60)
        print(f"ABLATION: {name}")
        print(f"DINO segmentation : {ablation['use_dino_segmentation']}")
        print(f"Tabular features: {ablation['use_tabular']}")
        print("█" * 60 + "\n")

        cfg = build_ablation_config(ablation)
        cfg.display()

        t0 = time.time()
        results = run_cv(df_train_val, cfg)
        elapsed = time.time() - t0

        summary = results["summary"]

        for fr in results["fold_results"]:
            all_rows.append({
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

        print(f"\nEvaluating {name} on held-out test set ({len(df_test)} trees)...")
        test_m = _train_and_eval_test(df_train_val, df_test, cfg, results["fold_results"], device)

        all_rows.append({
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

        print(f"\n {name} completed in {elapsed/60:.1f} min")
        print(f"CV  MAE = {summary['mean_val_mae']:.4f} ± {summary['std_val_mae']:.4f}")
        print(f"CV  R^2 = {summary['mean_val_r2']:.4f} ± {summary['std_val_r2']:.4f}")
        print(f"Test MAE = {test_m['mae']:.4f} | Test R^2 = {test_m['r2']:.4f}")

    results_df = pd.DataFrame(all_rows)
    output_path = Path("outputs") / "ablation_results.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)

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
    run_ablation()
