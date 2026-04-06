import sys
import time
from pathlib import Path
from datetime import datetime

import pandas as pd
import torch

sys.path.append(str(Path(__file__).parent))

from config.config import ModelConfig, TrainConfig, DataConfig, PathConfig
from config.config_cls import ClassificationConfig
from dataset.dataset import filter_trees_with_images, vitality_to_class
from evaluate.evaluate_cls import run_cls_cv, make_vitality_classes
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


def build_cls_ablation_config(ablation: dict) -> ClassificationConfig:
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

    cfg = ClassificationConfig(
        model=model_cfg,
        training=TrainConfig(lr=5e-5, epochs=60),
        data=DataConfig(),
        paths=PathConfig(),
        num_classes=5,
    )
    return cfg


def run_cls_ablation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Starting CLASSIFICATION ablation at {datetime.now().isoformat()}")
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

    cls_labels = make_vitality_classes(df_filtered["vitality"])
    df_train_val, df_test = train_test_split(
        df_filtered,
        test_size=0.1,
        random_state=42,
        stratify=cls_labels,
    )
    print(f"Train/val: {len(df_train_val)} trees | Test hold-out: {len(df_test)} trees")
    print("=" * 60)

    all_rows = []

    for ablation in ABLATION_CONFIGS:
        name = ablation["name"]
        print("\n" + "█" * 60)
        print(f"ABLATION (CLS): {name}")
        print(f"DINO segmentation : {ablation['use_dino_segmentation']}")
        print(f"Tabular features: {ablation['use_tabular']}")
        print("█" * 60 + "\n")

        cfg = build_cls_ablation_config(ablation)
        cfg.display()

        t0 = time.time()
        results = run_cls_cv(df_train_val, cfg, num_classes=cfg.num_classes)
        elapsed = time.time() - t0

        summary = results["summary"]

        for fr in results["fold_results"]:
            all_rows.append({
                "experiment": name,
                "use_dino": ablation["use_dino_segmentation"],
                "use_tabular": ablation["use_tabular"],
                "fold": fr["fold"],
                "val_loss": round(fr["best_val_loss"], 4),
                "val_accuracy": round(fr["best_val_acc"], 4),
                "val_f1": round(fr["best_val_f1"], 4),
                "best_epoch": fr["best_epoch"],
                "type": "fold",
            })

        all_rows.append({
            "experiment": name,
            "use_dino": ablation["use_dino_segmentation"],
            "use_tabular": ablation["use_tabular"],
            "fold": "mean±std",
            "val_loss": f"{summary['mean_val_loss']:.4f}±{summary['std_val_loss']:.4f}",
            "val_accuracy": f"{summary['mean_val_acc']:.4f}±{summary['std_val_acc']:.4f}",
            "val_f1": f"{summary['mean_val_f1']:.4f}±{summary['std_val_f1']:.4f}",
            "best_epoch": "",
            "type": "summary",
        })

        print(f"\n {name} completed in {elapsed/60:.1f} min")
        print(f"Accuracy = {summary['mean_val_acc']:.4f} ± {summary['std_val_acc']:.4f}")
        print(f"F1 = {summary['mean_val_f1']:.4f} ± {summary['std_val_f1']:.4f}")

    results_df = pd.DataFrame(all_rows)
    output_path = Path("outputs") / "ablation_cls_results.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)

    print("\n" + "=" * 60)
    print("CLASSIFICATION ABLATION COMPLETE")
    print("=" * 60)
    print(f"Results saved to {output_path.resolve()}")

    summary_df = results_df[results_df["type"] == "summary"][
        ["experiment", "use_dino", "use_tabular", "val_accuracy", "val_f1"]
    ]
    print("\n" + summary_df.to_string(index=False))
    print("=" * 60)

    return results_df


if __name__ == "__main__":
    run_cls_ablation()
