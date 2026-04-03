import sys
from pathlib import Path

import pandas as pd
import torch
sys.path.append(str(Path(__file__).parent))

from config.config import Config, EfficientNetAttention, ResNet50Mean, ViTAttention
from dataset.dataset import filter_trees_with_images
from evaluate.evaluate import run_cv

from sklearn.model_selection import train_test_split
from evaluate.evaluate import make_vitality_bins

COLUMN_NAME = {
        "N (°)": "N",
        "E (°)": "E",
        "circumference (cm)": "circumference_cm",
        "vitality (5 - highest)": "vitality",
        "fungal infection (3 - worst)": "fungal_infection",
    }

def main(config=None):
    if config is None:
        config = Config()
    config.display()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    print(f"Loading data from {config.paths.csv_path}")
    df = pd.read_csv(
        config.paths.csv_path,
        encoding=config.data.csv_encoding,
        sep=config.data.csv_sep
    )
    
    df = df.rename(columns=COLUMN_NAME)
    print(f"Loaded {len(df)} trees")
    
    df_filtered = filter_trees_with_images(
        df,
        config.paths.image_dir,
        image_extensions=config.data.image_extensions
    )
    print(f"Using {len(df_filtered)} trees for CV")
    
    df_train_val, df_test = train_test_split(
        df_filtered,
        test_size=0.1,
        random_state=42,
        stratify = make_vitality_bins(df_filtered["vitality"])
    )

    print(f"Train/val: {len(df_train_val)} trees")
    print(f"Test: {len(df_test)} trees")
    results = run_cv(df_train_val, config)
    
    best_fold = min(results["fold_results"], key=lambda x: x["best_val_mae"])
    best_fold_idx = best_fold["fold"]
    
    import json
    if config.model.use_tabular:
        best_fold_data = next(
            r for r in results["fold_results"] if r["fold"] == best_fold_idx
        )
        tab_stats = {
            "features" : list(config.model.tabular_features),
            "mean": eval(best_fold_data["tabular_mean"]),
            "std": eval(best_fold_data["tabular_std"])
        }

        stats_path = config.paths.output_dir / "tabular_stats.json"
        with open(stats_path, "w") as f:
            json.dump(tab_stats, f, indent=2)
        print(f"Tabular stats saved to {stats_path}")

    
    import shutil
    best_src  = config.paths.checkpoint_dir / f"fold_{best_fold_idx}_best.pt"
    best_dst  = config.paths.checkpoint_dir / "best_model.pt"
    shutil.copy(best_src, best_dst)
    print(f"Best fold: {best_fold_idx} (MAE: {best_fold['best_val_mae']:.4f})")
    print(f"Best weights saved to {best_dst}")

    test_csv_path = config.paths.output_dir / "test_set.csv"
    df_test.to_csv(test_csv_path, index=False)
    print(f"Test set saved to {test_csv_path}")

    summary = results["summary"]
    print("Final Result")
    print(f"MAE: {summary['mean_val_mae']:.4f} +- {summary['std_val_mae']:.4f}")
    print(f"R^2: {summary['mean_val_r2']:.4f} +- {summary['std_val_r2']:.4f}")
    print(f"Loss: {summary['mean_val_loss']:.4f} +- {summary['std_val_loss']:.4f}")
    
    return results

if __name__ == "__main__":
    config = Config()
    main(config)