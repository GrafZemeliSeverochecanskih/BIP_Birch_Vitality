import sys
from pathlib import Path

import pandas as pd
import torch
sys.path.append(str(Path(__file__).parent))

from config.config import Config, EfficientNetAttention, ResNet50Mean, ViTAttention
from dataset.dataset import filter_trees_with_images
from evaluate.evaluate import run_cv

COLUMN_NAME = {
        "N (°)": "N",
        "E (°)": "E",
        "circumference (cm)": "circumference_cm",
        "vitality (5 - highest)": "vitality",
        "fungal infection (3 - worst)": "fungal_infection",
    }

def main(config):
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
    
    results = run_cv(df_filtered, config)
    
    summary = results["summary"]
    print("Final Result")
    print(f"MAE: {summary["mean_val_mae"]:.4f} +- {summary["std_val_mae"]:.4f}")
    print(f"R^2: {summary["mean_val_r2"]:.4f} +- {summary["std_val_r2"]:.4f}")
    print(f"Loss: {summary["mean_val_loss"]:.4f} +- {summary["std_val_loss"]:.4f}")
    
    return results

if __name__ == "__main__":
    config = Config()
    main(config)