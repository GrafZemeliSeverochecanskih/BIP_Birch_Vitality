from dataset.dataset import compute_tabular_stats
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))  

from config.config import Config
from dataset.dataset import build_dataloaders, filter_trees_with_images
from model.model import BirchVitalityModel, build_model
from train.train import train_fold


def make_vitality_bins(vitality):
    return pd.cut(
        vitality,
        bins = [0, 2, 3, 4, 5],
        labels = ["low", "medium", "good", "high"]
    )
    
def run_cv(df, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Running {config.training.cv_folds}-fold CV on {len(df)} trees")
    
    vitality_bins = make_vitality_bins(df["vitality"])
    
    skf = StratifiedKFold(
        n_splits=config.training.cv_folds,
        shuffle=True,
        random_state=42
    )
    
    fold_results = list()
    all_histories = list()
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, vitality_bins), start=1):
        print("="*50)
        print(f"FOLD {fold} / {config.training.cv_folds}")
        print(f"Train: {len(train_idx)} trees | Val: {len(val_idx)} trees")
        print("="*50)
        
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)
        
        tabular_mean, tabular_std = dict(), dict()

        if config.model.use_tabular and config.model.tabular_features:
            tabular_mean, tabular_std = compute_tabular_stats(
                train_df, config.model.tabular_features
            )
            print(f"Tabular stats: {tabular_mean}")

        train_loader, val_loader = build_dataloaders(
            train_df,
            val_df,
            image_dir = config.paths.image_dir,
            batch_size = config.training.batch_size,
            num_workers = config.data.num_workers,
            image_size = config.data.image_size,
            image_extensions=config.data.image_extensions,
            augmentation=config.data.augmentation,
            tabular_mean=tabular_mean,
            tabular_std=tabular_std,
            tabular_features=config.model.tabular_features
        )
        
        n_tab = len(config.model.tabular_features) if config.model.use_tabular else 0

        model = BirchVitalityModel(
            backbone_name = config.model.backbone,
            aggregator_name = config.model.aggregator,
            hidden_dim = config.model.hidden_dim,
            dropout = config.model.dropout,
            pretrained = config.model.pretrained,
            freeze_backbone = config.model.freeze_backbone,
            use_dino=config.model.use_dino_segmentation,
            dino_seg_threshold=config.model.dino_seg_threshold,
            dino_seg_model=config.model.dino_seg_model,
            use_tabular=config.model.use_tabular,
            n_tabular_features=n_tab,
            tabular_hidden_dim=config.model.tabular_hidden_dim
        )
        
        result = train_fold(model, train_loader, val_loader, config, fold, device)
        fold_results.append({
            "fold": fold,
            "best_val_loss": result["best_val_loss"],
            "best_val_mae": result["best_val_mae"],
            "best_val_r2": result["best_val_r2"],
            "best_epoch": result["best_epoch"],
            "tabular_mean": str(tabular_mean),
            "tabular_std": str(tabular_std),
        })
        
        all_histories.append(result["history"])

        stats = train_df['vitality'].describe()[['mean', 'std', 'min', 'max']].to_dict()
        print(f"Train vitality: {stats}")
        stats = val_df['vitality'].describe()[['mean', 'std', 'min', 'max']].to_dict()
        print(f"Val vitality: {stats}")
        
        print(f"Fold {fold} best ->"
            f"loss: {result['best_val_loss']:.4f} |"
            f"MAE: {result['best_val_mae']:.4f} |"
            f"R^2: {result['best_val_r2']:.4f} |"
            f"(epoch: {result['best_epoch']})"
            )
        
    results_df = pd.DataFrame(fold_results)
    _save_logs(results_df, all_histories, config)
    summary = _print_summary(results_df, config)
        
    return {
        "fold_results": fold_results,
        "summary": summary,
        "histories": all_histories
    }
        
def _save_logs(results_df, histories, config):
    log_dir = config.paths.log_dir
    summary_path = log_dir / "cv_results.csv"
    results_df.to_csv(summary_path, index=False)
    print(f"Saved fold results")
    
    for fold_idx, history in enumerate(histories, start=1):
        n_epochs = len(history["train_loss"])
        history_df = pd.DataFrame({
            "epoch": list(range(1, n_epochs+1)),
            "train_loss": history["train_loss"],
            "train_mae": history["train_mae"],
            "train_r2": history["train_r2"],
            "val_loss": history["val_loss"],
            "val_mae": history["val_mae"],
            "val_r2": history["val_r2"],
        })
        history_path = log_dir / f"fold_{fold_idx}_history.csv"
        history_df.to_csv(history_path, index=False)
    print(f"Saved epoch histories -> {log_dir}/fold_*_history.csv")
    
def _print_summary(results_df, config):
    print("="*50)
    print("Cross-validation Summary")
    print(f"Backbone: {config.model.backbone}")
    print(f"Aggregator: {config.model.aggregator}")
    print("="*50)
    print(f"{'Fold':>5} | {'Val Loss':>9} | {'Val MAE':>8} |{'Val R^2':>7} |{'Epoch':>5}")
    print("="*50)
    
    for _, row in results_df.iterrows():
        print(f"{int(row['fold']):>5} |"
              f"{row['best_val_loss']:>9.4f} |"
              f"{row['best_val_mae']:>8.4f} |"
              f"{row['best_val_r2']:>7.4f} |"
              f"{int(row['best_epoch']):>5} |"
              )
    print("="*50)
    summary = {
        "mean_val_loss": results_df["best_val_loss"].mean(),
        "std_val_loss": results_df["best_val_loss"].std(),
        "mean_val_mae": results_df["best_val_mae"].mean(),
        "std_val_mae": results_df["best_val_mae"].std(),
        "mean_val_r2": results_df["best_val_r2"].mean(),
        "std_val_r2": results_df["best_val_r2"].std(),
    }
    
    print(f"{'Mean':>5} | "
          f"{summary['mean_val_loss']:>9.4f} |"
          f"{summary['mean_val_mae']:>8.4f} |"
          f"{summary['mean_val_r2']:>7.4f} |"
          )
    print(f"{'Std':>5} | "
          f"{summary['std_val_loss']:>9.4f} |"
          f"{summary['std_val_mae']:>8.4f} |"
          f"{summary['std_val_r2']:>7.4f} |"
          )
    print("="*50)
    print(f"Final result: MAE = {summary['mean_val_mae']:.4f} +- {summary['std_val_mae']:.4f}")
    return summary

if __name__ == "__main__":
    config = Config()
    config.display()
    
    df = pd.read_csv(
        config.paths.csv_path,
        encoding=config.data.csv_encoding,
        sep = config.data.csv_sep
    )
    df = df.rename(columns={
        "N (°)": "N",
        "E (°)": "E",
        "circumference (cm)": "circumference_cm",
        "vitality (5 - highest)": "vitality",
        "fungal infection (3 - worst)": "fungal_infection",
    })
    df_filtered = filter_trees_with_images(
        df,
        config.paths.image_dir,
        image_extensions=config.data.image_extensions
    )
    results = run_cv(df_filtered, config)
    print("evaluate.py sanity check completed")