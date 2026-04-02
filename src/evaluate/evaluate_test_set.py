import sys
from pathlib import Path

import pandas as pd
import torch
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from config.config import Config
from dataset.dataset import BirchDataset, collate_fn
from model.model import BirchVitalityModel
from train.train import mean_absolute_error, r2_score
from torch.utils.data import DataLoader

def evaluate_test(config=None):
    if config is None:
        config = Config()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    test_csv_path = config.paths.output_dir / "test_set.csv"
    if not test_csv_path.exists():
        raise FileNotFoundError(
            f"Test set not found at {test_csv_path}"
            "Run main.py first to generate the test split."
        )

    df_test = pd.read_csv(test_csv_path)
    print(f"Test set: {len(df_test)} trees")
    print(f"Vitality distribution:\n{df_test['vitality'].describe().round(3)}\n")

    test_dataset = BirchDataset(
        df_test,
        config.paths.image_dir,
        mode="val",
        image_size=config.data.image_size
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config.data.num_workers
    )

    checkpoint_path = config.paths.checkpoint_dir / "best_model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Best model not found at {checkpoint_path}"
            "Run main.py first to train the model"
        )
    
    model = BirchVitalityModel(
        backbone_name= config.model.backbone,
        aggregator_name=config.model.aggregator,
        hidden_dim=config.model.hidden_dim,
        dropout=config.model.dropout,
        pretrained=False
    )
    model.load_state_dict(torch.load(checkpoint_path, weights_only=False))
    model = model.to(device)
    model.eval()
    print(f"Loaded best model from {checkpoint_path}")

    all_preds = list()
    all_targets = list()
    all_ids = list()

    with torch.no_grad():
        for batch in test_loader:
            images = [img.to(device) for img in batch["images"]]
            targets = batch["vitality"].to(device)
            preds = model(images)
            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())
            all_ids.extend(batch["tree_id"])

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    mae = mean_absolute_error(all_preds, all_targets)
    r2 = r2_score(all_preds, all_targets)
    mse = ((all_preds - all_targets) ** 2).mean().item()
    rmse = mse ** 0.5

    print("Test results")
    print(f"Backbone: {config.model.backbone}")
    print(f"Aggregator: {config.model.aggregator}")
    print("="*50)
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R^2: {r2:.4f}")
    print("="*50)

    results_df = pd.DataFrame({
        "tree_id": all_ids,
        "true": all_targets.numpy().round(),
        "predicted": all_preds.numpy().round(),
        "error": (all_preds - all_targets).abs().numpy().round(2)
    }).sort_values("error", ascending=False)

    print("Per-tree prediction:")
    print(results_df.to_string(index=False))

    results_path = config.paths.output_dir / "test_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"Results saved to {results_path}")

    return {
        "mae": mae,
        "rmse": rmse,
        "r2":r2,
        "predictions": results_df
    }

if __name__ == "__main__":
    config = Config()
    evaluate_test(config)