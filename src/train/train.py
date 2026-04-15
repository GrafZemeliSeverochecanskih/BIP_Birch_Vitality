import csv
import time
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from torch.amp import autocast, GradScaler

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from model.model import BirchVitalityModel

def mean_absolute_error(preds, targets):
    return (preds - targets).abs().mean().item()

def r2_score(preds, targets):
    ss_res = ((targets - preds) ** 2).sum()
    ss_tot = ((targets - targets.mean()) ** 2).sum()
    if ss_tot == 0:
        return 0.0
    return (1 - ss_res/ss_tot).item()

def save_checkpoint(path, model, optimizer, scheduler, epoch, best_loss, scaler=None):
    # Only save trainable parameters — frozen backbone weights are unchanged from
    # their pretrained init and don't need to be written to disk every epoch.
    trainable_names = {name for name, p in model.named_parameters() if p.requires_grad}
    trainable_state = {k: v for k, v in model.state_dict().items() if k in trainable_names}
    d = {
        "epoch": epoch,
        "model_state": trainable_state,
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "best_loss": best_loss,
    }
    if scaler is not None:
        d["scaler_state"] = scaler.state_dict()
    tmp_path = str(path) + ".tmp"
    torch.save(d, tmp_path)
    Path(tmp_path).replace(path)

def load_checkpoint(path, model, optimizer, scheduler, scaler=None):
    try:
        checkpoint = torch.load(path, weights_only=False)
    except Exception as e:
        print(f"WARNING: checkpoint {path} is corrupt ({e}). Deleting and restarting fold from epoch 1.")
        Path(path).unlink(missing_ok=True)
        return None, None  # signals caller to start fresh

    # strict=False: frozen backbone stays as pretrained-init; only trainable params restored
    model.load_state_dict(checkpoint["model_state"], strict=False)
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    scheduler.load_state_dict(checkpoint["scheduler_state"])
    if scaler is not None and "scaler_state" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state"])
    return checkpoint["epoch"], checkpoint["best_loss"]

def predict_loader(model, loader, device):
    """Run inference over a loader and return per-tree (tree_id, pred, target) lists."""
    model.eval()
    tree_ids, preds, targets = [], [], []

    with torch.no_grad():
        for batch in loader:
            images = [img.to(device) for img in batch["images"]]
            tabular = batch["tabular"].to(device) if "tabular" in batch else None
            out = model(images, tabular).cpu()
            preds.extend(out.tolist())
            targets.extend(batch["vitality"].tolist())
            tree_ids.extend(batch["tree_id"])

    return tree_ids, preds, targets


def train_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    scaler=None,
):
    model.train()
    all_preds = list()
    all_targets = list()
    total_loss = 0.0
    n_batches = 0

    use_amp = scaler is not None
    for batch in loader:
        images = [img.to(device) for img in batch["images"]]
        targets = batch["vitality"].to(device)
        tabular = batch["tabular"].to(device) if "tabular" in batch else None

        optimizer.zero_grad()
        with autocast(device_type=device.type, enabled=use_amp):
            preds = model(images, tabular)
            loss = criterion(preds, targets)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        all_preds.append(preds.detach().cpu())
        all_targets.append(targets.detach().cpu())
        n_batches += 1

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    return {
        "loss": total_loss / n_batches,
        "mae": mean_absolute_error(all_preds, all_targets),
        "r2": r2_score(all_preds, all_targets)
    }
    
def val_epoch(
    model,
    loader,
    criterion,
    device  
):
    model.eval()
    all_preds = list()
    all_targets = list()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in loader:
            images = [img.to(device) for img in batch["images"]]
            targets = batch["vitality"].to(device)
            tabular = batch["tabular"].to(device) if "tabular" in batch else None
            
            preds = model(images, tabular)
            loss = criterion(preds, targets)
            
            total_loss += loss.item()
            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())
            n_batches += 1
    
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    
    return {
        "loss": total_loss / n_batches,
        "mae": mean_absolute_error(all_preds, all_targets),
        "r2": r2_score(all_preds, all_targets)
    }
    
class EarlyStopping:
    def __init__(self, patience = 10, checkpoint_path = None):
        self.patience = patience
        self.checkpoint_path = checkpoint_path
        self.best_loss = float("inf")
        self.best_metrics = {}
        self.counter = 0
        self.best_epoch = 0

    def step(self, val_loss, model, epoch, **metrics):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.best_metrics = dict(metrics)
            self.counter = 0
            self.best_epoch = epoch
            if self.checkpoint_path is not None:
                # Only save trainable params — frozen backbone unchanged from pretrained init
                trainable_names = {name for name, p in model.named_parameters() if p.requires_grad}
                trainable_state = {k: v for k, v in model.state_dict().items() if k in trainable_names}
                torch.save(trainable_state, self.checkpoint_path)
        else:
            self.counter += 1

        return self.counter >= self.patience

def train_fold(
    model,
    train_loader,
    val_loader,
    config,
    fold,
    device,
):
    model = model.to(device)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.training.lr,
        weight_decay=config.training.weight_decay
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5
    )
    
    criterion = nn.MSELoss()

    use_amp = getattr(config.training, "use_amp", False) and device.type == "cuda"
    scaler = GradScaler(device="cuda") if use_amp else None
    if use_amp:
        print("AMP (mixed precision) enabled.")

    checkpoint_path = config.paths.checkpoint_dir / f"fold_{fold}_best.pt"
    early_stopping = EarlyStopping(
        patience=config.training.patience,
        checkpoint_path=checkpoint_path
    )
    
    # Stream epoch history to disk — no in-memory accumulation
    history_path = config.paths.log_dir / f"fold_{fold}_history.csv"
    history_path.parent.mkdir(parents=True, exist_ok=True)

    start_epoch = 1
    resume_path = config.paths.checkpoint_dir / f"fold_{fold}_resume.pt"
    should_resume = getattr(config.training, "resume", True)

    if should_resume and resume_path.exists():
        print(f"Resume fold {fold} from {resume_path}")
        resumed_epoch, best_loss = load_checkpoint(resume_path, model, optimizer, scheduler, scaler)
        if resumed_epoch is not None:
            early_stopping.best_loss = best_loss
            start_epoch = resumed_epoch + 1
            print(f"Resumed from epoch {resumed_epoch}")
        else:
            print(f"Starting fold {fold} from scratch.")

    # Append to existing history if resuming, otherwise start fresh
    hist_mode = "a" if (start_epoch > 1 and history_path.exists()) else "w"
    hist_file = open(history_path, hist_mode, newline="")
    hist_writer = csv.writer(hist_file)
    if hist_mode == "w":
        hist_writer.writerow(["epoch", "train_loss", "train_mae", "train_r2", "val_loss", "val_mae", "val_r2"])

    print("=" * 50)
    print(f"Fold {fold} - training for up to {config.training.epochs} epochs")
    print("=" * 50)
    print(f"{'Epoch':>6} | {'Train Loss':>10} | {'Train MAE':>10} | {'Train R^2':>9} |"
          f"{'Val Loss':>9} | {'Val MAE':>9} | {'Val R^2':>8} |")
    print("="*70)

    epochs_run = 0
    for epoch in range(start_epoch, config.training.epochs + 1):
        t0 = time.time()
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device, scaler=scaler)
        val_metrics = val_epoch(model, val_loader, criterion, device)
        scheduler.step(val_metrics["loss"])

        hist_writer.writerow([
            epoch,
            round(train_metrics["loss"], 6), round(train_metrics["mae"], 6), round(train_metrics["r2"], 6),
            round(val_metrics["loss"], 6), round(val_metrics["mae"], 6), round(val_metrics["r2"], 6),
        ])
        hist_file.flush()
        epochs_run += 1

        elapsed = time.time() - t0
        print(
            f"{epoch:>6} | "
            f"{train_metrics['loss']:>10.4f} | "
            f"{train_metrics['mae']:>10.4f} | "
            f"{train_metrics['r2']:>10.4f} | "
            f"{val_metrics['loss']:>10.4f} | "
            f"{val_metrics['mae']:>10.4f} | "
            f"{val_metrics['r2']:>10.4f} | "
            f"({elapsed:.1f} s)"
        )

        stop = early_stopping.step(val_metrics["loss"], model, epoch, mae=val_metrics["mae"], r2=val_metrics["r2"])
        if stop:
            print(f"Early stopping at epoch {epoch}"
                  f"(best epoch: {early_stopping.best_epoch})"
                  f"(best val loss: {early_stopping.best_loss:.4f})"
                  )
            break

        save_checkpoint(resume_path, model, optimizer, scheduler, epoch, early_stopping.best_loss, scaler=scaler)

    hist_file.close()

    if checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, weights_only=True), strict=False)
        print(f"Loaded best weights from {checkpoint_path}")

    # If no epochs ran this session (fully resumed), compute metrics from the loaded model
    if epochs_run == 0:
        val_metrics = val_epoch(model, val_loader, criterion, device)
        best_mae = val_metrics["mae"]
        best_r2 = val_metrics["r2"]
    else:
        best_mae = early_stopping.best_metrics.get("mae", float("inf"))
        best_r2 = early_stopping.best_metrics.get("r2", float("-inf"))

    return {
        "best_val_loss": early_stopping.best_loss,
        "best_val_mae": best_mae,
        "best_val_r2": best_r2,
        "best_epoch": early_stopping.best_epoch,
    }
    
if __name__ == "__main__":
    print("train.py sanity check")
    print("="*40)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    model = BirchVitalityModel(
        backbone_name = "efficientnet_b0",
        aggregator_name = "attention",
        pretrained = False
    ).to(device)
    
    def mock_loader(n_batches=3, batch_size = 2):
        for _ in range(n_batches):
            images = [
                torch.rand(3, 3, 224, 224),
                torch.rand(5, 3, 224, 224)
            ]
            yield {
                "images": images,
                "vitality": torch.tensor([3.5, 4.0]),
                "tree_ids": [1, 2]
            }
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    train_metrics = train_epoch(model, mock_loader(), optimizer, criterion, device)
    val_metrics = val_epoch(model, mock_loader(), criterion, device)
    
    print(f"Train -> loss: {train_metrics['loss']:.4f} |"
          f"MAE: {train_metrics['mae']:.4f} | R^2: {train_metrics['r2']:.4f}")
    print(f"Validation -> loss: {val_metrics['loss']:.4f} |"
          f"MAE: {val_metrics['mae']:.4f} | R^2: {val_metrics['r2']:.4f}")
    
    print("train.py sanity check passed")
    