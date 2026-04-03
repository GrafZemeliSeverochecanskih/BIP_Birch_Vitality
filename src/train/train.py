import time
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

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

def save_checkpoint(path, model, optimizer, scheduler, epoch, best_loss):
    torch.save({
        "epoch":epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "best_loss": best_loss,
    }, path)

def load_checkpoint(path, model, optimizer, scheduler):
    checkpoint = torch.load(path, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    scheduler.load_state_dict(checkpoint["scheduler_state"])
    return checkpoint["epoch"], checkpoint["best_loss"]

def train_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device  
):
    model.train()
    all_preds = list()
    all_targets = list()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        images = [img.to(device) for img in batch["images"]]
        targets = batch["vitality"].to(device)
        tabular = batch["tabular"].to(device) if "tabular" in batch else None

        optimizer.zero_grad()
        preds = model(images)
        loss = criterion(preds, targets)
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
            
            preds = model(images)
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
        self.counter = 0
        self.best_epoch = 0
    
    def step(self, val_loss, model, epoch):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            self.best_epoch = epoch
            if self.checkpoint_path is not None:
                torch.save(model.state_dict(), self.checkpoint_path)
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
    
    checkpoint_path = config.paths.checkpoint_dir / f"fold_{fold}_best.pt"
    early_stopping = EarlyStopping(
        patience=config.training.patience,
        checkpoint_path=checkpoint_path
    )
    
    history = {
        "train_loss": list(),
        "train_mae": list(),
        "train_r2": list(),
        "val_loss": list(),
        "val_mae": list(),
        "val_r2": list(),
    }
    
    print("=" * 50)
    print(f"Fold {fold} - training for up to {config.training.epochs} epochs")
    print("=" * 50)
    print(f"{'Epoch':>6} | {'Train Loss':>10} | {'Train MAE':>10} | {'Train R^2':>9} |"
          f"{'Val Loss':>9} | {'Val MAE':>9} | {'Val R^2':>8} |")
    print("="*70)
    
    start_epoch = 1
    resume_path = config.paths.checkpoint_dir / f"fold_{fold}_resume.pt"

    if resume_path.exists():
        print(f"Resume fold {fold} from {resume_path}")
        start_epoch, best_loss = load_checkpoint(resume_path, model, optimizer, scheduler)
        early_stopping.best_loss = best_loss
        start_epoch += 1
        print(f"Resume from epoch {start_epoch - 1}")

    for epoch in range(start_epoch, config.training.epochs + 1):
        t0 = time.time()
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = val_epoch(model, val_loader, criterion, device)
        scheduler.step(val_metrics["loss"])
        
        history["train_loss"].append(train_metrics["loss"])
        history["train_mae"].append(train_metrics["mae"])
        history["train_r2"].append(train_metrics["r2"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_mae"].append(val_metrics["mae"])
        history["val_r2"].append(val_metrics["r2"])
        
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
        
        stop = early_stopping.step(val_metrics["loss"], model, epoch)
        if stop:
            print(f"Early stopping at epoch {epoch}"
                  f"(best epoch: {early_stopping.best_epoch})"
                  f"(best val loss: {early_stopping.best_loss:.4f})"
                  )
            break
        
        save_checkpoint(resume_path, model, optimizer, scheduler, epoch, early_stopping.best_loss)

    if checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
        print(f"Loaded best weights from {checkpoint_path}")
        
    best_epoch = early_stopping.best_epoch
    return {
        "best_val_loss": early_stopping.best_loss,
        "best_val_mae": history["val_mae"][best_epoch - 1],
        "best_val_r2": history["val_r2"][best_epoch - 1],
        "best_epoch": best_epoch,
        "history": history
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
    