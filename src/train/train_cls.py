"""
Classification training loop — mirrors train.py but uses CrossEntropyLoss
with inverse-frequency class weights, and reports accuracy + macro-F1.
"""
import time
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
import numpy as np
from torch.amp import autocast, GradScaler

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from model.model import BirchVitalityClassifier
from train.train import EarlyStopping, save_checkpoint, load_checkpoint


# ── Metrics ──────────────────────────────────────────────────────────

def accuracy(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Top-1 accuracy.  preds: (B, C) logits,  targets: (B,) long."""
    return (preds.argmax(dim=1) == targets).float().mean().item()


def macro_f1(preds: torch.Tensor, targets: torch.Tensor, num_classes: int) -> float:
    """Macro-averaged F1.  preds: (B, C) logits,  targets: (B,) long."""
    pred_labels = preds.argmax(dim=1)
    f1s = []
    for c in range(num_classes):
        tp = ((pred_labels == c) & (targets == c)).sum().float()
        fp = ((pred_labels == c) & (targets != c)).sum().float()
        fn = ((pred_labels != c) & (targets == c)).sum().float()
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        f1s.append(f1.item())
    return float(np.mean(f1s))


# ── Class weights ────────────────────────────────────────────────────

def compute_class_weights(labels, num_classes: int) -> torch.Tensor:
    """
    Inverse-frequency weights so that rare classes matter more.
    labels: iterable of int class indices.
    """
    counts = Counter(labels)
    total = sum(counts.values())
    weights = torch.zeros(num_classes)
    for c in range(num_classes):
        if counts[c] > 0:
            weights[c] = total / (num_classes * counts[c])
        else:
            weights[c] = 1.0
    return weights


# ── Epoch loops ──────────────────────────────────────────────────────

def train_cls_epoch(model, loader, optimizer, criterion, device, num_classes, scaler=None):
    model.train()
    all_preds, all_targets = [], []
    total_loss = 0.0
    n_batches = 0

    use_amp = scaler is not None
    for batch in loader:
        images = [img.to(device) for img in batch["images"]]
        targets = batch["label"].to(device)
        tabular = batch["tabular"].to(device) if "tabular" in batch else None

        optimizer.zero_grad()
        with autocast(device_type=device.type, enabled=use_amp):
            logits = model(images, tabular)  # (B, C)
            loss = criterion(logits, targets)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        all_preds.append(logits.detach().cpu())
        all_targets.append(targets.detach().cpu())
        n_batches += 1

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    return {
        "loss": total_loss / n_batches,
        "accuracy": accuracy(all_preds, all_targets),
        "f1": macro_f1(all_preds, all_targets, num_classes),
    }


def val_cls_epoch(model, loader, criterion, device, num_classes):
    model.eval()
    all_preds, all_targets = [], []
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in loader:
            images = [img.to(device) for img in batch["images"]]
            targets = batch["label"].to(device)
            tabular = batch["tabular"].to(device) if "tabular" in batch else None

            logits = model(images, tabular)
            loss = criterion(logits, targets)

            total_loss += loss.item()
            all_preds.append(logits.cpu())
            all_targets.append(targets.cpu())
            n_batches += 1

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    return {
        "loss": total_loss / n_batches,
        "accuracy": accuracy(all_preds, all_targets),
        "f1": macro_f1(all_preds, all_targets, num_classes),
    }


# ── Full fold training ───────────────────────────────────────────────

def train_cls_fold(
    model,
    train_loader,
    val_loader,
    config,
    fold,
    device,
    num_classes=5,
    class_weights=None,
):
    model = model.to(device)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.training.lr,
        weight_decay=config.training.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5,
    )

    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    else:
        criterion = nn.CrossEntropyLoss()

    use_amp = getattr(config.training, "use_amp", False) and device.type == "cuda"
    scaler = GradScaler(device="cuda") if use_amp else None
    if use_amp:
        print("AMP (mixed precision) enabled.")

    checkpoint_path = config.paths.checkpoint_dir / f"fold_{fold}_best.pt"
    early_stopping = EarlyStopping(
        patience=config.training.patience,
        checkpoint_path=checkpoint_path,
    )

    history = {
        "train_loss": [], "train_acc": [], "train_f1": [],
        "val_loss": [], "val_acc": [], "val_f1": [],
    }

    print("=" * 60)
    print(f"Fold {fold} — classification — up to {config.training.epochs} epochs")
    print("=" * 60)
    print(f"{'Epoch':>6} | {'Tr Loss':>8} | {'Tr Acc':>7} | {'Tr F1':>6} |"
          f"{'Va Loss':>8} | {'Va Acc':>7} | {'Va F1':>6} |")
    print("=" * 70)

    start_epoch = 1
    resume_path = config.paths.checkpoint_dir / f"fold_{fold}_resume.pt"
    should_resume = getattr(config.training, "resume", True)

    if should_resume and resume_path.exists():
        print(f"Resume fold {fold} from {resume_path}")
        start_epoch, best_loss = load_checkpoint(resume_path, model, optimizer, scheduler, scaler)
        early_stopping.best_loss = best_loss
        start_epoch += 1
        print(f"Resumed from epoch {start_epoch - 1}")

    for epoch in range(start_epoch, config.training.epochs + 1):
        t0 = time.time()
        train_m = train_cls_epoch(model, train_loader, optimizer, criterion, device, num_classes, scaler=scaler)
        val_m = val_cls_epoch(model, val_loader, criterion, device, num_classes)
        scheduler.step(val_m["loss"])

        history["train_loss"].append(train_m["loss"])
        history["train_acc"].append(train_m["accuracy"])
        history["train_f1"].append(train_m["f1"])
        history["val_loss"].append(val_m["loss"])
        history["val_acc"].append(val_m["accuracy"])
        history["val_f1"].append(val_m["f1"])

        elapsed = time.time() - t0

        print(
            f"{epoch:>6} | "
            f"{train_m['loss']:>8.4f} | "
            f"{train_m['accuracy']:>7.4f} | "
            f"{train_m['f1']:>6.4f} | "
            f"{val_m['loss']:>8.4f} | "
            f"{val_m['accuracy']:>7.4f} | "
            f"{val_m['f1']:>6.4f} | "
            f"({elapsed:.1f} s)"
        )

        stop = early_stopping.step(val_m["loss"], model, epoch, acc=val_m["accuracy"], f1=val_m["f1"])
        if stop:
            print(
                f"Early stopping at epoch {epoch} "
                f"(best epoch: {early_stopping.best_epoch}) "
                f"(best val loss: {early_stopping.best_loss:.4f})"
            )
            break

        save_checkpoint(resume_path, model, optimizer, scheduler, epoch, early_stopping.best_loss, scaler=scaler)

    if checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
        print(f"Loaded best weights from {checkpoint_path}")

    # If no epochs ran this session (fully resumed), compute metrics from the loaded model
    if not history["val_acc"]:
        val_m = val_cls_epoch(model, val_loader, criterion, device, num_classes)
        best_acc = val_m["accuracy"]
        best_f1 = val_m["f1"]
    else:
        best_acc = early_stopping.best_metrics.get("acc", 0.0)
        best_f1 = early_stopping.best_metrics.get("f1", 0.0)

    return {
        "best_val_loss": early_stopping.best_loss,
        "best_val_acc": best_acc,
        "best_val_f1": best_f1,
        "best_epoch": early_stopping.best_epoch,
        "history": history,
    }
