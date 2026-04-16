from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


def filter_trees_with_images(
    df,
    image_dir,
    image_extensions = (".jpg", ".jpeg", ".png")
):
    image_dir = Path(image_dir)
    has_images = list()
    for tree_id in df["ID"]:
        folder = image_dir / str(tree_id)
        if folder.exists():
            imgs = [
                p for p in folder.iterdir()
                if p.suffix.lower() in image_extensions
            ]
            has_images.append(len(imgs) > 0)
        else:
            has_images.append(False)
    
    df_filtered = df[has_images].reset_index(drop=True)
    n_removed = len(df) - len(df_filtered)
    removed_ids = df[~pd.Series(has_images).values]["ID"].tolist()
    print(f"Total trees: {len(df)}")
    print(f"Trees with images: {len(df_filtered)}")
    print(f"Trees removed: {n_removed} -> IDs: {removed_ids}")
    
    return df_filtered

def get_transforms(mode, image_size, augmentation="light"):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    if mode == "train" and augmentation == "light":
        return transforms.Compose(
            [
                transforms.Resize([image_size, image_size]),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ]
        )
    if mode == "train" and augmentation == "heavy":
        return transforms.Compose([
 
            transforms.RandomResizedCrop(
                size=image_size,
                scale=(0.5, 1.0),       
                ratio=(0.75, 1.33),
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(degrees=30),
 
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.3,
                hue=0.08,               
            ),
            transforms.RandomGrayscale(p=0.1),
 
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))
            ], p=0.4),
            transforms.RandomAdjustSharpness(sharpness_factor=2.0, p=0.3),
 
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
 
            transforms.RandomErasing(
                p=0.4,
                scale=(0.02, 0.2),      
                ratio=(0.3, 3.3),
                value=0,
            ),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

def compute_tabular_stats(
    df, 
    features
):
    mean = {f: float(df[f].mean()) for f in features}
    std = {f: float(df[f].std()) for f in features}
    return mean, std

class BirchDataset(Dataset):
    def __init__(
        self,
        df,
        image_dir,
        mode="train",
        image_size=224,
        image_extension=(".jpg", ".jpeg", ".png"),
        augmentation="light",
        tabular_features= (),
        tabular_mean = None,
        tabular_std = None,
        n_copies: int = 1,
    ):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = get_transforms(mode, image_size, augmentation)
        self.extensions = image_extension
        self.tabular_features = tabular_features
        self.tabular_mean = tabular_mean or {f: 0.0 for f in tabular_features}
        self.tabular_std = tabular_std or {f: 1.0 for f in tabular_features}
        # n_copies > 1: each tree appears N times per epoch, each time with a
        # fresh random augmentation.  Only applied in train mode (val ignores it).
        self.n_copies = n_copies if mode == "train" else 1
        self.image_paths = self._index_to_images()
    
    def _index_to_images(self):
        all_paths = list()
        for tree_id in self.df["ID"]:
            folder = self.image_dir / str(tree_id)
            paths = sorted([
                p for p in folder.iterdir() if p.suffix.lower() in self.extensions
            ])
            all_paths.append(paths)
        return all_paths

    def __len__(self):
        return len(self.df) * self.n_copies

    def __getitem__(self, index):
        real_index = index % len(self.df)
        row = self.df.iloc[real_index]
        paths = self.image_paths[real_index]
        vitality = torch.tensor(float(row["vitality"]), dtype=torch.float32)
        
        images = list()
        for p in paths:
            try:
                img = Image.open(p).convert("RGB")
                img = ImageOps.exif_transpose(img)
                img = self.transform(img)
                images.append(img)
            except Exception as e:
                print(f"Error loading image {p}: {e}")
        
        images = torch.stack(images)
        
        item = {
            "images": images,
            "vitality": vitality,
            "tree_id": int(row["ID"])
        }

        if self.tabular_features:
            tab = []
            for feat in self.tabular_features:
                val = float(row[feat]) if pd.notna(row.get(feat)) else 0.0
                mean = self.tabular_mean.get(feat, 0.0)
                std = self.tabular_std.get(feat, 1.0)
                tab.append((val - mean) / (std + 1e-8))
            item["tabular"] = torch.tensor(tab, dtype=torch.float32)
        
        return item
        
def collate_fn(batch):
    """
    Custom collate to handle variable-length image bags.
    Returns images as a list of tensors instead of a stacked tensor.
    """
    out = {
        "images": [item["images"] for item in batch],
        "vitality": torch.stack([item["vitality"] for item in batch]),
        "tree_id": [item["tree_id"] for item in batch]
    }

    if "tabular" in batch[0]:
        out["tabular"] = torch.stack([item["tabular"] for item in batch])
    
    return out
    
def build_dataloaders(
    train_df,
    val_df,
    image_dir,
    batch_size=16,
    num_workers=0,
    image_size=224,
    augmentation="light",
    tabular_features = tuple(),
    image_extensions = (".jpg", ".jpeg", ".png"),
    tabular_mean = None,
    tabular_std = None,
    n_copies: int = 1,
):
    shared = dict(
        image_dir = image_dir,
        image_size = image_size,
        augmentation = augmentation,
        tabular_features = tabular_features,
        tabular_mean = tabular_mean,
        tabular_std = tabular_std,
        image_extension = image_extensions,
    )

    train_dataset = BirchDataset(train_df, **shared, mode="train", n_copies=n_copies)
    val_dataset = BirchDataset(val_df, **shared, mode="val")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        collate_fn=collate_fn
        )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers
        )
    
    return train_loader, val_loader

import math

def vitality_to_class(v: float) -> int:
    """
    Convert continuous vitality to a 0-indexed class label.
    .5 values are rounded UP: 2.5->3, 3.5->4, 4.5->5
    Then 0-indexed:          1->0, 2->1, 3->2, 4->3, 5->4
    """
    return math.ceil(v) - 1


def vitality_to_class3(v: float) -> int:
    """
    Coarse 3-class mapping:
      0 — low     (vitality <= 1)
      1 — medium  (1 < vitality <= 3)
      2 — high    (vitality > 3)
    """
    if v <= 1.0:
        return 0
    elif v <= 3.0:
        return 1
    else:
        return 2


class BirchClassificationDataset(Dataset):
    """
    Same image-bag loading as BirchDataset but returns an integer class label
    instead of a continuous vitality value.
    """
    def __init__(
        self,
        df,
        image_dir,
        mode="train",
        image_size=224,
        image_extension=(".jpg", ".jpeg", ".png"),
        augmentation="light",
        tabular_features=(),
        tabular_mean=None,
        tabular_std=None,
        n_copies: int = 1,
        class_fn=None,
    ):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = get_transforms(mode, image_size, augmentation)
        self.extensions = image_extension
        self.tabular_features = tabular_features
        self.tabular_mean = tabular_mean or {f: 0.0 for f in tabular_features}
        self.tabular_std = tabular_std or {f: 1.0 for f in tabular_features}
        self.n_copies = n_copies if mode == "train" else 1
        self.class_fn = class_fn if class_fn is not None else vitality_to_class
        self.image_paths = self._index_to_images()

    def _index_to_images(self):
        all_paths = list()
        for tree_id in self.df["ID"]:
            folder = self.image_dir / str(tree_id)
            paths = sorted([
                p for p in folder.iterdir() if p.suffix.lower() in self.extensions
            ])
            all_paths.append(paths)
        return all_paths

    def __len__(self):
        return len(self.df) * self.n_copies

    def __getitem__(self, index):
        real_index = index % len(self.df)
        row = self.df.iloc[real_index]
        paths = self.image_paths[real_index]
        label = torch.tensor(self.class_fn(float(row["vitality"])), dtype=torch.long)

        images = list()
        for p in paths:
            try:
                img = Image.open(p).convert("RGB")
                img = ImageOps.exif_transpose(img)
                img = self.transform(img)
                images.append(img)
            except Exception as e:
                print(f"Error loading image {p}: {e}")

        images = torch.stack(images)

        item = {
            "images": images,
            "label": label,
            "tree_id": int(row["ID"]),
        }

        if self.tabular_features:
            tab = []
            for feat in self.tabular_features:
                val = float(row[feat]) if pd.notna(row.get(feat)) else 0.0
                mean = self.tabular_mean.get(feat, 0.0)
                std = self.tabular_std.get(feat, 1.0)
                tab.append((val - mean) / (std + 1e-8))
            item["tabular"] = torch.tensor(tab, dtype=torch.float32)

        return item


def collate_cls_fn(batch):
    """Collate for classification: labels as long tensor."""
    out = {
        "images": [item["images"] for item in batch],
        "label": torch.stack([item["label"] for item in batch]),
        "tree_id": [item["tree_id"] for item in batch],
    }
    if "tabular" in batch[0]:
        out["tabular"] = torch.stack([item["tabular"] for item in batch])
    return out


def build_cls_dataloaders(
    train_df,
    val_df,
    image_dir,
    batch_size=16,
    num_workers=0,
    image_size=224,
    augmentation="light",
    tabular_features=tuple(),
    image_extensions=(".jpg", ".jpeg", ".png"),
    tabular_mean=None,
    tabular_std=None,
    n_copies: int = 1,
    class_fn=None,
):
    shared = dict(
        image_dir=image_dir,
        image_size=image_size,
        augmentation=augmentation,
        tabular_features=tabular_features,
        tabular_mean=tabular_mean,
        tabular_std=tabular_std,
        image_extension=image_extensions,
        class_fn=class_fn,
    )

    train_dataset = BirchClassificationDataset(train_df, **shared, mode="train", n_copies=n_copies)
    val_dataset = BirchClassificationDataset(val_df, **shared, mode="val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_cls_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_cls_fn,
        num_workers=num_workers,
    )
    return train_loader, val_loader


if __name__ == "__main__":
    
    DATA_DIR = Path(r"D:\bip\data")
    IMAGE_DIR = DATA_DIR
    CSV_PATH = r"data\birch_trees_bratislava.csv"
    
    df = pd.read_csv(CSV_PATH, encoding="cp1250", sep=";")
    print(f"Loadeded {len(df)} trees from CSV")
    
    df = df.rename(columns={
        "N (°)": "N",
        "E (°)": "E",
        "circumference (cm)": "circumference_cm",
        "vitality (5 - highest)": "vitality",
        "fungal infection (3 - worst)": "fungal_infection",
    })
    
    df_filtered = filter_trees_with_images(df, IMAGE_DIR, image_extensions=(".jpg",))
    print(f"Using {len(df_filtered)} trees with images for training/validation")
    
    split = int(0.8 * len(df_filtered))
    train_df = df_filtered.iloc[:split]
    val_df = df_filtered.iloc[split:]
    
    train_loader, val_loader = build_dataloaders(
        train_df,
        val_df,
        IMAGE_DIR,
        batch_size=4,
    )
    
    batch = next(iter(train_loader))
    print(f"Batch keys: {list(batch.keys())}")
    print(f"Num trees in batch: {len(batch['images'])}")
    print(f"Vitality shape: {batch['vitality'].shape}")
    print(f"Tree IDs: {batch['tree_id']}")
    print(f"Sanity check passed!")