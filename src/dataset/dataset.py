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
    
class BirchDataset(Dataset):
    def __init__(
        self,
        df,
        image_dir,
        mode="train",
        image_size=224,
        image_extension=".jpg",
        augmentation="light"
    ):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = get_transforms(mode, image_size)
        self.extensions = image_extension
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
        return len(self.df)
    
    def __getitem__(self, index):
        row = self.df.iloc[index]
        paths = self.image_paths[index]
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
        
        return {
            "images": images,
            "vitality": vitality,
            "tree_id": int(row["ID"])
        }
        
def collate_fn(batch):
    """
    Custom collate to handle variable-length image bags.
    Returns images as a list of tensors instead of a stacked tensor.
    """
    return {
        "images": [item["images"] for item in batch],
        "vitality": torch.stack([item["vitality"] for item in batch]),
        "tree_id": [item["tree_id"] for item in batch]
    }
    
def build_dataloaders(
    train_df,
    val_df,
    image_dir,
    batch_size=16,
    num_workers=0,
    image_size=224,
    augmentation="light"
):
    train_dataset = BirchDataset(train_df, image_dir, mode="train", image_size=image_size, augmentation=augmentation)
    val_dataset = BirchDataset(val_df, image_dir, mode="val", image_size=image_size, augmentation=augmentation)
    
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