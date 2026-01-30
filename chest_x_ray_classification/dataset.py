from pathlib import Path
import os
import cv2
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
from torchvision import transforms
from loguru import logger
from tqdm import tqdm
import typer
from chest_x_ray_classification.config import PROCESSED_DATA_DIR, RAW_DATA_DIR


CLASSES = ['normal', 'pneumonia', 'tuberculosis']
CLASS_TO_IDX = {cls: i for i, cls in enumerate(CLASSES)}
IMG_SIZE = 320  


class CLAHETransform:
    """Applies Contrast Limited Adaptive Histogram Equalization."""
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img):
        # Convert PIL to Numpy/OpenCV
        img_np = np.array(img)
        
        # Ensure grayscale
        if len(img_np.shape) > 2:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        img_clahe = clahe.apply(img_np)
        
        # Convert back to PIL for downstream torchvision transforms
        return Image.fromarray(img_clahe)


class ResizePad:
    """Resizes max dimension to target and pads short dimension to maintain aspect ratio."""
    def __init__(self, target_size):
        self.target_size = target_size

    def __call__(self, img):
        w, h = img.size
        scale = self.target_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        img = img.resize((new_w, new_h), Image.BICUBIC)
        
        # Create black canvas
        new_img = Image.new("L", (self.target_size, self.target_size)) # "L" for Grayscale
        
        # Paste centered
        x_offset = (self.target_size - new_w) // 2
        y_offset = (self.target_size - new_h) // 2
        new_img.paste(img, (x_offset, y_offset))
        
        return new_img


class ChestXRayDataset(Dataset):
    def __init__(self, file_list, labels, transform=None):
        self.file_list = file_list
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        label = self.labels[idx]
        
        try:
            # Load as grayscale
            img = Image.open(img_path).convert('L') 
            
            if self.transform:
                img = self.transform(img)
                
            return img, torch.tensor(label, dtype=torch.long)
            
        except Exception as e:
            print(f" Warning: Corrupt file {img_path}, skipping/returning blank.")
            # Remediation Strategy: Return a blank image to avoid crashing, 
            # filtered out before init.
            return torch.zeros((1, IMG_SIZE, IMG_SIZE)), torch.tensor(label, dtype=torch.long)


def get_transforms(split):
    """
    Returns clinically plausible augmentations for train, 
    and deterministic preprocessing for val/test.
    """
    shared_pipeline = [
        CLAHETransform(),        
        ResizePad(IMG_SIZE),     
        transforms.ToTensor(),   
        # Normalize (Mean/Std for ImageNet is usually for RGB, we use 0.5 for Grayscale or calculate from data)
        transforms.Normalize(mean=[0.5], std=[0.5]) 
    ]
    
    if split == 'train':
        aug_pipeline = [
            transforms.RandomRotation(degrees=10),      
            transforms.RandomHorizontalFlip(p=0.5),     
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)), 
            # NO aggressive warping (e.g. PerspectiveTransform)
            # NO ColorJitter on Hue (invalid for X-ray)
        ]
        return transforms.Compose(aug_pipeline + shared_pipeline)
    else:
        return transforms.Compose(shared_pipeline)

def create_dataloaders(data_dir, batch_size=32, val_split=0.2, seed=42):
    """
    Returns train, val, AND test loaders.
    """
    train_files, train_labels = [], []
    test_files, test_labels = [], []
    
    print("Scanning files...")
    
    # 1. Load Data
    for split in ['train', 'val', 'test']:
        split_path = Path(data_dir) / split
        if not split_path.exists(): continue
        
        for cls_name in CLASSES:
            cls_folder = split_path / cls_name
            if not cls_folder.exists(): continue
            
            files = [str(f) for f in cls_folder.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
            labels = [CLASS_TO_IDX[cls_name]] * len(files)
            
            if split == 'test':
                test_files.extend(files)
                test_labels.extend(labels)
            else:
                train_files.extend(files)
                train_labels.extend(labels)

    # 2. Re-split the merged Training data into Train/Val
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_split, random_state=seed)
    train_idx, val_idx = next(sss.split(train_files, train_labels))
    
    final_train_files = [train_files[i] for i in train_idx]
    final_train_labels = [train_labels[i] for i in train_idx]
    
    final_val_files = [train_files[i] for i in val_idx]
    final_val_labels = [train_labels[i] for i in val_idx]
    
    print(f"Stats: Train={len(final_train_files)}, Val={len(final_val_files)}, Test={len(test_files)}")

    # 3. Create Datasets
    train_ds = ChestXRayDataset(final_train_files, final_train_labels, transform=get_transforms('train'))
    val_ds = ChestXRayDataset(final_val_files, final_val_labels, transform=get_transforms('val'))
    test_ds = ChestXRayDataset(test_files, test_labels, transform=get_transforms('test')) # No augmentation
    
    # 4. Create Loaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader, test_loader