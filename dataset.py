
import os
import random
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional

import torch
from torch.utils.data import Dataset, DataLoader

import config


# Augmentation helpers

def augment_add_noise(x: np.ndarray) -> np.ndarray:
    return x + np.random.normal(0, config.AUGMENT_NOISE_STD, x.shape)


def augment_flip(x: np.ndarray) -> np.ndarray:
    if random.random() < config.AUGMENT_FLIP_PROB:
        return np.flip(x, axis=-1).copy()
    return x


def augment_scale(x: np.ndarray) -> np.ndarray:
    lo, hi = config.AUGMENT_SCALE_RANGE
    scale = random.uniform(lo, hi)
    return x * scale


# Dataset

class BullyDataset(Dataset):
    """
    WiFi CSI bullying detection dataset.

    Expects .npy files with shape (samples, subcarriers, time)
    or a directory of per-sample .npy files with naming: <label>_<id>.npy
    where label is 0 (normal) or 1 (bullying).
    """

    def __init__(
        self,
        data_dir: Path,
        split: str = "train",          # "train" | "val" | "test"
        transform=None,
        augment: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.augment = augment and (split == "train")

        self.samples: List[np.ndarray] = []
        self.labels: List[int] = []

        self._load()

    # loading

    def _load(self):
        split_dir = self.data_dir / self.split
        x_file    = split_dir / "X.npy"
        y_file    = split_dir / "y.npy"

        if x_file.exists() and y_file.exists():
            # Format 1: pre-split X.npy / y.npy
            X = np.load(x_file).astype(np.float32)
            y = np.load(y_file).astype(np.int64)
            self.samples = [X[i] for i in range(len(X))]
            self.labels  = y.tolist()

        elif (self.data_dir / "X.npy").exists():
            # Format 2: single X.npy + y.npy at root (will be split by index)
            raise ValueError(
                "Found X.npy at root but not in split dirs. "
                "Run scripts/prepare_dataset.py first."
            )

        else:
            # Format 3: individual per-sample .npy files  <label>_<id>.npy
            files = sorted(split_dir.glob("*.npy"))
            if len(files) == 0:
                raise FileNotFoundError(
                    f"No data found in {split_dir}. "
                    "Run scripts/prepare_dataset.py to preprocess the dataset."
                )
            for f in files:
                label = int(f.stem.split("_")[0])
                arr   = np.load(f).astype(np.float32)
                self.samples.append(arr)
                self.labels.append(label)

        print(f"[{self.split}] loaded {len(self.samples)} samples | "
              f"classes: {dict(zip(*np.unique(self.labels, return_counts=True)))}")

    # augment

    def _apply_augment(self, x: np.ndarray) -> np.ndarray:
        x = augment_add_noise(x)
        x = augment_flip(x)
        x = augment_scale(x)
        return x

    # Dataset interface

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.samples[idx].copy()
        y = self.labels[idx]

        if self.augment:
            x = self._apply_augment(x)

        if self.transform:
            x = self.transform(x)

        # Ensure shape is (C, T) or (C, H, W) — at minimum 2D
        if x.ndim == 1:
            x = x[np.newaxis, :]  # (1, T)

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


# DataLoader factory

def get_dataloaders(
    data_dir: Optional[Path] = None,
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = config.NUM_WORKERS,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    if data_dir is None:
        data_dir = config.PREPROCESSED_DATA_DIR

    train_ds = BullyDataset(data_dir, split="train", augment=config.USE_AUGMENTATION)
    val_ds   = BullyDataset(data_dir, split="val",   augment=False)
    test_ds  = BullyDataset(data_dir, split="test",  augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=config.PIN_MEMORY,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=config.PIN_MEMORY,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=config.PIN_MEMORY,
    )

    return train_loader, val_loader, test_loader
