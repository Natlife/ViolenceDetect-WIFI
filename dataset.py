"""
  <data>/
    train/
    test/
    mean_std/
      mean_std_0.h5
      ...
      mean_std_7.h5
    train_list.csv
    test_list.csv
"""

import csv
import random
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import config

_mean_std_cache: dict = {}


def _load_mean_std(mean_std_dir: Path, label_idx: int) -> Tuple:
    if label_idx in _mean_std_cache:
        return _mean_std_cache[label_idx]

    h5_path = mean_std_dir / f"mean_std_{label_idx}.h5"
    if not h5_path.exists():
        _mean_std_cache[label_idx] = (None, None)
        return (None, None)

    try:
        import h5py
        with h5py.File(h5_path, "r") as f:
            keys = list(f.keys())
            mean_key = "mean" if "mean" in keys else keys[0]
            std_key  = "std"  if "std"  in keys else (keys[1] if len(keys) > 1 else keys[0])
            mean = f[mean_key][:].astype(np.float32)
            std  = f[std_key][:].astype(np.float32)
            std  = np.where(std < 1e-8, 1.0, std)
        _mean_std_cache[label_idx] = (mean, std)
        return (mean, std)
    except Exception as e:
        print(f"[WARN] Không đọc được {h5_path}: {e}")
        _mean_std_cache[label_idx] = (None, None)
        return (None, None)

def _read_h5_sample(h5_path: Path) -> np.ndarray:
    """Read csi signal -> shape (C, T)."""
    import h5py
    with h5py.File(h5_path, "r") as f:
        keys = list(f.keys())
        data_key = next(
            (k for k in ["data", "csi", "X", "csi_data", "amplitude"] if k in keys),
            keys[0]
        )
        arr = f[data_key][:].astype(np.float32)

    if arr.ndim == 1:
        arr = arr[np.newaxis, :]
    elif arr.ndim == 2:
        if arr.shape[0] > arr.shape[1]:
            arr = arr.T
    elif arr.ndim == 3:
        max_ax = np.argmax(arr.shape)
        if max_ax == 2:   # (..., T)
            arr = arr.reshape(-1, arr.shape[2])
        else:             # (T, ...)
            arr = arr.reshape(arr.shape[0], -1).T
    return arr  # (C, T)

def _normalize(arr: np.ndarray, mean, std) -> np.ndarray:
    if mean is None:
        m = arr.mean(axis=-1, keepdims=True)
        s = arr.std(axis=-1, keepdims=True) + 1e-8
        return (arr - m) / s
    m = np.array(mean).reshape(-1, 1) if np.ndim(mean) >= 1 else mean
    s = np.array(std).reshape(-1, 1)  if np.ndim(std)  >= 1 else std
    return (arr - m) / (s + 1e-8)


def _resize_time(arr: np.ndarray, target_T: int) -> np.ndarray:
    _, T = arr.shape
    if T == target_T:
        return arr
    try:
        import scipy.ndimage
        return scipy.ndimage.zoom(arr, (1, target_T / T), order=1)
    except ImportError:
        idx = np.linspace(0, T - 1, target_T)
        left = np.floor(idx).astype(int)
        right = np.minimum(left + 1, T - 1)
        frac = (idx - left)[np.newaxis, :]
        return arr[:, left] * (1 - frac) + arr[:, right] * frac


def _sliding_windows(arr: np.ndarray, window: int, stride: int) -> List[np.ndarray]:
    _, T = arr.shape
    wins = []
    s = 0
    while s + window <= T:
        wins.append(arr[:, s:s + window])
        s += stride
    if not wins:
        wins.append(_resize_time(arr, window))
    return wins

def _augment(x: np.ndarray) -> np.ndarray:
    x = x + np.random.normal(0, config.AUGMENT_NOISE_STD, x.shape).astype(np.float32)
    if random.random() < config.AUGMENT_FLIP_PROB:
        x = np.flip(x, axis=-1).copy()
    lo, hi = config.AUGMENT_SCALE_RANGE
    return x * random.uniform(lo, hi)

def _map_label(raw_label: int) -> int:
    if config.TASK == "binary":
        return 0 if raw_label == 1 else 1
    return raw_label - 1

class WiFiViolenceDataset(Dataset):

    def __init__(
        self,
        csv_path: Path,
        data_dir: Path,
        mean_std_dir: Path,
        split: str = "train",
        augment: bool = False,
        use_sliding_window: bool = True,
        val_ratio: float = 0.15,
    ):
        self.data_dir = Path(data_dir)
        self.mean_std_dir = Path(mean_std_dir)
        self.split = split
        self.augment = augment and (split == "train")
        self.use_sliding_window = use_sliding_window

        with open(csv_path, newline="", encoding="utf-8") as f:
            all_rows = list(csv.DictReader(f))

        if split in ("train", "val"):
            rng = np.random.default_rng(config.RANDOM_SEED)
            idx = np.arange(len(all_rows))
            rng.shuffle(idx)
            n_val = int(len(idx) * val_ratio)
            rows = [all_rows[i] for i in (idx[:n_val] if split == "val" else idx[n_val:])]
        else:
            rows = all_rows

        self.samples: List[Tuple[Path, int, int]] = []
        missing = 0
        for row in rows:
            fname = row["file"].strip()
            raw_label = int(row["label"])
            h5_file = self.data_dir / f"{fname}.h5"
            if not h5_file.exists():
                missing += 1
                continue
            self.samples.append((h5_file, _map_label(raw_label), raw_label))

        if missing:
            print(f"[WARN] [{split}] {missing}/{len(rows)} files không tìm thấy trong {self.data_dir}")

        # Log
        lbl_counts: dict = {}
        for _, lbl, _ in self.samples:
            lbl_counts[lbl] = lbl_counts.get(lbl, 0) + 1
        print(f"[{split}] {len(self.samples)} samples | labels: {dict(sorted(lbl_counts.items()))}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        h5_path, mapped_label, raw_label = self.samples[idx]

        try:
            arr = _read_h5_sample(h5_path)
        except Exception as e:
            print(f"[ERROR] {h5_path.name}: {e}")
            arr = np.zeros((config.NUM_SUBCARRIERS, config.WINDOW_SIZE), dtype=np.float32)

        mean, std = _load_mean_std(self.mean_std_dir, raw_label)
        arr = _normalize(arr, mean, std)

        if self.use_sliding_window:
            wins = _sliding_windows(arr, config.WINDOW_SIZE, config.STRIDE)
            arr = wins[random.randint(0, len(wins) - 1)] if self.split == "train" else wins[0]
        else:
            arr = _resize_time(arr, config.WINDOW_SIZE)

        C = arr.shape[0]
        if C > config.NUM_SUBCARRIERS:
            arr = arr[:config.NUM_SUBCARRIERS, :]
        elif C < config.NUM_SUBCARRIERS:
            arr = np.concatenate(
                [arr, np.zeros((config.NUM_SUBCARRIERS - C, arr.shape[1]), dtype=np.float32)], axis=0
            )

        # Augment
        if self.augment:
            arr = _augment(arr)

        return (
            torch.tensor(arr.astype(np.float32), dtype=torch.float32),
            torch.tensor(mapped_label, dtype=torch.long),
        )

def get_dataloaders(
    data_dir: Optional[Path] = None,
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = config.NUM_WORKERS,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    if data_dir is None:
        data_dir = config.DATA_ROOT

    data_dir = Path(data_dir)
    train_dir    = data_dir / "train"
    test_dir     = data_dir / "test"
    mean_std_dir = data_dir / "mean_std"
    train_csv    = data_dir / "train_list.csv"
    test_csv     = data_dir / "test_list.csv"

    for p in [train_dir, test_dir, train_csv, test_csv]:
        if not p.exists():
            raise FileNotFoundError(
                f"Not found: {p}\n"
                "Need exactly DATA_ROOT trong config.py:\n"
                "  train/  test/  mean_std/  train_list.csv  test_list.csv"
            )

    train_ds = WiFiViolenceDataset(
        csv_path=train_csv, data_dir=train_dir, mean_std_dir=mean_std_dir,
        split="train", augment=config.USE_AUGMENTATION, use_sliding_window=True,
    )
    val_ds = WiFiViolenceDataset(
        csv_path=train_csv, data_dir=train_dir, mean_std_dir=mean_std_dir,
        split="val", augment=False, use_sliding_window=True,
    )
    test_ds = WiFiViolenceDataset(
        csv_path=test_csv, data_dir=test_dir, mean_std_dir=mean_std_dir,
        split="test", augment=False, use_sliding_window=False,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=config.PIN_MEMORY, drop_last=True,
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