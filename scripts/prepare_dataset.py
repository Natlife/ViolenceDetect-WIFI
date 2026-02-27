"""
Download + preprocess dataset
save train/val/test splits as X.npy / y.npy.

Command:
  python scripts/prepare_dataset.py [--kaggle] [--raw_dir PATH] [--skip_download]

Kaggle API setup (one-time):
    Windows: place kaggle.json in C:\\Users\\<you>\\.kaggle\\
  -> pip install kaggle
"""

import sys
import argparse
import random
import shutil
from pathlib import Path

import numpy as np

# project root
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

config.ensure_dirs()

KAGGLE_DATASET = "laptype/wifi-bullydetect"


def download_kaggle(dest: Path):
    """Download dataset from Kaggle"""
    try:
        import kaggle
    except ImportError:
        print("kaggle package not found. Run: pip install kaggle")
        sys.exit(1)

    print(f"Downloading {KAGGLE_DATASET} → {dest} ...")
    dest.mkdir(parents=True, exist_ok=True)
    kaggle.api.dataset_download_files(KAGGLE_DATASET, path=str(dest), unzip=True)
    print("Download complete.")


def find_npy_files(raw_dir: Path):
    """Recursively find all .npy files and classify"""
    class_to_files = {}
    for cls_idx, cls_name in enumerate(config.CLASSES):
        # matching folder names -> "normal", "bullying", "0", "1"
        candidates = list(raw_dir.rglob(f"*{cls_name}*/*.npy"))
        if not candidates:
            candidates = list((raw_dir / cls_name).glob("*.npy"))
        if not candidates:
            # numeric subfolder
            candidates = list((raw_dir / str(cls_idx)).glob("*.npy"))
        class_to_files[cls_idx] = candidates

    return class_to_files


def load_sample(path: Path) -> np.ndarray:
    """Load a .npy file and normalize"""
    arr = np.load(path).astype(np.float32)

    # Expected shape: (subcarriers, time) or (time, subcarriers)
    # Standardize to (subcarriers, time)
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]   # (1, T)
    elif arr.ndim == 2 and arr.shape[0] > arr.shape[1]:
        arr = arr.T                # transpose if (T, C) → (C, T)

    # Resize time dimension to WINDOW_SIZE
    C, T = arr.shape
    if T != config.WINDOW_SIZE:
        # Simple linear interpolation
        import scipy.ndimage
        scale = config.WINDOW_SIZE / T
        arr   = scipy.ndimage.zoom(arr, (1, scale), order=1)

    # Normalize per-channel
    mean = arr.mean(axis=-1, keepdims=True)
    std  = arr.std(axis=-1, keepdims=True) + 1e-8
    arr  = (arr - mean) / std

    return arr  # (NUM_SUBCARRIERS, WINDOW_SIZE)


def sliding_windows(arr: np.ndarray, window: int, stride: int):
    """Generate windows from a long recording"""
    _, T = arr.shape
    windows = []
    start = 0
    while start + window <= T:
        windows.append(arr[:, start:start + window])
        start += stride
    return windows


def build_arrays(raw_dir: Path, use_sliding_window: bool = False):
    """
    Build X, y numpy arrays from raw .npy files
    """
    class_to_files = find_npy_files(raw_dir)

    all_files_exist = all(len(v) > 0 for v in class_to_files.values())
    if not all_files_exist:
        # flat structure: all files in raw_dir, named <label>_<id>.npy
        flat_files = list(raw_dir.rglob("*.npy"))
        if flat_files:
            print(f"Found {len(flat_files)} flat .npy files. Using stem prefix as label.")
            class_to_files = {i: [] for i in range(config.NUM_CLASSES)}
            for f in flat_files:
                try:
                    label = int(f.stem.split("_")[0])
                    if label in class_to_files:
                        class_to_files[label].append(f)
                except (ValueError, IndexError):
                    pass

    X_list, y_list = [], []
    for label, files in class_to_files.items():
        print(f"  Class {label} ({config.CLASSES[label]}): {len(files)} files")
        for f in files:
            try:
                arr = load_sample(f)
                if use_sliding_window:
                    wins = sliding_windows(arr, config.WINDOW_SIZE, config.STRIDE)
                    X_list.extend(wins)
                    y_list.extend([label] * len(wins))
                else:
                    X_list.append(arr)
                    y_list.append(label)
            except Exception as e:
                print(f"    Skip {f.name}: {e}")

    if not X_list:
        print("\nNo samples loaded. Check your data directory structure.")
        print("Expected either:")
        print("  data/raw/normal/*.npy  and  data/raw/bullying/*.npy")
        print("  OR  data/raw/<label>_<id>.npy  (label=0 or 1)")
        sys.exit(1)

    X = np.stack(X_list, axis=0)   # (N, C, T)
    y = np.array(y_list)
    print(f"\nTotal samples: {len(y)}  |  X shape: {X.shape}")
    return X, y


def split_and_save(X: np.ndarray, y: np.ndarray, out_dir: Path):
    """Split into train/val/test and save."""
    rng = np.random.default_rng(config.RANDOM_SEED)
    idx = np.arange(len(y))
    rng.shuffle(idx)

    n     = len(idx)
    n_tr  = int(n * config.TRAIN_RATIO)
    n_val = int(n * config.VAL_RATIO)

    splits = {
        "train": idx[:n_tr],
        "val":   idx[n_tr:n_tr + n_val],
        "test":  idx[n_tr + n_val:],
    }

    for split, sidx in splits.items():
        d = out_dir / split
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "X.npy", X[sidx])
        np.save(d / "y.npy", y[sidx])
        unique, counts = np.unique(y[sidx], return_counts=True)
        dist = dict(zip(unique.tolist(), counts.tolist()))
        print(f"  {split:>5}: {len(sidx):>5} samples | class dist: {dist}")

    print(f"\nDataset saved to {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Prepare WiFi-BullyDetect dataset")
    parser.add_argument("--kaggle",          action="store_true",  help="Download from Kaggle first")
    parser.add_argument("--raw_dir",         type=str, default=str(config.RAW_DATA_DIR))
    parser.add_argument("--out_dir",         type=str, default=str(config.PREPROCESSED_DATA_DIR))
    parser.add_argument("--skip_download",   action="store_true")
    parser.add_argument("--sliding_window",  action="store_true",  help="Apply sliding window augmentation")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)

    if args.kaggle and not args.skip_download:
        download_kaggle(raw_dir)

    print(f"\nBuilding dataset from: {raw_dir}")
    X, y = build_arrays(raw_dir, use_sliding_window=args.sliding_window)

    print("\nSplitting dataset ...")
    split_and_save(X, y, out_dir)

    print("\n✓ Dataset preparation complete!")


if __name__ == "__main__":
    main()
