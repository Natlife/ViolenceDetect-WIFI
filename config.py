"""
ndluong
"""

import os
import json
from pathlib import Path

# Detect environment
IS_COLAB = "COLAB_GPU" in os.environ or "google.colab" in str(os.environ.get("PATH", ""))

# Base paths (auto-detect)
BASE_DIR = Path(__file__).parent.resolve()

if IS_COLAB:
    # Mount Google Drive: from google.colab import drive; drive.mount('/content/drive')
    DRIVE_ROOT = Path("/content/drive/MyDrive/WiFi-BullyDetect")
    DATA_ROOT  = DRIVE_ROOT / "data"
    RESULT_DIR = DRIVE_ROOT / "results"
else:
    DATA_ROOT  = BASE_DIR / "data"
    RESULT_DIR = BASE_DIR / "results"

# Dataset paths 
RAW_DATA_DIR         = DATA_ROOT / "raw"
PREPROCESSED_DATA_DIR = DATA_ROOT / "preprocessed"

TRAIN_DATA_DIR = PREPROCESSED_DATA_DIR / "train"
TEST_DATA_DIR  = PREPROCESSED_DATA_DIR / "test"
VAL_DATA_DIR   = PREPROCESSED_DATA_DIR / "val"

# Result paths
WEIGHTS_DIR = RESULT_DIR / "weights"
PLOTS_DIR   = RESULT_DIR / "plots"
LOGS_DIR    = RESULT_DIR / "logs"

# Classes
CLASSES = ["normal", "bullying"]   # 0 = normal, 1 = bullying
NUM_CLASSES = len(CLASSES)

# CSI / Signal settings
NUM_SUBCARRIERS = 30          # number of WiFi subcarriers
WINDOW_SIZE     = 500         # time-window length (samples)
STRIDE          = 250         # sliding window stride
SAMPLING_RATE   = 100         # Hz

# Model hyperparameters
MODEL_NAME    = "DWT"         # choices: "DWT" | "Transformer" | "CNN"
EMBED_DIM     = 64
NUM_HEADS     = 4
NUM_LAYERS    = 3
DROPOUT       = 0.1
WAVELET       = "db4"         # PyWavelets wavelet name
WAVELET_LEVEL = 3

# Training settings
BATCH_SIZE     = 32
NUM_EPOCHS     = 100
LEARNING_RATE  = 1e-3
WEIGHT_DECAY   = 1e-4
EARLY_STOP     = 15           # patience
LR_SCHEDULER   = "cosine"     # "cosine" | "step" | "none"
LR_STEP_SIZE   = 20
LR_GAMMA       = 0.5

TRAIN_RATIO = 0.7
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15
RANDOM_SEED = 42

# Augmentation
USE_AUGMENTATION     = True
AUGMENT_NOISE_STD    = 0.01
AUGMENT_FLIP_PROB    = 0.5
AUGMENT_SCALE_RANGE  = (0.9, 1.1)

# Misc
NUM_WORKERS  = 0   # Windows requires 0; set to 4+ on Linux/Colab
PIN_MEMORY   = not IS_COLAB
DEVICE       = "auto"   # "auto" | "cpu" | "cuda" | "mps"

# Checkpoint
CHECKPOINT_NAME = "best_model.pth"
RESUME_TRAINING = False


def get_device():
    import torch
    if DEVICE == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(DEVICE)


def ensure_dirs():
    """Create all output directories if they don't exist."""
    for d in [DATA_ROOT, RAW_DATA_DIR, PREPROCESSED_DATA_DIR,
              TRAIN_DATA_DIR, TEST_DATA_DIR, VAL_DATA_DIR,
              WEIGHTS_DIR, PLOTS_DIR, LOGS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def to_dict():
    """Export config as dict (for logging)."""
    return {
        "IS_COLAB": IS_COLAB,
        "MODEL_NAME": MODEL_NAME,
        "NUM_CLASSES": NUM_CLASSES,
        "CLASSES": CLASSES,
        "NUM_SUBCARRIERS": NUM_SUBCARRIERS,
        "WINDOW_SIZE": WINDOW_SIZE,
        "STRIDE": STRIDE,
        "EMBED_DIM": EMBED_DIM,
        "NUM_HEADS": NUM_HEADS,
        "NUM_LAYERS": NUM_LAYERS,
        "DROPOUT": DROPOUT,
        "WAVELET": WAVELET,
        "WAVELET_LEVEL": WAVELET_LEVEL,
        "BATCH_SIZE": BATCH_SIZE,
        "NUM_EPOCHS": NUM_EPOCHS,
        "LEARNING_RATE": LEARNING_RATE,
        "WEIGHT_DECAY": WEIGHT_DECAY,
        "EARLY_STOP": EARLY_STOP,
        "TRAIN_RATIO": TRAIN_RATIO,
        "VAL_RATIO": VAL_RATIO,
        "TEST_RATIO": TEST_RATIO,
        "RANDOM_SEED": RANDOM_SEED,
    }


if __name__ == "__main__":
    ensure_dirs()
    print(json.dumps(to_dict(), indent=2))
    print(f"\nDevice: {get_device()}")
    print(f"Colab: {IS_COLAB}")
