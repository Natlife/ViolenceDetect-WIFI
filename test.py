"""
Usage:
  # Evaluate on test split using best checkpoint:
  python test.py --run_name DWT_20240101_120000

  # Predict on a single CSI file:
  python test.py --run_name DWT_20240101_120000 --input path/to/csi.npy
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import torch

import config
config.ensure_dirs()

from dataset import get_dataloaders, BullyDataset
from model.dwt_transformer import get_model
from evaluator import Evaluator


def predict_single(model: torch.nn.Module, csi_path: Path, device: torch.device):
    """Predict label for a single CSI .npy file."""
    arr = np.load(csi_path).astype(np.float32)

    # Normalize and ensure shape (C, T)
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]
    elif arr.ndim == 2 and arr.shape[0] > arr.shape[1]:
        arr = arr.T

    # Crop/pad to WINDOW_SIZE
    C, T = arr.shape
    if T > config.WINDOW_SIZE:
        arr = arr[:, :config.WINDOW_SIZE]
    elif T < config.WINDOW_SIZE:
        arr = np.pad(arr, ((0, 0), (0, config.WINDOW_SIZE - T)), mode="reflect")

    mean = arr.mean(axis=-1, keepdims=True)
    std  = arr.std(axis=-1, keepdims=True) + 1e-8
    arr  = (arr - mean) / std

    x = torch.tensor(arr).unsqueeze(0).to(device)  # (1, C, T)

    model.eval()
    with torch.no_grad():
        logits = model(x)
        probs  = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        pred   = probs.argmax()

    print(f"\nCSI file : {csi_path.name}")
    print(f"Prediction: {config.CLASSES[pred]} (class {pred})")
    for i, cls in enumerate(config.CLASSES):
        print(f"  P({cls}) = {probs[i]:.4f}")
    return pred, probs


def main():
    parser = argparse.ArgumentParser(description="Test WiFi-BullyDetect")
    parser.add_argument("--run_name",   type=str, required=True,
                        help="Name of the training run to load")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to specific .pth file (default: best.pth)")
    parser.add_argument("--input",      type=str, default=None,
                        help="Path to a single .npy CSI file for inference")
    parser.add_argument("--data_dir",   type=str, default=str(config.PREPROCESSED_DATA_DIR))
    parser.add_argument("--batch",      type=int, default=config.BATCH_SIZE)
    args = parser.parse_args()

    device = config.get_device()
    print(f"Device: {device}")

    # Model
    model = get_model(
        config.MODEL_NAME,
        in_channels   = config.NUM_SUBCARRIERS,
        num_classes   = config.NUM_CLASSES,
        embed_dim     = config.EMBED_DIM,
        num_heads     = config.NUM_HEADS,
        num_layers    = config.NUM_LAYERS,
        dropout       = config.DROPOUT,
        wavelet       = config.WAVELET,
        wavelet_level = config.WAVELET_LEVEL,
    )

    checkpoint = Path(args.checkpoint) if args.checkpoint else (
        config.WEIGHTS_DIR / args.run_name / "best.pth"
    )

    if not checkpoint.exists():
        print(f"Checkpoint not found: {checkpoint}")
        sys.exit(1)

    # Single file inference
    if args.input:
        from evaluator import Evaluator
        ckpt = torch.load(checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        predict_single(model, Path(args.input), device)
        return

    # Full test-set evaluation
    data_dir = Path(args.data_dir)
    _, _, test_loader = get_dataloaders(data_dir=data_dir, batch_size=args.batch)

    evaluator = Evaluator(
        model           = model,
        test_loader     = test_loader,
        device          = device,
        checkpoint_path = checkpoint,
        run_name        = args.run_name,
    )
    metrics = evaluator.evaluate()
    print("\nTest metrics:", metrics)


if __name__ == "__main__":
    main()
