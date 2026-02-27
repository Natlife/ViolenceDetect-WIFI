"""
Usage:
  python train.py
  python train.py --run_name experiment_01
  python train.py --epochs 50 --lr 0.0005 --batch 64
"""

import sys
import argparse
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

import config
config.ensure_dirs()

from dataset import get_dataloaders
from model.dwt_transformer import get_model
from trainer import Trainer
from evaluator import Evaluator


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def parse_args():
    p = argparse.ArgumentParser(description="Train WiFi-BullyDetect")
    p.add_argument("--run_name",  type=str, default=None,
                   help="Name for this run (default: timestamp)")
    p.add_argument("--model",     type=str, default=config.MODEL_NAME,
                   choices=["DWT", "Transformer", "CNN"])
    p.add_argument("--epochs",    type=int, default=config.NUM_EPOCHS)
    p.add_argument("--lr",        type=float, default=config.LEARNING_RATE)
    p.add_argument("--batch",     type=int, default=config.BATCH_SIZE)
    p.add_argument("--workers",   type=int, default=config.NUM_WORKERS)
    p.add_argument("--data_dir",  type=str, default=str(config.PREPROCESSED_DATA_DIR))
    p.add_argument("--eval_only", action="store_true",
                   help="Skip training, only evaluate best checkpoint")
    return p.parse_args()


def main():
    args = parse_args()

    # Override config with CLI args
    config.MODEL_NAME    = args.model
    config.NUM_EPOCHS    = args.epochs
    config.LEARNING_RATE = args.lr
    config.BATCH_SIZE    = args.batch
    config.NUM_WORKERS   = args.workers

    run_name = args.run_name or f"{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    data_dir = Path(args.data_dir)

    set_seed(config.RANDOM_SEED)
    device = config.get_device()
    print(f"\n{'='*60}")
    print(f"  WiFi-BullyDetect")
    print(f"  Run:    {run_name}")
    print(f"  Model:  {config.MODEL_NAME}")
    print(f"  Device: {device}")
    print(f"{'='*60}")

    # Data
    print("\nLoading data ...")
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir   = data_dir,
        batch_size = config.BATCH_SIZE,
        num_workers= config.NUM_WORKERS,
    )

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
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: {config.MODEL_NAME}  |  Parameters: {total_params:,}")

    history = None

    # Train
    if not args.eval_only:
        trainer = Trainer(
            model        = model,
            train_loader = train_loader,
            val_loader   = val_loader,
            device       = device,
            run_name     = run_name,
        )
        history = trainer.fit()

    # Evaluate
    print("\n--- Evaluating on test set ---")
    evaluator = Evaluator(
        model      = model,
        test_loader= test_loader,
        device     = device,
        run_name   = run_name,
    )
    metrics = evaluator.evaluate(history=history)

    print(f"\n✓ Done! Results saved to {config.RESULT_DIR / run_name}")
    return metrics


if __name__ == "__main__":
    main()
