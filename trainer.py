"""
Full training loop with:
 - Early stopping
 - LR scheduling
 - TensorBoard / CSV logging
 - Checkpoint save/resume
 - Metric tracking (acc, F1, confusion matrix)
"""

import os
import csv
import time
import json
import shutil
from pathlib import Path
from typing import Optional, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)

import config


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        run_name: str = "run",
    ):
        self.model        = model.to(device)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.device       = device
        self.run_name     = run_name

        # Paths
        config.ensure_dirs()
        self.weights_dir = config.WEIGHTS_DIR / run_name
        self.logs_dir    = config.LOGS_DIR    / run_name
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        self.best_path   = self.weights_dir / "best.pth"
        self.last_path   = self.weights_dir / "last.pth"
        self.csv_log     = self.logs_dir    / "training_log.csv"

        # Optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
        )

        # Scheduler
        if config.LR_SCHEDULER == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=config.NUM_EPOCHS, eta_min=1e-6
            )
        elif config.LR_SCHEDULER == "step":
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=config.LR_STEP_SIZE, gamma=config.LR_GAMMA
            )
        else:
            self.scheduler = None

        # State
        self.start_epoch  = 0
        self.best_val_acc = 0.0
        self.no_improve   = 0
        self.history: Dict = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_f1": []}

        # Resume
        if config.RESUME_TRAINING and self.last_path.exists():
            self._load_checkpoint(self.last_path)

        # CSV header
        if not self.csv_log.exists():
            with open(self.csv_log, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "val_f1", "lr"])

        # TensorBoard (optional)
        self.tb_writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.tb_writer = SummaryWriter(log_dir=str(self.logs_dir / "tb"))
            print("TensorBoard enabled →", self.logs_dir / "tb")
        except ImportError:
            pass

    # Checkpoint helpers

    def _save_checkpoint(self, path: Path, epoch: int, is_best: bool = False):
        state = {
            "epoch":        epoch,
            "model_state":  self.model.state_dict(),
            "optimizer":    self.optimizer.state_dict(),
            "scheduler":    self.scheduler.state_dict() if self.scheduler else None,
            "best_val_acc": self.best_val_acc,
            "history":      self.history,
            "config":       config.to_dict(),
        }
        torch.save(state, path)
        if is_best:
            shutil.copy(path, self.best_path)

    def _load_checkpoint(self, path: Path):
        print(f"Resuming from {path}")
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        if self.scheduler and ckpt.get("scheduler"):
            self.scheduler.load_state_dict(ckpt["scheduler"])
        self.start_epoch  = ckpt["epoch"] + 1
        self.best_val_acc = ckpt.get("best_val_acc", 0.0)
        self.history      = ckpt.get("history", self.history)

    # One epoch

    def _train_epoch(self) -> Dict:
        self.model.train()
        total_loss = 0.0
        all_preds, all_labels = [], []

        for X, y in self.train_loader:
            X, y = X.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(X)
            loss   = self.criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item() * len(y)
            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.cpu().numpy())

        n    = len(all_labels)
        loss = total_loss / n
        acc  = accuracy_score(all_labels, all_preds)
        return {"loss": loss, "acc": acc}

    def _eval_epoch(self, loader: DataLoader) -> Dict:
        self.model.eval()
        total_loss = 0.0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for X, y in loader:
                X, y = X.to(self.device), y.to(self.device)
                logits = self.model(X)
                loss   = self.criterion(logits, y)
                total_loss += loss.item() * len(y)
                preds = logits.argmax(dim=-1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(y.cpu().numpy())

        n    = len(all_labels)
        loss = total_loss / n
        acc  = accuracy_score(all_labels, all_preds)
        f1   = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        return {"loss": loss, "acc": acc, "f1": f1, "preds": all_preds, "labels": all_labels}

    # Main training loop

    def fit(self):
        print(f"\n{'='*60}")
        print(f"  Training: {self.run_name}")
        print(f"  Device:   {self.device}")
        print(f"  Epochs:   {config.NUM_EPOCHS}  |  Batch: {config.BATCH_SIZE}")
        print(f"{'='*60}\n")

        for epoch in range(self.start_epoch, config.NUM_EPOCHS):
            t0 = time.time()

            train_metrics = self._train_epoch()
            val_metrics   = self._eval_epoch(self.val_loader)

            if self.scheduler:
                self.scheduler.step()

            lr = self.optimizer.param_groups[0]["lr"]
            dt = time.time() - t0

            # Logging
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["train_acc"].append(train_metrics["acc"])
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_acc"].append(val_metrics["acc"])
            self.history["val_f1"].append(val_metrics["f1"])

            with open(self.csv_log, "a", newline="") as f:
                csv.writer(f).writerow([
                    epoch + 1,
                    f"{train_metrics['loss']:.4f}", f"{train_metrics['acc']:.4f}",
                    f"{val_metrics['loss']:.4f}",   f"{val_metrics['acc']:.4f}",
                    f"{val_metrics['f1']:.4f}",     f"{lr:.6f}",
                ])

            if self.tb_writer:
                self.tb_writer.add_scalars("Loss", {"train": train_metrics["loss"], "val": val_metrics["loss"]}, epoch)
                self.tb_writer.add_scalars("Acc",  {"train": train_metrics["acc"],  "val": val_metrics["acc"]},  epoch)
                self.tb_writer.add_scalar("LR", lr, epoch)

            print(f"Epoch {epoch+1:>3}/{config.NUM_EPOCHS} | "
                  f"Train loss {train_metrics['loss']:.4f} acc {train_metrics['acc']:.4f} | "
                  f"Val   loss {val_metrics['loss']:.4f} acc {val_metrics['acc']:.4f} f1 {val_metrics['f1']:.4f} | "
                  f"lr {lr:.2e} | {dt:.1f}s")

            # Checkpoint
            is_best = val_metrics["acc"] > self.best_val_acc
            if is_best:
                self.best_val_acc = val_metrics["acc"]
                self.no_improve   = 0
                print(f"  ✓ New best val acc: {self.best_val_acc:.4f} → saved to {self.best_path}")
            else:
                self.no_improve += 1

            self._save_checkpoint(self.last_path, epoch, is_best=is_best)

            # Early stopping
            if config.EARLY_STOP > 0 and self.no_improve >= config.EARLY_STOP:
                print(f"\nEarly stopping after {self.no_improve} epochs without improvement.")
                break

        print(f"\nTraining complete. Best val acc: {self.best_val_acc:.4f}")
        if self.tb_writer:
            self.tb_writer.close()

        return self.history
