"""
Load a trained checkpoint, run on test set, generate:
 - Classification report
 - Confusion matrix plot
 - ROC curve
 - Per-class accuracy
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, roc_curve, auc
)

import config


class Evaluator:
    def __init__(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        device: torch.device,
        checkpoint_path: Optional[Path] = None,
        run_name: str = "run",
    ):
        self.model       = model.to(device)
        self.test_loader = test_loader
        self.device      = device
        self.run_name    = run_name

        self.plots_dir = config.PLOTS_DIR / run_name
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        if checkpoint_path and checkpoint_path.exists():
            self._load(checkpoint_path)
        elif (config.WEIGHTS_DIR / run_name / "best.pth").exists():
            self._load(config.WEIGHTS_DIR / run_name / "best.pth")

    def _load(self, path: Path):
        print(f"Loading checkpoint: {path}")
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])

    def _predict(self):
        self.model.eval()
        all_preds, all_labels, all_probs = [], [], []

        with torch.no_grad():
            for X, y in self.test_loader:
                X = X.to(self.device)
                logits = self.model(X)
                probs  = torch.softmax(logits, dim=-1).cpu().numpy()
                preds  = logits.argmax(dim=-1).cpu().numpy()
                all_probs.extend(probs)
                all_preds.extend(preds)
                all_labels.extend(y.numpy())

        return np.array(all_labels), np.array(all_preds), np.array(all_probs)

    # Plots

    def _plot_confusion_matrix(self, y_true, y_pred):
        cm   = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=config.CLASSES, yticklabels=config.CLASSES, ax=ax
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix")
        fig.tight_layout()
        out = self.plots_dir / "confusion_matrix.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"Confusion matrix saved → {out}")

    def _plot_roc(self, y_true, y_probs):
        if config.NUM_CLASSES != 2:
            return  # ROC for binary only
        fpr, tpr, _ = roc_curve(y_true, y_probs[:, 1])
        roc_auc     = auc(fpr, tpr)
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
        ax.plot([0, 1], [0, 1], "k--")
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.set_title("ROC Curve")
        ax.legend()
        fig.tight_layout()
        out = self.plots_dir / "roc_curve.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"ROC curve saved → {out}")
        return roc_auc

    def _plot_training_history(self, history: dict):
        epochs = range(1, len(history["train_loss"]) + 1)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].plot(epochs, history["train_loss"], label="Train")
        axes[0].plot(epochs, history["val_loss"],   label="Val")
        axes[0].set_title("Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].legend()

        axes[1].plot(epochs, history["train_acc"], label="Train")
        axes[1].plot(epochs, history["val_acc"],   label="Val")
        axes[1].set_title("Accuracy")
        axes[1].set_xlabel("Epoch")
        axes[1].legend()

        fig.tight_layout()
        out = self.plots_dir / "training_history.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"Training history saved → {out}")

    # Main evaluate

    def evaluate(self, history: Optional[dict] = None) -> dict:
        y_true, y_pred, y_probs = self._predict()

        acc = accuracy_score(y_true, y_pred)
        f1  = f1_score(y_true, y_pred, average="macro", zero_division=0)
        report = classification_report(
            y_true, y_pred,
            target_names=config.CLASSES,
            digits=4,
        )

        print("\n" + "="*50)
        print(f"  Test Accuracy : {acc:.4f}")
        print(f"  Macro F1      : {f1:.4f}")
        print("="*50)
        print(report)

        # Save report
        report_path = self.plots_dir / "test_report.txt"
        with open(report_path, "w") as fp:
            fp.write(f"Accuracy: {acc:.4f}\nMacro F1: {f1:.4f}\n\n")
            fp.write(report)
        print(f"Report saved → {report_path}")

        # Plots
        self._plot_confusion_matrix(y_true, y_pred)
        roc_auc = self._plot_roc(y_true, y_probs)
        if history:
            self._plot_training_history(history)

        metrics = {"accuracy": acc, "macro_f1": f1}
        if roc_auc is not None:
            metrics["roc_auc"] = roc_auc

        # Save metrics JSON
        metrics_path = self.plots_dir / "metrics.json"
        with open(metrics_path, "w") as fp:
            json.dump({k: round(float(v), 6) for k, v in metrics.items()}, fp, indent=2)

        return metrics
