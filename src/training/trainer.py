"""
AgriAI Shared Training Pipeline
Modular, reusable PyTorch Trainer for BaselineCNN, LS-HGNet, and subsequent experiments.

Features:
- Deterministic execution support
- Per-epoch training and validation loops
- Checkpoint persistence for best validation metric (Macro F1)
- Structured training history tracking
- Pure separation of concerns (no test-set contamination)
"""

from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
import json
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class Trainer:
    """
    Standard PyTorch Trainer for classification models.
    """
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        device: str = "cpu",
        save_dir: str or Path = "results/baseline",
        seed: int = 42
    ):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed

        self.history: List[Dict[str, Any]] = []
        self.best_metric = -float("inf")
        self.best_epoch = -1

    def train_epoch(self, dataloader: DataLoader) -> Tuple[float, float]:
        """
        Runs one epoch of training.
        Returns:
            (average_train_loss, train_accuracy)
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in dataloader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    def validate_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Runs validation across the evaluation DataLoader.
        Returns:
            Dictionary with val_loss, val_acc, val_macro_f1, val_macro_precision, val_macro_recall
        """
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_targets = []
        total = 0

        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                running_loss += loss.item() * inputs.size(0)
                preds = torch.argmax(outputs, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                total += targets.size(0)

        val_loss = running_loss / total
        val_acc = float(accuracy_score(all_targets, all_preds))
        val_macro_f1 = float(f1_score(all_targets, all_preds, average="macro", zero_division=0))
        val_macro_prec = float(precision_score(all_targets, all_preds, average="macro", zero_division=0))
        val_macro_rec = float(recall_score(all_targets, all_preds, average="macro", zero_division=0))

        return {
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "val_macro_f1": val_macro_f1,
            "val_macro_precision": val_macro_prec,
            "val_macro_recall": val_macro_rec
        }

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        checkpoint_name: str = "best_model.pt",
        verbose: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Executes full multi-epoch training and validation loop.
        Saves best checkpoint based on validation Macro F1 score.
        """
        if verbose:
            print(f"Starting training for {epochs} epochs on device '{self.device}'...")
            print("-" * 75)

        for epoch in range(1, epochs + 1):
            t0 = time.time()

            train_loss, train_acc = self.train_epoch(train_loader)
            val_metrics = self.validate_epoch(val_loader)

            curr_lr = self.optimizer.param_groups[0]["lr"]

            if self.scheduler is not None:
                self.scheduler.step()

            duration = time.time() - t0

            epoch_record = {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "train_accuracy": round(train_acc, 4),
                "val_loss": round(val_metrics["val_loss"], 4),
                "val_accuracy": round(val_metrics["val_accuracy"], 4),
                "val_macro_f1": round(val_metrics["val_macro_f1"], 4),
                "val_macro_precision": round(val_metrics["val_macro_precision"], 4),
                "val_macro_recall": round(val_metrics["val_macro_recall"], 4),
                "learning_rate": curr_lr,
                "duration_seconds": round(duration, 2)
            }
            self.history.append(epoch_record)

            # Checkpoint save on best validation macro F1
            is_best = val_metrics["val_macro_f1"] > self.best_metric
            if is_best:
                self.best_metric = val_metrics["val_macro_f1"]
                self.best_epoch = epoch
                checkpoint_path = self.save_dir / checkpoint_name
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_macro_f1": val_metrics["val_macro_f1"],
                    "val_loss": val_metrics["val_loss"],
                    "seed": self.seed
                }, checkpoint_path)

            if verbose:
                best_tag = " [BEST]" if is_best else ""
                print(
                    f"Epoch {epoch:02d}/{epochs:02d} | "
                    f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:.2f}% | "
                    f"Val Loss: {val_metrics['val_loss']:.4f} Acc: {val_metrics['val_accuracy']*100:.2f}% "
                    f"F1: {val_metrics['val_macro_f1']*100:.2f}% | "
                    f"Time: {duration:.1f}s{best_tag}",
                    flush=True
                )

        # Save training history
        with open(self.save_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)

        if verbose:
            print("-" * 75, flush=True)
            print(f"Training completed. Best Val Macro F1: {self.best_metric*100:.2f}% at Epoch {self.best_epoch}", flush=True)
            print(f"Artifacts saved to: {self.save_dir}", flush=True)

        return self.history
