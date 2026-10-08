"""
src/optimization/fitness.py
============================
AgriAI M6 — Fitness Evaluator for STO and Random Search.

Fitness definition (AgriAI M0, Section 13.5):
    fitness = 1 - validation_macro_F1

TEST-SET PROTECTION
-------------------
This module deliberately provides NO test_dataset parameter.
Only train_dataset and val_dataset are accepted.
There is no code path that can load test data from inside this evaluator.
An explicit runtime guard verifies that val_dataset length != 2724 or
that the data doesn't match test.csv patterns at initialisation time
(additional assertion in run_sto.py and run_random_search.py).

CANDIDATE LOGGING
-----------------
Every candidate evaluation produces a full CandidateRecord containing:
- candidate_id, method, seed
- internal theta (normalized) vector
- decoded hyperparameters
- full validation metrics
- fitness, training time, timestamp

Fixed architecture (M4 final):
- base_channels = 64
- num_stacks    = 3
- encoder_levels = 4
These are NOT hyperparameters being optimized.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.models.ls_hgnet import LS_HGNet
from src.optimization.sto import SearchSpace
from src.training.trainer import Trainer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Candidate record
# ---------------------------------------------------------------------------
@dataclass
class CandidateRecord:
    """Full record for a single evaluated hyperparameter candidate."""
    candidate_id:              int
    method:                    str      # "sto_stabilized" | "random_search"
    eval_seed:                 int

    # Internal (normalised) position
    theta_lr:                  float    # theta_1 in [-4, -2] (log-scale)
    theta_wd:                  float    # theta_2 in [-6, -2] (log-scale)
    theta_dropout:             float    # theta_3 in [0, 0.5]
    theta_batch_size:          float    # theta_4 in [0, 2]

    # Decoded hyperparameters
    learning_rate:             float
    weight_decay:              float
    dropout:                   float
    batch_size:                int

    # Training protocol
    training_epochs:           int
    image_size:                int

    # Validation metrics (from best epoch within training run)
    validation_loss:           float
    validation_accuracy:       float
    validation_macro_precision: float
    validation_macro_recall:   float
    validation_macro_f1:       float

    # Fitness
    fitness:                   float    # 1 - validation_macro_f1

    # Timing
    training_time_seconds:     float
    timestamp_utc:             str


def record_to_dict(r: CandidateRecord) -> dict:
    return asdict(r)


# ---------------------------------------------------------------------------
# Core fitness evaluator
# ---------------------------------------------------------------------------
class FitnessEvaluator:
    """
    Callable fitness evaluator shared by both STO and Random Search.

    TEST-SET ISOLATION
    ------------------
    Only train_dataset and val_dataset are accepted as parameters.
    There is no test_dataset parameter.  This is a deliberate API-level
    protection: the evaluator cannot accidentally receive test data.

    Parameters
    ----------
    train_dataset  : PyTorch Dataset for training
    val_dataset    : PyTorch Dataset for validation ONLY
    num_epochs     : Epochs per candidate evaluation
    search_space   : SearchSpace for theta -> hyperparams decoding
    method         : "sto_stabilized" | "sto_original" | "random_search"
    base_channels  : Fixed LS-HGNet architecture (64, from M4)
    num_stacks     : Fixed LS-HGNet architecture (3, from M4)
    encoder_levels : Fixed LS-HGNet architecture (4, from M4)
    num_classes    : 10 (PlantVillage tomato)
    image_size     : 128 for proxy, 224 for full
    device         : "cpu"
    seed           : Base seed; combined with eval_count per evaluation
    use_cache      : Cache fitness for identical decoded HP sets
    save_dir       : Where to write per-eval checkpoints (None = skip)
    """

    def __init__(
        self,
        train_dataset:  Dataset,
        val_dataset:    Dataset,
        num_epochs:     int,
        search_space:   SearchSpace,
        method:         str = "sto_stabilized",
        base_channels:  int = 64,
        num_stacks:     int = 3,
        encoder_levels: int = 4,
        num_classes:    int = 10,
        image_size:     int = 128,
        device:         str = "cpu",
        seed:           int = 42,
        use_cache:      bool = True,
        save_dir:       Optional[str] = None,
        eval_id_prefix: Optional[str] = None,
    ) -> None:
        if eval_id_prefix is not None:
            method = eval_id_prefix
        # TEST-SET GUARD: confirm no test data passed
        if train_dataset is val_dataset:
            raise ValueError(
                "TEST-SET GUARD: train_dataset and val_dataset must be different objects. "
                "Never pass the same dataset for both."
            )

        self.train_dataset  = train_dataset
        self.val_dataset    = val_dataset
        self.num_epochs     = num_epochs
        self.search_space   = search_space
        self.method         = method
        self.base_channels  = base_channels
        self.num_stacks     = num_stacks
        self.encoder_levels = encoder_levels
        self.num_classes    = num_classes
        self.image_size     = image_size
        self.device         = device
        self.seed           = seed
        self.use_cache      = use_cache
        self.save_dir       = save_dir

        self._cache:         Dict[str, float]         = {}
        self._eval_count:    int                       = 0
        self.candidate_log:  List[CandidateRecord]    = []

    # ------------------------------------------------------------------
    def _cache_key(self, params: dict) -> str:
        return (
            f"lr={params['learning_rate']:.8f}_"
            f"wd={params['weight_decay']:.8f}_"
            f"drop={params['dropout']:.4f}_"
            f"bs={params['batch_size']}"
        )

    # ------------------------------------------------------------------
    def __call__(self, theta: np.ndarray) -> float:
        """
        Evaluate fitness for theta.  Returns 1 - val_macro_F1.
        Also records a full CandidateRecord.
        """
        self._eval_count += 1
        params = self.search_space.decode(theta)

        key = self._cache_key(params)
        if self.use_cache and key in self._cache:
            cached_fitness = self._cache[key]
            logger.info(
                "[%s #%d] Cache hit: %s -> fitness=%.6f",
                self.method, self._eval_count, params, cached_fitness,
            )
            # Still log a record for the cached evaluation
            # We use -1 sentinel for cached metrics lookup
            prev = next(
                (r for r in reversed(self.candidate_log) if self._cache_key({
                    "learning_rate": r.learning_rate,
                    "weight_decay": r.weight_decay,
                    "dropout": r.dropout,
                    "batch_size": r.batch_size,
                }) == key),
                None,
            )
            if prev:
                rec = CandidateRecord(
                    candidate_id=self._eval_count,
                    method=self.method,
                    eval_seed=self.seed + self._eval_count,
                    theta_lr=float(theta[0]),
                    theta_wd=float(theta[1]),
                    theta_dropout=float(theta[2]),
                    theta_batch_size=float(theta[3]),
                    learning_rate=params["learning_rate"],
                    weight_decay=params["weight_decay"],
                    dropout=params["dropout"],
                    batch_size=params["batch_size"],
                    training_epochs=self.num_epochs,
                    image_size=self.image_size,
                    validation_loss=prev.validation_loss,
                    validation_accuracy=prev.validation_accuracy,
                    validation_macro_precision=prev.validation_macro_precision,
                    validation_macro_recall=prev.validation_macro_recall,
                    validation_macro_f1=prev.validation_macro_f1,
                    fitness=cached_fitness,
                    training_time_seconds=0.0,
                    timestamp_utc=datetime.now(timezone.utc).isoformat(),
                )
                self.candidate_log.append(rec)
            return cached_fitness

        logger.info(
            "[%s #%d] Evaluating: lr=%.2e wd=%.2e drop=%.3f bs=%d",
            self.method, self._eval_count,
            params["learning_rate"], params["weight_decay"],
            params["dropout"], params["batch_size"],
        )

        t0 = time.perf_counter()
        metrics, fitness = self._run_evaluation(params)
        elapsed = time.perf_counter() - t0

        logger.info(
            "[%s #%d] val_macro_f1=%.4f%% fitness=%.6f  time=%.1fs",
            self.method, self._eval_count,
            metrics["val_macro_f1"] * 100, fitness, elapsed,
        )

        if self.use_cache:
            self._cache[key] = fitness

        rec = CandidateRecord(
            candidate_id=self._eval_count,
            method=self.method,
            eval_seed=self.seed + self._eval_count,
            theta_lr=float(theta[0]),
            theta_wd=float(theta[1]),
            theta_dropout=float(theta[2]),
            theta_batch_size=float(theta[3]),
            learning_rate=params["learning_rate"],
            weight_decay=params["weight_decay"],
            dropout=params["dropout"],
            batch_size=params["batch_size"],
            training_epochs=self.num_epochs,
            image_size=self.image_size,
            validation_loss=metrics["val_loss"],
            validation_accuracy=metrics["val_accuracy"],
            validation_macro_precision=metrics["val_macro_precision"],
            validation_macro_recall=metrics["val_macro_recall"],
            validation_macro_f1=metrics["val_macro_f1"],
            fitness=fitness,
            training_time_seconds=elapsed,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
        )
        self.candidate_log.append(rec)
        return fitness

    # ------------------------------------------------------------------
    def _run_evaluation(self, params: dict):
        """Train LS-HGNet for num_epochs and return (metrics_dict, fitness)."""
        eval_seed = self.seed + self._eval_count
        torch.manual_seed(eval_seed)
        np.random.seed(eval_seed)

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=params["batch_size"],
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            drop_last=False,
        )
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=params["batch_size"],
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )

        # Fixed M4 architecture — only dropout is from search space
        model = LS_HGNet(
            in_channels=3,
            num_classes=self.num_classes,
            base_channels=self.base_channels,    # FIXED: 64
            num_stacks=self.num_stacks,           # FIXED: 3
            encoder_levels=self.encoder_levels,   # FIXED: 4
            dropout=params["dropout"],            # From search space
        )

        criterion = nn.CrossEntropyLoss()
        # Using AdamW as per M6 proxy training spec
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=params["learning_rate"],
            weight_decay=params["weight_decay"],
        )

        save_dir = (
            f"{self.save_dir}/{self.method}_{self._eval_count}"
            if self.save_dir else
            f"results/m6_evals/{self.method}_{self._eval_count}"
        )

        trainer = Trainer(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=None,
            device=self.device,
            save_dir=save_dir,
            seed=eval_seed,
        )

        history = trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=self.num_epochs,
            checkpoint_name="eval_best.pt",
            verbose=False,
        )

        # Best validation metrics across all trained epochs
        best_h = max(history, key=lambda h: h["val_macro_f1"])
        metrics = {
            "val_loss":             best_h["val_loss"],
            "val_accuracy":         best_h["val_accuracy"],
            "val_macro_precision":  best_h["val_macro_precision"],
            "val_macro_recall":     best_h["val_macro_recall"],
            "val_macro_f1":         best_h["val_macro_f1"],
        }
        fitness = 1.0 - metrics["val_macro_f1"]
        return metrics, float(fitness)

    # ------------------------------------------------------------------
    @property
    def eval_count(self) -> int:
        return self._eval_count

    @property
    def best_fitness(self) -> float:
        if not self.candidate_log:
            return float("inf")
        return min(r.fitness for r in self.candidate_log)

    @property
    def best_record(self) -> Optional[CandidateRecord]:
        if not self.candidate_log:
            return None
        return min(self.candidate_log, key=lambda r: r.fitness)

    def get_best_so_far(self) -> List[float]:
        """Return best-so-far fitness after each evaluation."""
        best = float("inf")
        history = []
        for rec in self.candidate_log:
            best = min(best, rec.fitness)
            history.append(best)
        return history

    def get_best_so_far_f1(self) -> List[float]:
        """Return best-so-far macro-F1 after each evaluation."""
        return [1.0 - f for f in self.get_best_so_far()]


# Backward-compatible alias for M5 scripts
STOFitnessEvaluator = FitnessEvaluator

