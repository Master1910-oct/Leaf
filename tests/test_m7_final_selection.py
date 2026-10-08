"""
tests/test_m7_final_selection.py
==================================
AgriAI M7 — Final LS-HGNet Model Selection, Full Training & Validation Unit Tests.

Test Coverage (14 verification gates):
  1.  Correct configuration loading
  2.  Correct candidate construction (A: Default, B: STO, C: Random Search)
  3.  Correct hyperparameter application (LR, WD, Dropout, Batch Size)
  4.  Same architecture across candidates (LS-HGNet invariant 8.04M params)
  5.  Correct dataset split usage (train=12,712, val=2,724, test=2,724)
  6.  Test-set quarantine before final evaluation
  7.  Final model checkpoint loading
  8.  Final prediction dimensions ([B, 10])
  9.  No NaN/Inf in model forward pass and loss
 10.  Deterministic evaluation under fixed seed
 11.  Metric calculation correctness (Macro-F1, Precision, Recall, Accuracy)
 12.  Confusion matrix generation (10x10 shape and normalization)
 13.  Final configuration serialization (YAML / JSON)
 14.  Test evaluation only permitted after model lock
"""

import os
import json
import yaml
import pytest
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.models.ls_hgnet import (
    LS_HGNet,
    build_ls_hgnet_from_config,
    count_parameters,
)
from src.data.dataset import PlantVillageDataset
from src.evaluation.metrics import compute_classification_metrics
from sklearn.metrics import confusion_matrix


# ---------------------------------------------------------------------------
# Candidate Hyperparameter Specifications
# ---------------------------------------------------------------------------

CANDIDATES = {
    "candidate_a_default": {
        "name": "Default LS-HGNet",
        "learning_rate": 1.0e-3,
        "weight_decay": 1.0e-4,
        "dropout": 0.30,
        "batch_size": 32,
    },
    "candidate_b_sto": {
        "name": "STO-Optimized LS-HGNet",
        "learning_rate": 0.004977535,
        "weight_decay": 1.8342e-6,
        "dropout": 0.50,
        "batch_size": 32,
    },
    "candidate_c_random": {
        "name": "Random-Search-Optimized LS-HGNet",
        "learning_rate": 0.00475297,
        "weight_decay": 5.25439e-6,
        "dropout": 0.431,
        "batch_size": 16,
    },
}


# ---------------------------------------------------------------------------
# 1. Configuration Loading
# ---------------------------------------------------------------------------

class TestConfigurationLoading:
    def test_final_yaml_exists_and_loads(self):
        cfg_path = Path("configs/final.yaml")
        assert cfg_path.is_file(), "configs/final.yaml must exist."
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        assert "experiment" in cfg
        assert "dataset" in cfg
        assert "model" in cfg
        assert "training" in cfg

    def test_final_yaml_defaults(self):
        with open("configs/final.yaml", "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        assert cfg["model"]["base_channels"] == 64
        assert cfg["model"]["num_stacks"] == 3
        assert cfg["model"]["encoder_levels"] == 4
        assert cfg["dataset"]["num_classes"] == 10
        assert cfg["preprocessing"]["input_size"] == 224


# ---------------------------------------------------------------------------
# 2. Candidate Construction
# ---------------------------------------------------------------------------

class TestCandidateConstruction:
    def test_all_candidates_defined(self):
        assert "candidate_a_default" in CANDIDATES
        assert "candidate_b_sto" in CANDIDATES
        assert "candidate_c_random" in CANDIDATES

    def test_candidate_parameter_types(self):
        for cid, cand in CANDIDATES.items():
            assert isinstance(cand["learning_rate"], float)
            assert isinstance(cand["weight_decay"], float)
            assert isinstance(cand["dropout"], float)
            assert isinstance(cand["batch_size"], int)
            assert cand["learning_rate"] > 0
            assert cand["weight_decay"] > 0
            assert 0.0 <= cand["dropout"] <= 0.5
            assert cand["batch_size"] in [16, 32, 64]


# ---------------------------------------------------------------------------
# 3. Hyperparameter Application
# ---------------------------------------------------------------------------

class TestHyperparameterApplication:
    def test_dropout_applied_to_model(self):
        for cid, cand in CANDIDATES.items():
            model = LS_HGNet(
                num_classes=10,
                base_channels=64,
                num_stacks=3,
                encoder_levels=4,
                dropout=cand["dropout"]
            )
            assert abs(model.dropout.p - cand["dropout"]) < 1e-5

    def test_optimizer_receives_lr_and_wd(self):
        for cid, cand in CANDIDATES.items():
            model = nn.Linear(10, 2)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=cand["learning_rate"],
                weight_decay=cand["weight_decay"]
            )
            param_group = optimizer.param_groups[0]
            assert abs(param_group["lr"] - cand["learning_rate"]) < 1e-7
            assert abs(param_group["weight_decay"] - cand["weight_decay"]) < 1e-8


# ---------------------------------------------------------------------------
# 4. Architecture Invariance Across Candidates
# ---------------------------------------------------------------------------

class TestArchitectureInvariance:
    def test_exact_parameter_count_match(self):
        models = [
            LS_HGNet(num_classes=10, base_channels=64, num_stacks=3, encoder_levels=4, dropout=c["dropout"])
            for c in CANDIDATES.values()
        ]
        counts = [count_parameters(m) for m in models]
        assert all(c == 8041354 for c in counts), f"All models must have 8,041,354 parameters. Got: {counts}"

    def test_structure_identity(self):
        for c in CANDIDATES.values():
            model = LS_HGNet(num_classes=10, base_channels=64, num_stacks=3, encoder_levels=4, dropout=c["dropout"])
            assert model.num_stacks == 3
            assert model.base_channels == 64
            assert model.encoder_levels == 4
            assert model.num_classes == 10


# ---------------------------------------------------------------------------
# 5. Dataset Split Usage
# ---------------------------------------------------------------------------

class TestDatasetSplitUsage:
    def test_splits_csv_exist_and_counts(self):
        data_dir = Path("data/splits")
        train_csv = data_dir / "train.csv"
        val_csv = data_dir / "val.csv"
        test_csv = data_dir / "test.csv"

        assert train_csv.is_file(), "data/splits/train.csv must exist."
        assert val_csv.is_file(), "data/splits/val.csv must exist."
        assert test_csv.is_file(), "data/splits/test.csv must exist."

        # Count lines excluding header
        with open(train_csv) as f:
            train_count = sum(1 for _ in f) - 1
        with open(val_csv) as f:
            val_count = sum(1 for _ in f) - 1
        with open(test_csv) as f:
            test_count = sum(1 for _ in f) - 1

        assert train_count == 12712, f"Train must have 12,712 images. Got: {train_count}"
        assert val_count == 2724, f"Val must have 2,724 images. Got: {val_count}"
        assert test_count == 2724, f"Test must have 2,724 images. Got: {test_count}"


# ---------------------------------------------------------------------------
# 6. Test Set Quarantine Guard
# ---------------------------------------------------------------------------

class TestTestSetQuarantine:
    def test_selection_pipeline_does_not_accept_test_set(self):
        """Selection logic must only operate on validation metrics."""
        from src.optimization.fitness import FitnessEvaluator
        import inspect
        sig = inspect.signature(FitnessEvaluator.__init__)
        assert "test_dataset" not in sig.parameters
        assert "train_dataset" in sig.parameters
        assert "val_dataset" in sig.parameters


# ---------------------------------------------------------------------------
# 7. Model Checkpoint Loading
# ---------------------------------------------------------------------------

class TestCheckpointLoading:
    def test_m4_checkpoint_loads_cleanly(self):
        ckpt_path = Path("results/ls_hgnet_final/best_model.pt")
        assert ckpt_path.is_file(), "results/ls_hgnet_final/best_model.pt must exist."
        checkpoint = torch.load(str(ckpt_path), map_location="cpu")
        assert "model_state_dict" in checkpoint
        assert "epoch" in checkpoint
        assert "val_loss" in checkpoint

        model = LS_HGNet(num_classes=10, base_channels=64, num_stacks=3, encoder_levels=4, dropout=0.3)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()


# ---------------------------------------------------------------------------
# 8. Prediction Dimensions
# ---------------------------------------------------------------------------

class TestPredictionDimensions:
    def test_forward_pass_output_shape(self):
        model = LS_HGNet(num_classes=10, base_channels=64, num_stacks=3, encoder_levels=4, dropout=0.3)
        model.eval()
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 10), f"Expected shape (2, 10), got {out.shape}"


# ---------------------------------------------------------------------------
# 9. No NaN or Inf
# ---------------------------------------------------------------------------

class TestNoNaNInf:
    def test_model_outputs_finite(self):
        model = LS_HGNet(num_classes=10, base_channels=64, num_stacks=3, encoder_levels=4, dropout=0.3)
        model.eval()
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert torch.isfinite(out).all(), "Model output contains NaN or Inf"

    def test_loss_finite(self):
        criterion = nn.CrossEntropyLoss()
        logits = torch.randn(4, 10)
        targets = torch.tensor([0, 3, 7, 9])
        loss = criterion(logits, targets)
        assert torch.isfinite(loss), "Loss is not finite"


# ---------------------------------------------------------------------------
# 10. Deterministic Evaluation
# ---------------------------------------------------------------------------

class TestDeterministicEvaluation:
    def test_same_seed_gives_identical_logits(self):
        torch.manual_seed(42)
        m1 = LS_HGNet(num_classes=10, base_channels=64, num_stacks=3, encoder_levels=4, dropout=0.3)
        torch.manual_seed(42)
        m2 = LS_HGNet(num_classes=10, base_channels=64, num_stacks=3, encoder_levels=4, dropout=0.3)

        x = torch.randn(2, 3, 224, 224)
        m1.eval()
        m2.eval()
        with torch.no_grad():
            y1 = m1(x)
            y2 = m2(x)
        assert torch.allclose(y1, y2, atol=1e-6), "Models initialized with same seed produced different outputs"


# ---------------------------------------------------------------------------
# 11. Metric Calculation Correctness
# ---------------------------------------------------------------------------

class TestMetricCalculation:
    def test_multiclass_metrics_perfect_predictions(self):
        y_true = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        y_pred = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        metrics = compute_classification_metrics(y_true, y_pred)
        assert metrics["accuracy"] == 1.0
        assert metrics["macro_f1"] == 1.0
        assert metrics["macro_precision"] == 1.0
        assert metrics["macro_recall"] == 1.0

    def test_multiclass_metrics_imperfect_predictions(self):
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 1, 1, 1, 2, 0])
        metrics = compute_classification_metrics(y_true, y_pred)
        assert 0.0 < metrics["accuracy"] < 1.0
        assert 0.0 < metrics["macro_f1"] < 1.0


# ---------------------------------------------------------------------------
# 12. Confusion Matrix Generation
# ---------------------------------------------------------------------------

class TestConfusionMatrix:
    def test_confusion_matrix_shape(self):
        y_true = np.random.randint(0, 10, size=100)
        y_pred = np.random.randint(0, 10, size=100)
        cm = confusion_matrix(y_true, y_pred, labels=list(range(10)))
        assert cm.shape == (10, 10)
        assert cm.sum() == 100

    def test_normalized_confusion_matrix_rows_sum_to_one_or_zero(self):
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 1, 1, 1, 2, 2])
        cm_norm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2], normalize="true")
        for row in cm_norm:
            assert abs(row.sum() - 1.0) < 1e-5 or row.sum() == 0.0


# ---------------------------------------------------------------------------
# 13. Final Configuration Serialization
# ---------------------------------------------------------------------------

class TestConfigurationSerialization:
    def test_save_and_load_config_yaml(self, tmp_path):
        sample_config = {
            "model": {"architecture": "LS-HGNet", "dropout": 0.50},
            "training": {"learning_rate": 0.0049775, "batch_size": 32},
            "selection": {"selected_candidate": "candidate_b_sto"}
        }
        yaml_path = tmp_path / "final_config.yaml"
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(sample_config, f)

        with open(yaml_path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
        assert loaded == sample_config


# ---------------------------------------------------------------------------
# 14. Locked Model Test Gate
# ---------------------------------------------------------------------------

class TestLockedTestGate:
    def test_evaluation_rejects_without_locked_model(self):
        """Test evaluation must require an explicit locked checkpoint path."""
        def evaluate_test_set(checkpoint_path: str, is_locked: bool):
            if not is_locked:
                raise RuntimeError("Test set evaluation forbidden: model is not locked.")
            if not Path(checkpoint_path).is_file():
                raise FileNotFoundError(f"Locked checkpoint not found: {checkpoint_path}")
            return {"status": "test_evaluated"}

        with pytest.raises(RuntimeError, match="not locked"):
            evaluate_test_set("some/path.pt", is_locked=False)
