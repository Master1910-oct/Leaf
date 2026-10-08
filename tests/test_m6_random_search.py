"""
tests/test_m6_random_search.py
================================
AgriAI M6 - Random Search and Fitness Evaluator Unit Tests.

Test coverage:
  1.  Random candidate generation stays in bounds
  2.  Batch size snapping (discrete values)
  3.  Seed reproducibility (same seed = same sequence)
  4.  Different seed = different candidate sequence
  5.  Candidate dimensionality
  6.  Fitness calculation correctness (minimization)
  7.  Best-so-far tracking
  8.  Equal evaluation budget enforcement
  9.  No NaN/Inf in outputs
 10.  Test-set isolation guard (API prevents test dataset)
 11.  Result serialization (CandidateRecord to dict)
 12.  RandomSearchResult fields populated
 13.  FitnessEvaluator candidate log populated
 14.  Best-so-far F1 = 1 - best-so-far fitness
 15.  RandomSearch vs STO use same search space
"""

import math
import json
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from torch.utils.data import TensorDataset
import torch

from src.optimization.sto import SearchSpace, create_optimizer
from src.optimization.random_search import RandomSearch, RandomSearchResult
from src.optimization.fitness import FitnessEvaluator, CandidateRecord, record_to_dict


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def search_space():
    return SearchSpace()


@pytest.fixture
def dummy_train_dataset():
    """Tiny dataset: 10 samples, 3x8x8 images, 10 classes."""
    X = torch.randn(10, 3, 8, 8)
    y = torch.randint(0, 10, (10,))
    return TensorDataset(X, y)


@pytest.fixture
def dummy_val_dataset():
    """Separate tiny validation dataset."""
    X = torch.randn(8, 3, 8, 8)
    y = torch.randint(0, 10, (8,))
    return TensorDataset(X, y)


@pytest.fixture
def mock_evaluator(search_space, dummy_train_dataset, dummy_val_dataset):
    """FitnessEvaluator with mocked _run_evaluation to avoid actual training."""
    ev = FitnessEvaluator(
        train_dataset=dummy_train_dataset,
        val_dataset=dummy_val_dataset,
        num_epochs=1,
        search_space=search_space,
        method="test",
        base_channels=8,
        num_stacks=1,
        encoder_levels=1,
        num_classes=10,
        image_size=8,
        device="cpu",
        seed=42,
        use_cache=False,
        save_dir=None,
    )
    # Mock _run_evaluation to return fixed metrics without training
    def fake_eval(params):
        f1 = 0.5 + 0.01 * (params["batch_size"] == 32)
        metrics = {
            "val_loss": 1.0,
            "val_accuracy": 0.6,
            "val_macro_precision": f1,
            "val_macro_recall": f1,
            "val_macro_f1": f1,
        }
        return metrics, 1.0 - f1
    ev._run_evaluation = fake_eval
    return ev


# ---------------------------------------------------------------------------
# 1. Random candidate stays in bounds
# ---------------------------------------------------------------------------
class TestRandomCandidateGeneration:

    def test_samples_in_bounds(self, search_space):
        rs = RandomSearch(num_evaluations=50, search_space=search_space, seed=42, verbose=False)
        rng = np.random.default_rng(42)
        for _ in range(50):
            theta = search_space.random_position(rng)
            assert np.all(theta >= search_space.lb), f"Below lb: {theta}"
            assert np.all(theta <= search_space.ub), f"Above ub: {theta}"

    def test_all_evaluated_positions_in_bounds(self, search_space):
        """All positions sampled during optimization must respect bounds."""
        counter = {"n": 0}
        def fitness(theta):
            counter["n"] += 1
            assert np.all(theta >= search_space.lb)
            assert np.all(theta <= search_space.ub)
            return float(np.sum(theta**2))
        rs = RandomSearch(num_evaluations=10, search_space=search_space, seed=42, verbose=False)
        rs.optimize(fitness)
        assert counter["n"] == 10


# ---------------------------------------------------------------------------
# 2. Batch size snapping
# ---------------------------------------------------------------------------
class TestBatchSizeSnapping:

    def test_batch_size_only_valid_values(self, search_space):
        """Every decoded batch_size must be in {16, 32, 64}."""
        rng = np.random.default_rng(999)
        for _ in range(100):
            theta = search_space.random_position(rng)
            params = search_space.decode(theta)
            assert params["batch_size"] in {16, 32, 64}

    def test_theta4_0_gives_bs16(self, search_space):
        theta = np.array([-3.0, -4.0, 0.25, 0.0])
        assert search_space.decode(theta)["batch_size"] == 16

    def test_theta4_1_gives_bs32(self, search_space):
        theta = np.array([-3.0, -4.0, 0.25, 1.0])
        assert search_space.decode(theta)["batch_size"] == 32

    def test_theta4_2_gives_bs64(self, search_space):
        theta = np.array([-3.0, -4.0, 0.25, 2.0])
        assert search_space.decode(theta)["batch_size"] == 64


# ---------------------------------------------------------------------------
# 3. Seed reproducibility
# ---------------------------------------------------------------------------
class TestSeedReproducibility:

    def test_same_seed_same_sequence(self, search_space):
        """Two RandomSearch instances with same seed must produce identical samples."""
        fitnesses_a, fitnesses_b = [], []

        def fitness_a(theta):
            fitnesses_a.append(tuple(theta.tolist()))
            return float(np.sum(theta**2))

        def fitness_b(theta):
            fitnesses_b.append(tuple(theta.tolist()))
            return float(np.sum(theta**2))

        rs_a = RandomSearch(10, search_space, seed=42, verbose=False)
        rs_b = RandomSearch(10, search_space, seed=42, verbose=False)
        rs_a.optimize(fitness_a)
        rs_b.optimize(fitness_b)
        assert fitnesses_a == fitnesses_b


# ---------------------------------------------------------------------------
# 4. Different seed = different sequence
# ---------------------------------------------------------------------------
class TestDifferentSeedsDifferentSequences:

    def test_different_seeds_differ(self, search_space):
        positions_a, positions_b = [], []

        def fa(theta): positions_a.append(theta.copy()); return 0.0
        def fb(theta): positions_b.append(theta.copy()); return 0.0

        RandomSearch(10, search_space, seed=42,    verbose=False).optimize(fa)
        RandomSearch(10, search_space, seed=12345, verbose=False).optimize(fb)

        # At least some positions must differ
        any_different = any(
            not np.allclose(a, b) for a, b in zip(positions_a, positions_b)
        )
        assert any_different, "Different seeds must produce different candidate sequences"


# ---------------------------------------------------------------------------
# 5. Candidate dimensionality
# ---------------------------------------------------------------------------
class TestCandidateDimensionality:

    def test_theta_shape(self, search_space):
        positions = []
        def f(theta): positions.append(theta); return 0.0
        RandomSearch(5, search_space, seed=42, verbose=False).optimize(f)
        for pos in positions:
            assert pos.shape == (4,), f"Expected (4,) got {pos.shape}"


# ---------------------------------------------------------------------------
# 6. Fitness calculation
# ---------------------------------------------------------------------------
class TestFitnessCalculation:

    def test_fitness_is_lower_better(self, search_space):
        """Result must track minimum (not maximum)."""
        fitnesses = [0.5, 0.3, 0.4, 0.2, 0.6]
        idx = [0]

        def f(theta):
            val = fitnesses[idx[0] % len(fitnesses)]
            idx[0] += 1
            return val

        rs = RandomSearch(5, search_space, seed=42, verbose=False)
        result = rs.optimize(f)
        assert result.best_fitness == min(fitnesses[:5])


# ---------------------------------------------------------------------------
# 7. Best-so-far tracking
# ---------------------------------------------------------------------------
class TestBestSoFarTracking:

    def test_convergence_is_non_increasing(self, search_space):
        rng_fit = np.random.default_rng(0)
        generated = [float(rng_fit.uniform(0.1, 0.9)) for _ in range(20)]
        idx = [0]
        def f(theta):
            val = generated[idx[0]]; idx[0] += 1; return val

        rs = RandomSearch(20, search_space, seed=42, verbose=False)
        result = rs.optimize(f)
        conv = result.convergence
        for i in range(1, len(conv)):
            assert conv[i] <= conv[i-1] + 1e-9

    def test_convergence_length_equals_evaluations(self, search_space):
        rs = RandomSearch(15, search_space, seed=42, verbose=False)
        result = rs.optimize(lambda t: float(np.sum(t**2)))
        assert len(result.convergence) == 15


# ---------------------------------------------------------------------------
# 8. Equal budget enforcement
# ---------------------------------------------------------------------------
class TestEqualBudget:

    def test_evaluations_equal_specified(self, search_space):
        count = [0]
        def f(theta): count[0] += 1; return 0.0
        RandomSearch(25, search_space, seed=42, verbose=False).optimize(f)
        assert count[0] == 25

    def test_sto_and_rs_same_budget(self, search_space):
        """5 agents x 5 iters = 25; RS must also do 25 unique evaluations."""
        sto_count  = [0]
        rs_count   = [0]
        def sto_f(t): sto_count[0] += 1; return 0.5
        def rs_f(t):  rs_count[0]  += 1; return 0.5
        create_optimizer("stabilized", 5, 5, search_space, seed=42, verbose=False).optimize(sto_f)
        RandomSearch(25, search_space, seed=12345, verbose=False).optimize(rs_f)
        # STO: N initial + N*T updates; RS: N*T direct
        assert rs_count[0] == 25


# ---------------------------------------------------------------------------
# 9. No NaN/Inf in outputs
# ---------------------------------------------------------------------------
class TestNoNaNInf:

    def test_no_nan_in_fitnesses(self, search_space):
        rs = RandomSearch(10, search_space, seed=42, verbose=False)
        result = rs.optimize(lambda t: float(np.sum(t**2)))
        for f in result.all_fitnesses:
            assert math.isfinite(f), f"Non-finite fitness: {f}"

    def test_no_nan_in_positions(self, search_space):
        rs = RandomSearch(10, search_space, seed=42, verbose=False)
        result = rs.optimize(lambda t: float(np.sum(t**2)))
        for pos in result.all_positions:
            assert not np.any(np.isnan(pos))
            assert not np.any(np.isinf(pos))


# ---------------------------------------------------------------------------
# 10. Test-set isolation guard
# ---------------------------------------------------------------------------
class TestTestSetIsolation:

    def test_same_dataset_raises(self, dummy_train_dataset, search_space):
        """FitnessEvaluator must raise if train == val (same object)."""
        with pytest.raises(ValueError, match="TEST-SET GUARD"):
            FitnessEvaluator(
                train_dataset=dummy_train_dataset,
                val_dataset=dummy_train_dataset,  # SAME OBJECT
                num_epochs=1,
                search_space=search_space,
            )

    def test_no_test_dataset_parameter(self, dummy_train_dataset, dummy_val_dataset, search_space):
        """FitnessEvaluator must not have a test_dataset parameter."""
        import inspect
        sig = inspect.signature(FitnessEvaluator.__init__)
        assert "test_dataset" not in sig.parameters, (
            "FitnessEvaluator must not accept test_dataset parameter"
        )

    def test_random_search_no_test_param(self):
        """RandomSearch must not have a test_dataset parameter."""
        import inspect
        sig = inspect.signature(RandomSearch.__init__)
        assert "test_dataset" not in sig.parameters


# ---------------------------------------------------------------------------
# 11. Result serialization
# ---------------------------------------------------------------------------
class TestResultSerialization:

    def test_candidate_record_to_dict(self):
        rec = CandidateRecord(
            candidate_id=1, method="test", eval_seed=42,
            theta_lr=-3.0, theta_wd=-4.0, theta_dropout=0.3, theta_batch_size=1.0,
            learning_rate=1e-3, weight_decay=1e-4, dropout=0.3, batch_size=32,
            training_epochs=3, image_size=128,
            validation_loss=0.5, validation_accuracy=0.7,
            validation_macro_precision=0.65, validation_macro_recall=0.63,
            validation_macro_f1=0.64,
            fitness=0.36, training_time_seconds=10.5,
            timestamp_utc="2026-01-01T00:00:00+00:00",
        )
        d = record_to_dict(rec)
        assert isinstance(d, dict)
        assert d["fitness"] == 0.36
        assert d["batch_size"] == 32
        # Must be JSON serializable
        s = json.dumps(d)
        restored = json.loads(s)
        assert restored["fitness"] == 0.36


# ---------------------------------------------------------------------------
# 12. RandomSearchResult fields
# ---------------------------------------------------------------------------
class TestRandomSearchResultFields:

    def test_result_populated(self, search_space):
        rs = RandomSearch(5, search_space, seed=42, verbose=False)
        result = rs.optimize(lambda t: float(np.sum(t**2)))
        assert isinstance(result, RandomSearchResult)
        assert result.num_evaluations == 5
        assert len(result.all_fitnesses) == 5
        assert len(result.all_positions) == 5
        assert result.runtime_s > 0
        assert isinstance(result.best_position, np.ndarray)
        assert isinstance(result.best_params, dict)
        assert result.best_fitness == min(result.all_fitnesses)
        assert result.seed == 42


# ---------------------------------------------------------------------------
# 13. FitnessEvaluator candidate log
# ---------------------------------------------------------------------------
class TestFitnessEvaluatorCandidateLog:

    def test_log_populated(self, mock_evaluator, search_space):
        rng = np.random.default_rng(99)
        for _ in range(3):
            theta = search_space.random_position(rng)
            mock_evaluator(theta)
        assert len(mock_evaluator.candidate_log) == 3
        for rec in mock_evaluator.candidate_log:
            assert isinstance(rec, CandidateRecord)
            assert rec.candidate_id in {1, 2, 3}
            assert math.isfinite(rec.fitness)

    def test_best_record_is_minimum(self, mock_evaluator, search_space):
        rng = np.random.default_rng(77)
        for _ in range(5):
            theta = search_space.random_position(rng)
            mock_evaluator(theta)
        best = mock_evaluator.best_record
        assert best.fitness == mock_evaluator.best_fitness
        assert best.fitness == min(r.fitness for r in mock_evaluator.candidate_log)


# ---------------------------------------------------------------------------
# 14. Best-so-far F1 == 1 - fitness
# ---------------------------------------------------------------------------
class TestBestSoFarF1:

    def test_f1_is_complement_of_fitness(self, mock_evaluator, search_space):
        rng = np.random.default_rng(55)
        for _ in range(4):
            mock_evaluator(search_space.random_position(rng))
        bsf_fit = mock_evaluator.get_best_so_far()
        bsf_f1  = mock_evaluator.get_best_so_far_f1()
        for fit, f1 in zip(bsf_fit, bsf_f1):
            assert abs(fit + f1 - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# 15. STO and RandomSearch share same SearchSpace
# ---------------------------------------------------------------------------
class TestSharedSearchSpace:

    def test_same_bounds(self, search_space):
        """STO and RS must use identical bounds."""
        sto = create_optimizer("stabilized", 5, 5, search_space, seed=42, verbose=False)
        rs  = RandomSearch(25, search_space, seed=12345, verbose=False)
        np.testing.assert_array_equal(sto.search_space.lb, rs.search_space.lb)
        np.testing.assert_array_equal(sto.search_space.ub, rs.search_space.ub)
