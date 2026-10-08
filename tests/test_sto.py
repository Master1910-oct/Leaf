"""
tests/test_sto.py
==================
Unit tests for AgriAI M5 - STOA Hyperparameter Optimizer.

Test coverage:
  1. SearchSpace.decode() correctness (all dimensions)
  2. SearchSpace.clip() boundary enforcement
  3. SearchSpace.random_position() stays in bounds
  4. OriginalStrictSTO: smoke test on synthetic fitness function
  5. ProjectSpecificStabilizedSTO: smoke test on synthetic fitness function
  6. Sphere benchmark: stabilized variant converges within tolerance
  7. Rastrigin benchmark: stabilized variant evaluates without errors
  8. Benchmark: original_strict evaluates without crash (pathological behavior tolerated)
  9. OptimizationResult fields are correctly populated
 10. create_optimizer() factory function dispatches correctly
 11. Dimension extension: Omega4 == [Om1, Om2, Om3, Om1]
 12. Damping factor: r at k=2pi is exp(0.1*2pi) << 535 for stabilized
"""

import math
import pytest
import numpy as np

from src.optimization.sto import (
    SearchSpace,
    OptimizationResult,
    OriginalStrictSTO,
    ProjectSpecificStabilizedSTO,
    create_optimizer,
)
from src.optimization.benchmark import (
    benchmark_search_space,
    sphere,
    rastrigin,
    validate_optimizer_on_benchmarks,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def default_search_space():
    return SearchSpace()


@pytest.fixture
def benchmark_ss():
    return benchmark_search_space(dim=4, bound=5.12)


@pytest.fixture
def stabilized_optimizer(default_search_space):
    return ProjectSpecificStabilizedSTO(
        num_agents=5,
        num_iterations=5,
        search_space=default_search_space,
        seed=42,
        verbose=False,
    )


@pytest.fixture
def original_optimizer(default_search_space):
    return OriginalStrictSTO(
        num_agents=5,
        num_iterations=5,
        search_space=default_search_space,
        seed=42,
        verbose=False,
    )


# ---------------------------------------------------------------------------
# 1. SearchSpace.decode() correctness
# ---------------------------------------------------------------------------

class TestSearchSpaceDecode:

    def test_decode_lr_log_scale(self, default_search_space):
        """theta_1 = -3.0 -> lr = 10^(-3) = 0.001"""
        theta = np.array([-3.0, -4.0, 0.25, 1.0])
        params = default_search_space.decode(theta)
        assert abs(params["learning_rate"] - 1e-3) < 1e-10

    def test_decode_wd_log_scale(self, default_search_space):
        """theta_2 = -4.0 -> wd = 10^(-4) = 0.0001"""
        theta = np.array([-3.0, -4.0, 0.25, 1.0])
        params = default_search_space.decode(theta)
        assert abs(params["weight_decay"] - 1e-4) < 1e-12

    def test_decode_dropout_linear(self, default_search_space):
        """theta_3 = 0.3 -> dropout = 0.3"""
        theta = np.array([-3.0, -4.0, 0.3, 1.0])
        params = default_search_space.decode(theta)
        assert abs(params["dropout"] - 0.3) < 1e-6

    def test_decode_batch_size_16(self, default_search_space):
        """theta_4 = 0.0 -> round(0) = 0 -> batch_size = 16"""
        theta = np.array([-3.0, -4.0, 0.25, 0.0])
        params = default_search_space.decode(theta)
        assert params["batch_size"] == 16

    def test_decode_batch_size_32(self, default_search_space):
        """theta_4 = 1.0 -> round(1) = 1 -> batch_size = 32"""
        theta = np.array([-3.0, -4.0, 0.25, 1.0])
        params = default_search_space.decode(theta)
        assert params["batch_size"] == 32

    def test_decode_batch_size_64(self, default_search_space):
        """theta_4 = 2.0 -> round(2) = 2 -> batch_size = 64"""
        theta = np.array([-3.0, -4.0, 0.25, 2.0])
        params = default_search_space.decode(theta)
        assert params["batch_size"] == 64

    def test_decode_returns_all_keys(self, default_search_space):
        theta = np.array([-3.0, -4.0, 0.25, 1.0])
        params = default_search_space.decode(theta)
        assert set(params.keys()) == {"learning_rate", "weight_decay", "dropout", "batch_size"}


# ---------------------------------------------------------------------------
# 2. SearchSpace.clip() boundary enforcement
# ---------------------------------------------------------------------------

class TestSearchSpaceClip:

    def test_clip_above_ub(self, default_search_space):
        ss = default_search_space
        theta = np.array([0.0, 0.0, 1.0, 5.0])  # all out of range
        clipped = ss.clip(theta)
        assert np.all(clipped <= ss.ub)

    def test_clip_below_lb(self, default_search_space):
        ss = default_search_space
        theta = np.array([-10.0, -10.0, -1.0, -1.0])  # all out of range
        clipped = ss.clip(theta)
        assert np.all(clipped >= ss.lb)

    def test_clip_in_range_unchanged(self, default_search_space):
        ss = default_search_space
        theta = np.array([-3.0, -4.0, 0.25, 1.0])
        clipped = ss.clip(theta)
        np.testing.assert_array_almost_equal(clipped, theta)


# ---------------------------------------------------------------------------
# 3. SearchSpace.random_position() stays in bounds
# ---------------------------------------------------------------------------

class TestSearchSpaceRandom:

    def test_random_position_in_bounds(self, default_search_space):
        ss = default_search_space
        rng = np.random.default_rng(42)
        for _ in range(100):
            pos = ss.random_position(rng)
            assert np.all(pos >= ss.lb), f"Below lb: {pos}"
            assert np.all(pos <= ss.ub), f"Above ub: {pos}"

    def test_random_position_dim(self, default_search_space):
        ss = default_search_space
        rng = np.random.default_rng(42)
        pos = ss.random_position(rng)
        assert pos.shape == (4,)


# ---------------------------------------------------------------------------
# 4 & 5. Smoke tests: optimizers run without crashing
# ---------------------------------------------------------------------------

class TestOptimizerSmoke:

    @staticmethod
    def simple_fitness(theta: np.ndarray) -> float:
        """Trivial fitness: sum of squared theta values."""
        return float(np.sum(theta ** 2))

    def test_stabilized_smoke(self, stabilized_optimizer):
        result = stabilized_optimizer.optimize(self.simple_fitness)
        assert isinstance(result, OptimizationResult)
        assert result.variant == "stabilized"
        assert math.isfinite(result.best_fitness)

    def test_original_smoke(self, original_optimizer):
        """Original variant must run without crashing (pathologies expected)."""
        result = original_optimizer.optimize(self.simple_fitness)
        assert isinstance(result, OptimizationResult)
        assert result.variant == "original_strict"
        # Fitness may be large due to explosion, but must be finite or inf (not NaN)
        assert not math.isnan(result.best_fitness)


# ---------------------------------------------------------------------------
# 6. Sphere benchmark: stabilized converges within tolerance
# ---------------------------------------------------------------------------

class TestSphereConvergence:

    def test_stabilized_sphere_convergence(self, benchmark_ss):
        """Stabilized STO should find sphere minimum near 0 within 50 iters."""
        opt = ProjectSpecificStabilizedSTO(
            num_agents=10,
            num_iterations=50,
            search_space=benchmark_ss,
            seed=42,
            verbose=False,
        )
        result = opt.optimize(sphere)
        # Should converge substantially below initial random fitness
        # We accept tol=2.0 (conservative, since only 50 iters x 10 agents)
        assert result.best_fitness < 2.0, (
            f"Stabilized STO failed sphere convergence: best_fitness={result.best_fitness:.4f}"
        )

    def test_convergence_history_length(self, benchmark_ss):
        """Convergence history should have exactly num_iterations entries."""
        opt = ProjectSpecificStabilizedSTO(
            num_agents=5,
            num_iterations=10,
            search_space=benchmark_ss,
            seed=42,
            verbose=False,
        )
        result = opt.optimize(sphere)
        assert len(result.convergence) == 10

    def test_convergence_non_increasing(self, benchmark_ss):
        """Best fitness should be monotonically non-increasing."""
        opt = ProjectSpecificStabilizedSTO(
            num_agents=5,
            num_iterations=20,
            search_space=benchmark_ss,
            seed=42,
            verbose=False,
        )
        result = opt.optimize(sphere)
        conv = result.convergence
        for i in range(1, len(conv)):
            assert conv[i] <= conv[i-1] + 1e-9, (
                f"Convergence not monotone at iter {i}: {conv[i-1]} -> {conv[i]}"
            )


# ---------------------------------------------------------------------------
# 7. Rastrigin benchmark: runs without errors
# ---------------------------------------------------------------------------

class TestRastriginBenchmark:

    def test_rastrigin_runs(self, benchmark_ss):
        opt = ProjectSpecificStabilizedSTO(
            num_agents=10,
            num_iterations=30,
            search_space=benchmark_ss,
            seed=42,
            verbose=False,
        )
        result = opt.optimize(rastrigin)
        assert isinstance(result, OptimizationResult)
        assert math.isfinite(result.best_fitness)
        assert result.best_fitness >= 0.0  # Rastrigin is non-negative


# ---------------------------------------------------------------------------
# 8. Original strict: runs without Python crash
# ---------------------------------------------------------------------------

class TestOriginalStrictBenchmark:

    def test_original_sphere_no_crash(self, benchmark_ss):
        """Original variant may produce large values but must not crash Python."""
        opt = OriginalStrictSTO(
            num_agents=5,
            num_iterations=5,
            search_space=benchmark_ss,
            seed=42,
            verbose=False,
        )
        result = opt.optimize(sphere)
        assert isinstance(result, OptimizationResult)
        # fitness may be inf due to clipping saturation but must not be NaN
        assert not math.isnan(result.best_fitness)


# ---------------------------------------------------------------------------
# 9. OptimizationResult fields
# ---------------------------------------------------------------------------

class TestOptimizationResult:

    def test_result_fields_populated(self, benchmark_ss):
        opt = ProjectSpecificStabilizedSTO(
            num_agents=3,
            num_iterations=4,
            search_space=benchmark_ss,
            seed=42,
            verbose=False,
        )
        result = opt.optimize(sphere)

        assert result.variant == "stabilized"
        # initial eval (3) + iterations (4 x 3) = 15 total
        assert result.num_evaluations == 3 + 4 * 3
        assert len(result.all_fitnesses) == result.num_evaluations
        assert len(result.all_positions) == result.num_evaluations
        assert result.runtime_s > 0.0
        assert isinstance(result.best_position, np.ndarray)
        assert isinstance(result.best_params, dict)
        assert result.best_fitness == min(result.all_fitnesses)


# ---------------------------------------------------------------------------
# 10. Factory function
# ---------------------------------------------------------------------------

class TestCreateOptimizer:

    def test_factory_stabilized(self):
        opt = create_optimizer("stabilized", 5, 5)
        assert isinstance(opt, ProjectSpecificStabilizedSTO)
        assert opt.variant_name == "stabilized"

    def test_factory_original(self):
        opt = create_optimizer("original_strict", 5, 5)
        assert isinstance(opt, OriginalStrictSTO)
        assert opt.variant_name == "original_strict"

    def test_factory_invalid_raises(self):
        with pytest.raises(ValueError, match="Unknown STOA variant"):
            create_optimizer("invalid_variant", 5, 5)


# ---------------------------------------------------------------------------
# 11. Dimension extension: Omega4 == [Om1, Om2, Om3, Om1]
# ---------------------------------------------------------------------------

class TestDimensionalExtension:

    def test_cyclic_extension_pattern(self):
        """
        For any k, the stabilized spiral direction should satisfy:
        omega4[0] == omega4[3]  (Om1 is repeated at index 3)
        """
        opt = ProjectSpecificStabilizedSTO(
            num_agents=1, num_iterations=1,
            search_space=SearchSpace(), seed=42, verbose=False,
        )
        rng = np.random.default_rng(42)
        for _ in range(20):
            k = rng.uniform(0.0, 2.0 * math.pi)
            omega = opt._spiral_direction(k, dim=4)
            assert omega.shape == (4,)
            np.testing.assert_almost_equal(
                omega[0], omega[3],
                err_msg=f"omega[0]={omega[0]} != omega[3]={omega[3]} for k={k}"
            )

    def test_omega3_is_unit_vector(self):
        """Normalized Omega3 should have L2 norm close to 1 (within eps)."""
        opt = ProjectSpecificStabilizedSTO(
            num_agents=1, num_iterations=1,
            search_space=SearchSpace(), seed=42, verbose=False,
        )
        rng = np.random.default_rng(42)
        eps = ProjectSpecificStabilizedSTO.EPSILON
        for _ in range(20):
            k = rng.uniform(0.0, 2.0 * math.pi)
            b = ProjectSpecificStabilizedSTO.DAMPING_FACTOR
            r  = math.exp(b * k)
            xp = r * math.cos(k); yp = r * math.sin(k); zp = r * k
            raw = np.array([xp, yp, zp])
            norm = np.linalg.norm(raw) + eps
            omega3 = raw / norm
            assert abs(np.linalg.norm(omega3) - 1.0) < 0.01


# ---------------------------------------------------------------------------
# 12. Damping factor: exp explosion is suppressed
# ---------------------------------------------------------------------------

class TestDampingFactor:

    def test_stabilized_r_is_bounded(self):
        """
        For stabilized: max r at k=2pi = exp(0.1 * 2pi) ~ 1.87 << 535.49
        (535.49 is the original_strict value)
        """
        b   = ProjectSpecificStabilizedSTO.DAMPING_FACTOR
        k   = 2.0 * math.pi
        r_stabilized = math.exp(b * k)
        r_original   = math.exp(k)
        assert r_stabilized < 3.0,   f"Stabilized r too large: {r_stabilized}"
        assert r_original   > 500.0, f"Original r too small: {r_original}"

    def test_original_r_explosion(self):
        """Confirm original produces exp(2pi)~535 (the documented pathology)."""
        k = 2.0 * math.pi
        r_original = math.exp(1.0 * k)
        assert abs(r_original - 535.49) < 1.0
