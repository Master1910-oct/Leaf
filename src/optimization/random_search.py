"""
src/optimization/random_search.py
===================================
AgriAI M6 - Random Search Control Optimizer.

Random Search provides a non-metaheuristic baseline for comparison
with STO under an identical evaluation budget.

Design principles:
  - Same search space as STO (SearchSpace from sto.py)
  - Same evaluation budget (num_evaluations total)
  - Same FitnessEvaluator interface as STO
  - Independent random seed from STO
  - Genuinely random sampling (no structured population)
  - Best-so-far tracking identical to STO

Reference: AgriAI M0 research_spec.md Section 13.6 (random_search control)

IMPORTANT:
  - The test set is NEVER accessed.
  - Fitness = 1 - validation_macro_F1.
  - Random Search uses seed=12345 by default (different from STO seed=42).
  - Equal budget: num_evaluations must equal STO's (N * T).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

from src.optimization.sto import SearchSpace, OptimizationResult

logger = logging.getLogger(__name__)

FitnessFunc = Callable[[np.ndarray], float]


# ---------------------------------------------------------------------------
# Random Search Result
# ---------------------------------------------------------------------------
@dataclass
class RandomSearchResult:
    """Container for Random Search results."""
    best_position:   np.ndarray
    best_fitness:    float
    best_params:     dict
    convergence:     List[float]     # Best fitness after each evaluation
    all_positions:   List[np.ndarray]
    all_fitnesses:   List[float]
    num_evaluations: int
    runtime_s:       float
    seed:            int


# ---------------------------------------------------------------------------
# Random Search
# ---------------------------------------------------------------------------
class RandomSearch:
    """
    Random Search hyperparameter optimizer.

    Independently samples num_evaluations random configurations from the
    search space and evaluates each.  No structured exploration or
    exploitation.

    This class uses the SAME FitnessEvaluator interface as the STO
    optimizers to guarantee identical evaluation conditions.

    Parameters
    ----------
    num_evaluations : Total number of random configurations to evaluate.
                      MUST equal STO's (num_agents * num_iterations) for
                      fair equal-budget comparison.
    search_space    : Same SearchSpace used by STO.
    seed            : Random seed.  Use a DIFFERENT seed from STO to
                      prevent accidental correlation (default: 12345).
    verbose         : Log evaluation progress.
    """

    def __init__(
        self,
        num_evaluations: int,
        search_space:    SearchSpace,
        seed:            int = 12345,
        verbose:         bool = True,
    ) -> None:
        if num_evaluations < 1:
            raise ValueError("num_evaluations must be >= 1")
        self.num_evaluations = num_evaluations
        self.search_space    = search_space
        self.seed            = seed
        self.verbose         = verbose
        self._rng            = np.random.default_rng(seed)

    def optimize(self, fitness_fn: FitnessFunc) -> RandomSearchResult:
        """
        Run random search.

        Parameters
        ----------
        fitness_fn : callable
            Maps theta (np.ndarray shape (4,)) to fitness float.
            Lower = better.  fitness = 1 - val_macro_F1.

        Returns
        -------
        RandomSearchResult
        """
        ss = self.search_space
        N  = self.num_evaluations

        t_start = time.perf_counter()

        all_pos:    List[np.ndarray] = []
        all_fit:    List[float]      = []
        convergence: List[float]     = []
        best_fitness = float("inf")
        best_pos     = None

        if self.verbose:
            logger.info(
                "[RandomSearch] Starting: %d independent evaluations (seed=%d)",
                N, self.seed,
            )

        for i in range(N):
            # Sample uniformly random configuration
            theta = ss.random_position(self._rng)

            # Evaluate
            f = fitness_fn(theta)

            all_pos.append(theta.copy())
            all_fit.append(f)

            # Update best
            if f < best_fitness:
                best_fitness = f
                best_pos     = theta.copy()

            convergence.append(best_fitness)

            if self.verbose:
                logger.info(
                    "[RandomSearch] Eval %2d/%d  fitness=%.6f  best_so_far=%.6f  params=%s",
                    i + 1, N, f, best_fitness, ss.decode(theta),
                )

        runtime = time.perf_counter() - t_start

        result = RandomSearchResult(
            best_position   = best_pos if best_pos is not None else all_pos[0],
            best_fitness    = best_fitness,
            best_params     = ss.decode(best_pos if best_pos is not None else all_pos[0]),
            convergence     = convergence,
            all_positions   = all_pos,
            all_fitnesses   = all_fit,
            num_evaluations = N,
            runtime_s       = runtime,
            seed            = self.seed,
        )

        if self.verbose:
            logger.info(
                "[RandomSearch] Done. best_fitness=%.6f  val_macro_f1=%.4f%%  runtime=%.1fs",
                best_fitness, (1.0 - best_fitness) * 100, runtime,
            )
            logger.info(
                "[RandomSearch] Best params: %s", result.best_params,
            )

        return result
