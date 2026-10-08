"""
src/optimization/benchmark.py
==============================
Benchmark functions for verifying STO optimizer implementations.

Used in M5 Step 1 (pre-integration validation) to confirm that both
OriginalStrictSTO and ProjectSpecificStabilizedSTO produce sensible
optimization behavior on well-known synthetic test functions BEFORE
connecting to the LS-HGNet training pipeline.

Functions
---------
sphere(x)     : Sphere function  f(x) = sum(x_i^2)
                Global minimum: f(0,...,0) = 0
                Used to test basic convergence.

rastrigin(x)  : Rastrigin function  f(x) = A*n + sum(x_i^2 - A*cos(2*pi*x_i))
                Global minimum: f(0,...,0) = 0, A = 10
                Used to test exploration (many local minima).

Both functions are adapted to use the SearchSpace internal theta vector
(4D), with a custom SearchSpace defined for [-5.12, 5.12]^4 (standard
Rastrigin/Sphere benchmark bounds) so that the optimizer infrastructure
can be tested end-to-end without touching the real training pipeline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from src.optimization.sto import SearchSpace


# ---------------------------------------------------------------------------
# Standard benchmark search space: [-5.12, 5.12]^4 (linear, no decode)
# ---------------------------------------------------------------------------
def benchmark_search_space(dim: int = 4, bound: float = 5.12) -> SearchSpace:
    """
    Return a SearchSpace with symmetric bounds [-bound, +bound]^dim.
    decode() returns theta directly (no log-space remapping).
    Used only for benchmark validation; NOT used in real STO training.
    """
    class _BenchmarkSearchSpace(SearchSpace):
        def decode(self, theta: np.ndarray) -> dict:
            """For benchmarks, decode just returns the raw theta values."""
            theta = self.clip(theta)
            return {f"x{i}": float(theta[i]) for i in range(len(theta))}

    lb = np.full(dim, -bound)
    ub = np.full(dim, +bound)
    ss = _BenchmarkSearchSpace(
        lb=lb,
        ub=ub,
        dim=dim,
        batch_size_values=(16, 32, 64),  # unused for benchmarks
    )
    return ss


# ---------------------------------------------------------------------------
# Sphere function
# ---------------------------------------------------------------------------
def sphere(x: np.ndarray) -> float:
    """
    Sphere function.  f(x) = sum(x_i^2).

    Properties
    ----------
    - Unimodal (single global minimum)
    - Global minimum: f(0,...,0) = 0
    - Smooth and convex: simplest convergence test

    Parameters
    ----------
    x : np.ndarray, any shape
        Position vector (internal theta or real coordinates).

    Returns
    -------
    float: fitness value (lower = better)
    """
    return float(np.sum(x ** 2))


# ---------------------------------------------------------------------------
# Rastrigin function
# ---------------------------------------------------------------------------
def rastrigin(x: np.ndarray, A: float = 10.0) -> float:
    """
    Rastrigin function.  f(x) = A*n + sum(x_i^2 - A*cos(2*pi*x_i)).

    Properties
    ----------
    - Highly multimodal (many local minima)
    - Global minimum: f(0,...,0) = 0
    - Tests exploration capability

    Parameters
    ----------
    x : np.ndarray, any shape
        Position vector.
    A : float, default 10.0
        Rastrigin amplitude parameter.

    Returns
    -------
    float: fitness value (lower = better)
    """
    n = len(x)
    return float(A * n + np.sum(x ** 2 - A * np.cos(2.0 * math.pi * x)))


# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------
def validate_optimizer_on_benchmarks(
    variant: str = "stabilized",
    num_agents: int = 10,
    num_iterations: int = 50,
    seed: int = 42,
    dim: int = 4,
    bound: float = 5.12,
    tol_sphere: float = 0.1,
    verbose: bool = True,
) -> dict:
    """
    Run both benchmark functions with a given STO variant and return results.

    Parameters
    ----------
    variant        : "original_strict" | "stabilized"
    num_agents     : Population size
    num_iterations : Number of iterations
    seed           : Random seed
    dim            : Problem dimensionality (default 4, matches AgriAI)
    bound          : Symmetric search bound [-bound, +bound]^dim
    tol_sphere     : Acceptance threshold for Sphere best_fitness
    verbose        : Print results

    Returns
    -------
    dict with keys: sphere_result, rastrigin_result, sphere_passed
    """
    from src.optimization.sto import create_optimizer

    ss = benchmark_search_space(dim=dim, bound=bound)

    # Sphere
    opt_sphere = create_optimizer(
        variant=variant,
        num_agents=num_agents,
        num_iterations=num_iterations,
        search_space=ss,
        seed=seed,
        verbose=False,
    )
    sphere_result = opt_sphere.optimize(sphere)

    # Rastrigin
    opt_rastrigin = create_optimizer(
        variant=variant,
        num_agents=num_agents,
        num_iterations=num_iterations,
        search_space=ss,
        seed=seed,
        verbose=False,
    )
    rastrigin_result = opt_rastrigin.optimize(rastrigin)

    sphere_passed = sphere_result.best_fitness < tol_sphere

    if verbose:
        print(f"\n{'='*60}")
        print(f"Benchmark Validation: variant='{variant}'")
        print(f"{'='*60}")
        print(f"Sphere:")
        print(f"  Best fitness : {sphere_result.best_fitness:.6f}")
        print(f"  Best position: {sphere_result.best_position}")
        print(f"  Evaluations  : {sphere_result.num_evaluations}")
        print(f"  PASSED       : {sphere_passed} (tol={tol_sphere})")
        print(f"Rastrigin:")
        print(f"  Best fitness : {rastrigin_result.best_fitness:.6f}")
        print(f"  Best position: {rastrigin_result.best_position}")
        print(f"  Evaluations  : {rastrigin_result.num_evaluations}")
        print(f"{'='*60}\n")

    return {
        "sphere_result":    sphere_result,
        "rastrigin_result": rastrigin_result,
        "sphere_passed":    sphere_passed,
        "variant":          variant,
    }
