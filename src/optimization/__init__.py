"""
src/optimization/__init__.py
AgriAI M5/M6 - STOA Hyperparameter Optimization Package
"""

from src.optimization.sto import (
    SearchSpace,
    OptimizationResult,
    OriginalStrictSTO,
    ProjectSpecificStabilizedSTO,
    create_optimizer,
    FitnessFunc,
)
from src.optimization.benchmark import (
    benchmark_search_space,
    sphere,
    rastrigin,
    validate_optimizer_on_benchmarks,
)
from src.optimization.fitness import (
    FitnessEvaluator,
    CandidateRecord,
    record_to_dict,
)
from src.optimization.random_search import (
    RandomSearch,
    RandomSearchResult,
)

__all__ = [
    # STO
    "SearchSpace",
    "OptimizationResult",
    "OriginalStrictSTO",
    "ProjectSpecificStabilizedSTO",
    "create_optimizer",
    "FitnessFunc",
    # Benchmarks
    "benchmark_search_space",
    "sphere",
    "rastrigin",
    "validate_optimizer_on_benchmarks",
    # Fitness
    "FitnessEvaluator",
    "CandidateRecord",
    "record_to_dict",
    # Random Search
    "RandomSearch",
    "RandomSearchResult",
]
