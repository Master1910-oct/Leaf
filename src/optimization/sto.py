"""
src/optimization/sto.py
=======================
Sooty Tern Optimization Algorithm (STOA) -- AgriAI M5 Implementation

TWO STRICTLY SEPARATED IMPLEMENTATIONS
---------------------------------------
A. OriginalStrictSTO
    Source-derived STOA formulation as reported by Dhiman & Kaur (2019).
    Implements the literal equations from the literature WITHOUT modification,
    stabilization, or engineering corrections.  Used ONLY for ablation study
    EXP_003_original to empirically document known numerical pathologies.

    VERIFICATION STATUS: Literature-Verified, Publisher-Unverified.
    The equations are consistent with independently consulted secondary
    literature (ResearchGate, PubMed/NIH, AIP-affiliated repositories,
    peer-reviewed survey papers).  The original publisher PDF was NOT
    directly inspected.  If a primary-source discrepancy is later found,
    equations must be updated before publication.

B. ProjectSpecificStabilizedSTO
    Project-modified stabilized/additive STOA variant (Option A: Consistent
    Best-Guided Formulation). This is an EXPLICIT PROJECT-SPECIFIC
    ENGINEERING MODIFICATION and MUST NOT be described as the original
    Dhiman & Kaur (2019) STOA.  Modifications:
      1. Damped spiral radius: b = 0.1  (prevents exp(2pi)~535 explosion)
      2. L2-normalized 3D spiral direction vector Omega_3
      3. Cyclic 3D->4D extension: Omega_4 = [Om1, Om2, Om3, Om1]^T
      4. Additive position update (eliminates zero-bias of multiplicative form)
      5. Consistent P_best guidance throughout (no P_rand)
    Reference: AgriAI M0 Research Specification, Section 13.5 (research_spec.md)

Both classes share the same public API so the optimizer variant can be
selected purely by configuration without changing call sites.

IMPORTANT: The test set is NEVER accessed by either optimizer.
           Only the validation macro-F1 is used as the fitness signal.
"""

from __future__ import annotations

import logging
import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
FitnessFunc = Callable[[np.ndarray], float]


# ---------------------------------------------------------------------------
# Search space definition
# ---------------------------------------------------------------------------
@dataclass
class SearchSpace:
    """
    4-dimensional hyperparameter search space.

    Internal (optimisation) representation
    .  theta_1 in [-4, -2]  -> lr = 10^theta_1           (log-space)
    .  theta_2 in [-6, -2]  -> wd = 10^theta_2           (log-space)
    .  theta_3 in [0, 0.5]  -> dropout = theta_3          (linear)
    .  theta_4 in [0, 2]    -> batch_size in {16,32,64}  (discrete via NN)

    The arrays lb and ub define the internal search bounds.
    """
    lb: np.ndarray = field(default_factory=lambda: np.array([-4.0, -6.0, 0.0, 0.0]))
    ub: np.ndarray = field(default_factory=lambda: np.array([-2.0, -2.0, 0.5, 2.0]))
    dim: int = 4
    batch_size_values: Tuple[int, ...] = (16, 32, 64)

    def decode(self, theta: np.ndarray) -> dict:
        """
        Convert an internal position vector theta to the actual hyperparameter dict.

        Parameters
        ----------
        theta : np.ndarray, shape (4,)
            Internal position after boundary clamping.

        Returns
        -------
        dict with keys: learning_rate, weight_decay, dropout, batch_size
        """
        theta = self.clip(theta)
        lr    = float(10.0 ** theta[0])
        wd    = float(10.0 ** theta[1])
        drop  = float(np.clip(theta[2], 0.0, 0.5))
        # nearest-neighbour mapping: [0,2] -> {0,1,2} -> {16,32,64}
        idx   = int(round(float(np.clip(theta[3], 0.0, 2.0))))
        bs    = self.batch_size_values[idx]
        return {
            "learning_rate": lr,
            "weight_decay":  wd,
            "dropout":       drop,
            "batch_size":    bs,
        }

    def clip(self, theta: np.ndarray) -> np.ndarray:
        """Boundary handling: clip to [lb, ub]."""
        return np.clip(theta, self.lb, self.ub)

    def reflect_clip(self, theta: np.ndarray) -> np.ndarray:
        """
        Reflected boundary handler (M7.1 addition -- backward compatible).

        For each dimension: if the new position overshoots a boundary, it is
        reflected back into the feasible region by the overshoot amount.  If
        the reflected value still falls outside (double-overshoot), it is
        clipped.  Dimension 3 (batch_size, discrete) uses plain clip because
        reflection on a 3-point discrete space is ill-defined.

        This replaces the plain clip for dims 0-2 to prevent the boundary
        attraction collapse observed in M6 (Findings 3, 4, 5 of STO_AUDIT.md).
        """
        result = theta.copy()
        for d in range(self.dim - 1):   # dims 0,1,2  (LR, WD, dropout)
            lb_d, ub_d = self.lb[d], self.ub[d]
            v = result[d]
            if v > ub_d:
                v = 2.0 * ub_d - v
                if v < lb_d:
                    v = lb_d
            elif v < lb_d:
                v = 2.0 * lb_d - v
                if v > ub_d:
                    v = ub_d
            result[d] = v
        # dim 3: batch_size -- plain clip
        result[3] = float(np.clip(result[3], self.lb[3], self.ub[3]))
        return result

    def random_position(self, rng: np.random.Generator) -> np.ndarray:
        """Sample a uniformly random position within [lb, ub]."""
        return rng.uniform(self.lb, self.ub)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class OptimizationResult:
    """Container for the result of a single STO run."""
    best_position:   np.ndarray        # Internal theta* vector
    best_fitness:    float             # Minimum fitness found (1 - macro-F1)
    best_params:     dict              # Decoded hyperparameter dict
    convergence:     List[float]       # Best fitness at each iteration end
    all_positions:   List[np.ndarray]  # theta of every evaluated agent
    all_fitnesses:   List[float]       # fitness of every evaluated agent
    num_evaluations: int
    runtime_s:       float
    variant:         str               # "original_strict" | "stabilized"


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------
class _BaseSTOA(ABC):
    """
    Internal ABC shared by both STOA variants.

    Subclasses implement _spiral_direction() and _position_update().
    The outer optimization loop is identical for both variants.
    """

    def __init__(
        self,
        num_agents:     int,
        num_iterations: int,
        search_space:   SearchSpace,
        seed:           Optional[int] = 42,
        verbose:        bool = True,
    ) -> None:
        self.num_agents     = num_agents
        self.num_iterations = num_iterations
        self.search_space   = search_space
        self.seed           = seed
        self.verbose        = verbose
        self._rng           = np.random.default_rng(seed)

    @property
    @abstractmethod
    def variant_name(self) -> str:
        ...

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def optimize(
        self,
        fitness_fn:  FitnessFunc,
        boundary_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> OptimizationResult:
        """
        Run the STOA optimisation loop.

        Parameters
        ----------
        fitness_fn : callable
            Maps an internal theta vector (np.ndarray, shape (4,)) to a scalar
            fitness value.  Lower = better.
            Fitness = 1 - validation_macro_F1.
        boundary_fn : callable, optional
            Maps a theta vector to a feasible theta vector (boundary handling).
            Defaults to ``search_space.clip`` when None.
            Pass ``search_space.reflect_clip`` for M7.1 reflected boundaries.

        Returns
        -------
        OptimizationResult
        """
        ss = self.search_space
        N  = self.num_agents
        T  = self.num_iterations

        t_start = time.perf_counter()

        # -- Initialisation -----------------------------------------------
        population = np.array([ss.random_position(self._rng) for _ in range(N)])
        fitnesses  = np.full(N, np.inf)
        all_pos: List[np.ndarray] = []
        all_fit: List[float]      = []
        convergence: List[float]  = []

        if self.verbose:
            logger.info(
                "[%s] Starting: %d agents x %d iters = %d evaluations",
                self.variant_name, N, T, N * T,
            )

        # -- Evaluate initial population -----------------------------------
        for i in range(N):
            f = fitness_fn(population[i])
            fitnesses[i] = f
            all_pos.append(population[i].copy())
            all_fit.append(f)

        best_idx     = int(np.argmin(fitnesses))
        best_pos     = population[best_idx].copy()
        best_fitness = float(fitnesses[best_idx])

        if self.verbose:
            logger.info(
                "[%s] Initial best fitness=%.6f  params=%s",
                self.variant_name, best_fitness, ss.decode(best_pos),
            )

        # -- Main loop -----------------------------------------------------
        for t in range(1, T + 1):
            S_A = 2.0 * (1.0 - t / T)  # Linearly decaying collision param

            for i in range(N):
                r_vec = self._rng.uniform(0.0, 1.0, ss.dim)

                # Exploration displacement D_s(t)
                C_sa = S_A * population[i]
                M_s  = 0.5 * r_vec * (best_pos - population[i])
                D_s  = C_sa + M_s

                # Spiral direction (variant-specific)
                k     = self._rng.uniform(0.0, 2.0 * math.pi)
                omega = self._spiral_direction(k, ss.dim)

                # Position update (variant-specific)
                new_pos = self._position_update(population[i], best_pos, D_s, omega)

                # Boundary handling (caller-supplied or default clip)
                _bfn = boundary_fn if boundary_fn is not None else ss.clip
                new_pos = _bfn(new_pos)

                # Evaluate
                f = fitness_fn(new_pos)
                population[i] = new_pos
                fitnesses[i]  = f
                all_pos.append(new_pos.copy())
                all_fit.append(f)

                # Update global best
                if f < best_fitness:
                    best_fitness = f
                    best_pos     = new_pos.copy()

            convergence.append(best_fitness)

            if self.verbose:
                logger.info(
                    "[%s] Iter %2d/%d  best_fitness=%.6f  params=%s",
                    self.variant_name, t, T, best_fitness, ss.decode(best_pos),
                )

        runtime = time.perf_counter() - t_start

        result = OptimizationResult(
            best_position   = best_pos,
            best_fitness    = best_fitness,
            best_params     = ss.decode(best_pos),
            convergence     = convergence,
            all_positions   = all_pos,
            all_fitnesses   = all_fit,
            num_evaluations = len(all_fit),
            runtime_s       = runtime,
            variant         = self.variant_name,
        )

        if self.verbose:
            logger.info(
                "[%s] Done. best_fitness=%.6f  evals=%d  runtime=%.1fs",
                self.variant_name, best_fitness, len(all_fit), runtime,
            )
            logger.info("[%s] Best params: %s", self.variant_name, result.best_params)

        return result

    # ------------------------------------------------------------------
    # Variant-specific hooks
    # ------------------------------------------------------------------
    @abstractmethod
    def _spiral_direction(self, k: float, dim: int) -> np.ndarray:
        """Return spiral perturbation direction for current step."""
        ...

    @abstractmethod
    def _position_update(
        self,
        pos:      np.ndarray,
        best_pos: np.ndarray,
        D_s:      np.ndarray,
        omega:    np.ndarray,
    ) -> np.ndarray:
        """Compute new agent position."""
        ...


# ===========================================================================
# A. ORIGINAL / STRICT STOA
# ===========================================================================
class OriginalStrictSTO(_BaseSTOA):
    """
    Original STOA - Source-Derived Formulation (Dhiman & Kaur, 2019).

    VERIFICATION STATUS: Literature-Verified, Publisher-Unverified.

    Implements the literal equations EXACTLY as documented in secondary
    literature without any modification, stabilization, or numerical
    correction.  Used ONLY for ablation study EXP_003_original to empirically
    document known numerical pathologies under controlled conditions.

    Attacking phase (Spiral):
        r = u * exp(k * v),  u = v = 1
        x' = r * cos(k),  y' = r * sin(k),  z' = r * k
        scalar_factor = x' + y' + z'

    Position update (Eq. 7-8, Dhiman & Kaur 2019):
        P(t+1) = (D_s(t) o scalar_factor) o P_best(t)
    """

    @property
    def variant_name(self) -> str:
        return "original_strict"

    def _spiral_direction(self, k: float, dim: int) -> np.ndarray:
        """
        Original scalar spiral factor broadcast to all dimensions.

        NOTE: Produces r = exp(k) with k in [0, 2pi] giving r~535 at k=2pi.
        This is the documented exponential explosion pathology reproduced
        faithfully for ablation study EXP_003_original.
        """
        u, v = 1.0, 1.0
        r    = u * math.exp(k * v)
        xp   = r * math.cos(k)
        yp   = r * math.sin(k)
        zp   = r * k
        scalar = xp + yp + zp
        return np.full(dim, scalar, dtype=np.float64)

    def _position_update(
        self,
        pos:      np.ndarray,
        best_pos: np.ndarray,
        D_s:      np.ndarray,
        omega:    np.ndarray,
    ) -> np.ndarray:
        """
        Original multiplicative Hadamard position update.
        P(t+1) = (D_s(t) o omega) o P_best(t)

        NOTE: Zero-bias pathology: collapses when P_best -> 0.
        Reproduced faithfully for ablation.
        """
        return (D_s * omega) * best_pos


# ===========================================================================
# B. PROJECT-SPECIFIC STABILIZED STOA
# ===========================================================================
class ProjectSpecificStabilizedSTO(_BaseSTOA):
    """
    Project-Modified Stabilized/Additive STOA Variant.
    (Option A: Consistent Best-Guided Formulation)

    THIS IS AN EXPLICIT PROJECT-SPECIFIC ENGINEERING MODIFICATION.
    It MUST NOT be described as the original Dhiman & Kaur (2019) STOA.

    All design choices (damping, normalization, additive displacement,
    dimensional extension) are engineering decisions introduced by this
    project to ensure practical numerical stability.

    Reference: AgriAI M0 Research Specification, Section 13.5

    Modifications vs. OriginalStrictSTO
    ------------------------------------
    1. Damped spiral: r = exp(b*k), b=0.1  (eliminates exp(2pi)~535)
    2. Normalized 3D direction: Om3 = [x',y',z']^T / (||...||_2 + eps)
    3. Cyclic 3D->4D: Om4 = [Om1, Om2, Om3, Om1]^T
    4. Additive update: P(t+1) = P_best + D_s o Om4  (no zero-bias)
    5. P_best used consistently (no P_rand)
    """

    DAMPING_FACTOR: float = 0.1   # b - project damping modification
    EPSILON:        float = 1e-8  # Numerical safety for L2-norm

    @property
    def variant_name(self) -> str:
        return "stabilized"

    def _spiral_direction(self, k: float, dim: int) -> np.ndarray:
        """
        Normalized and damped 3D spiral direction, cyclically extended to dim.

        Step 1: Damped spiral radius
            r = exp(b * k),  b = 0.1

        Step 2: 3D spiral coordinates
            x' = r * cos(k),  y' = r * sin(k),  z' = r * k

        Step 3: Normalize
            Om3 = [x', y', z']^T / (||[x', y', z']||_2 + eps)
            Om3 = [Om1, Om2, Om3]^T

        Step 4: Cyclic 3D->D extension (D=4)
            Om_D[i] = Om3[i mod 3]  for i in {0,...,D-1}
            -> Om4 = [Om1, Om2, Om3, Om1]^T in R^4
        """
        b      = self.DAMPING_FACTOR
        r      = math.exp(b * k)
        xp     = r * math.cos(k)
        yp     = r * math.sin(k)
        zp     = r * k
        omega3 = np.array([xp, yp, zp], dtype=np.float64)
        norm   = np.linalg.norm(omega3) + self.EPSILON
        omega3 = omega3 / norm  # [Om1, Om2, Om3]

        # Cyclic extension to dim dimensions
        omega_d = np.array([omega3[i % 3] for i in range(dim)], dtype=np.float64)
        return omega_d

    def _position_update(
        self,
        pos:      np.ndarray,
        best_pos: np.ndarray,
        D_s:      np.ndarray,
        omega:    np.ndarray,
    ) -> np.ndarray:
        """
        Additive position update (project modification - no zero-bias).

        P(t+1) = P_best(t) + D_s(t) o Om4

        Reference: research_spec.md Section 13.5.1 Step 4.
        """
        return best_pos + D_s * omega


# ===========================================================================
# Factory function
# ===========================================================================
def create_optimizer(
    variant:        str,
    num_agents:     int,
    num_iterations: int,
    search_space:   Optional[SearchSpace] = None,
    seed:           Optional[int] = 42,
    verbose:        bool = True,
) -> _BaseSTOA:
    """
    Factory to construct the requested STOA optimizer variant.

    Parameters
    ----------
    variant        : "original_strict" | "stabilized"
    num_agents     : Population size N
    num_iterations : Number of iterations T
    search_space   : SearchSpace (default: AgriAI 4D space)
    seed           : Random seed
    verbose        : Enable iteration logging

    Returns
    -------
    _BaseSTOA instance
    """
    if search_space is None:
        search_space = SearchSpace()

    if variant == "original_strict":
        return OriginalStrictSTO(
            num_agents=num_agents,
            num_iterations=num_iterations,
            search_space=search_space,
            seed=seed,
            verbose=verbose,
        )
    elif variant == "stabilized":
        return ProjectSpecificStabilizedSTO(
            num_agents=num_agents,
            num_iterations=num_iterations,
            search_space=search_space,
            seed=seed,
            verbose=verbose,
        )
    else:
        raise ValueError(
            f"Unknown STOA variant: '{variant}'. "
            "Choose 'original_strict' or 'stabilized'."
        )
