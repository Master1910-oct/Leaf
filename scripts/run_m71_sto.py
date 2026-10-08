"""
scripts/run_m71_sto.py
======================
AgriAI M7.1 -- Corrected / Refined STO Hyperparameter Search.

Addresses all Critical and High findings from STO_AUDIT.md:
  FINDING 1 (Critical) : image_size 128 -> 224
  FINDING 2 (Critical) : eval_epochs 3  -> 5
  FINDING 3 (High)     : premature convergence monitored + reflected boundary
  FINDING 4 (High)     : additive update boundary collapse -> reflect_clip
  FINDING 5 (Medium)   : WD boundary pegging -> reflect_clip on WD dim too

ALL outputs go to results/m7_1_sto_refinement/ ONLY.
M7 frozen artifacts are never touched.
Test set is never accessed.

RESUMABILITY
------------
After every completed candidate evaluation the script persists:
  checkpoints/state.json     -- STO state (population, best, rng state)
  checkpoints/candidates.json -- All completed candidate records so far
Rerunning the script automatically resumes from the latest checkpoint.

BOUNDARY MONITORING
-------------------
After every STO iteration, population statistics per dimension are logged
and a BOUNDARY_COLLAPSE flag is emitted if collapse is detected.

Usage
-----
.venv\\Scripts\\python scripts/run_m71_sto.py
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

# -- project root on path ----------------------------------------------------
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.data.dataset import PlantVillageDataset
from src.data.transforms import get_train_transforms, get_eval_transforms
from src.models.ls_hgnet import LS_HGNet
from src.optimization.sto import SearchSpace, create_optimizer
from src.training.trainer import Trainer

# ----------------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("run_m71_sto")

# ----------------------------------------------------------------------------
# Constants -- M7.1 approved configuration
# ----------------------------------------------------------------------------
M71_IMAGE_SIZE    = 224
M71_EVAL_EPOCHS   = 5
M71_NUM_AGENTS    = 10
M71_NUM_ITERS     = 10
M71_TOTAL_EVALS   = M71_NUM_AGENTS * M71_NUM_ITERS  # 100 main + 10 init = 110
M71_SUBSET_FRAC   = 0.30
M71_SEED          = 42
M71_VARIANT       = "stabilized"
M71_OUTPUT_DIR    = Path("results/m7_1_sto_refinement")
M71_CKPT_DIR      = M71_OUTPUT_DIR / "checkpoints"

# Default baseline reference (M7 Candidate A -- FROZEN)
DEFAULT_BASELINE = {
    "learning_rate": 0.001,
    "weight_decay":  0.0001,
    "dropout":       0.30,
    "batch_size":    32,
    "val_macro_f1":  0.8816,   # Frozen M7 result
}

# Fixed M4 architecture
ARCH = dict(base_channels=64, num_stacks=3, encoder_levels=4, num_classes=10)

# ----------------------------------------------------------------------------
# Search space verification
# ----------------------------------------------------------------------------
def verify_default_in_search_space(ss: SearchSpace) -> None:
    """Assert the Default baseline lies strictly inside the search bounds."""
    lb, ub = ss.lb, ss.ub
    default_theta = np.array([
        math.log10(DEFAULT_BASELINE["learning_rate"]),   # -3.0  in [-4,-2] [OK]
        math.log10(DEFAULT_BASELINE["weight_decay"]),    # -4.0  in [-6,-2] [OK]
        DEFAULT_BASELINE["dropout"],                     #  0.30 in [0,0.5] [OK]
        1.0,                                             # BS=32 -> idx=1    [OK]
    ])
    for d, (v, l, u, name) in enumerate(zip(
        default_theta, lb, ub, ["LR(log)", "WD(log)", "dropout", "BS_idx"]
    )):
        assert l <= v <= u, (
            f"SEARCH SPACE VIOLATION: Default {name}={v:.4f} not in [{l},{u}]"
        )
    logger.info("Search space verification: Default baseline lies inside bounds. OK.")

# ----------------------------------------------------------------------------
# Stratified subset creation
# ----------------------------------------------------------------------------
def create_stratified_subset(
    train_csv: str,
    output_dir: Path,
    subset_frac: float = 0.30,
    seed: int = 42,
) -> Tuple[List[int], Path]:
    """
    Create a fixed stratified 30% subset from the training CSV.
    Saves the subset manifest CSV to output_dir/sto_train_subset.csv.

    Returns
    -------
    (indices, subset_csv_path)
    """
    import pandas as pd
    from sklearn.model_selection import StratifiedShuffleSplit

    subset_csv = output_dir / "sto_train_subset.csv"

    df = pd.read_csv(train_csv)
    labels = df["class_idx"].tolist()
    n_total = len(df)
    n_subset = max(10, int(round(n_total * subset_frac)))

    sss = StratifiedShuffleSplit(n_splits=1, train_size=n_subset, random_state=seed)
    subset_idx, _ = next(sss.split(range(n_total), labels))
    subset_idx = sorted(subset_idx.tolist())

    subset_df = df.iloc[subset_idx].copy().reset_index(drop=True)
    subset_df.to_csv(subset_csv, index=False)

    # Record manifest metadata
    class_counts = subset_df.groupby("class_idx").size().to_dict()
    meta = {
        "source_csv":    str(train_csv),
        "random_seed":   seed,
        "subset_frac":   subset_frac,
        "total_training_samples": n_total,
        "subset_samples": len(subset_idx),
        "samples_per_class": {str(k): int(v) for k, v in sorted(class_counts.items())},
        "subset_csv":    str(subset_csv),
        "created_utc":   datetime.now(timezone.utc).isoformat(),
    }
    with open(output_dir / "sto_train_subset_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    logger.info(
        "Stratified subset: %d / %d images (%.1f%%) | seed=%d",
        len(subset_idx), n_total, 100.0 * len(subset_idx) / n_total, seed,
    )
    for cls, cnt in sorted(class_counts.items()):
        logger.info("  class %d: %d images", cls, cnt)

    return subset_idx, subset_csv

# ----------------------------------------------------------------------------
# Checkpoint I/O
# ----------------------------------------------------------------------------
STATE_FILE      = M71_CKPT_DIR / "state.json"
CANDIDATES_FILE = M71_CKPT_DIR / "candidates.json"


def load_checkpoint() -> Optional[dict]:
    """Return saved STO state dict if it exists, else None."""
    if STATE_FILE.exists():
        with open(STATE_FILE, encoding="utf-8") as f:
            return json.load(f)
    return None


def save_state(
    iteration:      int,
    population:     np.ndarray,
    best_pos:       np.ndarray,
    best_fitness:   float,
    rng_state:      dict,
    eval_count:     int,
) -> None:
    """Persist STO population state after each evaluation."""
    state = {
        "iteration":    iteration,
        "eval_count":   eval_count,
        "best_fitness": best_fitness,
        "best_pos":     best_pos.tolist(),
        "population":   population.tolist(),
        "rng_state":    rng_state,
        "saved_utc":    datetime.now(timezone.utc).isoformat(),
    }
    M71_CKPT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = STATE_FILE.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    tmp.replace(STATE_FILE)


def load_candidates() -> List[dict]:
    if CANDIDATES_FILE.exists():
        with open(CANDIDATES_FILE, encoding="utf-8") as f:
            return json.load(f)
    return []


def append_candidate(rec: dict) -> None:
    """Atomically append a candidate record to the incremental log."""
    M71_CKPT_DIR.mkdir(parents=True, exist_ok=True)
    candidates = load_candidates()
    candidates.append(rec)
    tmp = CANDIDATES_FILE.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(candidates, f, indent=2)
    tmp.replace(CANDIDATES_FILE)

# ----------------------------------------------------------------------------
# Boundary collapse monitor
# ----------------------------------------------------------------------------
DIM_NAMES = ["LR(log10)", "WD(log10)", "dropout", "BS_idx"]
COLLAPSE_THRESHOLD = 0.05   # population std < 5% of dim range => collapse

def monitor_boundary(population: np.ndarray, ss: SearchSpace, iteration: int) -> None:
    """Log per-dimension population statistics and flag boundary collapse."""
    ranges = ss.ub - ss.lb
    logger.info("-- Boundary monitor after iteration %d --", iteration)
    collapsed_dims = []
    for d in range(ss.dim):
        vals = population[:, d]
        dmin, dmax = float(vals.min()), float(vals.max())
        dmean, dmed = float(vals.mean()), float(np.median(vals))
        dstd = float(vals.std())
        near_lb = abs(dmin - ss.lb[d]) < 0.02 * ranges[d]
        near_ub = abs(dmax - ss.ub[d]) < 0.02 * ranges[d]
        boundary_flag = ""
        if dstd < COLLAPSE_THRESHOLD * ranges[d]:
            boundary_flag = "  *** BOUNDARY_COLLAPSE ***"
            collapsed_dims.append(DIM_NAMES[d])
        logger.info(
            "  %s: min=%.4f max=%.4f mean=%.4f median=%.4f std=%.4f"
            "  near_lb=%s near_ub=%s%s",
            DIM_NAMES[d], dmin, dmax, dmean, dmed, dstd,
            near_lb, near_ub, boundary_flag,
        )
    if collapsed_dims:
        logger.warning(
            "BOUNDARY_COLLAPSE detected in dims: %s (iter %d)",
            collapsed_dims, iteration,
        )

# ----------------------------------------------------------------------------
# Single candidate evaluation
# ----------------------------------------------------------------------------

def evaluate_candidate(
    candidate_id:    int,
    theta:           np.ndarray,
    ss:              SearchSpace,
    train_dataset,
    val_dataset,
    eval_seed:       int,
    save_base:       Path,
    device:          str = "cpu",
) -> dict:
    """
    Train LS-HGNet for M71_EVAL_EPOCHS and return a full candidate record.
    Uses M71_IMAGE_SIZE and M71_EVAL_EPOCHS (already baked into the datasets).
    """
    params = ss.decode(theta)

    logger.info(
        "[Candidate %d] lr=%.2e  wd=%.2e  dropout=%.3f  bs=%d",
        candidate_id,
        params["learning_rate"],
        params["weight_decay"],
        params["dropout"],
        params["batch_size"],
    )

    torch.manual_seed(eval_seed)
    np.random.seed(eval_seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=params["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=params["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    model = LS_HGNet(
        in_channels=3,
        num_classes=ARCH["num_classes"],
        base_channels=ARCH["base_channels"],
        num_stacks=ARCH["num_stacks"],
        encoder_levels=ARCH["encoder_levels"],
        dropout=params["dropout"],
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=params["learning_rate"],
        weight_decay=params["weight_decay"],
    )

    save_dir = save_base / f"eval_{candidate_id:04d}"

    t0 = time.perf_counter()
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=None,
        device=device,
        save_dir=str(save_dir),
        seed=eval_seed,
    )
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=M71_EVAL_EPOCHS,
        checkpoint_name="eval_best.pt",
        verbose=True,
    )
    elapsed = time.perf_counter() - t0

    best_h = max(history, key=lambda h: h["val_macro_f1"])
    fitness = 1.0 - best_h["val_macro_f1"]

    rec = {
        "candidate_id":              candidate_id,
        "iteration":                 None,    # filled by caller
        "theta_lr":                  float(theta[0]),
        "theta_wd":                  float(theta[1]),
        "theta_dropout":             float(theta[2]),
        "theta_batch_size":          float(theta[3]),
        "learning_rate":             params["learning_rate"],
        "weight_decay":              params["weight_decay"],
        "dropout":                   params["dropout"],
        "batch_size":                params["batch_size"],
        "epochs":                    M71_EVAL_EPOCHS,
        "image_size":                M71_IMAGE_SIZE,
        "eval_seed":                 eval_seed,
        "validation_accuracy":       round(best_h["val_accuracy"], 6),
        "validation_macro_precision":round(best_h["val_macro_precision"], 6),
        "validation_macro_recall":   round(best_h["val_macro_recall"], 6),
        "validation_macro_f1":       round(best_h["val_macro_f1"], 6),
        "fitness":                   round(fitness, 6),
        "runtime_seconds":           round(elapsed, 2),
        "timestamp_utc":             datetime.now(timezone.utc).isoformat(),
    }

    logger.info(
        "[Candidate %d] val_macro_f1=%.4f%%  fitness=%.6f  time=%.1fs",
        candidate_id,
        best_h["val_macro_f1"] * 100,
        fitness,
        elapsed,
    )
    return rec

# ----------------------------------------------------------------------------
# Main optimization loop -- with full resumability
# ----------------------------------------------------------------------------

def run_m71_sto(
    train_csv:    str,
    val_csv:      str,
    root_dir:     str,
    output_dir:   Path,
    device:       str = "cpu",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    M71_CKPT_DIR.mkdir(parents=True, exist_ok=True)
    evals_dir = output_dir / "evals"
    evals_dir.mkdir(parents=True, exist_ok=True)

    # -- file logger ----------------------------------------------------------
    fh = logging.FileHandler(output_dir / "m7_1_sto_run.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logging.getLogger().addHandler(fh)

    logger.info("=" * 72)
    logger.info("AgriAI M7.1 -- Corrected STO Hyperparameter Search")
    logger.info("=" * 72)
    logger.info("Image size     : %d", M71_IMAGE_SIZE)
    logger.info("Eval epochs    : %d", M71_EVAL_EPOCHS)
    logger.info("Agents (N)     : %d", M71_NUM_AGENTS)
    logger.info("Iterations (T) : %d", M71_NUM_ITERS)
    logger.info("Total evals    : %d init + %d main = %d",
                M71_NUM_AGENTS, M71_NUM_AGENTS * M71_NUM_ITERS,
                M71_NUM_AGENTS + M71_NUM_AGENTS * M71_NUM_ITERS)
    logger.info("Subset frac    : %.0f%%", M71_SUBSET_FRAC * 100)
    logger.info("Seed           : %d", M71_SEED)
    logger.info("Boundary fn    : reflect_clip")
    logger.info("Variant        : %s", M71_VARIANT)
    logger.info("Output dir     : %s", output_dir)
    logger.info("=" * 72)
    logger.info("DEFAULT BASELINE (frozen M7 Candidate A):")
    logger.info("  LR=%.4e  WD=%.4e  dropout=%.2f  BS=%d  val_macro_f1=%.2f%%",
                DEFAULT_BASELINE["learning_rate"], DEFAULT_BASELINE["weight_decay"],
                DEFAULT_BASELINE["dropout"], DEFAULT_BASELINE["batch_size"],
                DEFAULT_BASELINE["val_macro_f1"] * 100)
    logger.info("=" * 72)

    # -- search space + verification ------------------------------------------
    ss = SearchSpace()
    verify_default_in_search_space(ss)

    # -- stratified subset (always fixed, same subset for all candidates) ----
    subset_idx, subset_csv_path = create_stratified_subset(
        train_csv=train_csv,
        output_dir=output_dir,
        subset_frac=M71_SUBSET_FRAC,
        seed=M71_SEED,
    )

    # -- datasets -------------------------------------------------------------
    logger.info("Loading datasets at image_size=%d ...", M71_IMAGE_SIZE)
    train_transform = get_train_transforms(image_size=M71_IMAGE_SIZE)
    val_transform   = get_eval_transforms(image_size=M71_IMAGE_SIZE)

    full_train_ds = PlantVillageDataset(
        manifest_csv=train_csv, root_dir=root_dir, transform=train_transform
    )
    # Test-set guard: do NOT load test split here
    val_dataset = PlantVillageDataset(
        manifest_csv=val_csv, root_dir=root_dir, transform=val_transform
    )

    # Fixed 30% stratified subset -- same for every candidate
    train_dataset = Subset(full_train_ds, subset_idx)

    logger.info("Train subset : %d images", len(train_dataset))
    logger.info("Val split    : %d images (FULL -- never subsampled)", len(val_dataset))

    # TEST-SET GUARD: confirm val_dataset is not the same object as train
    assert full_train_ds is not val_dataset, "TEST-SET GUARD: datasets must differ"
    assert "test" not in str(val_csv).lower() or "val" in str(val_csv).lower(), \
        f"TEST-SET GUARD: val_csv looks like test split: {val_csv}"

    # -- load or init STO state -----------------------------------------------
    checkpoint = load_checkpoint()
    completed_candidates = load_candidates()
    completed_ids = {c["candidate_id"] for c in completed_candidates}

    rng = np.random.default_rng(M71_SEED)

    if checkpoint is not None:
        logger.info(
            "RESUMING from checkpoint: iter=%d  eval_count=%d  best_fitness=%.6f",
            checkpoint["iteration"], checkpoint["eval_count"], checkpoint["best_fitness"],
        )
        population  = np.array(checkpoint["population"])
        best_pos    = np.array(checkpoint["best_pos"])
        best_fitness = float(checkpoint["best_fitness"])
        start_iter  = int(checkpoint["iteration"])
        eval_count  = int(checkpoint["eval_count"])
        # Restore RNG bit_generator state
        try:
            rng_state = checkpoint["rng_state"]
            rng.bit_generator.state = {
                "bit_generator": rng_state["bit_generator"],
                "state": {k: np.array(v) for k, v in rng_state["state"].items()},
                "has_uint32": rng_state.get("has_uint32", 0),
                "uinteger":   rng_state.get("uinteger",   0),
            }
        except Exception as e:
            logger.warning("Could not restore RNG state (%s); using seed-derived RNG.", e)
    else:
        logger.info("No checkpoint found -- starting fresh M7.1 STO search.")
        population   = np.array([ss.random_position(rng) for _ in range(M71_NUM_AGENTS)])
        best_pos     = population[0].copy()
        best_fitness = math.inf
        start_iter   = 0
        eval_count   = 0

    boundary_fn = ss.reflect_clip

    convergence: List[float] = []
    all_records: List[dict]  = list(completed_candidates)

    # -- helper: get serialisable RNG state ----------------------------------
    def _rng_state_dict() -> dict:
        st = rng.bit_generator.state
        return {
            "bit_generator": st["bit_generator"],
            "state": {k: (v.tolist() if hasattr(v, "tolist") else v)
                      for k, v in st["state"].items()},
            "has_uint32": int(st.get("has_uint32", 0)),
            "uinteger":   int(st.get("uinteger", 0)),
        }

    # -- Phase 1: evaluate initial population (if not already done) ----------
    if start_iter == 0:
        logger.info("-- Phase 1: Evaluating initial population (%d agents) --", M71_NUM_AGENTS)
        fitnesses = np.full(M71_NUM_AGENTS, math.inf)
        for i in range(M71_NUM_AGENTS):
            candidate_id = i + 1
            if candidate_id in completed_ids:
                prev = next(c for c in completed_candidates if c["candidate_id"] == candidate_id)
                fitnesses[i] = prev["fitness"]
                logger.info("[Candidate %d] Skipped (already in checkpoint)", candidate_id)
                continue

            eval_count += 1
            eval_seed = M71_SEED + eval_count
            rec = evaluate_candidate(
                candidate_id=candidate_id,
                theta=population[i],
                ss=ss,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                eval_seed=eval_seed,
                save_base=evals_dir,
                device=device,
            )
            rec["iteration"] = 0
            fitnesses[i] = rec["fitness"]

            if rec["fitness"] < best_fitness:
                best_fitness = rec["fitness"]
                best_pos = population[i].copy()

            all_records.append(rec)
            append_candidate(rec)
            save_state(0, population, best_pos, best_fitness, _rng_state_dict(), eval_count)

        # Recompute fitnesses from checkpoint where available
        for i in range(M71_NUM_AGENTS):
            cid = i + 1
            if cid in completed_ids and fitnesses[i] == math.inf:
                prev = next(c for c in completed_candidates if c["candidate_id"] == cid)
                fitnesses[i] = prev["fitness"]

        best_idx = int(np.argmin(fitnesses))
        if fitnesses[best_idx] < best_fitness:
            best_fitness = float(fitnesses[best_idx])
            best_pos = population[best_idx].copy()

        logger.info(
            "Initial population done. Best fitness=%.6f  best_val_f1=%.4f%%",
            best_fitness, (1.0 - best_fitness) * 100,
        )
        monitor_boundary(population, ss, iteration=0)
    else:
        # Restore fitnesses from checkpoint candidates for the stored population
        fitnesses = np.full(M71_NUM_AGENTS, math.inf)
        logger.info("Restoring fitnesses from checkpoint candidates.")

    # -- Phase 2: main STO loop -----------------------------------------------
    logger.info("-- Phase 2: STO main loop (%d iterations x %d agents) --",
                M71_NUM_ITERS, M71_NUM_AGENTS)

    t_run_start = time.perf_counter()

    for t in range(max(1, start_iter + 1), M71_NUM_ITERS + 1):
        S_A = 2.0 * (1.0 - t / M71_NUM_ITERS)   # linearly decaying collision param
        logger.info("-- Iteration %d/%d  S_A=%.4f  current_best_f1=%.4f%% --",
                    t, M71_NUM_ITERS, S_A, (1.0 - best_fitness) * 100)

        for i in range(M71_NUM_AGENTS):
            candidate_id = M71_NUM_AGENTS + (t - 1) * M71_NUM_AGENTS + i + 1

            if candidate_id in completed_ids:
                prev = next(c for c in completed_candidates if c["candidate_id"] == candidate_id)
                fitnesses[i] = prev["fitness"]
                population[i] = np.array([
                    prev["theta_lr"], prev["theta_wd"],
                    prev["theta_dropout"], prev["theta_batch_size"],
                ])
                logger.info("[Candidate %d] Skipped (already in checkpoint)", candidate_id)
                continue

            r_vec = rng.uniform(0.0, 1.0, ss.dim)

            # Spiral (stabilized variant)
            k = rng.uniform(0.0, 2.0 * math.pi)
            b = 0.1
            r = math.exp(b * k)
            xp, yp, zp = r * math.cos(k), r * math.sin(k), r * k
            omega3 = np.array([xp, yp, zp])
            norm = np.linalg.norm(omega3) + 1e-8
            omega3 /= norm
            omega = np.array([omega3[i % 3] for i in range(ss.dim)])

            # Displacement
            C_sa = S_A * population[i]
            M_s  = 0.5 * r_vec * (best_pos - population[i])
            D_s  = C_sa + M_s

            # Position update (additive -- stabilized)
            new_pos = best_pos + D_s * omega

            # Boundary: REFLECT (not clip) for dims 0-2
            new_pos = boundary_fn(new_pos)

            eval_count += 1
            eval_seed = M71_SEED + eval_count
            rec = evaluate_candidate(
                candidate_id=candidate_id,
                theta=new_pos,
                ss=ss,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                eval_seed=eval_seed,
                save_base=evals_dir,
                device=device,
            )
            rec["iteration"] = t

            population[i] = new_pos
            fitnesses[i]  = rec["fitness"]

            if rec["fitness"] < best_fitness:
                best_fitness = rec["fitness"]
                best_pos     = new_pos.copy()
                logger.info(
                    "[Candidate %d] ** NEW BEST **  val_macro_f1=%.4f%%",
                    candidate_id, (1.0 - best_fitness) * 100,
                )

            all_records.append(rec)
            append_candidate(rec)
            save_state(t, population, best_pos, best_fitness, _rng_state_dict(), eval_count)

        convergence.append(best_fitness)
        monitor_boundary(population, ss, iteration=t)

        logger.info(
            "Iter %d/%d done. best_fitness=%.6f  best_val_f1=%.4f%%  evals_so_far=%d",
            t, M71_NUM_ITERS, best_fitness, (1.0 - best_fitness) * 100, eval_count,
        )

    total_runtime = time.perf_counter() - t_run_start
    best_params = ss.decode(best_pos)

    logger.info("=" * 72)
    logger.info("M7.1 STO SEARCH COMPLETE")
    logger.info("=" * 72)
    logger.info("Total evaluations  : %d", eval_count)
    logger.info("Best fitness       : %.6f", best_fitness)
    logger.info("Best val Macro-F1  : %.4f%%", (1.0 - best_fitness) * 100)
    logger.info("Default baseline   : %.4f%%", DEFAULT_BASELINE["val_macro_f1"] * 100)
    logger.info("Delta vs default   : %+.4f pp",
                ((1.0 - best_fitness) - DEFAULT_BASELINE["val_macro_f1"]) * 100)
    logger.info("Best hyperparams   :")
    for k, v in best_params.items():
        logger.info("  %-20s : %s", k, v)
    logger.info("Runtime            : %.1f s (%.2f h)", total_runtime, total_runtime / 3600)
    logger.info("Test set accessed  : NO")
    logger.info("M7 artifacts modified: NO")
    logger.info("=" * 72)

    # -- Save final outputs ---------------------------------------------------
    _save_results(
        output_dir=output_dir,
        all_records=all_records,
        best_pos=best_pos,
        best_fitness=best_fitness,
        best_params=best_params,
        convergence=convergence,
        eval_count=eval_count,
        total_runtime=total_runtime,
    )


def _save_results(
    output_dir:    Path,
    all_records:   List[dict],
    best_pos:      np.ndarray,
    best_fitness:  float,
    best_params:   dict,
    convergence:   List[float],
    eval_count:    int,
    total_runtime: float,
) -> None:
    """Persist all M7.1 final output artifacts."""
    import pandas as pd

    best_val_f1 = 1.0 - best_fitness

    # -- sto_search_results.json ----------------------------------------------
    summary = {
        "experiment":         "m7_1_sto_refinement",
        "variant":            M71_VARIANT,
        "boundary_handler":   "reflect_clip",
        "image_size":         M71_IMAGE_SIZE,
        "eval_epochs":        M71_EVAL_EPOCHS,
        "num_agents":         M71_NUM_AGENTS,
        "num_iterations":     M71_NUM_ITERS,
        "subset_fraction":    M71_SUBSET_FRAC,
        "seed":               M71_SEED,
        "total_evaluations":  eval_count,
        "best_fitness":       round(best_fitness, 6),
        "best_val_macro_f1":  round(best_val_f1, 6),
        "best_params":        best_params,
        "best_position":      best_pos.tolist(),
        "convergence":        convergence,
        "convergence_f1":     [round(1.0 - f, 6) for f in convergence],
        "default_baseline_f1": DEFAULT_BASELINE["val_macro_f1"],
        "delta_vs_default_pp": round((best_val_f1 - DEFAULT_BASELINE["val_macro_f1"]) * 100, 4),
        "runtime_seconds":    round(total_runtime, 1),
        "test_set_accessed":  False,
        "m7_artifacts_modified": False,
        "completed_utc":      datetime.now(timezone.utc).isoformat(),
    }

    with open(output_dir / "sto_search_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(output_dir / "best_config.json", "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2)

    with open(output_dir / "candidate_results.json", "w", encoding="utf-8") as f:
        json.dump(all_records, f, indent=2)

    # -- sto_search_results.csv -----------------------------------------------
    df = pd.DataFrame(all_records)
    df.to_csv(output_dir / "sto_search_results.csv", index=False)

    # -- top 3 candidates -----------------------------------------------------
    top3_dir = output_dir / "top_candidates"
    top3_dir.mkdir(parents=True, exist_ok=True)
    sorted_recs = sorted(all_records, key=lambda r: r["validation_macro_f1"], reverse=True)
    top3 = sorted_recs[:3]
    with open(top3_dir / "top_candidates.json", "w", encoding="utf-8") as f:
        json.dump(top3, f, indent=2)
    top3_df = pd.DataFrame(top3)[
        ["candidate_id", "learning_rate", "weight_decay", "dropout", "batch_size",
         "validation_macro_f1", "validation_accuracy", "runtime_seconds"]
    ]
    top3_df.to_csv(top3_dir / "top_candidates.csv", index=False)

    # -- convergence plot -----------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        all_f1 = [r["validation_macro_f1"] * 100 for r in all_records]
        bsf = []
        b = 0.0
        for v in all_f1:
            b = max(b, v)
            bsf.append(b)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(range(1, len(all_f1) + 1), all_f1, ".", alpha=0.4, color="#6B7280",
                label="Per-candidate F1")
        ax.plot(range(1, len(bsf) + 1), bsf, "-", linewidth=2, color="#2563EB",
                label="Best-so-far F1")
        ax.axhline(DEFAULT_BASELINE["val_macro_f1"] * 100, color="#DC2626",
                   linestyle="--", linewidth=1.5, label=f"Default baseline ({DEFAULT_BASELINE['val_macro_f1']*100:.2f}%)")
        ax.set_xlabel("Evaluation #", fontsize=11)
        ax.set_ylabel("Validation Macro-F1 (%)", fontsize=11)
        ax.set_title("M7.1 STO Convergence -- Corrected Protocol (224px / 5 epochs / reflect_clip)", fontsize=11)
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(output_dir / "convergence.png", dpi=150)
        plt.close()
        logger.info("Saved convergence.png")
    except Exception as e:
        logger.warning("Could not save convergence plot: %s", e)

    # -- STO_SEARCH_RESULTS.md -------------------------------------------------
    _write_final_report(output_dir, summary, top3, all_records)


def _write_final_report(
    output_dir:  Path,
    summary:     dict,
    top3:        List[dict],
    all_records: List[dict],
) -> None:
    best_f1 = summary["best_val_macro_f1"] * 100
    delta   = summary["delta_vs_default_pp"]
    delta_str = f"{delta:+.4f}"

    lines = [
        "# M7.1 -- STO Search Results Report",
        "",
        "## Configuration",
        "",
        f"| Parameter         | Value |",
        f"|-------------------|-------|",
        f"| Experiment        | M7.1 -- Corrected STO |",
        f"| Variant           | {summary['variant']} |",
        f"| Boundary handler  | {summary['boundary_handler']} |",
        f"| Image size        | {summary['image_size']}x{summary['image_size']} |",
        f"| Eval epochs       | {summary['eval_epochs']} |",
        f"| Agents (N)        | {summary['num_agents']} |",
        f"| Iterations (T)    | {summary['num_iterations']} |",
        f"| Total evaluations | {summary['total_evaluations']} |",
        f"| Training subset   | {summary['subset_fraction']*100:.0f}% stratified |",
        f"| Seed              | {summary['seed']} |",
        "",
        "## Results Summary",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Best STO val Macro-F1 | **{best_f1:.4f}%** |",
        f"| Default baseline      | **{summary['default_baseline_f1']*100:.4f}%** |",
        f"| Delta (STO - Default) | **{delta_str} pp** |",
        f"| Total evaluations     | {summary['total_evaluations']} |",
        f"| Runtime               | {summary['runtime_seconds']:.1f} s ({summary['runtime_seconds']/3600:.2f} h) |",
        f"| Test set accessed     | **NO** |",
        f"| M7 artifacts modified | **NO** |",
        "",
        "## Best STO Hyperparameters",
        "",
        "| Hyperparameter | Value |",
        "|----------------|-------|",
        f"| Learning rate  | {summary['best_params']['learning_rate']:.6e} |",
        f"| Weight decay   | {summary['best_params']['weight_decay']:.6e} |",
        f"| Dropout        | {summary['best_params']['dropout']:.4f} |",
        f"| Batch size     | {summary['best_params']['batch_size']} |",
        "",
        "## Default Baseline (M7 Candidate A -- FROZEN)",
        "",
        "| Hyperparameter | Value |",
        "|----------------|-------|",
        f"| Learning rate  | {DEFAULT_BASELINE['learning_rate']:.6e} |",
        f"| Weight decay   | {DEFAULT_BASELINE['weight_decay']:.6e} |",
        f"| Dropout        | {DEFAULT_BASELINE['dropout']:.2f} |",
        f"| Batch size     | {DEFAULT_BASELINE['batch_size']} |",
        f"| Val Macro-F1   | {DEFAULT_BASELINE['val_macro_f1']*100:.4f}% (frozen) |",
        "",
        "## Top 3 STO Candidates",
        "",
        "| Rank | ID | LR | WD | Dropout | Macro-F1 | Accuracy | Runtime |",
        "|------|----|----|-----|---------|----------|----------|---------|",
    ]
    for rank, rec in enumerate(top3, 1):
        lines.append(
            f"| {rank} | {rec['candidate_id']} "
            f"| {rec['learning_rate']:.4e} "
            f"| {rec['weight_decay']:.4e} "
            f"| {rec['dropout']:.3f} "
            f"| {rec['validation_macro_f1']*100:.4f}% "
            f"| {rec['validation_accuracy']*100:.4f}% "
            f"| {rec['runtime_seconds']:.1f}s |"
        )

    lines += [
        "",
        "## Integrity Checks",
        "",
        "- [ ] Test set (`data/splits/test.csv`) was NOT accessed",
        "- [ ] `results/m7_final_selection/` was NOT modified",
        "- [ ] All outputs written to `results/m7_1_sto_refinement/` only",
        "- [ ] Validation Macro-F1 was the ONLY fitness signal",
        "",
        "## Next Steps",
        "",
        "> WAITING FOR EXPLICIT USER APPROVAL before any confirmation training of top candidates.",
        "",
        "*End of STO_SEARCH_RESULTS.md*",
    ]

    with open(output_dir / "STO_SEARCH_RESULTS.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info("Saved STO_SEARCH_RESULTS.md")


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AgriAI M7.1 -- Corrected STO Search")
    p.add_argument("--train_csv",   default="data/train.csv")
    p.add_argument("--val_csv",     default="data/val.csv")
    p.add_argument("--root_dir",    default="d:/Leaf")
    p.add_argument("--output_dir",  default=str(M71_OUTPUT_DIR))
    p.add_argument("--device",      default="cpu")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_m71_sto(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        root_dir=args.root_dir,
        output_dir=Path(args.output_dir),
        device=args.device,
    )
