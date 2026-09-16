"""
Experiment 15 — Uncertainty-Range Sensitivity & Monte-Carlo Convergence
========================================================================

Purpose
-------
Assess:

1. Sensitivity of the estimated 09B-v3 improvement to the assumed
   motor-parameter uncertainty range.
2. Convergence/stability of Monte-Carlo estimates as sample size increases.

This experiment extends the validated protocol of Experiments 10 and 13.

Frozen elements
---------------
- 09A / 09B-v3 controller gains
- 08A identified digital twin
- DC motor equations
- RK4 integration
- DT = 0.002 s
- simulation horizon = 10 s
- target reference = 30 rad/s
- load torque = 0.05 N m
- voltage limits = [0, 12] V
- 09B reference governor = 18 rad/s^2
- zero initial state
- no retuning
- no re-identification
- no added anti-windup

Experimental design
-------------------
Uncertainty ranges:
    ±5%, ±10%, ±15%, ±20%

Nested Monte-Carlo sample sizes:
    N = 100, 250, 500, 1000, 5000

Each uncertainty range receives one deterministic master population
of 5000 realizations. The convergence subsets are the first N members
of that master population.

Outputs
-------
results/15_uncertainty_range_summary.csv
results/15_monte_carlo_convergence.csv
results/15_monte_carlo_samples.csv
results/15_experiment_metadata.json
results/15_summary.txt

Figures:
results/15_uncertainty_range_summary.png
results/15_monte_carlo_convergence.png
results/15_win_rate_by_uncertainty.png
results/15_improvement_distributions.png

Usage
-----
Smoke test:

    python simulations/15_uncertainty_range_convergence.py --smoke-test

Full experiment:

    python simulations/15_uncertainty_range_convergence.py

Optional worker control:

    python simulations/15_uncertainty_range_convergence.py --workers 4
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path


from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================================
# Configuration
# ============================================================================

RESULTS_DIR = Path("results")

DT = 0.002
SIM_TIME = 10.0

REFERENCE = 30.0
LOAD_TORQUE = 0.05

V_MIN = 0.0
V_MAX = 12.0

GOVERNOR_RATE = 18.0

# ============================================================================
# Validated post-governor evaluation phase
# ============================================================================

PHASE_START = REFERENCE / GOVERNOR_RATE

TRUE_NOMINAL = {
    "R": 2.0,
    "L": 0.5,
    "Kt": 0.1,
    "Ke": 0.1,
    "J": 0.02,
    "b": 0.01,
}

IDENTIFIED = {
    "R": 2.011540,
    "L": 0.481997,
    "Kt": 0.105805,
    "Ke": 0.098059,
    "J": 0.021550,
    "b": 0.010715,
}

PARAMETERS = ["R", "L", "Kt", "Ke", "J", "b"]

# Frozen controller from 09A / 09B-v3.
K_AUG = np.array(
    [[4.97875849, 6.45871589, -11.78057976]],
    dtype=float,
)

UNCERTAINTY_LEVELS = [0.05, 0.10, 0.15, 0.20]
CONVERGENCE_SIZES = [100, 250, 500, 1000, 5000]

# Independent deterministic seeds for the four uncertainty populations.
BASE_SEEDS = {
    0.05: 20261505,
    0.10: 20261010,
    0.15: 20261515,
    0.20: 20261520,
}

FULL_N = max(CONVERGENCE_SIZES)

# Settling tolerance follows the existing experiment convention:
# 2% of final/reference target.
SETTLING_TOLERANCE = 0.02 * REFERENCE

SATURATION_TOLERANCE = 1e-10


# ============================================================================
# Data structures
# ============================================================================

@dataclass
class Metrics:
    rms: float
    iae: float
    ise: float
    max_abs_error: float
    peak_speed: float
    overshoot_pct: float
    settling_time: float
    max_voltage: float
    rms_voltage: float
    rms_current: float
    saturation_time: float
    saturation_pct: float


# ============================================================================
# Motor dynamics
# ============================================================================

def derivatives(
    x: np.ndarray,
    voltage: float,
    load_torque: float,
    p: dict[str, float],
) -> np.ndarray:
    """Continuous-time DC motor dynamics."""

    current, speed = x

    di = (
        -p["R"] / p["L"] * current
        -p["Ke"] / p["L"] * speed
        +voltage / p["L"]
    )

    domega = (
        p["Kt"] / p["J"] * current
        -p["b"] / p["J"] * speed
        -load_torque / p["J"]
    )

    return np.array([di, domega], dtype=float)


# ============================================================================
# Frozen controller
# ============================================================================

def frozen_controller(
    x: np.ndarray,
    z: float,
) -> tuple[float, float]:

    x_aug = np.array(
        [x[0], x[1], z],
        dtype=float,
    )

    u_unsaturated = (-K_AUG @ x_aug).item()

    u = float(
        np.clip(
            u_unsaturated,
            V_MIN,
            V_MAX,
        )
    )

    return u, u_unsaturated


# ============================================================================
# Reference governor
# ============================================================================

def governed_reference(t: float) -> float:
    """Frozen 09B-v3 reference governor."""

    return min(
        REFERENCE,
        GOVERNOR_RATE * t,
    )


# ============================================================================
# Simulation
# ============================================================================

def simulate(
    plant: dict[str, float],
    use_governor: bool,
) -> dict[str, np.ndarray]:
    """
    Simulate the frozen controller using complete closed-loop RK4.

    The controller and integral state are evaluated consistently at every
    RK4 substage, matching the corrected Experiment 13 protocol.
    """

    n_steps = int(round(SIM_TIME / DT)) + 1

    t = np.linspace(
        0.0,
        SIM_TIME,
        n_steps,
    )

    x = np.zeros(
        (n_steps, 2),
        dtype=float,
    )

    z = np.zeros(
        n_steps,
        dtype=float,
    )

    voltage = np.zeros(
        n_steps,
        dtype=float,
    )

    voltage_unsat = np.zeros(
        n_steps,
        dtype=float,
    )

    reference = np.zeros(
        n_steps,
        dtype=float,
    )

    def closed_loop_derivative(
        ti: float,
        xi: np.ndarray,
        zi: float,
    ) -> tuple[np.ndarray, float, float, float]:

        ri = (
            governed_reference(ti)
            if use_governor
            else REFERENCE
        )

        ui, uui = frozen_controller(
            xi,
            zi,
        )

        dxi = derivatives(
            xi,
            ui,
            LOAD_TORQUE,
            plant,
        )

        dzi = ri - xi[1]

        return (
            dxi,
            dzi,
            ui,
            uui,
        )

    reference[0] = (
        governed_reference(0.0)
        if use_governor
        else REFERENCE
    )

    for k in range(n_steps - 1):

        tk = t[k]

        d1, dz1, u1, uu1 = closed_loop_derivative(
            tk,
            x[k],
            z[k],
        )

        voltage[k] = u1
        voltage_unsat[k] = uu1

        reference[k] = (
            governed_reference(tk)
            if use_governor
            else REFERENCE
        )

        d2, dz2, _, _ = closed_loop_derivative(
            tk + DT / 2.0,
            x[k] + DT * d1 / 2.0,
            z[k] + DT * dz1 / 2.0,
        )

        d3, dz3, _, _ = closed_loop_derivative(
            tk + DT / 2.0,
            x[k] + DT * d2 / 2.0,
            z[k] + DT * dz2 / 2.0,
        )

        d4, dz4, _, _ = closed_loop_derivative(
            tk + DT,
            x[k] + DT * d3,
            z[k] + DT * dz3,
        )

        x[k + 1] = x[k] + (
            DT / 6.0
        ) * (
            d1
            + 2.0 * d2
            + 2.0 * d3
            + d4
        )

        z[k + 1] = z[k] + (
            DT / 6.0
        ) * (
            dz1
            + 2.0 * dz2
            + 2.0 * dz3
            + dz4
        )

    reference[-1] = (
        governed_reference(t[-1])
        if use_governor
        else REFERENCE
    )

    _, _, voltage[-1], voltage_unsat[-1] = (
        closed_loop_derivative(
            t[-1],
            x[-1],
            z[-1],
        )
    )

    return {
        "t": t,
        "current": x[:, 0],
        "speed": x[:, 1],
        "integral": z,
        "voltage": voltage,
        "voltage_unsat": voltage_unsat,
        "reference": reference,
    }

# ============================================================================
# Frozen digital-twin cache
# ============================================================================

TWIN_09A = None
TWIN_09B = None
TWIN_PHASE_MASK = None


def initialize_twin_cache() -> None:
    global TWIN_09A
    global TWIN_09B
    global TWIN_PHASE_MASK

    twin_09a = simulate(
        IDENTIFIED,
        use_governor=False,
    )

    twin_09b = simulate(
        IDENTIFIED,
        use_governor=True,
    )

    TWIN_09A = twin_09a["speed"].copy()
    TWIN_09B = twin_09b["speed"].copy()

    TWIN_PHASE_MASK = (
        twin_09a["t"] >= PHASE_START
    )



# ============================================================================
# Parallel worker initialization
# ============================================================================

def initialize_worker() -> None:
    """
    Initialize the frozen digital-twin cache once per worker process.
    """
    initialize_twin_cache()


def evaluate_worker(
    task: tuple[int, float, dict[str, float]],
) -> dict:
    """
    Evaluate one deterministic Monte-Carlo realization.

    The worker receives an already-generated plant realization so that
    multiprocessing does not alter the deterministic sampling protocol.
    """

    realization, uncertainty, plant = task

    return evaluate_realization(
        realization,
        uncertainty,
        plant,
    )

# ============================================================================
# Metrics
# ============================================================================
def calculate_metrics(
    result: dict,
) -> dict:

    t = result["t"]
    speed = result["speed"]
    current = result["current"]
    voltage = result["voltage"]

    # ------------------------------------------------------------------------
    # Full trajectory quantities
    # ------------------------------------------------------------------------

    full_error = REFERENCE - speed

    peak_speed = float(np.max(speed))

    overshoot = max(
        0.0,
        (peak_speed - REFERENCE) / REFERENCE * 100.0,
    )

    # ------------------------------------------------------------------------
    # Common post-governor phase
    # ------------------------------------------------------------------------

    mask = t >= PHASE_START

    tp = t[mask]
    speedp = speed[mask]
    currentp = current[mask]
    voltagep = voltage[mask]

    error = REFERENCE - speedp

    rms_error = float(
        np.sqrt(np.mean(error ** 2))
    )

    iae = float(
        np.trapezoid(
            np.abs(error),
            tp,
        )
    )

    ise = float(
        np.trapezoid(
            error ** 2,
            tp,
        )
    )

    rms_voltage = float(
        np.sqrt(np.mean(voltagep ** 2))
    )

    rms_current = float(
        np.sqrt(np.mean(currentp ** 2))
    )

    saturation = np.abs(
        voltagep - V_MAX
    ) <= SATURATION_TOLERANCE

    saturation_duration = float(
        np.sum(saturation) * DT
    )

    # ------------------------------------------------------------------------
    # Settling time: full trajectory, fixed 30 rad/s target
    # ------------------------------------------------------------------------

    tolerance = SETTLING_TOLERANCE

    within_band = (
        np.abs(full_error)
        <= tolerance
    )

    settling_time = SIM_TIME

    for index in range(len(t)):

        if within_band[index:].all():

            settling_time = float(
                t[index]
            )

            break

    return {
        "rms_error": rms_error,
        "iae": iae,
        "ise": ise,
        "max_abs_error": float(
            np.max(np.abs(error))
        ),
        "peak_speed": peak_speed,
        "overshoot_pct": overshoot,
        "settling_time": settling_time,
        "max_voltage": float(
            np.max(np.abs(voltagep))
        ),
        "rms_voltage": rms_voltage,
        "rms_current": rms_current,
        "saturation_duration": saturation_duration,
    }


# ============================================================================
# Parameter generation
# ============================================================================

def generate_master_population(
    uncertainty: float,
    seed: int,
    n_samples: int,
) -> list[dict[str, float]]:

    rng = np.random.default_rng(seed)

    perturbations = rng.uniform(
        -uncertainty,
        uncertainty,
        size=(
            n_samples,
            len(PARAMETERS),
        ),
    )

    population = []

    for row in perturbations:

        plant = {}

        for j, name in enumerate(PARAMETERS):

            plant[name] = (
                TRUE_NOMINAL[name]
                * (1.0 + row[j])
            )

        population.append(plant)

    return population


# ============================================================================
# Metric comparison
# ============================================================================

def improvement_pct(
    baseline: float,
    proposed: float,
) -> float:

    if baseline == 0.0:
        return np.nan

    return (
        (baseline - proposed)
        / baseline
        * 100.0
    )


def evaluate_realization(
    realization: int,
    uncertainty: float,
    plant: dict,
) -> dict:

    global TWIN_09A
    global TWIN_09B
    global TWIN_PHASE_MASK

    # ------------------------------------------------------------------------
    # True plant simulations
    # ------------------------------------------------------------------------

    true_09a = simulate(
        plant,
        use_governor=False,
    )

    true_09b = simulate(
        plant,
        use_governor=True,
    )

    metrics_09a = calculate_metrics(
        true_09a
    )

    metrics_09b = calculate_metrics(
        true_09b
    )

    # ------------------------------------------------------------------------
    # Percentage improvement
    # ------------------------------------------------------------------------

    def improvement(
        baseline: float,
        proposed: float,
    ) -> float:

        if abs(baseline) < 1e-15:
            return np.nan

        return (
            (baseline - proposed)
            / baseline
            * 100.0
        )

    rms_improvement = improvement(
        metrics_09a["rms_error"],
        metrics_09b["rms_error"],
    )

    iae_improvement = improvement(
        metrics_09a["iae"],
        metrics_09b["iae"],
    )

    ise_improvement = improvement(
        metrics_09a["ise"],
        metrics_09b["ise"],
    )

    overshoot_improvement = improvement(
        metrics_09a["overshoot_pct"],
        metrics_09b["overshoot_pct"],
    )

    settling_improvement = improvement(
        metrics_09a["settling_time"],
        metrics_09b["settling_time"],
    )

    saturation_improvement = improvement(
        metrics_09a["saturation_duration"],
        metrics_09b["saturation_duration"],
    )

    rms_voltage_improvement = improvement(
        metrics_09a["rms_voltage"],
        metrics_09b["rms_voltage"],
    )

    rms_current_improvement = improvement(
        metrics_09a["rms_current"],
        metrics_09b["rms_current"],
    )

    # ------------------------------------------------------------------------
    # Twin -> true validation error
    # ------------------------------------------------------------------------

    true_speed_09a = true_09a["speed"]
    true_speed_09b = true_09b["speed"]

    mask = TWIN_PHASE_MASK

    twin_true_09a_rmse = float(
        np.sqrt(
            np.mean(
                (
                    TWIN_09A[mask]
                    - true_speed_09a[mask]
                ) ** 2
            )
        )
    )

    twin_true_09b_rmse = float(
        np.sqrt(
            np.mean(
                (
                    TWIN_09B[mask]
                    - true_speed_09b[mask]
                ) ** 2
            )
        )
    )

    twin_true_improvement = improvement(
        twin_true_09a_rmse,
        twin_true_09b_rmse,
    )

    return {
        "uncertainty_pct": uncertainty * 100.0,
        "realization": realization,

        "rms_improvement_pct":
            rms_improvement,

        "iae_improvement_pct":
            iae_improvement,

        "ise_improvement_pct":
            ise_improvement,

        "overshoot_improvement_pct":
            overshoot_improvement,

        "settling_improvement_pct":
            settling_improvement,

        "saturation_improvement_pct":
            saturation_improvement,

        "rms_voltage_improvement_pct":
            rms_voltage_improvement,

        "rms_current_improvement_pct":
            rms_current_improvement,

        "twin_true_speed_rmse_improvement_pct":
            twin_true_improvement,

        "baseline_rms_error":
            metrics_09a["rms_error"],

        "proposed_rms_error":
            metrics_09b["rms_error"],

        "baseline_iae":
            metrics_09a["iae"],

        "proposed_iae":
            metrics_09b["iae"],

        "baseline_ise":
            metrics_09a["ise"],

        "proposed_ise":
            metrics_09b["ise"],

        "baseline_overshoot_pct":
            metrics_09a["overshoot_pct"],

        "proposed_overshoot_pct":
            metrics_09b["overshoot_pct"],

        "baseline_settling_time":
            metrics_09a["settling_time"],

        "proposed_settling_time":
            metrics_09b["settling_time"],

        "baseline_saturation_duration":
            metrics_09a["saturation_duration"],

        "proposed_saturation_duration":
            metrics_09b["saturation_duration"],

        "twin_true_09a_speed_rmse":
            twin_true_09a_rmse,

        "twin_true_09b_speed_rmse":
            twin_true_09b_rmse,
    }

# ============================================================================
# Summary statistics
# ============================================================================

IMPROVEMENT_COLUMNS = [
    "rms_improvement_pct",
    "iae_improvement_pct",
    "ise_improvement_pct",
    "overshoot_improvement_pct",
    "settling_improvement_pct",
    "saturation_improvement_pct",
    "rms_voltage_improvement_pct",
    "rms_current_improvement_pct",
    "twin_true_speed_rmse_improvement_pct",
]


def summarize(
    df: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for uncertainty, group in df.groupby(
        "uncertainty_pct"
    ):

        for n in sorted(
            group["realization"].unique()
        ):

            subset = group[
                group["realization"] <= n
            ]

            row = {
                "uncertainty_pct": uncertainty,
                "N": int(n),
            }

            for column in IMPROVEMENT_COLUMNS:

                values = subset[column].dropna()

                row[
                    f"{column}_mean"
                ] = float(
                    values.mean()
                )

                row[
                    f"{column}_median"
                ] = float(
                    values.median()
                )

                row[
                    f"{column}_std"
                ] = float(
                    values.std(
                        ddof=1
                    )
                )

                row[
                    f"{column}_win_rate_pct"
                ] = float(
                    np.mean(
                        values > 0.0
                    )
                    * 100.0
                )

                if len(values) > 1:

                    sem = (
                        values.std(
                            ddof=1
                        )
                        / np.sqrt(
                            len(values)
                        )
                    )

                    margin = (
                        1.96 * sem
                    )

                    row[
                        f"{column}_ci95_low"
                    ] = float(
                        values.mean()
                        - margin
                    )

                    row[
                        f"{column}_ci95_high"
                    ] = float(
                        values.mean()
                        + margin
                    )

                else:

                    row[
                        f"{column}_ci95_low"
                    ] = np.nan

                    row[
                        f"{column}_ci95_high"
                    ] = np.nan

            rows.append(row)

    return pd.DataFrame(rows)


# ============================================================================
# Figures
# ============================================================================

def plot_uncertainty_range_summary(
    summary: pd.DataFrame,
) -> None:

    full = summary[
        summary["N"] == FULL_N
    ].copy()

    metrics = [
        (
            "rms_improvement_pct_mean",
            "RMS error",
        ),
        (
            "iae_improvement_pct_mean",
            "IAE",
        ),
        (
            "ise_improvement_pct_mean",
            "ISE",
        ),
        (
            "overshoot_improvement_pct_mean",
            "Overshoot",
        ),
        (
            "settling_improvement_pct_mean",
            "Settling time",
        ),
        (
            "twin_true_speed_rmse_improvement_pct_mean",
            "Twin→true speed RMSE",
        ),
    ]

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    for column, label in metrics:

        ax.plot(
            full["uncertainty_pct"],
            full[column],
            marker="o",
            label=label,
        )

    ax.axhline(
        0.0,
        linewidth=1.0,
    )

    ax.set_xlabel(
        "Parameter uncertainty range (%)"
    )

    ax.set_ylabel(
        "Mean improvement (%)"
    )

    ax.set_title(
        "Experiment 15: Uncertainty-Range Sensitivity"
    )

    ax.legend(
        loc="best",
        fontsize=8,
    )

    ax.grid(
        alpha=0.3
    )

    fig.tight_layout()

    fig.savefig(
        RESULTS_DIR
        / "15_uncertainty_range_summary.png",
        dpi=300,
    )

    plt.close(fig)


def plot_convergence(
    summary: pd.DataFrame,
) -> None:

    metrics = [
        (
            "rms_improvement_pct_mean",
            "RMS error improvement",
        ),
        (
            "iae_improvement_pct_mean",
            "IAE improvement",
        ),
        (
            "twin_true_speed_rmse_improvement_pct_mean",
            "Twin→true RMSE improvement",
        ),
    ]

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    for uncertainty in UNCERTAINTY_LEVELS:

        subset = summary[
            summary["uncertainty_pct"]
            == uncertainty * 100.0
        ]

        ax.plot(
            subset["N"],
            subset[
                metrics[0][0]
            ],
            marker="o",
            label=f"±{uncertainty * 100:.0f}%",
        )

    ax.set_xscale("log")

    ax.set_xlabel(
        "Monte-Carlo sample size N"
    )

    ax.set_ylabel(
        "Mean RMS error improvement (%)"
    )

    ax.set_title(
        "Experiment 15: Monte-Carlo Convergence"
    )

    ax.legend()

    ax.grid(
        alpha=0.3
    )

    fig.tight_layout()

    fig.savefig(
        RESULTS_DIR
        / "15_monte_carlo_convergence.png",
        dpi=300,
    )

    plt.close(fig)


def plot_win_rates(
    summary: pd.DataFrame,
) -> None:

    full = summary[
        summary["N"] == FULL_N
    ].copy()

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.plot(
        full["uncertainty_pct"],
        full[
            "rms_improvement_pct_win_rate_pct"
        ],
        marker="o",
        label="RMS error",
    )

    ax.plot(
        full["uncertainty_pct"],
        full[
            "iae_improvement_pct_win_rate_pct"
        ],
        marker="o",
        label="IAE",
    )

    ax.plot(
        full["uncertainty_pct"],
        full[
            "overshoot_improvement_pct_win_rate_pct"
        ],
        marker="o",
        label="Overshoot",
    )

    ax.plot(
        full["uncertainty_pct"],
        full[
            "settling_improvement_pct_win_rate_pct"
        ],
        marker="o",
        label="Settling time",
    )

    ax.set_xlabel(
        "Parameter uncertainty range (%)"
    )

    ax.set_ylabel(
        "09B improvement win rate (%)"
    )

    ax.set_ylim(
        0,
        105,
    )

    ax.set_title(
        "Experiment 15: Win Rate Across Uncertainty Ranges"
    )

    ax.legend()

    ax.grid(
        alpha=0.3
    )

    fig.tight_layout()

    fig.savefig(
        RESULTS_DIR
        / "15_win_rate_by_uncertainty.png",
        dpi=300,
    )

    plt.close(fig)


def plot_distributions(
    samples: pd.DataFrame,
) -> None:

    columns = [
        (
            "rms_improvement_pct",
            "RMS error improvement (%)",
        ),
        (
            "iae_improvement_pct",
            "IAE improvement (%)",
        ),
        (
            "overshoot_improvement_pct",
            "Overshoot improvement (%)",
        ),
        (
            "settling_improvement_pct",
            "Settling-time improvement (%)",
        ),
    ]

    fig, ax = plt.subplots(
        figsize=(11, 7)
    )

    positions = []

    values = []

    labels = []

    position = 1

    for uncertainty in UNCERTAINTY_LEVELS:

        subset = samples[
            samples["uncertainty_pct"]
            == uncertainty * 100.0
        ]

        for column, label in columns:

            values.append(
                subset[column].dropna()
            )

            positions.append(
                position
            )

            labels.append(
                f"±{uncertainty * 100:.0f}%\n{label}"
            )

            position += 1

        position += 1

    ax.boxplot(
        values,
        positions=positions,
        showfliers=False,
    )

    ax.axhline(
        0.0,
        linewidth=1.0,
    )

    ax.set_xticks(
        positions
    )

    ax.set_xticklabels(
        labels,
        rotation=75,
        ha="right",
        fontsize=7,
    )

    ax.set_ylabel(
        "Improvement (%)"
    )

    ax.set_title(
        "Experiment 15: Improvement Distributions"
    )

    ax.grid(
        axis="y",
        alpha=0.3,
    )

    fig.tight_layout()

    fig.savefig(
        RESULTS_DIR
        / "15_improvement_distributions.png",
        dpi=300,
    )

    plt.close(fig)


# ============================================================================
# Metadata
# ============================================================================

def build_metadata(
    seeds: dict[str, int],
    runtime_seconds: float,
    smoke_test: bool,
    workers: int,
    n_samples: int,
    run_mode: str,
) -> dict:

    try:
        import matplotlib
        import scipy

        scipy_version = scipy.__version__
        matplotlib_version = matplotlib.__version__

    except Exception:

        scipy_version = "unknown"
        matplotlib_version = "unknown"

    try:
        import pandas

        pandas_version = pandas.__version__

    except Exception:

        pandas_version = "unknown"

    metadata = {
        "experiment": "15",
        "title": (
            "Uncertainty-Range Sensitivity "
            "& Monte-Carlo Convergence"
        ),
        "timestamp": time.strftime(
            "%Y-%m-%dT%H:%M:%S%z"
        ),
        "platform": platform.platform(),
        "python": sys.version,
        "numpy": np.__version__,
        "scipy": scipy_version,
        "pandas": pandas_version,
        "matplotlib": matplotlib_version,
        "cpu_count": os.cpu_count(),
        "workers": workers,
        "samples_per_uncertainty_level": n_samples,
        "total_realizations": len(UNCERTAINTY_LEVELS) * n_samples if run_mode == "production" else n_samples,
        "run_mode": run_mode,
        "smoke_test": smoke_test,
        "dt": DT,
        "simulation_time": SIM_TIME,
        "reference": REFERENCE,
        "load_torque": LOAD_TORQUE,
        "voltage_min": V_MIN,
        "voltage_max": V_MAX,
        "governor_rate": GOVERNOR_RATE,
        "parameters": PARAMETERS,
        "true_nominal": TRUE_NOMINAL,
        "identified_parameters": IDENTIFIED,
        "K_AUG": K_AUG.tolist(),
        "uncertainty_levels": UNCERTAINTY_LEVELS,
        "convergence_sizes": CONVERGENCE_SIZES,
        "master_sample_size": FULL_N,
        "seeds": seeds,
        "settling_tolerance": SETTLING_TOLERANCE,
        "runtime_seconds": runtime_seconds,
        "protocol": {
            "controller_retuning": False,
            "reidentification": False,
            "explicit_anti_windup": False,
            "integrator_equation": "z_dot = r - omega",
            "integration": "fixed-step RK4",
            "sampling": "independent uniform parameter perturbations",
            "convergence": "nested prefixes of deterministic 5000-sample master population",
        },
    }

    return metadata


# ============================================================================
# Main execution
# ============================================================================

def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help=(
            "Run a tiny protocol validation instead of "
            "the full experiment."
        ),
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel worker processes.",
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help=(
            "Override the number of Monte-Carlo realizations per "
            "uncertainty level for benchmarking."
        ),
    )


    args = parser.parse_args()

    if args.samples is not None and args.samples < 1:
        parser.error("--samples must be at least 1.")


    if args.workers < 1:

        parser.error(
            "--workers must be at least 1."
        )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    start = time.perf_counter()

    # ------------------------------------------------------------------------
    # Initialize frozen digital-twin cache
    #
    # Serial execution:
    #     cache belongs to the main process.
    #
    # Parallel execution:
    #     each worker initializes its own cache through initialize_worker().
    # ------------------------------------------------------------------------

    if args.workers == 1:

        initialize_twin_cache()

    # ------------------------------------------------------------------------
    # Experiment configuration
    # ------------------------------------------------------------------------

    if args.smoke_test:

        uncertainty_levels = [0.10]
        n_samples = 3
    elif args.samples is not None:
        uncertainty_levels = [0.10]
        n_samples = args.samples

        print()
        print("=" * 72)
        print(
            "EXPERIMENT 15 — BENCHMARK RUN"
        )
        print("=" * 72)
        print(
            f"Benchmarking {n_samples:,} realizations at ±10%."
        )
        print(
            f"Worker processes: {args.workers}"
        )

    else:

        uncertainty_levels = UNCERTAINTY_LEVELS
        n_samples = FULL_N

        print()
        print("=" * 72)
        print(
            "EXPERIMENT 15 — UNCERTAINTY RANGE "
            "& MONTE-CARLO CONVERGENCE"
        )
        print("=" * 72)

        print(
            f"Uncertainty levels: "
            f"{[int(x * 100) for x in uncertainty_levels]}%"
        )

        print(
            f"Master samples per level: "
            f"{n_samples}"
        )

        print(
            f"Total plant realizations: "
            f"{len(uncertainty_levels) * n_samples:,}"
        )

        print(
            "Each realization evaluates paired "
            "09A and 09B-v3."
        )

        print(
            f"Worker processes: "
            f"{args.workers}"
        )

    # ------------------------------------------------------------------------
    # Monte-Carlo evaluation
    # ------------------------------------------------------------------------

    all_rows = []

    for uncertainty in uncertainty_levels:

        seed = BASE_SEEDS[
            uncertainty
        ]

        population = generate_master_population(
            uncertainty,
            seed,
            n_samples,
        )

        print()
        print(
            f"Uncertainty ±{uncertainty * 100:.0f}% "
            f"| seed={seed}"
        )

        # --------------------------------------------------------------------
        # Construct deterministic tasks.
        #
        # The population is generated before multiprocessing, so the random
        # sampling protocol is completely independent of worker scheduling.
        # --------------------------------------------------------------------

        tasks = [
            (
                idx,
                uncertainty,
                plant,
            )
            for idx, plant in enumerate(
                population,
                start=1,
            )
        ]

        # --------------------------------------------------------------------
        # Serial execution
        # --------------------------------------------------------------------

        if args.workers == 1:

            rows = []

            for task in tasks:

                row = evaluate_worker(
                    task
                )

                rows.append(
                    row
                )

                idx = task[0]

                if (
                    idx <= 3
                    or idx % 500 == 0
                    or idx == n_samples
                ):

                    print(
                        f"  realization "
                        f"{idx:>5}/{n_samples}"
                    )

        # --------------------------------------------------------------------
        # Parallel execution
        # --------------------------------------------------------------------

        else:

            print(
                f"  parallel workers: "
                f"{args.workers}"
            )

            with ProcessPoolExecutor(
                max_workers=args.workers,
                initializer=initialize_worker,
            ) as executor:

                rows = list(
                    executor.map(
                        evaluate_worker,
                        tasks,
                        chunksize=4,
                    )
                )

            print(
                f"  completed "
                f"{len(rows):,}/{n_samples}"
            )

        # --------------------------------------------------------------------
        # Collect this uncertainty level.
        # --------------------------------------------------------------------

        all_rows.extend(
            rows
        )

    # ------------------------------------------------------------------------
    # Assemble results
    # ------------------------------------------------------------------------

    samples = pd.DataFrame(
        all_rows
    )

    expected_rows = (
        len(uncertainty_levels)
        * n_samples
    )

    if len(samples) != expected_rows:

        raise RuntimeError(
            "Experiment 15 produced an unexpected "
            f"number of rows: "
            f"{len(samples)} "
            f"(expected {expected_rows})."
        )

    samples.to_csv(
        RESULTS_DIR
        / "15_monte_carlo_samples.csv",
        index=False,
    )

    # ------------------------------------------------------------------------
    # Smoke-test / benchmark reporting
    # ------------------------------------------------------------------------

    if args.smoke_test:

        print()
        print(
            "Smoke-test metrics:"
        )

        print(
            samples[
                [
                    "realization",
                    "uncertainty_pct",
                    "rms_improvement_pct",
                    "iae_improvement_pct",
                    "overshoot_improvement_pct",
                    "settling_improvement_pct",
                    "twin_true_speed_rmse_improvement_pct",
                ]
            ].to_string(
                index=False
            )
        )

        runtime = (
            time.perf_counter()
            - start
        )

        metadata = build_metadata(
            {
                "10%": BASE_SEEDS[0.10]
            },
            runtime,
            True,
            args.workers,
            n_samples,
            "smoke_test",
        )

        with open(
            RESULTS_DIR
            / "15_experiment_metadata.json",
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                metadata,
                f,
                indent=2,
            )

        print()
        print(
            f"Smoke test completed in "
            f"{runtime:.2f} s."
        )

        print(
            "No production figures were generated."
        )

        return

    if args.samples is not None:
        runtime = time.perf_counter() - start

        metadata = build_metadata(
            {"10%": BASE_SEEDS[0.10]},
            runtime,
            False,
            args.workers,
            n_samples,
            "benchmark",
        )

        with open(
            RESULTS_DIR / "15_experiment_metadata.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(metadata, f, indent=2)

        print()
        print("Benchmark metrics:")
        print(
            samples[[
                "realization",
                "uncertainty_pct",
                "rms_improvement_pct",
                "iae_improvement_pct",
                "overshoot_improvement_pct",
                "settling_improvement_pct",
                "twin_true_speed_rmse_improvement_pct",
            ]].to_string(index=False)
        )
        print()
        print(f"Benchmark completed in {runtime:.2f} s.")
        print("No production summary or figures were generated.")
        return

    # ------------------------------------------------------------------------
    # Statistical summary
    # ------------------------------------------------------------------------

    summary = summarize(
        samples
    )

    summary.to_csv(
        RESULTS_DIR
        / "15_monte_carlo_convergence.csv",
        index=False,
    )

    full_summary = summary[
        summary["N"] == FULL_N
    ].copy()

    full_summary.to_csv(
        RESULTS_DIR
        / "15_uncertainty_range_summary.csv",
        index=False,
    )

    # ------------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------------

    plot_uncertainty_range_summary(
        summary
    )

    plot_convergence(
        summary
    )

    plot_win_rates(
        summary
    )

    plot_distributions(
        samples
    )

    # ------------------------------------------------------------------------
    # Runtime and metadata
    # ------------------------------------------------------------------------

    runtime = (
        time.perf_counter()
        - start
    )

    metadata = build_metadata(
        {
            f"{int(k * 100)}%": v
            for k, v in BASE_SEEDS.items()
        },
        runtime,
        False,
        args.workers,
        n_samples,
        "production",
    )

    with open(
        RESULTS_DIR
        / "15_experiment_metadata.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
        )

    # ------------------------------------------------------------------------
    # Human-readable summary
    # ------------------------------------------------------------------------

    with open(
        RESULTS_DIR
        / "15_summary.txt",
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "EXPERIMENT 15 — UNCERTAINTY-RANGE "
            "SENSITIVITY & MONTE-CARLO CONVERGENCE\n"
        )

        f.write(
            "=" * 72
            + "\n\n"
        )

        f.write(
            f"Total realizations: "
            f"{len(samples):,}\n"
        )

        f.write(
            f"Runtime: {runtime:.3f} s\n\n"
        )

        f.write(
            "FULL N=5000 RESULTS\n"
        )

        f.write(
            "-" * 72
            + "\n"
        )

        for _, row in full_summary.iterrows():

            uncertainty = row[
                "uncertainty_pct"
            ]

            f.write(
                f"\nUncertainty: ±{uncertainty:.0f}%\n"
            )

            for metric in [
                "rms_improvement_pct",
                "iae_improvement_pct",
                "ise_improvement_pct",
                "overshoot_improvement_pct",
                "settling_improvement_pct",
                "saturation_improvement_pct",
                "rms_voltage_improvement_pct",
                "rms_current_improvement_pct",
                "twin_true_speed_rmse_improvement_pct",
            ]:

                f.write(
                    f"  {metric}: "
                    f"mean={row[metric + '_mean']:.4f}, "
                    f"median={row[metric + '_median']:.4f}, "
                    f"win={row[metric + '_win_rate_pct']:.2f}%\n"
                )

        f.write(
            "\n\nCONVERGENCE SAMPLE SIZES\n"
        )

        f.write(
            "-" * 72
            + "\n"
        )

        f.write(
            summary[
                [
                    "uncertainty_pct",
                    "N",
                    "rms_improvement_pct_mean",
                    "rms_improvement_pct_std",
                    "iae_improvement_pct_mean",
                    "twin_true_speed_rmse_improvement_pct_mean",
                ]
            ].to_string(
                index=False
            )
        )

    # ------------------------------------------------------------------------
    # Final console report
    # ------------------------------------------------------------------------

    print()
    print("=" * 72)
    print(
        "EXPERIMENT 15 COMPLETE"
    )
    print("=" * 72)

    print(
        f"Runtime: {runtime:.2f} s"
    )

    print()
    print(
        "Generated:"
    )

    for path in [
        "15_uncertainty_range_summary.csv",
        "15_monte_carlo_convergence.csv",
        "15_monte_carlo_samples.csv",
        "15_experiment_metadata.json",
        "15_summary.txt",
        "15_uncertainty_range_summary.png",
        "15_monte_carlo_convergence.png",
        "15_win_rate_by_uncertainty.png",
        "15_improvement_distributions.png",
    ]:

        print(
            f"  results/{path}"
        )

if __name__ == "__main__":
    main()