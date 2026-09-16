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
# Metrics
# ============================================================================

def calculate_metrics(
    result: dict[str, np.ndarray],
) -> Metrics:

    t = result["t"]
    speed = result["speed"]
    current = result["current"]
    voltage = result["voltage"]
    reference = result["reference"]

    error = reference - speed

    rms = float(
        np.sqrt(
            np.mean(error**2)
        )
    )

    iae = float(
        np.trapezoid(
            np.abs(error),
            t,
        )
    )

    ise = float(
        np.trapezoid(
            error**2,
            t,
        )
    )

    max_abs_error = float(
        np.max(np.abs(error))
    )

    peak_speed = float(
        np.max(speed)
    )

    overshoot_pct = max(
        0.0,
        float(
            (peak_speed - REFERENCE)
            / REFERENCE
            * 100.0
        ),
    )

    # Settling is measured relative to the final target after the
    # governed reference has reached 30 rad/s.
    post_governor = t >= (
        REFERENCE / GOVERNOR_RATE
    )

    post_t = t[post_governor]
    post_speed = speed[post_governor]

    within = (
        np.abs(
            post_speed - REFERENCE
        )
        <= SETTLING_TOLERANCE
    )

    settling_time = SIM_TIME

    if np.any(within):

        indices = np.where(within)[0]

        for idx in indices:

            if np.all(within[idx:]):

                settling_time = float(
                    post_t[idx]
                )

                break

    max_voltage = float(
        np.max(voltage)
    )

    rms_voltage = float(
        np.sqrt(
            np.mean(voltage**2)
        )
    )

    rms_current = float(
        np.sqrt(
            np.mean(current**2)
        )
    )

    saturated = (
        (voltage <= V_MIN + SATURATION_TOLERANCE)
        | (voltage >= V_MAX - SATURATION_TOLERANCE)
    )

    saturation_time = float(
        np.sum(saturated) * DT
    )

    saturation_pct = (
        saturation_time
        / SIM_TIME
        * 100.0
    )

    return Metrics(
        rms=rms,
        iae=iae,
        ise=ise,
        max_abs_error=max_abs_error,
        peak_speed=peak_speed,
        overshoot_pct=overshoot_pct,
        settling_time=settling_time,
        max_voltage=max_voltage,
        rms_voltage=rms_voltage,
        rms_current=rms_current,
        saturation_time=saturation_time,
        saturation_pct=saturation_pct,
    )


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
    plant: dict[str, float],
    realization_id: int,
    uncertainty: float,
) -> dict[str, float]:

    baseline = simulate(
        plant,
        use_governor=False,
    )

    proposed = simulate(
        plant,
        use_governor=True,
    )

    baseline_metrics = calculate_metrics(
        baseline
    )

    proposed_metrics = calculate_metrics(
        proposed
    )

    row = {
        "uncertainty_pct": uncertainty * 100.0,
        "realization": realization_id,
    }

    for name in PARAMETERS:
        row[f"{name}_value"] = plant[name]
        row[
            f"{name}_perturbation_pct"
        ] = (
            plant[name]
            / TRUE_NOMINAL[name]
            - 1.0
        ) * 100.0

    metrics = [
        ("rms", "rms_improvement_pct"),
        ("iae", "iae_improvement_pct"),
        ("ise", "ise_improvement_pct"),
        (
            "overshoot_pct",
            "overshoot_improvement_pct",
        ),
        (
            "settling_time",
            "settling_improvement_pct",
        ),
        (
            "saturation_time",
            "saturation_improvement_pct",
        ),
        (
            "rms_voltage",
            "rms_voltage_improvement_pct",
        ),
        (
            "rms_current",
            "rms_current_improvement_pct",
        ),
    ]

    for metric_name, improvement_name in metrics:

        baseline_value = getattr(
            baseline_metrics,
            metric_name,
        )

        proposed_value = getattr(
            proposed_metrics,
            metric_name,
        )

        row[
            f"baseline_{metric_name}"
        ] = baseline_value

        row[
            f"proposed_{metric_name}"
        ] = proposed_value

        row[
            improvement_name
        ] = improvement_pct(
            baseline_value,
            proposed_value,
        )

    # Twin → true speed RMSE.
    # The frozen identified twin is simulated under the same governor.
    twin = simulate(
        IDENTIFIED,
        use_governor=True,
    )

    true_speed = proposed["speed"]
    twin_speed = twin["speed"]

    twin_true_rmse = float(
        np.sqrt(
            np.mean(
                (true_speed - twin_speed) ** 2
            )
        )
    )

    # Baseline twin-vs-true value is calculated with the same plant
    # and unguided controller for paired comparison.
    twin_baseline = simulate(
        IDENTIFIED,
        use_governor=False,
    )

    twin_true_baseline_rmse = float(
        np.sqrt(
            np.mean(
                (
                    baseline["speed"]
                    - twin_baseline["speed"]
                ) ** 2
            )
        )
    )

    row[
        "baseline_twin_true_speed_rmse"
    ] = twin_true_baseline_rmse

    row[
        "proposed_twin_true_speed_rmse"
    ] = twin_true_rmse

    row[
        "twin_true_speed_rmse_improvement_pct"
    ] = improvement_pct(
        twin_true_baseline_rmse,
        twin_true_rmse,
    )

    return row


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
                group["realization"] < n
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
        help="Run a tiny protocol validation instead of the full experiment.",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Reserved for future parallel execution. Current engine uses deterministic serial execution.",
    )

    args = parser.parse_args()

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if args.workers != 1:

        print(
            "NOTE: deterministic serial engine selected; "
            "--workers is currently reserved."
        )

    start = time.perf_counter()
    initialize_twin_cache()

    if args.smoke_test:

        uncertainty_levels = [0.10]
        n_samples = 3

        print()
        print("=" * 72)
        print(
            "EXPERIMENT 15 — SMOKE TEST"
        )
        print("=" * 72)
        print(
            "Testing 3 realizations at ±10%."
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
            f"Master samples per level: {n_samples}"
        )

        print(
            f"Total plant realizations: "
            f"{len(uncertainty_levels) * n_samples:,}"
        )

        print(
            "Each realization evaluates paired 09A and 09B-v3."
        )

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

        for idx, plant in enumerate(
            population,
            start=1,
        ):

            row = evaluate_realization(
                idx,
                uncertainty,
                plant,
            )

            all_rows.append(
                row
            )

            if (
                idx <= 3
                or idx % 500 == 0
                or idx == n_samples
            ):

                print(
                    f"  realization "
                    f"{idx:>5}/{n_samples}"
                )

    samples = pd.DataFrame(
        all_rows
    )

    samples.to_csv(
        RESULTS_DIR
        / "15_monte_carlo_samples.csv",
        index=False,
    )

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

    # Human-readable summary.
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