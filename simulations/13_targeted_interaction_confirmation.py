"""
Experiment 13 — Targeted Interaction Confirmation

Purpose
-------
Confirm the strongest candidate parameter interactions identified in
Experiments 11–12 using a new, larger Monte-Carlo study.

This experiment:
- runs 1,000 NEW independent ±10% plant-parameter realizations;
- keeps the 09A and 09B-v3 controller gains frozen;
- keeps the 08A identified digital twin frozen;
- performs NO gain retuning and NO re-identification;
- focuses statistical testing on candidate interactions from Experiment 12.

Primary candidate interactions
------------------------------
R × Kt
R × b
Kt × b
Kt × Ke
Kt × J

Primary outcomes
----------------
RMS error
Overshoot
Settling time
Twin→true speed RMSE

Input/reference
---------------
The simulation protocol is reproduced from Experiment 10:
DT = 0.002 s
SIM_TIME = 10 s
REFERENCE = 30 rad/s
LOAD_TORQUE = 0.05 Nm
V_MIN = 0 V
V_MAX = 12 V
Governor rate = 18 rad/s^2

Outputs
-------
results/13_targeted_interaction_summary.csv
results/13_targeted_interaction_statistics.csv
results/13_targeted_interaction_summary.txt
results/13_targeted_monte_carlo.csv

Interpretation
--------------
This is a confirmation study for candidate interactions inside the ±10%
uncertainty domain. It does not establish causality or behavior outside
the sampled uncertainty region, and it is not hardware validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SEED = 20260913
N_SAMPLES = 1000

DT = 0.002
SIM_TIME = 10.0
REFERENCE = 30.0
LOAD_TORQUE = 0.05

V_MIN = 0.0
V_MAX = 12.0

GOVERNOR_RATE = 18.0
GOVERNOR_COMPLETION_TIME = REFERENCE / GOVERNOR_RATE
PHASE_START = GOVERNOR_COMPLETION_TIME

UNCERTAINTY = 0.10

RESULTS_DIR = Path("results")

BOOTSTRAP_RESAMPLES = 3000
PERMUTATION_RESAMPLES = 3000
ALPHA = 0.05

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

# Frozen controller from 09A / 09B-v3.
K_AUG = np.array(
    [[4.97875849, 6.45871589, -11.78057976]],
    dtype=float,
)

CANDIDATE_INTERACTIONS = [
    ("R", "Kt"),
    ("R", "b"),
    ("Kt", "b"),
    ("Kt", "Ke"),
    ("Kt", "J"),
]

OUTCOMES = [
    ("RMS error", "rms_improvement_pct"),
    ("Overshoot", "overshoot_improvement_pct"),
    ("Settling time", "settling_improvement_pct"),
    ("Twin→true speed RMSE", "twin_true_speed_rmse_improvement_pct"),
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Motor dynamics
# ---------------------------------------------------------------------------

def derivatives(
    x: np.ndarray,
    voltage: float,
    load_torque: float,
    p: dict,
) -> np.ndarray:
    """Continuous-time DC motor dynamics."""
    current, speed = x

    di = (
        -p["R"] / p["L"] * current
        -p["Ke"] / p["L"] * speed
        + voltage / p["L"]
    )

    domega = (
        p["Kt"] / p["J"] * current
        -p["b"] / p["J"] * speed
        -load_torque / p["J"]
    )

    return np.array([di, domega], dtype=float)


# ---------------------------------------------------------------------------
# Controllers
# ---------------------------------------------------------------------------

def frozen_controller(
    x: np.ndarray,
    z: float,
    reference: float,
) -> tuple[float, float]:
    """Frozen integral state-feedback controller."""
    x_aug = np.array([x[0], x[1], z], dtype=float)

    u_unsaturated = (-K_AUG @ x_aug).item()
    u = float(np.clip(u_unsaturated, V_MIN, V_MAX))

    return u, u_unsaturated


def governed_reference(t: float) -> float:
    """Frozen 09B-v3 reference governor."""
    return min(REFERENCE, GOVERNOR_RATE * t)


# ---------------------------------------------------------------------------
# Fixed-step RK4 simulation
# ---------------------------------------------------------------------------

def simulate(
    plant: dict,
    use_governor: bool,
    initial_state: np.ndarray | None = None,
) -> dict:
    """Reproduce the frozen Experiment-10 simulation protocol."""
    n_steps = int(round(SIM_TIME / DT)) + 1
    t = np.linspace(0.0, SIM_TIME, n_steps)

    x = np.zeros((n_steps, 2), dtype=float)
    z = np.zeros(n_steps, dtype=float)
    voltage = np.zeros(n_steps, dtype=float)
    voltage_unsat = np.zeros(n_steps, dtype=float)
    reference = np.zeros(n_steps, dtype=float)

    if initial_state is not None:
        x[0] = np.asarray(initial_state, dtype=float)

    reference[0] = (
        governed_reference(0.0)
        if use_governor
        else REFERENCE
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
            ri,
        )

        dxi = derivatives(
            xi,
            ui,
            LOAD_TORQUE,
            plant,
        )

        dzi = ri - xi[1]

        return dxi, dzi, ui, uui

    for k in range(n_steps - 1):
        tk = t[k]

        # Record controller values at the beginning of the step.
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

        # RK4: evaluate the complete closed-loop dynamics at
        # each RK4 substage, matching Experiment 10.
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

    # Final recorded values.
    dlast, dzlast, voltage[-1], voltage_unsat[-1] = (
        closed_loop_derivative(
            t[-1],
            x[-1],
            z[-1],
        )
    )

    reference[-1] = (
        governed_reference(t[-1])
        if use_governor
        else REFERENCE
    )

    return {
        "t": t,
        "current": x[:, 0],
        "speed": x[:, 1],
        "z": z,
        "voltage": voltage,
        "voltage_unsat": voltage_unsat,
        "reference": reference,
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def settling_time(
    t: np.ndarray,
    error: np.ndarray,
    reference: float,
) -> float:
    """2% settling time relative to the final reference."""
    band = 0.02 * max(abs(reference), 1e-12)
    inside = np.abs(error) <= band

    for i in range(len(t)):
        if inside[i] and np.all(inside[i:]):
            return float(t[i])

    return float("nan")


def compute_metrics(
    sim: dict,
    phase_start: float = 0.0,
) -> Metrics:
    t = sim["t"]
    speed = sim["speed"]
    current = sim["current"]
    voltage = sim["voltage"]

    mask = t >= phase_start
    tp = t[mask]
    speedp = speed[mask]
    currentp = current[mask]
    voltagep = voltage[mask]

    error = REFERENCE - speedp

    rms = float(np.sqrt(np.mean(error**2)))
    iae = float(np.trapezoid(np.abs(error), tp))
    ise = float(np.trapezoid(error**2, tp))

    peak_speed = float(np.max(speed))
    overshoot_pct = max(
        0.0,
        (peak_speed - REFERENCE) / REFERENCE * 100.0,
    )

    settling = settling_time(
        t,
        REFERENCE - speed,
        REFERENCE,
    )

    max_voltage = float(np.max(np.abs(voltagep)))
    rms_voltage = float(np.sqrt(np.mean(voltagep**2)))
    rms_current = float(np.sqrt(np.mean(currentp**2)))

    saturated = (
        np.isclose(voltagep, V_MIN, atol=1e-10)
        | np.isclose(voltagep, V_MAX, atol=1e-10)
    )

    saturation_time = float(
        np.sum(saturated) * DT
    )

    phase_duration = float(
        tp[-1] - tp[0] + DT
    )

    saturation_pct = (
        100.0 * saturation_time / phase_duration
    )

    max_abs_error = float(
        np.max(np.abs(error))
    )

    return Metrics(
        rms=rms,
        iae=iae,
        ise=ise,
        max_abs_error=max_abs_error,
        peak_speed=peak_speed,
        overshoot_pct=overshoot_pct,
        settling_time=settling,
        max_voltage=max_voltage,
        rms_voltage=rms_voltage,
        rms_current=rms_current,
        saturation_time=saturation_time,
        saturation_pct=saturation_pct,
    )


# ---------------------------------------------------------------------------
# Sampling and experiment
# ---------------------------------------------------------------------------

def sample_plants(
    rng: np.random.Generator,
) -> list[dict]:
    """Generate independent ±10% parameter perturbations."""
    plants = []

    for _ in range(N_SAMPLES):
        plant = {}

        for name, nominal in TRUE_NOMINAL.items():
            factor = rng.uniform(
                1.0 - UNCERTAINTY,
                1.0 + UNCERTAINTY,
            )
            plant[name] = nominal * factor

        plants.append(plant)

    return plants


def improvement(a: float, b: float) -> float:
    if not np.isfinite(a) or not np.isfinite(b):
        return np.nan

    if abs(a) < 1e-12:
        return np.nan

    return 100.0 * (a - b) / a


def run_monte_carlo() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    plants = sample_plants(rng)

    twin_09a = simulate(
        IDENTIFIED,
        use_governor=False,
    )
    twin_09b = simulate(
        IDENTIFIED,
        use_governor=True,
    )

    rows = []

    for idx, plant in enumerate(plants, start=1):
        true_09a = simulate(
            plant,
            use_governor=False,
        )
        true_09b = simulate(
            plant,
            use_governor=True,
        )

        m09a = compute_metrics(
            true_09a,
            phase_start=PHASE_START,
        )
        m09b = compute_metrics(
            true_09b,
            phase_start=PHASE_START,
        )

        mask = true_09a["t"] >= PHASE_START

        transfer_09a = float(
            np.sqrt(
                np.mean(
                    (
                        twin_09a["speed"][mask]
                        - true_09a["speed"][mask]
                    ) ** 2
                )
            )
        )

        transfer_09b = float(
            np.sqrt(
                np.mean(
                    (
                        twin_09b["speed"][mask]
                        - true_09b["speed"][mask]
                    ) ** 2
                )
            )
        )

        rows.append(
            {
                "sample": idx,
                **plant,

                "09A_rms": m09a.rms,
                "09B_rms": m09b.rms,
                "rms_improvement_pct": improvement(
                    m09a.rms,
                    m09b.rms,
                ),

                "09A_iae": m09a.iae,
                "09B_iae": m09b.iae,
                "iae_improvement_pct": improvement(
                    m09a.iae,
                    m09b.iae,
                ),

                "09A_ise": m09a.ise,
                "09B_ise": m09b.ise,
                "ise_improvement_pct": improvement(
                    m09a.ise,
                    m09b.ise,
                ),

                "09A_overshoot_pct": m09a.overshoot_pct,
                "09B_overshoot_pct": m09b.overshoot_pct,
                "overshoot_improvement_pct": improvement(
                    m09a.overshoot_pct,
                    m09b.overshoot_pct,
                ),

                "09A_settling_s": m09a.settling_time,
                "09B_settling_s": m09b.settling_time,
                "settling_improvement_pct": improvement(
                    m09a.settling_time,
                    m09b.settling_time,
                ),

                "09A_saturation_s": m09a.saturation_time,
                "09B_saturation_s": m09b.saturation_time,
                "saturation_improvement_pct": improvement(
                    m09a.saturation_time,
                    m09b.saturation_time,
                ),

                "09A_rms_voltage": m09a.rms_voltage,
                "09B_rms_voltage": m09b.rms_voltage,
                "rms_voltage_improvement_pct": improvement(
                    m09a.rms_voltage,
                    m09b.rms_voltage,
                ),

                "09A_rms_current": m09a.rms_current,
                "09B_rms_current": m09b.rms_current,
                "rms_current_improvement_pct": improvement(
                    m09a.rms_current,
                    m09b.rms_current,
                ),

                "09A_twin_true_speed_rmse": transfer_09a,
                "09B_twin_true_speed_rmse": transfer_09b,
                "twin_true_speed_rmse_improvement_pct": improvement(
                    transfer_09a,
                    transfer_09b,
                ),
            }
        )

        if idx % 100 == 0:
            print(f"  Completed {idx}/{N_SAMPLES} realizations")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Interaction statistics
# ---------------------------------------------------------------------------

def standardized_parameters(
    df: pd.DataFrame,
) -> pd.DataFrame:
    z = pd.DataFrame(index=df.index)

    for parameter, nominal in TRUE_NOMINAL.items():
        z[parameter] = (
            (
                df[parameter].to_numpy(dtype=float)
                - nominal
            )
            / (0.10 * nominal)
        )

    return z


def candidate_design_matrix(
    z: pd.DataFrame,
) -> tuple[np.ndarray, list[str]]:
    names = ["Intercept"]
    columns = [np.ones(len(z))]

    for parameter in TRUE_NOMINAL:
        names.append(parameter)
        columns.append(
            z[parameter].to_numpy(dtype=float)
        )

    for a, b in CANDIDATE_INTERACTIONS:
        names.append(f"{a}×{b}")
        columns.append(
            z[a].to_numpy(dtype=float)
            * z[b].to_numpy(dtype=float)
        )

    return np.column_stack(columns), names


def fit_model(
    X: np.ndarray,
    y: np.ndarray,
):
    beta, _, rank, _ = np.linalg.lstsq(
        X,
        y,
        rcond=None,
    )

    fitted = X @ beta
    residuals = y - fitted

    sse = float(np.sum(residuals**2))
    sst = float(
        np.sum((y - np.mean(y))**2)
    )

    r2 = (
        np.nan
        if sst == 0
        else 1.0 - sse / sst
    )

    n, k = X.shape

    adjusted_r2 = (
        np.nan
        if n <= k or not np.isfinite(r2)
        else 1.0
        - (1.0 - r2)
        * (n - 1)
        / (n - k)
    )

    return beta, r2, adjusted_r2, rank


def bootstrap_interaction_ci(
    X: np.ndarray,
    y: np.ndarray,
    interaction_indices: list[int],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(y)
    estimates = np.empty(
        (
            BOOTSTRAP_RESAMPLES,
            len(interaction_indices),
        )
    )

    for i in range(BOOTSTRAP_RESAMPLES):
        indices = rng.integers(
            0,
            n,
            size=n,
        )

        beta, _, _, _ = fit_model(
            X[indices],
            y[indices],
        )

        estimates[i] = beta[
            interaction_indices
        ]

    low = np.percentile(
        estimates,
        100 * ALPHA / 2,
        axis=0,
    )

    high = np.percentile(
        estimates,
        100 * (1 - ALPHA / 2),
        axis=0,
    )

    return low, high


def permutation_pvalues(
    X: np.ndarray,
    y: np.ndarray,
    interaction_indices: list[int],
    observed: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    counts = np.zeros(
        len(interaction_indices),
        dtype=int,
    )

    observed_abs = np.abs(observed)

    for _ in range(PERMUTATION_RESAMPLES):
        y_perm = rng.permutation(y)

        beta_perm, _, _, _ = fit_model(
            X,
            y_perm,
        )

        counts += (
            np.abs(
                beta_perm[
                    interaction_indices
                ]
            )
            >= observed_abs
        )

    return (
        counts + 1
    ) / (
        PERMUTATION_RESAMPLES + 1
    )


def holm_adjust(
    p_values: np.ndarray,
) -> np.ndarray:
    p_values = np.asarray(
        p_values,
        dtype=float,
    )

    adjusted = np.full_like(
        p_values,
        np.nan,
    )

    finite = np.where(
        np.isfinite(p_values)
    )[0]

    if len(finite) == 0:
        return adjusted

    order = finite[
        np.argsort(
            p_values[finite]
        )
    ]

    running = 0.0
    m = len(order)

    for rank, idx in enumerate(order):
        value = (
            (m - rank)
            * p_values[idx]
        )
        running = max(
            running,
            value,
        )
        adjusted[idx] = min(
            running,
            1.0,
        )

    return adjusted


def rank_biserial_from_differences(
    differences: np.ndarray,
) -> float:
    differences = np.asarray(
        differences,
        dtype=float,
    )

    differences = differences[
        np.isfinite(differences)
    ]
    differences = differences[
        differences != 0
    ]

    if len(differences) == 0:
        return np.nan

    ranks = rankdata(
        np.abs(differences),
        method="average",
    )

    positive = np.sum(
        ranks[differences > 0]
    )
    negative = np.sum(
        ranks[differences < 0]
    )

    total = positive + negative

    if total == 0:
        return np.nan

    return float(
        (positive - negative) / total
    )


def analyze_interactions(
    df: pd.DataFrame,
) -> pd.DataFrame:
    z = standardized_parameters(df)
    rng = np.random.default_rng(SEED + 1000)

    rows = []

    for outcome_label, outcome_column in OUTCOMES:
        y_all = df[
            outcome_column
        ].to_numpy(dtype=float)

        mask = np.isfinite(y_all)

        y = y_all[mask]
        z_valid = z.loc[
            mask
        ].reset_index(drop=True)

        X, names = candidate_design_matrix(
            z_valid
        )

        beta, r2, adjusted_r2, rank = fit_model(
            X,
            y,
        )

        interaction_indices = [
            i
            for i, name in enumerate(names)
            if "×" in name
        ]

        observed = beta[
            interaction_indices
        ]

        boot_low, boot_high = (
            bootstrap_interaction_ci(
                X,
                y,
                interaction_indices,
                rng,
            )
        )

        raw_p = permutation_pvalues(
            X,
            y,
            interaction_indices,
            observed,
            rng,
        )

        holm_p = holm_adjust(raw_p)

        for pos, idx in enumerate(
            interaction_indices
        ):
            rows.append(
                {
                    "outcome": outcome_label,
                    "outcome_key": outcome_column,
                    "interaction": names[idx],
                    "coefficient": beta[idx],
                    "bootstrap_ci95_low": boot_low[pos],
                    "bootstrap_ci95_high": boot_high[pos],
                    "permutation_p_raw": raw_p[pos],
                    "permutation_p_holm": holm_p[pos],
                    "significant_holm": bool(
                        holm_p[pos] < ALPHA
                    ),
                    "n": len(y),
                    "model_rank": rank,
                    "r2": r2,
                    "adjusted_r2": adjusted_r2,
                }
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def create_outcome_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for label, column in OUTCOMES:
        values = df[column].dropna().to_numpy(
            dtype=float
        )

        rows.append(
            {
                "metric": label,
                "n": len(values),
                "mean_improvement_pct": (
                    np.mean(values)
                    if len(values)
                    else np.nan
                ),
                "median_improvement_pct": (
                    np.median(values)
                    if len(values)
                    else np.nan
                ),
                "std_improvement_pct": (
                    np.std(values, ddof=1)
                    if len(values) > 1
                    else np.nan
                ),
                "p05_improvement_pct": (
                    np.percentile(values, 5)
                    if len(values)
                    else np.nan
                ),
                "p95_improvement_pct": (
                    np.percentile(values, 95)
                    if len(values)
                    else np.nan
                ),
                "win_rate_pct": (
                    100.0 * np.mean(values > 0)
                    if len(values)
                    else np.nan
                ),
                "negative_cases": (
                    int(np.sum(values < 0))
                    if len(values)
                    else 0
                ),
            }
        )

    return pd.DataFrame(rows)


def write_report(
    outcome_summary: pd.DataFrame,
    interaction_stats: pd.DataFrame,
) -> None:
    path = (
        RESULTS_DIR
        / "13_targeted_interaction_summary.txt"
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            "EXPERIMENT 13 — TARGETED INTERACTION CONFIRMATION\n"
        )
        f.write("=" * 78 + "\n\n")

        f.write("Protocol\n")
        f.write("--------\n")
        f.write(
            f"Monte-Carlo realizations: {N_SAMPLES}\n"
        )
        f.write(
            f"Parameter uncertainty: ±{UNCERTAINTY * 100:.1f}%\n"
        )
        f.write(
            f"Random seed: {SEED}\n"
        )
        f.write(
            "Controller gains: FROZEN from 09A / 09B-v3\n"
        )
        f.write(
            "Digital twin: FROZEN from 08A\n"
        )
        f.write(
            "Gain retuning: NONE\n"
        )
        f.write(
            "Re-identification: NONE\n"
        )
        f.write(
            "Interaction correction: Holm within each outcome\n"
        )
        f.write(
            f"Bootstrap resamples: {BOOTSTRAP_RESAMPLES:,}\n"
        )
        f.write(
            f"Permutation resamples: {PERMUTATION_RESAMPLES:,}\n\n"
        )

        f.write("Outcome summary\n")
        f.write("---------------\n")

        for _, row in outcome_summary.iterrows():
            f.write(
                f"\n{row['metric']}\n"
            )
            f.write(
                f"  n = {int(row['n'])}\n"
            )
            f.write(
                f"  mean improvement = "
                f"{row['mean_improvement_pct']:.3f}%\n"
            )
            f.write(
                f"  median improvement = "
                f"{row['median_improvement_pct']:.3f}%\n"
            )
            f.write(
                f"  std = "
                f"{row['std_improvement_pct']:.3f}%\n"
            )
            f.write(
                f"  5th–95th percentile = "
                f"{row['p05_improvement_pct']:.3f}% to "
                f"{row['p95_improvement_pct']:.3f}%\n"
            )
            f.write(
                f"  win rate = "
                f"{row['win_rate_pct']:.2f}%\n"
            )
            f.write(
                f"  negative cases = "
                f"{int(row['negative_cases'])}\n"
            )

        f.write("\n\nCandidate interaction results\n")
        f.write("-----------------------------\n")

        for outcome_label, _ in OUTCOMES:
            f.write(
                f"\n{outcome_label}\n"
            )

            sub = interaction_stats[
                interaction_stats["outcome"]
                == outcome_label
            ].copy()

            sub["abs_beta"] = (
                sub["coefficient"].abs()
            )

            sub = sub.sort_values(
                "abs_beta",
                ascending=False,
            )

            for _, row in sub.iterrows():
                marker = (
                    " *"
                    if row["significant_holm"]
                    else ""
                )

                f.write(
                    f"  {row['interaction']}: "
                    f"beta={row['coefficient']:.4f}, "
                    f"CI95=["
                    f"{row['bootstrap_ci95_low']:.4f}, "
                    f"{row['bootstrap_ci95_high']:.4f}], "
                    f"pHolm="
                    f"{row['permutation_p_holm']:.6g}"
                    f"{marker}\n"
                )

        significant = interaction_stats[
            interaction_stats[
                "significant_holm"
            ]
        ]

        f.write("\n\nConfirmed interactions\n")
        f.write("----------------------\n")

        if len(significant):
            for _, row in significant.iterrows():
                f.write(
                    f"- {row['outcome']}: "
                    f"{row['interaction']} "
                    f"(beta={row['coefficient']:.4f}, "
                    f"Holm p={row['permutation_p_holm']:.6g})\n"
                )
        else:
            f.write(
                "No candidate interaction survived Holm correction.\n"
            )

        f.write("\n\nInterpretation guardrails\n")
        f.write("-------------------------\n")
        f.write(
            "Experiment 13 is a targeted confirmation study for candidate "
            "interactions identified in Experiments 11–12.\n"
        )
        f.write(
            "A significant interaction supports a reproducible statistical "
            "association within the sampled ±10% parameter domain; it does "
            "not establish causality or behavior outside that domain.\n"
        )
        f.write(
            "The experiment remains computational and does not constitute "
            "hardware validation.\n"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 78)
    print(
        "EXPERIMENT 13 — TARGETED INTERACTION CONFIRMATION"
    )
    print("=" * 78)
    print(
        f"Monte-Carlo realizations: {N_SAMPLES}"
    )
    print(
        f"Parameter uncertainty: ±{UNCERTAINTY * 100:.1f}%"
    )
    print(
        f"Random seed: {SEED}"
    )
    print(
        "Controller gains: FROZEN from 09A / 09B-v3"
    )
    print(
        "Digital twin: FROZEN from 08A"
    )
    print(
        "No gain retuning. No re-identification."
    )
    print(
        f"Bootstrap resamples: {BOOTSTRAP_RESAMPLES:,}"
    )
    print(
        f"Permutation resamples: {PERMUTATION_RESAMPLES:,}"
    )
    print()

    print(
        "Candidate interactions:"
    )
    print(
        "  "
        + ", ".join(
            f"{a}×{b}"
            for a, b in CANDIDATE_INTERACTIONS
        )
    )
    print()

    print(
        "RUNNING NEW MONTE-CARLO SIMULATIONS"
    )
    print("-" * 78)

    df = run_monte_carlo()

    raw_path = (
        RESULTS_DIR
        / "13_targeted_monte_carlo.csv"
    )
    df.to_csv(
        raw_path,
        index=False,
    )

    outcome_summary = create_outcome_summary(
        df
    )

    interaction_stats = analyze_interactions(
        df
    )

    outcome_summary.to_csv(
        RESULTS_DIR
        / "13_targeted_interaction_summary.csv",
        index=False,
    )

    interaction_stats.to_csv(
        RESULTS_DIR
        / "13_targeted_interaction_statistics.csv",
        index=False,
    )

    write_report(
        outcome_summary,
        interaction_stats,
    )

    print()
    print(
        "OUTCOME SUMMARY"
    )
    print("-" * 78)

    with pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        180,
        "display.float_format",
        "{:.3f}".format,
    ):
        print(
            outcome_summary.to_string(
                index=False
            )
        )

    print()
    print(
        "CANDIDATE INTERACTION RESULTS"
    )
    print("-" * 78)

    with pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        180,
        "display.float_format",
        "{:.5f}".format,
    ):
        print(
            interaction_stats[
                [
                    "outcome",
                    "interaction",
                    "coefficient",
                    "bootstrap_ci95_low",
                    "bootstrap_ci95_high",
                    "permutation_p_holm",
                    "significant_holm",
                    "r2",
                    "adjusted_r2",
                ]
            ].to_string(
                index=False
            )
        )

    print()
    print(
        "OUTPUTS"
    )
    print("-" * 78)

    for output in [
        raw_path,
        RESULTS_DIR
        / "13_targeted_interaction_summary.csv",
        RESULTS_DIR
        / "13_targeted_interaction_statistics.csv",
        RESULTS_DIR
        / "13_targeted_interaction_summary.txt",
    ]:
        print(output)

    print()
    print(
        "EXPERIMENT 13 COMPLETE"
    )


if __name__ == "__main__":
    main()
