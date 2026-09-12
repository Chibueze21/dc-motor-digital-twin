"""
Experiment 10 — Uncertainty Robustness of Constraint-Aware Digital-Twin Control

Purpose
-------
Evaluate whether the performance improvement demonstrated by 09B-v3 persists
when the true DC-motor parameters vary around the nominal true plant.

Research protocol
-----------------
09A:
    Frozen integral state-feedback controller designed from the identified
    digital twin. No reference governor.

09B:
    Same frozen controller gains, plus the frozen 18 rad/s^2 reference governor.
    No gain retuning.

For each Monte-Carlo realization, the plant parameters are independently
sampled from ±10% around the nominal true plant.

The identified digital twin and controller are NOT re-identified or retuned.

Evaluation:
    - Full-horizon RMS tracking error
    - Phase-matched RMS tracking error
    - IAE
    - ISE
    - Overshoot
    - Settling time
    - Saturation duration
    - RMS voltage
    - RMS current
    - Twin-to-plant speed-transfer RMSE

Outputs
-------
results/10_uncertainty_robustness_summary.csv
results/10_uncertainty_robustness_distributions.png
results/10_uncertainty_robustness_improvement.png
results/10_uncertainty_robustness_summary.txt
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SEED = 20260912
N_SAMPLES = 100

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


# ---------------------------------------------------------------------------
# Frozen identified digital twin: 08A
# ---------------------------------------------------------------------------

IDENTIFIED = {
    "R": 2.011540,
    "L": 0.481997,
    "Kt": 0.105805,
    "Ke": 0.098059,
    "J": 0.021550,
    "b": 0.010715,
}

# Frozen controller from 09A / 09B-v3.
# Designed from the identified digital twin, desired poles [-4, -5, -6].
K_AUG = np.array([[4.97875849, 6.45871589, -11.78057976]], dtype=float)


# Nominal true plant used by 09A/09B-v3.
TRUE_NOMINAL = {
    "R": 2.0,
    "L": 0.5,
    "Kt": 0.1,
    "Ke": 0.1,
    "J": 0.02,
    "b": 0.01,
}


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

def derivatives(x: np.ndarray, voltage: float, load_torque: float, p: dict) -> np.ndarray:
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
    """
    Frozen integral state-feedback controller.

    x_aug = [current, speed, integral(reference - speed)]
    u_unsat = -K_aug x_aug + reference feedforward term represented by
              the integral action.

    Returns:
        saturated voltage, unsaturated voltage
    """
    x_aug = np.array([x[0], x[1], z], dtype=float)

    # 09A/09B-v3 controller law.
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
    """
    Simulate closed-loop motor with the frozen controller.

    A fixed-step RK4 integrator is used so every Monte-Carlo realization
    follows exactly the same numerical protocol.
    """
    n = int(round(SIM_TIME / DT)) + 1
    t = np.linspace(0.0, SIM_TIME, n)

    x = np.zeros((n, 2), dtype=float)
    z = np.zeros(n, dtype=float)
    u = np.zeros(n, dtype=float)
    u_unsat = np.zeros(n, dtype=float)
    r = np.zeros(n, dtype=float)

    if initial_state is None:
        x[0] = np.array([0.0, 0.0], dtype=float)
    else:
        x[0] = np.asarray(initial_state, dtype=float)

    r[0] = governed_reference(0.0) if use_governor else REFERENCE

    def closed_loop_derivative(
        ti: float,
        xi: np.ndarray,
        zi: float,
    ) -> tuple[np.ndarray, float, float, float]:
        ri = governed_reference(ti) if use_governor else REFERENCE
        ui, uui = frozen_controller(xi, zi, ri)
        dxi = derivatives(xi, ui, LOAD_TORQUE, plant)
        dzi = ri - xi[1]
        return dxi, dzi, ui, uui

    for k in range(n - 1):
        tk = t[k]

        # Record controller values at the beginning of the step.
        d1, dz1, u1, uu1 = closed_loop_derivative(tk, x[k], z[k])
        u[k] = u1
        u_unsat[k] = uu1
        r[k] = governed_reference(tk) if use_governor else REFERENCE

        # RK4.
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

        x[k + 1] = x[k] + DT * (d1 + 2*d2 + 2*d3 + d4) / 6.0
        z[k + 1] = z[k] + DT * (dz1 + 2*dz2 + 2*dz3 + dz4) / 6.0

    # Final recorded values.
    dlast, dzlast, u[-1], u_unsat[-1] = closed_loop_derivative(
        t[-1], x[-1], z[-1]
    )
    r[-1] = governed_reference(t[-1]) if use_governor else REFERENCE

    return {
        "t": t,
        "current": x[:, 0],
        "speed": x[:, 1],
        "z": z,
        "voltage": u,
        "voltage_unsat": u_unsat,
        "reference": r,
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def settling_time(t: np.ndarray, error: np.ndarray, reference: float) -> float:
    """
    2% settling time relative to the final reference.

    The signal must remain inside the 2% band thereafter.
    """
    band = 0.02 * max(abs(reference), 1e-12)
    inside = np.abs(error) <= band

    for i in range(len(t)):
        if inside[i] and np.all(inside[i:]):
            return float(t[i])

    return float("nan")


def compute_metrics(sim: dict, phase_start: float = 0.0) -> Metrics:
    t = sim["t"]
    speed = sim["speed"]
    current = sim["current"]
    voltage = sim["voltage"]

    mask = t >= phase_start
    tp = t[mask]
    speedp = speed[mask]
    currentp = current[mask]
    voltagep = voltage[mask]

    # For the phase-matched comparison, both 09A and 09B are evaluated
    # against the fixed 30 rad/s target after the governor has completed.
    error = REFERENCE - speedp

    rms = float(np.sqrt(np.mean(error**2)))
    iae = float(np.trapezoid(np.abs(error), tp))
    ise = float(np.trapezoid(error**2, tp))

    peak_speed = float(np.max(speed))
    overshoot_pct = max(0.0, (peak_speed - REFERENCE) / REFERENCE * 100.0)

    settling = settling_time(t, REFERENCE - speed, REFERENCE)

    max_voltage = float(np.max(np.abs(voltagep)))
    rms_voltage = float(np.sqrt(np.mean(voltagep**2)))
    rms_current = float(np.sqrt(np.mean(currentp**2)))

    saturated = np.isclose(voltagep, V_MIN, atol=1e-10) | np.isclose(
        voltagep, V_MAX, atol=1e-10
    )
    saturation_time = float(np.sum(saturated) * DT)
    phase_duration = float(tp[-1] - tp[0] + DT)
    saturation_pct = 100.0 * saturation_time / phase_duration

    max_abs_error = float(np.max(np.abs(error)))

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
# Sampling
# ---------------------------------------------------------------------------

def sample_plants(rng: np.random.Generator) -> list[dict]:
    """Generate independent ±10% parameter perturbations."""
    plants = []

    for _ in range(N_SAMPLES):
        plant = {}
        for name, nominal in TRUE_NOMINAL.items():
            factor = rng.uniform(1.0 - UNCERTAINTY, 1.0 + UNCERTAINTY)
            plant[name] = nominal * factor
        plants.append(plant)

    return plants


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)
    plants = sample_plants(rng)

    rows = []

    # Nominal identified twin is used as the reference digital twin.
    twin_09a = simulate(IDENTIFIED, use_governor=False)
    twin_09b = simulate(IDENTIFIED, use_governor=True)

    for idx, plant in enumerate(plants, start=1):
        # Baseline: frozen 09A.
        true_09a = simulate(plant, use_governor=False)

        # Proposed: frozen 09B-v3.
        true_09b = simulate(plant, use_governor=True)

        m09a = compute_metrics(true_09a, phase_start=PHASE_START)
        m09b = compute_metrics(true_09b, phase_start=PHASE_START)

        # Twin-to-true transfer error over the common post-governor window.
        mask = true_09a["t"] >= PHASE_START

        twin_speed_09a = twin_09a["speed"][mask]
        true_speed_09a = true_09a["speed"][mask]
        twin_speed_09b = twin_09b["speed"][mask]
        true_speed_09b = true_09b["speed"][mask]

        transfer_09a = float(
            np.sqrt(np.mean((twin_speed_09a - true_speed_09a) ** 2))
        )
        transfer_09b = float(
            np.sqrt(np.mean((twin_speed_09b - true_speed_09b) ** 2))
        )

        def improvement(a: float, b: float) -> float:
            if abs(a) < 1e-12:
                return np.nan
            return 100.0 * (a - b) / a

        rows.append(
            {
                "sample": idx,
                **plant,

                "09A_rms": m09a.rms,
                "09B_rms": m09b.rms,
                "rms_improvement_pct": improvement(m09a.rms, m09b.rms),

                "09A_iae": m09a.iae,
                "09B_iae": m09b.iae,
                "iae_improvement_pct": improvement(m09a.iae, m09b.iae),

                "09A_ise": m09a.ise,
                "09B_ise": m09b.ise,
                "ise_improvement_pct": improvement(m09a.ise, m09b.ise),

                "09A_overshoot_pct": m09a.overshoot_pct,
                "09B_overshoot_pct": m09b.overshoot_pct,
                "overshoot_improvement_pct": improvement(
                    m09a.overshoot_pct, m09b.overshoot_pct
                ),

                "09A_settling_s": m09a.settling_time,
                "09B_settling_s": m09b.settling_time,
                "settling_improvement_pct": improvement(
                    m09a.settling_time, m09b.settling_time
                ),

                "09A_saturation_s": m09a.saturation_time,
                "09B_saturation_s": m09b.saturation_time,
                "saturation_improvement_pct": improvement(
                    m09a.saturation_time, m09b.saturation_time
                ),

                "09A_rms_voltage": m09a.rms_voltage,
                "09B_rms_voltage": m09b.rms_voltage,
                "rms_voltage_improvement_pct": improvement(
                    m09a.rms_voltage, m09b.rms_voltage
                ),

                "09A_rms_current": m09a.rms_current,
                "09B_rms_current": m09b.rms_current,
                "rms_current_improvement_pct": improvement(
                    m09a.rms_current, m09b.rms_current
                ),

                "09A_twin_true_speed_rmse": transfer_09a,
                "09B_twin_true_speed_rmse": transfer_09b,
                "twin_true_speed_rmse_improvement_pct": improvement(
                    transfer_09a, transfer_09b
                ),
            }
        )

    # Use pandas only for compact tabulation/export.
    import pandas as pd

    df = pd.DataFrame(rows)

    csv_path = RESULTS_DIR / "10_uncertainty_robustness_summary.csv"
    df.to_csv(csv_path, index=False)

    # -----------------------------------------------------------------------
    # Statistical summary
    # -----------------------------------------------------------------------

    metrics = [
        ("RMS error", "rms_improvement_pct"),
        ("IAE", "iae_improvement_pct"),
        ("ISE", "ise_improvement_pct"),
        ("Overshoot", "overshoot_improvement_pct"),
        ("Settling time", "settling_improvement_pct"),
        ("Saturation duration", "saturation_improvement_pct"),
        ("RMS voltage", "rms_voltage_improvement_pct"),
        ("RMS current", "rms_current_improvement_pct"),
        ("Twin→true speed RMSE", "twin_true_speed_rmse_improvement_pct"),
    ]

    summary_lines = []
    summary_lines.append("=" * 78)
    summary_lines.append("EXPERIMENT 10 — UNCERTAINTY ROBUSTNESS")
    summary_lines.append("=" * 78)
    summary_lines.append(f"Monte-Carlo samples: {N_SAMPLES}")
    summary_lines.append(f"Parameter uncertainty: ±{UNCERTAINTY * 100:.1f}%")
    summary_lines.append(f"Random seed: {SEED}")
    summary_lines.append(f"Phase-matched start: t >= {PHASE_START:.6f} s")
    summary_lines.append("Controller gains: FROZEN from 09A / 09B-v3")
    summary_lines.append("Digital twin parameters: FROZEN from 08A")
    summary_lines.append("No gain retuning performed.")
    summary_lines.append("")

    for label, column in metrics:
        values = df[column].dropna().to_numpy()

        summary_lines.append(f"{label}")
        summary_lines.append("-" * 78)
        summary_lines.append(f"Mean improvement:   {np.mean(values):9.3f}%")
        summary_lines.append(f"Median improvement: {np.median(values):9.3f}%")
        summary_lines.append(f"Std. deviation:    {np.std(values, ddof=1):9.3f}%")
        summary_lines.append(f"5th percentile:    {np.percentile(values, 5):9.3f}%")
        summary_lines.append(f"95th percentile:   {np.percentile(values, 95):9.3f}%")
        summary_lines.append(
            f"09B better:        {np.mean(values > 0) * 100:9.1f}% of samples"
        )
        summary_lines.append("")

    # Robustness assessment focused on the primary claims.
    primary = {
        "RMS": df["rms_improvement_pct"],
        "IAE": df["iae_improvement_pct"],
        "Overshoot": df["overshoot_improvement_pct"],
        "Settling": df["settling_improvement_pct"],
        "Saturation": df["saturation_improvement_pct"],
    }

    summary_lines.append("=" * 78)
    summary_lines.append("PRIMARY ROBUSTNESS ASSESSMENT")
    summary_lines.append("=" * 78)

    for label, values in primary.items():
        values = values.dropna().to_numpy()
        robust_fraction = np.mean(values > 0) * 100.0
        summary_lines.append(
            f"{label:12s}: 09B improves in {robust_fraction:6.1f}% of realizations"
        )

    summary_lines.append("")
    summary_lines.append(
        "Interpretation: positive improvement means the 09B-v3 constraint-aware"
    )
    summary_lines.append(
        "strategy outperformed the frozen 09A baseline for that metric."
    )
    summary_lines.append(
        "This experiment tests robustness of the previously demonstrated effect;"
    )
    summary_lines.append(
        "it does not establish hardware/experimental validation."
    )

    summary_text = "\n".join(summary_lines)

    summary_path = RESULTS_DIR / "10_uncertainty_robustness_summary.txt"
    summary_path.write_text(summary_text, encoding="utf-8")

    print(summary_text)
    print()
    print(f"Saved: {csv_path}")
    print(f"Saved: {summary_path}")

    # -----------------------------------------------------------------------
    # Publication-oriented plots
    # -----------------------------------------------------------------------

    # Each publication figure is intentionally generated as a separate plot.
    plot_pairs = [
        ("09A_rms", "09B_rms", "Phase-matched RMS error [rad/s]",
         "Phase-Matched RMS Tracking Error"),
        ("09A_iae", "09B_iae", "IAE",
         "Integral Absolute Error"),
        ("09A_overshoot_pct", "09B_overshoot_pct", "Overshoot [%]",
         "Overshoot"),
        ("09A_settling_s", "09B_settling_s", "Settling time [s]",
         "Settling Time"),
        ("09A_saturation_s", "09B_saturation_s", "Saturation duration [s]",
         "Actuator Saturation Duration"),
        ("09A_twin_true_speed_rmse", "09B_twin_true_speed_rmse",
         "Twin→true speed RMSE [rad/s]",
         "Digital-Twin-to-True-Plant Speed Transfer Error"),
    ]

    for a_col, b_col, ylabel, title in plot_pairs:
        fig = plt.figure(figsize=(7.5, 5.5))
        ax = fig.add_subplot(1, 1, 1)
        ax.boxplot(
            [df[a_col].dropna(), df[b_col].dropna()],
            tick_labels=["09A", "09B-v3"],
        )
        ax.set_ylabel(ylabel)
        ax.set_title(
            f"{title}\nMonte-Carlo: ±{UNCERTAINTY * 100:.0f}% uncertainty"
        )
        ax.grid(True, alpha=0.25)
        fig.tight_layout()

        safe_name = {
            "09A_rms": "rms",
            "09A_iae": "iae",
            "09A_overshoot_pct": "overshoot",
            "09A_settling_s": "settling",
            "09A_saturation_s": "saturation",
            "09A_twin_true_speed_rmse": "twin_true_speed_rmse",
        }[a_col]

        figure_path = (
            RESULTS_DIR / f"10_uncertainty_robustness_{safe_name}.png"
        )
        fig.savefig(figure_path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {figure_path}")

    # Improvement distributions: one figure per primary metric.
    improvement_cols = [
        ("rms_improvement_pct", "RMS improvement [%]",
         "Distribution of RMS Improvement"),
        ("iae_improvement_pct", "IAE improvement [%]",
         "Distribution of IAE Improvement"),
        ("overshoot_improvement_pct", "Overshoot improvement [%]",
         "Distribution of Overshoot Improvement"),
        ("settling_improvement_pct", "Settling-time improvement [%]",
         "Distribution of Settling-Time Improvement"),
        ("saturation_improvement_pct", "Saturation-duration improvement [%]",
         "Distribution of Saturation-Duration Improvement"),
    ]

    for column, xlabel, title in improvement_cols:
        values = df[column].dropna().to_numpy()

        fig = plt.figure(figsize=(7.5, 5.5))
        ax = fig.add_subplot(1, 1, 1)
        ax.hist(values, bins=16)
        ax.axvline(0.0, linestyle="--", linewidth=1.2)
        ax.axvline(np.mean(values), linestyle="-", linewidth=1.2)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")
        ax.set_title(
            f"{title}\nMonte-Carlo: ±{UNCERTAINTY * 100:.0f}% uncertainty"
        )
        ax.grid(True, alpha=0.25)
        fig.tight_layout()

        safe_name = column.replace("_improvement_pct", "")
        figure_path = (
            RESULTS_DIR / f"10_uncertainty_robustness_{safe_name}_distribution.png"
        )
        fig.savefig(figure_path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {figure_path}")



if __name__ == "__main__":
    main()
