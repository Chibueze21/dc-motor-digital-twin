
"""
Experiment 09B-v3
=================

Constraint-Aware Digital-Twin Control with Phase-Matched Evaluation

Research question:
    Can a reference-governed integral state-feedback controller,
    designed exclusively from a frozen identified digital twin,
    improve transient performance and actuator usage on the true plant?

IMPORTANT:
    This version is an EVALUATION CORRECTION of 09B-v2.

    The following are frozen:
        - 08A identified motor parameters
        - Integral-augmented state-feedback controller
        - Controller gains
        - Desired closed-loop poles
        - Reference governor rate
        - Voltage limits
        - True motor parameters
        - Simulation horizon
        - Time step
        - Load torque

    No controller retuning is performed.
    No parameter identification is repeated.
    No true-plant information is used in controller design.

Main methodological correction:
    09B-v2 compared the proposed controller's post-governor
    tracking metrics against the baseline's complete 0-10 s metrics.

    09B-v3 evaluates BOTH controllers over the same
    post-governor interval:

        t >= GOVERNOR_COMPLETION_TIME

    This produces an apples-to-apples phase-matched comparison.

Outputs:
    results/09B_constraint_aware_control_v3.png
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import place_poles


# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PLOT_PATH = RESULTS_DIR / "09B_constraint_aware_control_v3.png"


# ============================================================================
# FROZEN DC MOTOR MODEL
# ============================================================================

class DCMotorParameters:
    def __init__(
        self,
        R=2.0,
        L=0.5,
        Kt=0.1,
        Ke=0.1,
        J=0.02,
        b=0.01,
    ):
        self.R = R
        self.L = L
        self.Kt = Kt
        self.Ke = Ke
        self.J = J
        self.b = b


class DCMotor:
    """
    Continuous-time permanent-magnet DC motor.

    State:
        x[0] = armature current [A]
        x[1] = angular velocity [rad/s]

    Input:
        voltage [V]

    Disturbance:
        load torque [N*m]
    """

    def __init__(self, parameters=None):
        self.p = parameters or DCMotorParameters()

    def matrices(self):
        p = self.p

        A = np.array([
            [-p.R / p.L, -p.Ke / p.L],
            [ p.Kt / p.J, -p.b / p.J],
        ])

        B = np.array([
            [1.0 / p.L],
            [0.0],
        ])

        E = np.array([
            [0.0],
            [-1.0 / p.J],
        ])

        return A, B, E

    def derivatives(self, x, voltage, load_torque=0.0):
        A, B, E = self.matrices()

        x = np.asarray(x, dtype=float).reshape(2, 1)

        dx = (
            A @ x
            + B * voltage
            + E * load_torque
        )

        return dx.ravel()


# ============================================================================
# FROZEN PARAMETERS
# ============================================================================

TRUE_PARAMS = DCMotorParameters(
    R=2.0,
    L=0.5,
    Kt=0.1,
    Ke=0.1,
    J=0.02,
    b=0.01,
)

# Frozen 08A identification result.
IDENTIFIED_PARAMS = DCMotorParameters(
    R=2.011540,
    L=0.481997,
    Kt=0.105805,
    Ke=0.098059,
    J=0.021550,
    b=0.010715,
)


# ============================================================================
# EXPERIMENT SETTINGS
# ============================================================================

REFERENCE = 30.0
LOAD_TORQUE = 0.05

T_FINAL = 10.0
DT = 0.002

VOLTAGE_MIN = 0.0
VOLTAGE_MAX = 12.0

DESIRED_POLES = np.array([
    -4.0,
    -5.0,
    -6.0,
])

# Frozen 09B governor design.
GOVERNOR_RATE = 18.0

# For a 0 -> 30 rad/s ramp:
GOVERNOR_COMPLETION_TIME = REFERENCE / GOVERNOR_RATE


# ============================================================================
# CONTROLLER DESIGN
# ============================================================================

def design_integral_state_feedback(motor):
    """
    Design integral-augmented state feedback.

    Augmented state:

        x_aug = [current, speed, integral_error]

    where:

        integral_error_dot = reference - speed

    Control law:

        u = -K_aug x_aug

    """

    A, B, _ = motor.matrices()

    C = np.array([
        [0.0, 1.0],
    ])

    A_aug = np.block([
        [A, np.zeros((2, 1))],
        [-C, np.zeros((1, 1))],
    ])

    B_aug = np.vstack([
        B,
        [[0.0]],
    ])

    controllability = np.column_stack([
        B_aug,
        A_aug @ B_aug,
        A_aug @ A_aug @ B_aug,
    ])

    controllability_rank = np.linalg.matrix_rank(
        controllability
    )

    result = place_poles(
        A_aug,
        B_aug,
        DESIRED_POLES,
    )

    K_aug = result.gain_matrix

    closed_loop_poles = np.linalg.eigvals(
        A_aug - B_aug @ K_aug
    )

    return (
        A_aug,
        B_aug,
        K_aug,
        controllability_rank,
        closed_loop_poles,
    )


# ============================================================================
# REFERENCE GOVERNOR
# ============================================================================

def governed_reference(time):
    """
    Rate-limited reference.

    0 <= r_g <= 30 rad/s

    dr_g/dt <= 18 rad/s^2
    """

    return min(
        REFERENCE,
        GOVERNOR_RATE * time,
    )


# ============================================================================
# SIMULATION
# ============================================================================

def simulate(
    motor,
    K_aug,
    use_governor,
):
    """
    Simulate the closed-loop system using fixed-step RK4.

    Controller state:
        [current, speed, integral_error]

    Integral state:

        z_dot = reference_command - speed

    Voltage is saturated to [0, 12] V.
    """

    time = np.arange(
        0.0,
        T_FINAL + DT / 2.0,
        DT,
    )

    n = len(time)

    states = np.zeros((n, 2))
    integral_error = np.zeros(n)

    voltage = np.zeros(n)
    reference_command = np.zeros(n)

    x = np.array([
        0.0,
        0.0,
    ])

    z = 0.0

    for k, t in enumerate(time):

        if use_governor:
            r = governed_reference(t)
        else:
            r = REFERENCE

        reference_command[k] = r

        # ------------------------------------------------------------
        # Controller
        # ------------------------------------------------------------

        x_aug = np.array([
            x[0],
            x[1],
            z,
        ])

        u_unsaturated = (
            -K_aug @ x_aug
        ).item()

        u = np.clip(
            u_unsaturated,
            VOLTAGE_MIN,
            VOLTAGE_MAX,
        )

        voltage[k] = u
        states[k] = x
        integral_error[k] = z

        # ------------------------------------------------------------
        # RK4 integration
        # ------------------------------------------------------------

        def dynamics(x_local, z_local):
            if use_governor:
                r_local = governed_reference(t)
            else:
                r_local = REFERENCE

            x_dot = motor.derivatives(
                x_local,
                u,
                LOAD_TORQUE,
            )

            z_dot = r_local - x_local[1]

            return x_dot, z_dot

        k1_x, k1_z = dynamics(x, z)

        k2_x, k2_z = dynamics(
            x + 0.5 * DT * k1_x,
            z + 0.5 * DT * k1_z,
        )

        k3_x, k3_z = dynamics(
            x + 0.5 * DT * k2_x,
            z + 0.5 * DT * k2_z,
        )

        k4_x, k4_z = dynamics(
            x + DT * k3_x,
            z + DT * k3_z,
        )

        x = x + (
            DT / 6.0
        ) * (
            k1_x
            + 2.0 * k2_x
            + 2.0 * k3_x
            + k4_x
        )

        z = z + (
            DT / 6.0
        ) * (
            k1_z
            + 2.0 * k2_z
            + 2.0 * k3_z
            + k4_z
        )

    return {
        "time": time,
        "current": states[:, 0],
        "speed": states[:, 1],
        "integral_error": integral_error,
        "voltage": voltage,
        "reference": reference_command,
    }


# ============================================================================
# METRICS
# ============================================================================

def settling_time(
    time,
    speed,
    reference,
    tolerance_fraction=0.02,
):
    """
    2% settling time relative to the final reference.

    The trajectory must remain inside the tolerance band
    for the remainder of the simulation.
    """

    tolerance = tolerance_fraction * abs(reference)

    error = np.abs(
        speed - reference
    )

    inside = error <= tolerance

    for i in range(len(time)):
        if inside[i] and np.all(inside[i:]):
            return time[i]

    return np.nan


def calculate_metrics(
    simulation,
    evaluation_start=0.0,
):
    """
    Calculate metrics over a specified evaluation interval.

    This function is deliberately reusable so that 09A and 09B
    can be evaluated over exactly the same time interval.
    """

    time = simulation["time"]
    speed = simulation["speed"]
    voltage = simulation["voltage"]
    current = simulation["current"]
    reference = simulation["reference"]

    mask = time >= evaluation_start

    t = time[mask]
    y = speed[mask]
    u = voltage[mask]
    i = current[mask]
    r = reference[mask]

    # Final target is always 30 rad/s.
    tracking_error = REFERENCE - y

    rms_error = np.sqrt(
        np.mean(tracking_error ** 2)
    )

    iae = np.trapezoid(
        np.abs(tracking_error),
        t,
    )

    ise = np.trapezoid(
        tracking_error ** 2,
        t,
    )

    maximum_abs_error = np.max(
        np.abs(tracking_error)
    )

    peak_speed = np.max(y)

    overshoot = max(
        0.0,
        (
            (peak_speed - REFERENCE)
            / REFERENCE
        ) * 100.0,
    )

    settling = settling_time(
        t,
        y,
        REFERENCE,
    )

    maximum_voltage = np.max(
        np.abs(u)
    )

    rms_voltage = np.sqrt(
        np.mean(u ** 2)
    )

    voltage_squared_integral = np.trapezoid(
        u ** 2,
        t,
    )

    maximum_current = np.max(
        np.abs(i)
    )

    rms_current = np.sqrt(
        np.mean(i ** 2)
    )

    saturated = (
        np.isclose(
            u,
            VOLTAGE_MAX,
            atol=1e-10,
        )
        |
        np.isclose(
            u,
            VOLTAGE_MIN,
            atol=1e-10,
        )
    )

    saturation_time = np.sum(
        saturated
    ) * DT

    evaluation_duration = (
        t[-1] - t[0]
    )

    saturation_percentage = (
        100.0
        * saturation_time
        / evaluation_duration
    )

    return {
        "rms_error": rms_error,
        "iae": iae,
        "ise": ise,
        "maximum_abs_error": maximum_abs_error,
        "peak_speed": peak_speed,
        "overshoot": overshoot,
        "settling_time": settling,
        "maximum_voltage": maximum_voltage,
        "rms_voltage": rms_voltage,
        "voltage_squared_integral": voltage_squared_integral,
        "maximum_current": maximum_current,
        "rms_current": rms_current,
        "saturation_time": saturation_time,
        "saturation_percentage": saturation_percentage,
    }


def calculate_governed_phase_metrics(
    simulation,
):
    """
    Metrics specifically for the governed acceleration phase.

    This is descriptive rather than a conventional final-reference
    tracking metric because the commanded reference is itself moving.
    """

    time = simulation["time"]
    speed = simulation["speed"]
    reference = simulation["reference"]

    mask = time <= GOVERNOR_COMPLETION_TIME

    t = time[mask]
    y = speed[mask]
    r = reference[mask]

    error = r - y

    rms_error = np.sqrt(
        np.mean(error ** 2)
    )

    iae = np.trapezoid(
        np.abs(error),
        t,
    )

    ise = np.trapezoid(
        error ** 2,
        t,
    )

    return {
        "rms_error": rms_error,
        "iae": iae,
        "ise": ise,
    }


# ============================================================================
# PARAMETER REPORT
# ============================================================================

def print_parameter_table():
    print("\n=== FROZEN MODEL PARAMETERS ===\n")

    print(
        f"{'Parameter':<12}"
        f"{'True':>14}"
        f"{'08A ID':>14}"
        f"{'Error %':>14}"
    )

    print("-" * 54)

    true_values = {
        "R": TRUE_PARAMS.R,
        "L": TRUE_PARAMS.L,
        "Kt": TRUE_PARAMS.Kt,
        "Ke": TRUE_PARAMS.Ke,
        "J": TRUE_PARAMS.J,
        "b": TRUE_PARAMS.b,
    }

    identified_values = {
        "R": IDENTIFIED_PARAMS.R,
        "L": IDENTIFIED_PARAMS.L,
        "Kt": IDENTIFIED_PARAMS.Kt,
        "Ke": IDENTIFIED_PARAMS.Ke,
        "J": IDENTIFIED_PARAMS.J,
        "b": IDENTIFIED_PARAMS.b,
    }

    for name in true_values:

        true_value = true_values[name]
        identified_value = identified_values[name]

        error_percent = (
            abs(
                identified_value
                - true_value
            )
            / abs(true_value)
        ) * 100.0

        print(
            f"{name:<12}"
            f"{true_value:>14.6f}"
            f"{identified_value:>14.6f}"
            f"{error_percent:>13.3f}%"
        )


# ============================================================================
# PRINT METRICS
# ============================================================================

def print_metrics(
    title,
    metrics,
):
    print(f"\n=== {title} ===")

    print(
        f"RMS tracking error:       "
        f"{metrics['rms_error']:.6f} rad/s"
    )

    print(
        f"IAE:                      "
        f"{metrics['iae']:.6f}"
    )

    print(
        f"ISE:                      "
        f"{metrics['ise']:.6f}"
    )

    print(
        f"Maximum abs error:        "
        f"{metrics['maximum_abs_error']:.6f} rad/s"
    )

    print(
        f"Peak speed:               "
        f"{metrics['peak_speed']:.6f} rad/s"
    )

    print(
        f"Overshoot:                "
        f"{metrics['overshoot']:.4f} %"
    )

    print(
        f"Settling time:             "
        f"{metrics['settling_time']:.6f} s"
    )

    print(
        f"Maximum voltage:          "
        f"{metrics['maximum_voltage']:.6f} V"
    )

    print(
        f"RMS voltage:              "
        f"{metrics['rms_voltage']:.6f} V"
    )

    print(
        f"Voltage squared integral: "
        f"{metrics['voltage_squared_integral']:.6f}"
    )

    print(
        f"Maximum current:          "
        f"{metrics['maximum_current']:.6f} A"
    )

    print(
        f"RMS current:              "
        f"{metrics['rms_current']:.6f} A"
    )

    print(
        f"Time at saturation:       "
        f"{metrics['saturation_time']:.6f} s"
    )

    print(
        f"Saturation percentage:    "
        f"{metrics['saturation_percentage']:.3f} %"
    )


# ============================================================================
# IMPROVEMENT
# ============================================================================

def percentage_improvement(
    baseline,
    proposed,
):
    """
    Percentage improvement for metrics where lower is better.
    """

    if abs(baseline) < 1e-15:
        return np.nan

    return (
        (baseline - proposed)
        / abs(baseline)
    ) * 100.0


def print_comparison(
    baseline,
    proposed,
    baseline_name,
    proposed_name,
    title,
):
    print(f"\n=== {title} ===")

    print(
        f"{'Metric':<30}"
        f"{baseline_name:>16}"
        f"{proposed_name:>16}"
        f"{'Improvement':>16}"
    )

    print("-" * 80)

    comparisons = [
        (
            "RMS tracking error",
            baseline["rms_error"],
            proposed["rms_error"],
        ),
        (
            "IAE",
            baseline["iae"],
            proposed["iae"],
        ),
        (
            "ISE",
            baseline["ise"],
            proposed["ise"],
        ),
        (
            "Overshoot (%)",
            baseline["overshoot"],
            proposed["overshoot"],
        ),
        (
            "Settling time (s)",
            baseline["settling_time"],
            proposed["settling_time"],
        ),
        (
            "Saturation time (s)",
            baseline["saturation_time"],
            proposed["saturation_time"],
        ),
        (
            "Saturation (%)",
            baseline["saturation_percentage"],
            proposed["saturation_percentage"],
        ),
        (
            "RMS voltage (V)",
            baseline["rms_voltage"],
            proposed["rms_voltage"],
        ),
        (
            "RMS current (A)",
            baseline["rms_current"],
            proposed["rms_current"],
        ),
        (
            "Voltage squared integral",
            baseline["voltage_squared_integral"],
            proposed["voltage_squared_integral"],
        ),
    ]

    for name, base_value, proposed_value in comparisons:

        improvement = percentage_improvement(
            base_value,
            proposed_value,
        )

        print(
            f"{name:<30}"
            f"{base_value:>16.6f}"
            f"{proposed_value:>16.6f}"
            f"{improvement:>15.3f}%"
        )


# ============================================================================
# PLOT
# ============================================================================

def create_plot(
    baseline,
    proposed,
):
    time = baseline["time"]

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(11, 11),
        sharex=True,
    )

    # ------------------------------------------------------------
    # Speed
    # ------------------------------------------------------------

    axes[0].plot(
        time,
        baseline["speed"],
        label="09A baseline — true plant",
    )

    axes[0].plot(
        time,
        proposed["speed"],
        label="09B-v3 proposed — true plant",
    )

    axes[0].plot(
        time,
        proposed["reference"],
        "--",
        label="09B governed reference",
    )

    axes[0].axvline(
        GOVERNOR_COMPLETION_TIME,
        linestyle=":",
        label="Governor completion",
    )

    axes[0].axhline(
        REFERENCE,
        linestyle="--",
        alpha=0.5,
        label="Final target",
    )

    axes[0].set_ylabel(
        "Speed [rad/s]"
    )

    axes[0].set_title(
        "09B-v3 Constraint-Aware Digital-Twin Control"
    )

    axes[0].grid(True)
    axes[0].legend()

    # ------------------------------------------------------------
    # Voltage
    # ------------------------------------------------------------

    axes[1].plot(
        time,
        baseline["voltage"],
        label="09A baseline",
    )

    axes[1].plot(
        time,
        proposed["voltage"],
        label="09B-v3 proposed",
    )

    axes[1].axhline(
        VOLTAGE_MAX,
        linestyle="--",
        label="Voltage limit",
    )

    axes[1].axvline(
        GOVERNOR_COMPLETION_TIME,
        linestyle=":",
    )

    axes[1].set_ylabel(
        "Voltage [V]"
    )

    axes[1].grid(True)
    axes[1].legend()

    # ------------------------------------------------------------
    # Current
    # ------------------------------------------------------------

    axes[2].plot(
        time,
        baseline["current"],
        label="09A baseline",
    )

    axes[2].plot(
        time,
        proposed["current"],
        label="09B-v3 proposed",
    )

    axes[2].axvline(
        GOVERNOR_COMPLETION_TIME,
        linestyle=":",
    )

    axes[2].set_ylabel(
        "Current [A]"
    )

    axes[2].set_xlabel(
        "Time [s]"
    )

    axes[2].grid(True)
    axes[2].legend()

    fig.tight_layout()

    fig.savefig(
        PLOT_PATH,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def main():

    print("=" * 72)
    print("=== EXPERIMENT 09B-v3: CONSTRAINT-AWARE DIGITAL-TWIN CONTROL ===")
    print("=" * 72)

    print(
        "\nPurpose:"
        "\nEvaluate reference governance using the frozen 08A model."
        "\n"
        "\n09B-v3 is an evaluation correction of 09B-v2."
        "\nNo parameter optimization is performed."
        "\nNo controller retuning is performed."
        "\nNo true-plant information is used in controller design."
    )

    print_parameter_table()

    # ------------------------------------------------------------
    # Controller synthesis
    # ------------------------------------------------------------

    identified_motor = DCMotor(
        IDENTIFIED_PARAMS
    )

    (
        A_aug,
        B_aug,
        K_aug,
        controllability_rank,
        closed_loop_poles,
    ) = design_integral_state_feedback(
        identified_motor
    )

    print("\n=== 09B-v3 CONTROLLER DESIGN ===")

    print(
        "\nAugmented state:"
        "\n[current, speed, integral_error]"
    )

    print(
        "\nDesired closed-loop poles:"
    )

    print(DESIRED_POLES)

    print(
        "\nAugmented controllability rank:"
    )

    print(
        f"{controllability_rank}/3"
    )

    print(
        "\nK_aug ="
    )

    print(K_aug)

    print(
        "\nClosed-loop poles from identified model:"
    )

    print(
        np.sort_complex(
            closed_loop_poles
        )
    )

    print(
        "\nReference governor rate:"
        f"\n{GOVERNOR_RATE:.3f} rad/s^2"
    )

    print(
        "\nExpected governor completion time:"
        f"\n{GOVERNOR_COMPLETION_TIME:.6f} s"
    )

    # ------------------------------------------------------------
    # Simulate baseline 09A on identified twin
    # ------------------------------------------------------------

    baseline_twin = simulate(
        identified_motor,
        K_aug,
        use_governor=False,
    )

    # ------------------------------------------------------------
    # Simulate baseline 09A on true plant
    # ------------------------------------------------------------

    true_motor = DCMotor(
        TRUE_PARAMS
    )

    baseline_true = simulate(
        true_motor,
        K_aug,
        use_governor=False,
    )

    # ------------------------------------------------------------
    # Simulate proposed 09B on identified twin
    # ------------------------------------------------------------

    proposed_twin = simulate(
        identified_motor,
        K_aug,
        use_governor=True,
    )

    # ------------------------------------------------------------
    # Simulate proposed 09B on true plant
    # ------------------------------------------------------------

    proposed_true = simulate(
        true_motor,
        K_aug,
        use_governor=True,
    )

    # ============================================================
    # WHOLE-EXPERIMENT METRICS
    # ============================================================

    baseline_twin_full = calculate_metrics(
        baseline_twin,
        evaluation_start=0.0,
    )

    baseline_true_full = calculate_metrics(
        baseline_true,
        evaluation_start=0.0,
    )

    proposed_twin_full = calculate_metrics(
        proposed_twin,
        evaluation_start=0.0,
    )

    proposed_true_full = calculate_metrics(
        proposed_true,
        evaluation_start=0.0,
    )

    print_metrics(
        "09A BASELINE — IDENTIFIED DIGITAL TWIN — FULL 0-10 s",
        baseline_twin_full,
    )

    print_metrics(
        "09A BASELINE — TRUE PLANT — FULL 0-10 s",
        baseline_true_full,
    )

    print_metrics(
        "09B-v3 PROPOSED — IDENTIFIED DIGITAL TWIN — FULL 0-10 s",
        proposed_twin_full,
    )

    print_metrics(
        "09B-v3 PROPOSED — TRUE PLANT — FULL 0-10 s",
        proposed_true_full,
    )

    # ============================================================
    # GOVERNED-PHASE METRICS
    # ============================================================

    baseline_twin_governed = (
        calculate_governed_phase_metrics(
            baseline_twin
        )
    )

    baseline_true_governed = (
        calculate_governed_phase_metrics(
            baseline_true
        )
    )

    proposed_twin_governed = (
        calculate_governed_phase_metrics(
            proposed_twin
        )
    )

    proposed_true_governed = (
        calculate_governed_phase_metrics(
            proposed_true
        )
    )

    print(
        "\n=== GOVERNED ACCELERATION PHASE ==="
    )

    print(
        f"Phase: 0 -> "
        f"{GOVERNOR_COMPLETION_TIME:.6f} s"
    )

    print(
        "\nThe governed-phase error is measured against"
        "\nthe moving reference and is therefore descriptive."
    )

    print(
        "\n09A true plant governed-phase RMS error:"
        f" {baseline_true_governed['rms_error']:.6f} rad/s"
    )

    print(
        "09B-v3 true plant governed-phase RMS error:"
        f" {proposed_true_governed['rms_error']:.6f} rad/s"
    )

    # ============================================================
    # PHASE-MATCHED EVALUATION
    # ============================================================

    evaluation_start = GOVERNOR_COMPLETION_TIME

    baseline_true_post = calculate_metrics(
        baseline_true,
        evaluation_start=evaluation_start,
    )

    proposed_true_post = calculate_metrics(
        proposed_true,
        evaluation_start=evaluation_start,
    )

    baseline_twin_post = calculate_metrics(
        baseline_twin,
        evaluation_start=evaluation_start,
    )

    proposed_twin_post = calculate_metrics(
        proposed_twin,
        evaluation_start=evaluation_start,
    )

    print(
        "\n"
        + "=" * 72
    )

    print(
        "=== PHASE-MATCHED EVALUATION ==="
    )

    print(
        "=" * 72
    )

    print(
        "\nEvaluation window:"
        f"\nt >= {evaluation_start:.6f} s"
    )

    print(
        "\nIMPORTANT:"
        "\nBoth 09A and 09B-v3 are evaluated over"
        "\nthe exact same post-governor time window."
        "\nThis corrects the comparison ambiguity from v2."
    )

    print_metrics(
        "09A BASELINE — TRUE PLANT — PHASE MATCHED",
        baseline_true_post,
    )

    print_metrics(
        "09B-v3 PROPOSED — TRUE PLANT — PHASE MATCHED",
        proposed_true_post,
    )

    # ============================================================
    # PRIMARY PHASE-MATCHED COMPARISON
    # ============================================================

    print_comparison(
        baseline_true_post,
        proposed_true_post,
        "09A baseline",
        "09B-v3",
        "PRIMARY PHASE-MATCHED TRUE-PLANT COMPARISON",
    )

    # ============================================================
    # DIGITAL-TWIN PHASE-MATCHED COMPARISON
    # ============================================================

    print_comparison(
        baseline_twin_post,
        proposed_twin_post,
        "09A twin",
        "09B-v3 twin",
        "PHASE-MATCHED DIGITAL-TWIN COMPARISON",
    )

    # ============================================================
    # TWIN -> TRUE TRANSFER
    # ============================================================

    def transfer_metrics(
        twin,
        true,
    ):
        time = twin["time"]

        mask = time >= evaluation_start

        speed_difference = (
            twin["speed"][mask]
            - true["speed"][mask]
        )

        current_difference = (
            twin["current"][mask]
            - true["current"][mask]
        )

        voltage_difference = (
            twin["voltage"][mask]
            - true["voltage"][mask]
        )

        return {
            "speed_rmse": np.sqrt(
                np.mean(
                    speed_difference ** 2
                )
            ),
            "speed_max": np.max(
                np.abs(speed_difference)
            ),
            "current_rmse": np.sqrt(
                np.mean(
                    current_difference ** 2
                )
            ),
            "current_max": np.max(
                np.abs(current_difference)
            ),
            "voltage_rmse": np.sqrt(
                np.mean(
                    voltage_difference ** 2
                )
            ),
            "voltage_max": np.max(
                np.abs(voltage_difference)
            ),
        }

    baseline_transfer = transfer_metrics(
        baseline_twin,
        baseline_true,
    )

    proposed_transfer = transfer_metrics(
        proposed_twin,
        proposed_true,
    )

    print(
        "\n=== PHASE-MATCHED DIGITAL-TWIN -> TRUE-PLANT TRANSFER ==="
    )

    print(
        "\n09A baseline:"
    )

    print(
        f"Speed transfer RMSE:       "
        f"{baseline_transfer['speed_rmse']:.6f} rad/s"
    )

    print(
        f"Maximum speed difference:  "
        f"{baseline_transfer['speed_max']:.6f} rad/s"
    )

    print(
        f"Current transfer RMSE:     "
        f"{baseline_transfer['current_rmse']:.6f} A"
    )

    print(
        f"Maximum current difference:"
        f" {baseline_transfer['current_max']:.6f} A"
    )

    print(
        f"Voltage transfer RMSE:     "
        f"{baseline_transfer['voltage_rmse']:.6f} V"
    )

    print(
        f"Maximum voltage difference:"
        f" {baseline_transfer['voltage_max']:.6f} V"
    )

    print(
        "\n09B-v3:"
    )

    print(
        f"Speed transfer RMSE:       "
        f"{proposed_transfer['speed_rmse']:.6f} rad/s"
    )

    print(
        f"Maximum speed difference:  "
        f"{proposed_transfer['speed_max']:.6f} rad/s"
    )

    print(
        f"Current transfer RMSE:     "
        f"{proposed_transfer['current_rmse']:.6f} A"
    )

    print(
        f"Maximum current difference:"
        f" {proposed_transfer['current_max']:.6f} A"
    )

    print(
        f"Voltage transfer RMSE:     "
        f"{proposed_transfer['voltage_rmse']:.6f} V"
    )

    print(
        f"Maximum voltage difference:"
        f" {proposed_transfer['voltage_max']:.6f} V"
    )

    # ============================================================
    # TRANSFER IMPROVEMENT
    # ============================================================

    print(
        "\n=== PHASE-MATCHED TWIN-TO-TRUE TRANSFER CHANGE ==="
    )

    # print(
    #     f"Speed transfer RMSE:"
    #     f" {percentage_improvement ("
    #     f"{baseline_transfer['speed_rmse']},"
    #     f"{proposed_transfer['speed_rmse']}"
    #     f"):.3f}%"
    # )

    # print(
    #     f"Maximum speed difference:"
    #     f" {percentage_improvement("
    #     f"{baseline_transfer['speed_max']},"
    #     f"{proposed_transfer['speed_max']}"
    #     f"):.3f}%"
    # )

    # print(
    #     f"Current transfer RMSE:"
    #     f" {percentage_improvement("
    #     f"{baseline_transfer['current_rmse']},"
    #     f"{proposed_transfer['current_rmse']}"
    #     f"):.3f}%"
    # )

    # print(
    #     f"Voltage transfer RMSE:"
    #     f" {percentage_improvement("
    #     f"{baseline_transfer['voltage_rmse']},"
    #     f"{proposed_transfer['voltage_rmse']}"
    #     f"):.3f}%"
    # )
    speed_transfer_improvement = percentage_improvement(
        baseline_transfer["speed_rmse"],
        proposed_transfer["speed_rmse"],
    )

    speed_max_improvement = percentage_improvement(
        baseline_transfer["speed_max"],
        proposed_transfer["speed_max"],
    )

    current_transfer_improvement = percentage_improvement(
        baseline_transfer["current_rmse"],
        proposed_transfer["current_rmse"],
    )

    voltage_transfer_improvement = percentage_improvement(
        baseline_transfer["voltage_rmse"],
        proposed_transfer["voltage_rmse"],
    )

    print(
        f"Speed transfer RMSE:"
        f" {speed_transfer_improvement:.3f}%"
    )

    print(
        f"Maximum speed difference:"
        f" {speed_max_improvement:.3f}%"
    )

    print(
        f"Current transfer RMSE:"
        f" {current_transfer_improvement:.3f}%"
    )

    print(
        f"Voltage transfer RMSE:"
        f" {voltage_transfer_improvement:.3f}%"
    )

    # ============================================================
    # SCIENTIFIC ASSESSMENT
    # ============================================================

    overshoot_improvement = percentage_improvement(
        baseline_true_full["overshoot"],
        proposed_true_full["overshoot"],
    )

    settling_improvement = percentage_improvement(
        baseline_true_full["settling_time"],
        proposed_true_full["settling_time"],
    )

    saturation_improvement = percentage_improvement(
        baseline_true_full["saturation_time"],
        proposed_true_full["saturation_time"],
    )

    phase_rms_improvement = percentage_improvement(
        baseline_true_post["rms_error"],
        proposed_true_post["rms_error"],
    )

    phase_iae_improvement = percentage_improvement(
        baseline_true_post["iae"],
        proposed_true_post["iae"],
    )

    phase_ise_improvement = percentage_improvement(
        baseline_true_post["ise"],
        proposed_true_post["ise"],
    )

    print(
        "\n"
        + "=" * 72
    )

    print(
        "=== 09B-v3 SCIENTIFIC ASSESSMENT ==="
    )

    print(
        "=" * 72
    )

    print(
        "\nFull-horizon final tracking: "
        "SUCCESS"
    )

    print(
        "Transient performance: "
        f"{'IMPROVED' if overshoot_improvement > 0 and settling_improvement > 0 else 'NOT IMPROVED'}"
    )

    print(
        "Actuator constraint handling: "
        f"{'IMPROVED' if saturation_improvement > 0 else 'NOT IMPROVED'}"
    )

    print(
        "Phase-matched RMS tracking: "
        f"{'IMPROVED' if phase_rms_improvement > 0 else 'NOT IMPROVED'}"
    )

    print(
        "Phase-matched IAE: "
        f"{'IMPROVED' if phase_iae_improvement > 0 else 'NOT IMPROVED'}"
    )

    print(
        "Phase-matched ISE: "
        f"{'IMPROVED' if phase_ise_improvement > 0 else 'NOT IMPROVED'}"
    )

    # ============================================================
    # INTERPRETATION
    # ============================================================

    print(
        "\n=== SCIENTIFIC INTERPRETATION ==="
    )

    print(
        """
09A applies the full 30 rad/s reference immediately.
The actuator therefore spends a substantial portion of
the initial transient at its 12 V limit.

09B-v3 retains the exact same state-feedback gains but
introduces a rate-limited reference command.

The governor deliberately changes the commanded trajectory
during the initial acceleration phase. Consequently, whole-
experiment RMS tracking error is not interpreted as a
standalone measure of controller quality.

The principal fair comparison is therefore performed over
the identical post-governor interval:

    t >= 1.666667 s

Both 09A and 09B-v3 are evaluated over this same interval.

This phase-matched evaluation separates:

    1. deliberate reference shaping during acceleration,
    2. subsequent tracking of the final 30 rad/s target,
    3. actuator constraint handling,
    4. digital-twin-to-true-plant transfer fidelity.

No true-plant information is used by the governor.
No parameter identification is repeated.
No controller gain retuning is performed.
"""
    )

    # ============================================================
    # PLOT
    # ============================================================

    create_plot(
        baseline_true,
        proposed_true,
    )

    print(
        f"\nSaved plot:"
        f"\n{PLOT_PATH}"
    )

    print(
        "\n"
        + "=" * 72
    )

    print(
        "=== EXPERIMENT 09B-v3 COMPLETE ==="
    )

    print(
        "=" * 72
    )


if __name__ == "__main__":
    main()

