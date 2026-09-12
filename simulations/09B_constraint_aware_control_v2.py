"""
Experiment 09B-v2: Constraint-Aware Digital-Twin Control
==========================================================

Purpose
-------
Evaluate a reference-governed integral state-feedback controller designed
from the frozen 08A identified DC-motor model.

09A baseline
------------
Integral state feedback with a fixed 30 rad/s reference.

09B proposed controller
-----------------------
The SAME integral state-feedback controller gains are retained.

A rate-limited reference governor is introduced to reduce actuator
saturation during acceleration.

Scientific protocol
-------------------
- 08A identified parameters are frozen.
- True motor parameters are frozen.
- Controller gains are designed from the identified model only.
- No true-plant parameter fitting is performed.
- No controller retuning is performed on the true plant.
- Actuator voltage remains physically limited to 0--12 V.
- Both 09A and 09B are evaluated on:
      1. the identified digital twin
      2. the true motor

v2 evaluation improvements
--------------------------
The total 0--10 s RMS tracking error can be misleading because 09B
deliberately changes the commanded reference during acceleration.

Therefore v2 reports:

    - total RMS tracking error
    - governed-phase RMS tracking error
    - post-governor RMS tracking error
    - IAE
    - ISE
    - overshoot
    - settling time
    - saturation duration
    - saturation percentage
    - RMS voltage
    - voltage energy / effort
    - current metrics
    - twin-to-true transfer fidelity

Research question
-----------------
Can constraint-aware reference governance reduce saturation-driven
transient degradation while preserving accurate controller transfer
from the identified digital twin to the true motor?

Outputs
-------
results/09B_constraint_aware_control_v2.png
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import place_poles

from src.motor import DCMotor, DCMotorParameters


# ============================================================================
# CONFIGURATION
# ============================================================================

REFERENCE = 30.0
LOAD_TORQUE = 0.05

SIMULATION_TIME = 10.0
DT = 0.002

VOLTAGE_MIN = 0.0
VOLTAGE_MAX = 12.0

DESIRED_POLES = np.array([-4.0, -5.0, -6.0])

# Reference governor rate.
#
# The reference begins at 0 rad/s and moves toward 30 rad/s at this rate.
#
# 30 / 18 = 1.6667 s to reach the final command.
GOVERNOR_RATE = 18.0  # rad/s^2


# ============================================================================
# OUTPUT
# ============================================================================

OUTPUT_DIR = Path("results")

OUTPUT_FILE = (
    OUTPUT_DIR
    / "09B_constraint_aware_control_v2.png"
)


# ============================================================================
# FROZEN 08A IDENTIFIED PARAMETERS
# ============================================================================

IDENTIFIED_PARAMETERS = DCMotorParameters(
    R=2.011540,
    L=0.481997,
    Kt=0.105805,
    Ke=0.098059,
    J=0.021550,
    b=0.010715,
)


# ============================================================================
# FROZEN TRUE PARAMETERS
# ============================================================================

TRUE_PARAMETERS = DCMotorParameters(
    R=2.0,
    L=0.5,
    Kt=0.1,
    Ke=0.1,
    J=0.02,
    b=0.01,
)


# ============================================================================
# MODEL HELPERS
# ============================================================================

def get_speed_output_matrix():
    """Return C for y = Cx, where y is angular speed."""
    return np.array([
        [0.0, 1.0]
    ])


def augmented_matrices(motor):
    """
    Construct the integral-augmented state-space model.

    Augmented state:

        xa = [current, speed, integral_error]

    Integral state:

        z_dot = reference - speed

    Therefore:

        A_aug = [[A, 0],
                  [-C, 0]]

        B_aug = [[B],
                  [0]]

        E_aug = [[E],
                  [0]]
    """

    A, B, E = motor.matrices()

    C = get_speed_output_matrix()

    A_aug = np.block([
        [
            A,
            np.zeros((2, 1))
        ],
        [
            -C,
            np.zeros((1, 1))
        ]
    ])

    B_aug = np.vstack([
        B,
        np.zeros((1, 1))
    ])

    E_aug = np.vstack([
        E,
        np.zeros((1, 1))
    ])

    return A_aug, B_aug, E_aug


def design_integral_controller(motor):
    """
    Design integral state-feedback controller.

    The controller is synthesized using the frozen identified model.
    """

    A_aug, B_aug, _ = augmented_matrices(motor)

    controllability = np.column_stack([
        B_aug,
        A_aug @ B_aug,
        A_aug @ A_aug @ B_aug,
    ])

    rank = np.linalg.matrix_rank(
        controllability
    )

    if rank != 3:
        raise RuntimeError(
            f"Augmented system is not controllable: "
            f"rank={rank}/3"
        )

    result = place_poles(
        A_aug,
        B_aug,
        DESIRED_POLES
    )

    K_aug = result.gain_matrix

    closed_loop_poles = np.linalg.eigvals(
        A_aug - B_aug @ K_aug
    )

    return (
        K_aug,
        closed_loop_poles,
        rank,
    )


# ============================================================================
# REFERENCE GOVERNOR
# ============================================================================

def reference_governor(
    current_reference,
    target_reference,
    dt,
):
    """
    Rate-limit the commanded reference.

    The final target remains exactly 30 rad/s.

    This is intentionally independent of the true plant state.
    """

    maximum_step = (
        GOVERNOR_RATE * dt
    )

    error = (
        target_reference
        - current_reference
    )

    if abs(error) <= maximum_step:
        return target_reference

    return (
        current_reference
        + np.sign(error)
        * maximum_step
    )


# ============================================================================
# CONTROLLER
# ============================================================================

def controller_voltage(
    K_aug,
    current,
    speed,
    integral_error,
):
    """
    Integral state-feedback control law.

        u_raw = -K_aug * x_aug

    The physical actuator constraint is then applied.
    """

    state = np.array([
        current,
        speed,
        integral_error,
    ])

    raw_voltage = float(
        -(K_aug @ state)[0]
    )

    saturated_voltage = np.clip(
        raw_voltage,
        VOLTAGE_MIN,
        VOLTAGE_MAX,
    )

    return (
        raw_voltage,
        saturated_voltage,
    )


# ============================================================================
# SIMULATION
# ============================================================================

def simulate(
    motor,
    K_aug,
    governed=False,
):
    """
    Simulate the motor.

    governed=False
        09A baseline.

    governed=True
        09B reference-governed controller.
    """

    time = np.arange(
        0.0,
        SIMULATION_TIME + DT,
        DT,
    )

    n = len(time)

    current = np.zeros(n)
    speed = np.zeros(n)

    integral_error = np.zeros(n)

    raw_voltage = np.zeros(n)
    voltage = np.zeros(n)

    command_reference = np.zeros(n)

    x = np.zeros(2)

    z = 0.0

    r_governed = 0.0

    governor_complete_time = np.nan

    for k in range(n):

        # --------------------------------------------------------------
        # Reference
        # --------------------------------------------------------------

        if governed:

            r_governed = reference_governor(
                r_governed,
                REFERENCE,
                DT,
            )

            r = r_governed

            if (
                np.isnan(governor_complete_time)
                and np.isclose(
                    r,
                    REFERENCE,
                    atol=1e-12,
                )
            ):
                governor_complete_time = time[k]

        else:

            r = REFERENCE

        command_reference[k] = r

        # --------------------------------------------------------------
        # Store state
        # --------------------------------------------------------------

        current[k] = x[0]
        speed[k] = x[1]
        integral_error[k] = z

        # --------------------------------------------------------------
        # Control
        # --------------------------------------------------------------

        raw_u, saturated_u = controller_voltage(
            K_aug,
            x[0],
            x[1],
            z,
        )

        raw_voltage[k] = raw_u
        voltage[k] = saturated_u

        # --------------------------------------------------------------
        # Motor dynamics
        # --------------------------------------------------------------

        dx = motor.derivatives(
            x,
            saturated_u,
            LOAD_TORQUE,
        )

        # --------------------------------------------------------------
        # Integral state
        # --------------------------------------------------------------

        dz = r - x[1]

        # --------------------------------------------------------------
        # Euler integration
        # --------------------------------------------------------------

        if k < n - 1:

            x = x + DT * dx

            z = z + DT * dz

    return {
        "time": time,
        "current": current,
        "speed": speed,
        "integral_error": integral_error,
        "raw_voltage": raw_voltage,
        "voltage": voltage,
        "reference": command_reference,
        "governor_complete_time": governor_complete_time,
    }


# ============================================================================
# METRICS
# ============================================================================

def calculate_metrics(result):
    """
    Calculate comprehensive closed-loop metrics.
    """

    time = result["time"]
    speed = result["speed"]
    voltage = result["voltage"]
    current = result["current"]
    reference = result["reference"]

    dt = float(
        time[1] - time[0]
    )

    # ------------------------------------------------------------------
    # Actual target tracking
    # ------------------------------------------------------------------

    tracking_error = (
        REFERENCE - speed
    )

    absolute_error = np.abs(
        tracking_error
    )

    squared_error = (
        tracking_error ** 2
    )

    # ------------------------------------------------------------------
    # Total metrics
    # ------------------------------------------------------------------

    final_speed = float(
        speed[-1]
    )

    final_error = float(
        REFERENCE - final_speed
    )

    rms_tracking_error = float(
        np.sqrt(
            np.mean(
                squared_error
            )
        )
    )

    iae = float(
        np.sum(
            absolute_error
        ) * dt
    )

    ise = float(
        np.sum(
            squared_error
        ) * dt
    )

    maximum_abs_error = float(
        np.max(
            absolute_error
        )
    )

    # ------------------------------------------------------------------
    # Peak and overshoot
    # ------------------------------------------------------------------

    peak_speed = float(
        np.max(speed)
    )

    overshoot = max(
        0.0,
        (
            peak_speed
            - REFERENCE
        )
        / REFERENCE
        * 100.0,
    )

    # ------------------------------------------------------------------
    # Settling time
    # ------------------------------------------------------------------

    settling_band = (
        0.02 * REFERENCE
    )

    outside = np.where(
        absolute_error
        > settling_band
    )[0]

    if len(outside) == 0:

        settling_time = 0.0

    elif outside[-1] >= len(time) - 1:

        settling_time = np.nan

    else:

        settling_time = float(
            time[
                outside[-1] + 1
            ]
        )

    # ------------------------------------------------------------------
    # Governor completion
    # ------------------------------------------------------------------

    governor_complete_time = (
        result[
            "governor_complete_time"
        ]
    )

    if np.isnan(
        governor_complete_time
    ):

        post_governor_mask = (
            np.ones_like(
                time,
                dtype=bool,
            )
        )

        governor_phase_mask = (
            np.zeros_like(
                time,
                dtype=bool,
            )
        )

    else:

        governor_phase_mask = (
            time
            <= governor_complete_time
        )

        post_governor_mask = (
            time
            > governor_complete_time
        )

    # ------------------------------------------------------------------
    # Phase-specific tracking
    # ------------------------------------------------------------------

    if np.any(
        governor_phase_mask
    ):

        governed_rms_error = float(
            np.sqrt(
                np.mean(
                    squared_error[
                        governor_phase_mask
                    ]
                )
            )
        )

    else:

        governed_rms_error = np.nan

    if np.any(
        post_governor_mask
    ):

        post_governor_rms_error = float(
            np.sqrt(
                np.mean(
                    squared_error[
                        post_governor_mask
                    ]
                )
            )
        )

        post_governor_iae = float(
            np.sum(
                absolute_error[
                    post_governor_mask
                ]
            )
            * dt
        )

        post_governor_ise = float(
            np.sum(
                squared_error[
                    post_governor_mask
                ]
            )
            * dt
        )

    else:

        post_governor_rms_error = np.nan
        post_governor_iae = np.nan
        post_governor_ise = np.nan

    # ------------------------------------------------------------------
    # Saturation
    # ------------------------------------------------------------------

    saturation_mask = (
        np.isclose(
            voltage,
            VOLTAGE_MIN,
        )
        |
        np.isclose(
            voltage,
            VOLTAGE_MAX,
        )
    )

    saturation_time = float(
        np.sum(
            saturation_mask
        ) * dt
    )

    saturation_percentage = (
        saturation_time
        / (time[-1] - time[0])
        * 100.0
    )

    # ------------------------------------------------------------------
    # Control effort
    # ------------------------------------------------------------------

    voltage_squared_integral = float(
        np.sum(
            voltage ** 2
        ) * dt
    )

    voltage_absolute_integral = float(
        np.sum(
            np.abs(voltage)
        ) * dt
    )

    # ------------------------------------------------------------------
    # Electrical metrics
    # ------------------------------------------------------------------

    maximum_voltage = float(
        np.max(voltage)
    )

    rms_voltage = float(
        np.sqrt(
            np.mean(
                voltage ** 2
            )
        )
    )

    maximum_current = float(
        np.max(current)
    )

    rms_current = float(
        np.sqrt(
            np.mean(
                current ** 2
            )
        )
    )

    return {
        "final_speed": final_speed,
        "final_error": final_error,
        "rms_tracking_error": rms_tracking_error,
        "iae": iae,
        "ise": ise,
        "maximum_abs_error": maximum_abs_error,
        "peak_speed": peak_speed,
        "overshoot": overshoot,
        "settling_time": settling_time,
        "governed_rms_error": governed_rms_error,
        "post_governor_rms_error": post_governor_rms_error,
        "post_governor_iae": post_governor_iae,
        "post_governor_ise": post_governor_ise,
        "maximum_voltage": maximum_voltage,
        "rms_voltage": rms_voltage,
        "maximum_current": maximum_current,
        "rms_current": rms_current,
        "saturation_time": saturation_time,
        "saturation_percentage": saturation_percentage,
        "voltage_squared_integral":
            voltage_squared_integral,
        "voltage_absolute_integral":
            voltage_absolute_integral,
        "governor_complete_time":
            governor_complete_time,
    }


# ============================================================================
# TRANSFER METRICS
# ============================================================================

def calculate_transfer_metrics(
    twin,
    true,
):
    """
    Compare digital-twin and true-plant responses.
    """

    speed_difference = (
        twin["speed"]
        - true["speed"]
    )

    current_difference = (
        twin["current"]
        - true["current"]
    )

    voltage_difference = (
        twin["voltage"]
        - true["voltage"]
    )

    return {
        "speed_rmse": float(
            np.sqrt(
                np.mean(
                    speed_difference ** 2
                )
            )
        ),
        "speed_max_difference": float(
            np.max(
                np.abs(
                    speed_difference
                )
            )
        ),
        "current_rmse": float(
            np.sqrt(
                np.mean(
                    current_difference ** 2
                )
            )
        ),
        "current_max_difference": float(
            np.max(
                np.abs(
                    current_difference
                )
            )
        ),
        "voltage_rmse": float(
            np.sqrt(
                np.mean(
                    voltage_difference ** 2
                )
            )
        ),
        "voltage_max_difference": float(
            np.max(
                np.abs(
                    voltage_difference
                )
            )
        ),
    }


# ============================================================================
# PRINTING
# ============================================================================

def print_metrics(
    title,
    metrics,
):
    """Print comprehensive metrics."""

    print(
        f"\n=== {title} ==="
    )

    print(
        f"Final speed:             "
        f"{metrics['final_speed']:.6f} rad/s"
    )

    print(
        f"Final error:             "
        f"{metrics['final_error']:.6f} rad/s"
    )

    print(
        f"Total RMS tracking error:"
        f" {metrics['rms_tracking_error']:.6f} rad/s"
    )

    print(
        f"IAE:                     "
        f"{metrics['iae']:.6f}"
    )

    print(
        f"ISE:                     "
        f"{metrics['ise']:.6f}"
    )

    print(
        f"Maximum abs error:       "
        f"{metrics['maximum_abs_error']:.6f} rad/s"
    )

    print(
        f"Peak speed:              "
        f"{metrics['peak_speed']:.6f} rad/s"
    )

    print(
        f"Overshoot:               "
        f"{metrics['overshoot']:.4f} %"
    )

    print(
        f"Settling time:            "
        f"{metrics['settling_time']:.6f} s"
    )

    print(
        f"Governed-phase RMS error:"
        f" {metrics['governed_rms_error']:.6f} rad/s"
    )

    print(
        f"Post-governor RMS error: "
        f"{metrics['post_governor_rms_error']:.6f} rad/s"
    )

    print(
        f"Post-governor IAE:       "
        f"{metrics['post_governor_iae']:.6f}"
    )

    print(
        f"Post-governor ISE:       "
        f"{metrics['post_governor_ise']:.6f}"
    )

    print(
        f"Maximum voltage:         "
        f"{metrics['maximum_voltage']:.6f} V"
    )

    print(
        f"RMS voltage:             "
        f"{metrics['rms_voltage']:.6f} V"
    )

    print(
        f"Voltage squared integral:"
        f" {metrics['voltage_squared_integral']:.6f}"
    )

    print(
        f"Maximum current:         "
        f"{metrics['maximum_current']:.6f} A"
    )

    print(
        f"RMS current:             "
        f"{metrics['rms_current']:.6f} A"
    )

    print(
        f"Time at saturation:      "
        f"{metrics['saturation_time']:.6f} s"
    )

    print(
        f"Saturation percentage:   "
        f"{metrics['saturation_percentage']:.3f} %"
    )

    if np.isnan(
        metrics["governor_complete_time"]
    ):

        print(
            "Governor completion time:"
            " N/A"
        )

    else:

        print(
            f"Governor completion time:"
            f" {metrics['governor_complete_time']:.6f} s"
        )


def print_transfer_metrics(
    title,
    metrics,
):
    """Print twin-to-true transfer metrics."""

    print(
        f"\n=== {title} ==="
    )

    print(
        f"Speed transfer RMSE:       "
        f"{metrics['speed_rmse']:.6f} rad/s"
    )

    print(
        f"Maximum speed difference:  "
        f"{metrics['speed_max_difference']:.6f} rad/s"
    )

    print(
        f"Current transfer RMSE:     "
        f"{metrics['current_rmse']:.6f} A"
    )

    print(
        f"Maximum current difference:"
        f" {metrics['current_max_difference']:.6f} A"
    )

    print(
        f"Voltage transfer RMSE:     "
        f"{metrics['voltage_rmse']:.6f} V"
    )

    print(
        f"Maximum voltage difference:"
        f" {metrics['voltage_max_difference']:.6f} V"
    )


# ============================================================================
# IMPROVEMENT
# ============================================================================

def percentage_reduction(
    baseline,
    proposed,
):
    """
    Percentage reduction from baseline to proposed.

    Positive = reduction/improvement.
    Negative = increase/degradation.
    """

    if (
        baseline is None
        or np.isnan(baseline)
        or abs(baseline) < 1e-12
    ):

        return np.nan

    return (
        100.0
        * (baseline - proposed)
        / baseline
    )


def print_improvement(
    label,
    baseline,
    proposed,
):
    """Print percentage change."""

    change = percentage_reduction(
        baseline,
        proposed,
    )

    print(
        f"{label:<30}"
        f"{change:>10.3f}%"
    )


# ============================================================================
# PLOTTING
# ============================================================================

def create_plot(
    baseline_twin,
    baseline_true,
    proposed_twin,
    proposed_true,
):
    """Create 09B-v2 comparison figure."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(11, 14),
        sharex=True,
    )

    time = baseline_twin[
        "time"
    ]

    # ------------------------------------------------------------------
    # Speed
    # ------------------------------------------------------------------

    axes[0].plot(
        time,
        baseline_twin["speed"],
        label="09A Twin",
    )

    axes[0].plot(
        time,
        baseline_true["speed"],
        "--",
        label="09A True Plant",
    )

    axes[0].plot(
        time,
        proposed_twin["speed"],
        label="09B Twin",
    )

    axes[0].plot(
        time,
        proposed_true["speed"],
        "--",
        label="09B True Plant",
    )

    axes[0].plot(
        time,
        np.full_like(
            time,
            REFERENCE,
        ),
        ":",
        label="Target",
    )

    axes[0].set_ylabel(
        "Speed [rad/s]"
    )

    axes[0].set_title(
        "09B-v2 Constraint-Aware Digital-Twin Control"
    )

    axes[0].grid(True)
    axes[0].legend()

    # ------------------------------------------------------------------
    # Voltage
    # ------------------------------------------------------------------

    axes[1].plot(
        time,
        baseline_true["voltage"],
        label="09A True Plant",
    )

    axes[1].plot(
        time,
        proposed_true["voltage"],
        label="09B True Plant",
    )

    axes[1].axhline(
        VOLTAGE_MAX,
        linestyle=":",
        label="12 V limit",
    )

    axes[1].set_ylabel(
        "Voltage [V]"
    )

    axes[1].grid(True)
    axes[1].legend()

    # ------------------------------------------------------------------
    # Governed reference
    # ------------------------------------------------------------------

    axes[2].plot(
        time,
        proposed_twin["reference"],
        label="09B governed reference",
    )

    axes[2].plot(
        time,
        np.full_like(
            time,
            REFERENCE,
        ),
        ":",
        label="Final target",
    )

    axes[2].set_ylabel(
        "Reference [rad/s]"
    )

    axes[2].grid(True)
    axes[2].legend()

    # ------------------------------------------------------------------
    # Twin-to-true speed deviation
    # ------------------------------------------------------------------

    speed_difference_09a = (
        baseline_twin["speed"]
        - baseline_true["speed"]
    )

    speed_difference_09b = (
        proposed_twin["speed"]
        - proposed_true["speed"]
    )

    axes[3].plot(
        time,
        speed_difference_09a,
        label="09A Twin − True",
    )

    axes[3].plot(
        time,
        speed_difference_09b,
        label="09B Twin − True",
    )

    axes[3].axhline(
        0.0,
        linestyle=":",
    )

    axes[3].set_ylabel(
        "Speed difference [rad/s]"
    )

    axes[3].set_xlabel(
        "Time [s]"
    )

    axes[3].grid(True)
    axes[3].legend()

    fig.tight_layout()

    fig.savefig(
        OUTPUT_FILE,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================================
# MAIN
# ============================================================================

def main():

    print(
        "=" * 72
    )

    print(
        "=== EXPERIMENT 09B-v2: "
        "CONSTRAINT-AWARE DIGITAL-TWIN CONTROL ==="
    )

    print(
        "=" * 72
    )

    print(
        "\nPurpose:"
        "\nEvaluate reference governance using the frozen 08A model."
    )

    print(
        "\nNo parameter optimization is performed."
        "\nNo controller retuning is performed on the true plant."
    )

    # ------------------------------------------------------------------
    # Frozen parameters
    # ------------------------------------------------------------------

    print(
        "\n=== FROZEN MODEL PARAMETERS ==="
    )

    print(
        "\nParameter           True        08A ID       Error %"
        "\n----------------------------------------------------"
    )

    true_values = {
        "R": TRUE_PARAMETERS.R,
        "L": TRUE_PARAMETERS.L,
        "Kt": TRUE_PARAMETERS.Kt,
        "Ke": TRUE_PARAMETERS.Ke,
        "J": TRUE_PARAMETERS.J,
        "b": TRUE_PARAMETERS.b,
    }

    identified_values = {
        "R": IDENTIFIED_PARAMETERS.R,
        "L": IDENTIFIED_PARAMETERS.L,
        "Kt": IDENTIFIED_PARAMETERS.Kt,
        "Ke": IDENTIFIED_PARAMETERS.Ke,
        "J": IDENTIFIED_PARAMETERS.J,
        "b": IDENTIFIED_PARAMETERS.b,
    }

    for name in [
        "R",
        "L",
        "Kt",
        "Ke",
        "J",
        "b",
    ]:

        error = (
            abs(
                identified_values[name]
                - true_values[name]
            )
            / abs(
                true_values[name]
            )
            * 100.0
        )

        print(
            f"{name:<8}"
            f"{true_values[name]:>14.6f}"
            f"{identified_values[name]:>14.6f}"
            f"{error:>14.3f}%"
        )

    # ------------------------------------------------------------------
    # Motors
    # ------------------------------------------------------------------

    true_motor = DCMotor(
        TRUE_PARAMETERS
    )

    identified_motor = DCMotor(
        IDENTIFIED_PARAMETERS
    )

    # ------------------------------------------------------------------
    # Controller
    # ------------------------------------------------------------------

    (
        K_aug,
        closed_loop_poles,
        controllability_rank,
    ) = design_integral_controller(
        identified_motor
    )

    print(
        "\n=== 09B-v2 CONTROLLER DESIGN ==="
    )

    print(
        "\nAugmented state:"
        "\n[current, speed, integral_error]"
    )

    print(
        "\nDesired closed-loop poles:"
    )

    print(
        DESIRED_POLES
    )

    print(
        "\nAugmented controllability rank:"
    )

    print(
        f"{controllability_rank}/3"
    )

    print(
        "\nK_aug ="
    )

    print(
        K_aug
    )

    print(
        "\nClosed-loop poles from identified model:"
    )

    print(
        closed_loop_poles
    )

    print(
        "\nReference governor rate:"
    )

    print(
        f"{GOVERNOR_RATE:.3f} rad/s^2"
    )

    print(
        "\nExpected governor completion time:"
    )

    print(
        f"{REFERENCE / GOVERNOR_RATE:.6f} s"
    )

    # ------------------------------------------------------------------
    # 09A
    # ------------------------------------------------------------------

    baseline_twin = simulate(
        identified_motor,
        K_aug,
        governed=False,
    )

    baseline_true = simulate(
        true_motor,
        K_aug,
        governed=False,
    )

    # ------------------------------------------------------------------
    # 09B
    # ------------------------------------------------------------------

    proposed_twin = simulate(
        identified_motor,
        K_aug,
        governed=True,
    )

    proposed_true = simulate(
        true_motor,
        K_aug,
        governed=True,
    )

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    baseline_twin_metrics = calculate_metrics(
        baseline_twin
    )

    baseline_true_metrics = calculate_metrics(
        baseline_true
    )

    proposed_twin_metrics = calculate_metrics(
        proposed_twin
    )

    proposed_true_metrics = calculate_metrics(
        proposed_true
    )

    # ------------------------------------------------------------------
    # Print 09A
    # ------------------------------------------------------------------

    print_metrics(
        "09A BASELINE — IDENTIFIED DIGITAL TWIN",
        baseline_twin_metrics,
    )

    print_metrics(
        "09A BASELINE — TRUE PLANT",
        baseline_true_metrics,
    )

    # ------------------------------------------------------------------
    # Print 09B
    # ------------------------------------------------------------------

    print_metrics(
        "09B-v2 PROPOSED — IDENTIFIED DIGITAL TWIN",
        proposed_twin_metrics,
    )

    print_metrics(
        "09B-v2 PROPOSED — TRUE PLANT",
        proposed_true_metrics,
    )

    # ------------------------------------------------------------------
    # Transfer
    # ------------------------------------------------------------------

    baseline_transfer = (
        calculate_transfer_metrics(
            baseline_twin,
            baseline_true,
        )
    )

    proposed_transfer = (
        calculate_transfer_metrics(
            proposed_twin,
            proposed_true,
        )
    )

    print_transfer_metrics(
        "09A DIGITAL-TWIN → TRUE-PLANT TRANSFER",
        baseline_transfer,
    )

    print_transfer_metrics(
        "09B-v2 DIGITAL-TWIN → TRUE-PLANT TRANSFER",
        proposed_transfer,
    )

    # ------------------------------------------------------------------
    # Improvement
    # ------------------------------------------------------------------

    print(
        "\n=== 09A → 09B-v2 IMPROVEMENT "
        "ON TRUE PLANT ==="
    )

    print(
        f"{'Metric':<30}"
        f"{'Change':>10}"
    )

    print(
        "-" * 42
    )

    print_improvement(
        "Overshoot",
        baseline_true_metrics[
            "overshoot"
        ],
        proposed_true_metrics[
            "overshoot"
        ],
    )

    print_improvement(
        "Settling time",
        baseline_true_metrics[
            "settling_time"
        ],
        proposed_true_metrics[
            "settling_time"
        ],
    )

    print_improvement(
        "Saturation time",
        baseline_true_metrics[
            "saturation_time"
        ],
        proposed_true_metrics[
            "saturation_time"
        ],
    )

    print_improvement(
        "Total RMS error",
        baseline_true_metrics[
            "rms_tracking_error"
        ],
        proposed_true_metrics[
            "rms_tracking_error"
        ],
    )

    print_improvement(
        "IAE",
        baseline_true_metrics[
            "iae"
        ],
        proposed_true_metrics[
            "iae"
        ],
    )

    print_improvement(
        "ISE",
        baseline_true_metrics[
            "ise"
        ],
        proposed_true_metrics[
            "ise"
        ],
    )

    print_improvement(
        "RMS voltage",
        baseline_true_metrics[
            "rms_voltage"
        ],
        proposed_true_metrics[
            "rms_voltage"
        ],
    )

    print_improvement(
        "RMS current",
        baseline_true_metrics[
            "rms_current"
        ],
        proposed_true_metrics[
            "rms_current"
        ],
    )

    print_improvement(
        "Voltage squared effort",
        baseline_true_metrics[
            "voltage_squared_integral"
        ],
        proposed_true_metrics[
            "voltage_squared_integral"
        ],
    )

    print(
        "\n=== TWIN-TO-TRUE TRANSFER CHANGE ==="
    )

    print_improvement(
        "Speed transfer RMSE",
        baseline_transfer[
            "speed_rmse"
        ],
        proposed_transfer[
            "speed_rmse"
        ],
    )

    print_improvement(
        "Maximum speed difference",
        baseline_transfer[
            "speed_max_difference"
        ],
        proposed_transfer[
            "speed_max_difference"
        ],
    )

    print_improvement(
        "Current transfer RMSE",
        baseline_transfer[
            "current_rmse"
        ],
        proposed_transfer[
            "current_rmse"
        ],
    )

    print_improvement(
        "Voltage transfer RMSE",
        baseline_transfer[
            "voltage_rmse"
        ],
        proposed_transfer[
            "voltage_rmse"
        ],
    )

    # ------------------------------------------------------------------
    # Scientific assessments
    # ------------------------------------------------------------------

    tracking_success = (
        abs(
            proposed_true_metrics[
                "final_error"
            ]
        ) < 0.1
        and not np.isnan(
            proposed_true_metrics[
                "settling_time"
            ]
        )
        and proposed_true_metrics[
            "settling_time"
        ] < 7.0
    )

    transient_improved = (
        proposed_true_metrics[
            "overshoot"
        ]
        <
        baseline_true_metrics[
            "overshoot"
        ]
        and
        proposed_true_metrics[
            "settling_time"
        ]
        <
        baseline_true_metrics[
            "settling_time"
        ]
    )

    constraint_improved = (
        proposed_true_metrics[
            "saturation_time"
        ]
        <
        baseline_true_metrics[
            "saturation_time"
        ]
    )

    transfer_success = (
        proposed_transfer[
            "speed_rmse"
        ] < 0.5
        and
        proposed_transfer[
            "speed_max_difference"
        ] < 1.5
    )

    post_governor_quality = (
        proposed_true_metrics[
            "post_governor_rms_error"
        ]
        <
        baseline_true_metrics[
            "rms_tracking_error"
        ]
    )

    # ------------------------------------------------------------------
    # Assessment
    # ------------------------------------------------------------------

    print(
        "\n=== 09B-v2 SCIENTIFIC ASSESSMENT ==="
    )

    if tracking_success:

        print(
            "Closed-loop final tracking: SUCCESS"
        )

    else:

        print(
            "Closed-loop final tracking: "
            "REQUIRES REVIEW"
        )

    if transient_improved:

        print(
            "Transient performance: IMPROVED"
        )

    else:

        print(
            "Transient performance: "
            "NOT IMPROVED"
        )

    if constraint_improved:

        print(
            "Actuator constraint handling: IMPROVED"
        )

    else:

        print(
            "Actuator constraint handling: "
            "NOT IMPROVED"
        )

    if transfer_success:

        print(
            "Digital-twin → true-plant transfer: SUCCESS"
        )

    else:

        print(
            "Digital-twin → true-plant transfer: "
            "REQUIRES REVIEW"
        )

    if post_governor_quality:

        print(
            "Post-governor tracking quality: "
            "IMPROVED"
        )

    else:

        print(
            "Post-governor tracking quality: "
            "NOT BETTER THAN 09A TOTAL RMS"
        )

    # ------------------------------------------------------------------
    # Interpretation
    # ------------------------------------------------------------------

    print(
        "\n=== SCIENTIFIC INTERPRETATION ==="
    )

    print(
        "\n09A applies the full 30 rad/s reference immediately."
        "\nThe actuator therefore spends a substantial portion of the"
        "\ninitial transient at its 12 V limit."
    )

    print(
        "\n09B-v2 retains the exact same state-feedback gains but"
        "\nintroduces a rate-limited reference command."
    )

    print(
        "\nThe purpose is not to make the total 0--10 s RMS error"
        "\nartificially smaller. Instead, the experiment explicitly"
        "\nseparates the governed acceleration phase from subsequent"
        "\ntracking of the final 30 rad/s target."
    )

    print(
        "\nThe primary constraint-aware metrics are therefore:"
        "\n  * overshoot"
        "\n  * settling time"
        "\n  * saturation duration"
        "\n  * actuator effort"
        "\n  * post-governor tracking"
        "\n  * twin-to-true-plant transfer fidelity"
    )

    print(
        "\nNo true-plant information is used by the governor."
        "\nNo parameter identification is repeated."
        "\nNo controller gain retuning is performed."
    )

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------

    create_plot(
        baseline_twin,
        baseline_true,
        proposed_twin,
        proposed_true,
    )

    print(
        f"\nSaved plot: "
        f"{OUTPUT_FILE.resolve()}"
    )

    print(
        "\n" + "=" * 72
    )

    print(
        "=== EXPERIMENT 09B-v2 COMPLETE ==="
    )

    print(
        "=" * 72
    )


if __name__ == "__main__":
    main()