
"""
Experiment 09A
==============
Digital-Twin-Based Closed-Loop Control

Purpose
-------
Design an integral-augmented state-feedback controller using the
FROZEN 08A identified DC-motor model, then independently evaluate
that controller on:

    1. The identified digital twin
    2. The true DC motor plant

No parameter optimization is performed in this experiment.
No controller retuning is performed on the true plant.

Research question
-----------------
Can a controller designed entirely from the identified digital twin
transfer successfully to the underlying true plant?

Architecture
------------
    08A identified parameters
            |
            v
    Identified state-space model
            |
            v
    Integral-augmented pole placement
            |
            v
       Frozen K_aug
            |
       +----+----+
       |         |
       v         v
   Digital     True
     Twin      Plant

The controller is NOT redesigned using the true plant.

Interpretation
--------------
Two different questions are evaluated separately:

1. Closed-loop tracking performance:
   How well does the controller track the 30 rad/s reference?

2. Digital-twin transfer fidelity:
   How closely does the true plant reproduce the response predicted
   by the identified digital twin?

The second quantity is the primary validation objective of 09A.

Author: Project One - DC Motor Digital Twin & Control Lab
"""

from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.signal import place_poles


# ---------------------------------------------------------------------
# PROJECT PATH
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.motor import DCMotor, DCMotorParameters


# ---------------------------------------------------------------------
# EXPERIMENT CONFIGURATION
# ---------------------------------------------------------------------

SIMULATION_TIME = 10.0
DT = 0.002

REFERENCE_SPEED = 30.0
LOAD_TORQUE = 0.05

VOLTAGE_MIN = 0.0
VOLTAGE_MAX = 12.0

# Desired closed-loop poles.
#
# These poles are used to synthesize the controller from the
# IDENTIFIED digital-twin model.
DESIRED_POLES = np.array([
    -4.0,
    -5.0,
    -6.0,
])


# ---------------------------------------------------------------------
# FROZEN 08A IDENTIFIED PARAMETERS
# ---------------------------------------------------------------------
#
# These values are copied directly from Experiment 08A.
#
# IMPORTANT:
# No optimization or re-identification is performed here.
# ---------------------------------------------------------------------

IDENTIFIED_PARAMETERS = DCMotorParameters(
    R=2.011540,
    L=0.481997,
    Kt=0.105805,
    Ke=0.098059,
    J=0.021550,
    b=0.010715,
)


# ---------------------------------------------------------------------
# TRUE MOTOR PARAMETERS
# ---------------------------------------------------------------------

TRUE_PARAMETERS = DCMotorParameters(
    R=2.0,
    L=0.5,
    Kt=0.1,
    Ke=0.1,
    J=0.02,
    b=0.01,
)


# ---------------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------------

def augment_integral_state(A, B, C):
    """
    Construct the integral-augmented state-space model.

    Original system:

        x_dot = A x + B u

    Output:

        y = C x

    Integral state:

        z_dot = r - y
              = r - C x

    Therefore:

        [x_dot]   [ A   0 ] [x]   [B]u
        [z_dot] = [-C   0 ] [z] + [0]r
    """

    n = A.shape[0]

    A_aug = np.zeros((n + 1, n + 1))
    A_aug[:n, :n] = A
    A_aug[n, :n] = -C

    B_aug = np.zeros((n + 1, 1))
    B_aug[:n, :] = B

    return A_aug, B_aug


def design_integral_state_feedback(motor):
    """
    Design integral-augmented state feedback using the supplied model.

    The supplied model MUST be the identified digital-twin model.

    Control law:

        u = -K_aug x_aug

    where:

        x_aug = [current, speed, integral_error]^T

    and:

        integral_error = integral(reference - speed) dt
    """

    A, B, _ = motor.matrices()

    C = np.array([
        [0.0, 1.0],
    ])

    A_aug, B_aug = augment_integral_state(
        A,
        B,
        C,
    )

    # -------------------------------------------------------------
    # Controllability matrix
    # -------------------------------------------------------------

    controllability = np.hstack([
        B_aug,
        A_aug @ B_aug,
        A_aug @ A_aug @ B_aug,
    ])

    controllability_rank = np.linalg.matrix_rank(
        controllability
    )

    if controllability_rank != 3:
        raise RuntimeError(
            "Augmented identified model is not controllable."
        )

    # -------------------------------------------------------------
    # Pole placement
    # -------------------------------------------------------------

    placement = place_poles(
        A_aug,
        B_aug,
        DESIRED_POLES,
    )

    K_aug = placement.gain_matrix

    closed_loop_poles = np.linalg.eigvals(
        A_aug - B_aug @ K_aug
    )

    return (
        A_aug,
        B_aug,
        K_aug,
        closed_loop_poles,
        controllability_rank,
    )


def controller_voltage(
    state,
    integral_state,
    reference,
    K_aug,
):
    """
    Integral-augmented state-feedback controller.

        u = -K_aug [current, speed, integral_error]^T

    The reference is retained as an explicit argument for clarity
    and future controller extensions.

    The actuator is constrained to:

        0 <= u <= 12 V
    """

    augmented_state = np.array([
        state[0],
        state[1],
        integral_state,
    ])

    voltage = float(
        -(
            K_aug
            @ augmented_state.reshape(-1, 1)
        )[0, 0]
    )

    return np.clip(
        voltage,
        VOLTAGE_MIN,
        VOLTAGE_MAX,
    )


def simulate_closed_loop(
    motor,
    K_aug,
    reference,
    load_torque,
):
    """
    Simulate one closed-loop system.

    State vector:

        [current, speed, integral_error]

    The controller is identical for the digital twin and true plant.
    """

    t_eval = np.arange(
        0.0,
        SIMULATION_TIME + DT / 2.0,
        DT,
    )

    def dynamics(t, state_aug):

        current = state_aug[0]
        speed = state_aug[1]
        integral_error = state_aug[2]

        x = np.array([
            current,
            speed,
        ])

        voltage = controller_voltage(
            x,
            integral_error,
            reference,
            K_aug,
        )

        motor_dx = motor.derivatives(
            x,
            voltage,
            load_torque,
        )

        tracking_error = reference - speed

        return np.array([
            motor_dx[0],
            motor_dx[1],
            tracking_error,
        ])

    initial_state = np.array([
        0.0,
        0.0,
        0.0,
    ])

    solution = solve_ivp(
        dynamics,
        (
            0.0,
            SIMULATION_TIME,
        ),
        initial_state,
        t_eval=t_eval,
        rtol=1e-8,
        atol=1e-10,
    )

    if not solution.success:
        raise RuntimeError(
            f"Closed-loop simulation failed: {solution.message}"
        )

    current = solution.y[0]
    speed = solution.y[1]
    integral_state = solution.y[2]

    voltage = np.array([
        controller_voltage(
            np.array([i, w]),
            z,
            reference,
            K_aug,
        )
        for i, w, z in zip(
            current,
            speed,
            integral_state,
        )
    ])

    error = reference - speed

    return {
        "time": solution.t,
        "current": current,
        "speed": speed,
        "integral_state": integral_state,
        "voltage": voltage,
        "error": error,
    }


def calculate_metrics(result):
    """
    Calculate closed-loop performance metrics.
    """

    time = result["time"]
    speed = result["speed"]
    error = result["error"]
    voltage = result["voltage"]
    current = result["current"]

    final_speed = speed[-1]
    final_error = error[-1]

    peak_speed = np.max(speed)

    if REFERENCE_SPEED != 0.0:
        overshoot = max(
            0.0,
            (
                (peak_speed - REFERENCE_SPEED)
                / abs(REFERENCE_SPEED)
                * 100.0
            ),
        )
    else:
        overshoot = 0.0

    rms_error = np.sqrt(
        np.mean(error ** 2)
    )

    max_abs_error = np.max(
        np.abs(error)
    )

    max_voltage = np.max(
        np.abs(voltage)
    )

    rms_voltage = np.sqrt(
        np.mean(voltage ** 2)
    )

    max_current = np.max(
        np.abs(current)
    )

    rms_current = np.sqrt(
        np.mean(current ** 2)
    )

    # -------------------------------------------------------------
    # 2% settling band
    # -------------------------------------------------------------

    settling_band = 0.02 * abs(
        REFERENCE_SPEED
    )

    outside_band = np.where(
        np.abs(error) > settling_band
    )[0]

    if len(outside_band) == 0:

        settling_time = 0.0

    elif outside_band[-1] < len(time) - 1:

        settling_time = time[
            outside_band[-1] + 1
        ]

    else:

        settling_time = np.nan

    # -------------------------------------------------------------
    # Time spent at actuator saturation
    # -------------------------------------------------------------

    saturation_tolerance = 1e-8

    saturated = (
        np.abs(voltage - VOLTAGE_MIN)
        <= saturation_tolerance
    ) | (
        np.abs(voltage - VOLTAGE_MAX)
        <= saturation_tolerance
    )

    saturation_fraction = (
        np.mean(saturated)
    )

    saturation_time = (
        saturation_fraction
        * SIMULATION_TIME
    )

    return {
        "final_speed": final_speed,
        "final_error": final_error,
        "rms_error": rms_error,
        "max_abs_error": max_abs_error,
        "peak_speed": peak_speed,
        "overshoot": overshoot,
        "settling_time": settling_time,
        "max_voltage": max_voltage,
        "rms_voltage": rms_voltage,
        "max_current": max_current,
        "rms_current": rms_current,
        "saturation_time": saturation_time,
    }


def print_metrics(name, metrics):
    """
    Print closed-loop performance metrics.
    """

    print()
    print(f"=== {name} ===")

    print(
        f"Final speed:        "
        f"{metrics['final_speed']:.6f} rad/s"
    )

    print(
        f"Final error:        "
        f"{metrics['final_error']:.6f} rad/s"
    )

    print(
        f"RMS tracking error: "
        f"{metrics['rms_error']:.6f} rad/s"
    )

    print(
        f"Maximum abs error:  "
        f"{metrics['max_abs_error']:.6f} rad/s"
    )

    print(
        f"Peak speed:         "
        f"{metrics['peak_speed']:.6f} rad/s"
    )

    print(
        f"Overshoot:          "
        f"{metrics['overshoot']:.4f} %"
    )

    print(
        f"Settling time:      "
        f"{metrics['settling_time']:.6f} s"
    )

    print(
        f"Maximum voltage:    "
        f"{metrics['max_voltage']:.6f} V"
    )

    print(
        f"RMS voltage:        "
        f"{metrics['rms_voltage']:.6f} V"
    )

    print(
        f"Maximum current:    "
        f"{metrics['max_current']:.6f} A"
    )

    print(
        f"RMS current:        "
        f"{metrics['rms_current']:.6f} A"
    )

    print(
        f"Time at saturation: "
        f"{metrics['saturation_time']:.6f} s"
    )


def assess_tracking_performance(metrics):
    """
    Assess absolute closed-loop tracking performance.

    This assessment is deliberately separate from model-transfer
    fidelity.
    """

    final_error = abs(
        metrics["final_error"]
    )

    if (
        final_error < 0.1
        and metrics["settling_time"] < 7.0
    ):
        return "Successful closed-loop tracking"

    if (
        final_error < 0.5
        and metrics["settling_time"] < 10.0
    ):
        return "Acceptable closed-loop tracking"

    return "Tracking performance requires further investigation"


def assess_transfer_fidelity(
    speed_rmse,
    maximum_speed_difference,
):
    """
    Assess digital-twin-to-true-plant transfer fidelity.

    These thresholds concern the difference between the predicted
    digital-twin response and the true-plant response.

    They do NOT measure the absolute reference-tracking error.
    """

    if (
        speed_rmse < 0.5
        and maximum_speed_difference < 1.5
    ):
        return (
            "Successful controller transfer "
            "with small model-to-plant deviation"
        )

    if (
        speed_rmse < 1.0
        and maximum_speed_difference < 3.0
    ):
        return (
            "Acceptable controller transfer "
            "with moderate model-to-plant deviation"
        )

    return (
        "Significant digital-twin-to-plant "
        "transfer deviation"
    )


# ---------------------------------------------------------------------
# MAIN EXPERIMENT
# ---------------------------------------------------------------------

def main():

    print("=" * 72)
    print(
        "=== EXPERIMENT 09A: "
        "DIGITAL-TWIN CLOSED-LOOP CONTROL ==="
    )
    print("=" * 72)

    print()
    print("Purpose:")
    print(
        "Design a controller using the frozen 08A "
        "identified model"
    )
    print(
        "and independently test it on the true motor."
    )
    print()
    print(
        "No parameter optimization is performed."
    )
    print(
        "No controller retuning is performed on the true plant."
    )

    # -------------------------------------------------------------
    # Construct models
    # -------------------------------------------------------------

    true_motor = DCMotor(
        TRUE_PARAMETERS
    )

    identified_motor = DCMotor(
        IDENTIFIED_PARAMETERS
    )

    # -------------------------------------------------------------
    # Parameter comparison
    # -------------------------------------------------------------

    print()
    print("=== FROZEN MODEL PARAMETERS ===")
    print()

    print(
        f"{'Parameter':<10}"
        f"{'True':>14}"
        f"{'08A ID':>14}"
        f"{'Error %':>14}"
    )

    print("-" * 52)

    parameter_pairs = [
        (
            "R",
            TRUE_PARAMETERS.R,
            IDENTIFIED_PARAMETERS.R,
        ),
        (
            "L",
            TRUE_PARAMETERS.L,
            IDENTIFIED_PARAMETERS.L,
        ),
        (
            "Kt",
            TRUE_PARAMETERS.Kt,
            IDENTIFIED_PARAMETERS.Kt,
        ),
        (
            "Ke",
            TRUE_PARAMETERS.Ke,
            IDENTIFIED_PARAMETERS.Ke,
        ),
        (
            "J",
            TRUE_PARAMETERS.J,
            IDENTIFIED_PARAMETERS.J,
        ),
        (
            "b",
            TRUE_PARAMETERS.b,
            IDENTIFIED_PARAMETERS.b,
        ),
    ]

    for name, true_value, identified_value in parameter_pairs:

        error_percent = (
            abs(
                identified_value
                - true_value
            )
            / abs(true_value)
            * 100.0
        )

        print(
            f"{name:<10}"
            f"{true_value:>14.6f}"
            f"{identified_value:>14.6f}"
            f"{error_percent:>14.3f}%"
        )

    # -------------------------------------------------------------
    # State-space models
    # -------------------------------------------------------------

    A_true, B_true, E_true = (
        true_motor.matrices()
    )

    A_id, B_id, E_id = (
        identified_motor.matrices()
    )

    print()
    print("=== TRUE STATE-SPACE MODEL ===")
    print()

    print("A_true =")
    print(A_true)

    print()
    print("B_true =")
    print(B_true)

    print()
    print("E_true =")
    print(E_true)

    print()
    print("=== IDENTIFIED DIGITAL-TWIN MODEL ===")
    print()

    print("A_identified =")
    print(A_id)

    print()
    print("B_identified =")
    print(B_id)

    print()
    print("E_identified =")
    print(E_id)

    # -------------------------------------------------------------
    # Controller design
    # -------------------------------------------------------------
    #
    # IMPORTANT:
    # The controller is designed ONLY from the identified model.
    # -------------------------------------------------------------

    (
        A_aug,
        B_aug,
        K_aug,
        designed_poles,
        controllability_rank,
    ) = design_integral_state_feedback(
        identified_motor
    )

    print()
    print("=== DIGITAL-TWIN CONTROLLER DESIGN ===")
    print()

    print("Augmented state:")
    print(
        "[current, speed, integral_error]"
    )

    print()
    print("Desired closed-loop poles:")
    print(DESIRED_POLES)

    print()
    print("Augmented controllability rank:")
    print(
        f"{controllability_rank}/3"
    )

    print()
    print("K_aug =")
    print(K_aug)

    print()
    print(
        "Closed-loop poles from identified model:"
    )
    print(designed_poles)

    # -------------------------------------------------------------
    # Simulate identified digital twin
    # -------------------------------------------------------------

    twin_result = simulate_closed_loop(
        identified_motor,
        K_aug,
        REFERENCE_SPEED,
        LOAD_TORQUE,
    )

    # -------------------------------------------------------------
    # Simulate true motor
    # -------------------------------------------------------------

    true_result = simulate_closed_loop(
        true_motor,
        K_aug,
        REFERENCE_SPEED,
        LOAD_TORQUE,
    )

    # -------------------------------------------------------------
    # Calculate metrics
    # -------------------------------------------------------------

    twin_metrics = calculate_metrics(
        twin_result
    )

    true_metrics = calculate_metrics(
        true_result
    )

    print_metrics(
        "IDENTIFIED DIGITAL-TWIN CLOSED LOOP",
        twin_metrics,
    )

    print_metrics(
        "TRUE-PLANT CLOSED LOOP",
        true_metrics,
    )

    # -------------------------------------------------------------
    # Digital-twin → true-plant transfer
    # -------------------------------------------------------------

    speed_difference = (
        true_result["speed"]
        - twin_result["speed"]
    )

    current_difference = (
        true_result["current"]
        - twin_result["current"]
    )

    voltage_difference = (
        true_result["voltage"]
        - twin_result["voltage"]
    )

    speed_transfer_rmse = np.sqrt(
        np.mean(
            speed_difference ** 2
        )
    )

    current_transfer_rmse = np.sqrt(
        np.mean(
            current_difference ** 2
        )
    )

    voltage_transfer_rmse = np.sqrt(
        np.mean(
            voltage_difference ** 2
        )
    )

    maximum_speed_transfer_error = np.max(
        np.abs(speed_difference)
    )

    maximum_current_transfer_error = np.max(
        np.abs(current_difference)
    )

    maximum_voltage_transfer_error = np.max(
        np.abs(voltage_difference)
    )

    print()
    print(
        "=== DIGITAL-TWIN → TRUE-PLANT TRANSFER ==="
    )
    print()

    print(
        f"Speed transfer RMSE:       "
        f"{speed_transfer_rmse:.6f} rad/s"
    )

    print(
        f"Maximum speed difference:  "
        f"{maximum_speed_transfer_error:.6f} rad/s"
    )

    print(
        f"Current transfer RMSE:     "
        f"{current_transfer_rmse:.6f} A"
    )

    print(
        f"Maximum current difference:"
        f" {maximum_current_transfer_error:.6f} A"
    )

    print(
        f"Voltage transfer RMSE:     "
        f"{voltage_transfer_rmse:.6f} V"
    )

    print(
        f"Maximum voltage difference:"
        f" {maximum_voltage_transfer_error:.6f} V"
    )

    # -------------------------------------------------------------
    # Separate assessments
    # -------------------------------------------------------------

    tracking_assessment = (
        assess_tracking_performance(
            true_metrics
        )
    )

    transfer_assessment = (
        assess_transfer_fidelity(
            speed_transfer_rmse,
            maximum_speed_transfer_error,
        )
    )

    print()
    print(
        "=== CLOSED-LOOP TRACKING ASSESSMENT ==="
    )
    print()

    print(
        tracking_assessment
    )

    print()
    print(
        "=== DIGITAL-TWIN CONTROL TRANSFER "
        "ASSESSMENT ==="
    )
    print()

    print(
        transfer_assessment
    )

    print()
    print("Interpretation:")
    print(
        "The tracking assessment measures absolute "
        "reference-following performance."
    )
    print(
        "The transfer assessment measures how closely "
        "the true plant follows"
    )
    print(
        "the response predicted by the identified "
        "digital twin."
    )

    print()
    print(
        "Note: actuator saturation is retained as a "
        "physical constraint."
    )

    # -------------------------------------------------------------
    # Save plot
    # -------------------------------------------------------------

    results_dir = (
        PROJECT_ROOT / "results"
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        results_dir
        / "09A_digital_twin_closed_loop_control.png"
    )

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(10, 11),
        sharex=True,
    )

    # -------------------------------------------------------------
    # Speed
    # -------------------------------------------------------------

    axes[0].plot(
        twin_result["time"],
        twin_result["speed"],
        label="Identified digital twin",
    )

    axes[0].plot(
        true_result["time"],
        true_result["speed"],
        "--",
        label="True motor",
    )

    axes[0].axhline(
        REFERENCE_SPEED,
        linestyle=":",
        label="Reference",
    )

    axes[0].set_ylabel(
        "Speed [rad/s]"
    )

    axes[0].set_title(
        "09A — Digital-Twin-Based "
        "Closed-Loop Control"
    )

    axes[0].grid(True)
    axes[0].legend()

    # -------------------------------------------------------------
    # Tracking error
    # -------------------------------------------------------------

    axes[1].plot(
        twin_result["time"],
        twin_result["error"],
        label="Twin tracking error",
    )

    axes[1].plot(
        true_result["time"],
        true_result["error"],
        "--",
        label="True-plant tracking error",
    )

    axes[1].axhline(
        0.0,
        linestyle=":",
    )

    axes[1].set_ylabel(
        "Error [rad/s]"
    )

    axes[1].grid(True)
    axes[1].legend()

    # -------------------------------------------------------------
    # Control voltage
    # -------------------------------------------------------------

    axes[2].plot(
        twin_result["time"],
        twin_result["voltage"],
        label="Twin control voltage",
    )

    axes[2].plot(
        true_result["time"],
        true_result["voltage"],
        "--",
        label="True-plant control voltage",
    )

    axes[2].axhline(
        VOLTAGE_MAX,
        linestyle=":",
        label="Voltage limit",
    )

    axes[2].set_ylabel(
        "Voltage [V]"
    )

    axes[2].set_xlabel(
        "Time [s]"
    )

    axes[2].grid(True)
    axes[2].legend()

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    print()
    print(
        f"Saved plot: {output_path}"
    )

    print()
    print("=" * 72)
    print("=== EXPERIMENT 09A COMPLETE ===")
    print("=" * 72)


if __name__ == "__main__":
    main()

