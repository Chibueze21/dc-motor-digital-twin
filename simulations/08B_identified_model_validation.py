"""
Experiment 08B: Independent Validation of the Identified DC Motor

Purpose
-------
Validate the DC motor parameters identified in Experiment 08A
using a completely different voltage excitation trajectory.

The parameters below are FROZEN from the 08A identification result.
No parameter estimation is performed in this experiment.

The validation compares:

    1. True physical motor
    2. Identified digital twin
    3. Noisy measurements

The purpose is to determine whether the identified digital twin
generalizes to an excitation trajectory that was not used during
parameter estimation.

Frozen 08A identified parameters:

    R  = 2.011540 Ohm
    L  = 0.481997 H
    Kt = 0.105805 N*m/A
    Ke = 0.098059 V*s/rad
    J  = 0.021550 kg*m^2
    b  = 0.010715 N*m*s/rad
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from src.motor import DCMotor, DCMotorParameters


# ============================================================
# Experiment configuration
# ============================================================

simulation_time = 10.0
Ts = 0.002

load_torque = 0.05

current_noise_std = 0.02
speed_noise_std = 0.10

random_seed = 84


# ============================================================
# True physical motor
# ============================================================

true_motor = DCMotor()

true_parameters = true_motor.p

A_true, B_true, E_true = true_motor.matrices()


# ============================================================
# Frozen identified parameters from Experiment 08A
# ============================================================

identified_parameters = DCMotorParameters(
    R=2.011540,
    L=0.481997,
    Kt=0.105805,
    Ke=0.098059,
    J=0.021550,
    b=0.010715,
)

identified_motor = DCMotor(
    identified_parameters
)

A_identified, B_identified, E_identified = (
    identified_motor.matrices()
)


# ============================================================
# Independent validation excitation
# ============================================================

def validation_voltage(t):
    """
    Independent excitation trajectory.

    This trajectory is deliberately different from the
    excitation used in Experiment 08A.
    """

    if t < 0.5:
        return 0.0
    elif t < 1.5:
        return 7.0
    elif t < 2.5:
        return 2.0
    elif t < 4.0:
        return 11.0
    elif t < 5.5:
        return 4.0
    elif t < 7.0:
        return 9.0
    elif t < 8.5:
        return 1.0
    else:
        return 8.0


# ============================================================
# Time vector
# ============================================================

time = np.arange(
    0.0,
    simulation_time + Ts,
    Ts,
)


initial_state = np.array([
    0.0,
    0.0,
])


voltage = np.array([
    validation_voltage(t)
    for t in time
])


# ============================================================
# Simulate true physical motor
# ============================================================

def true_dynamics(t, x):
    return true_motor.derivatives(
        x,
        voltage=validation_voltage(t),
        load_torque=load_torque,
    )


true_solution = solve_ivp(
    true_dynamics,
    [time[0], time[-1]],
    initial_state,
    t_eval=time,
    rtol=1e-9,
    atol=1e-11,
)


if not true_solution.success:
    raise RuntimeError(
        "True motor validation simulation failed."
    )


true_current = true_solution.y[0]
true_speed = true_solution.y[1]


# ============================================================
# Simulate identified digital twin
# ============================================================

def identified_dynamics(t, x):
    return identified_motor.derivatives(
        x,
        voltage=validation_voltage(t),
        load_torque=load_torque,
    )


identified_solution = solve_ivp(
    identified_dynamics,
    [time[0], time[-1]],
    initial_state,
    t_eval=time,
    rtol=1e-9,
    atol=1e-11,
)


if not identified_solution.success:
    raise RuntimeError(
        "Identified digital-twin simulation failed."
    )


identified_current = identified_solution.y[0]
identified_speed = identified_solution.y[1]


# ============================================================
# Generate independent noisy measurements
# ============================================================

rng = np.random.default_rng(random_seed)

measured_current = (
    true_current
    + rng.normal(
        0.0,
        current_noise_std,
        size=true_current.shape,
    )
)

measured_speed = (
    true_speed
    + rng.normal(
        0.0,
        speed_noise_std,
        size=true_speed.shape,
    )
)


# ============================================================
# Validation metrics
# ============================================================

def rmse(reference, estimate):
    return np.sqrt(
        np.mean(
            (reference - estimate) ** 2
        )
    )


def r_squared(reference, estimate):
    residual_sum = np.sum(
        (reference - estimate) ** 2
    )

    total_sum = np.sum(
        (reference - np.mean(reference)) ** 2
    )

    if total_sum == 0.0:
        return np.nan

    return 1.0 - (
        residual_sum / total_sum
    )


def maximum_absolute_error(
    reference,
    estimate,
):
    return np.max(
        np.abs(
            reference - estimate
        )
    )


current_error = (
    identified_current - true_current
)

speed_error = (
    identified_speed - true_speed
)


current_rmse = rmse(
    true_current,
    identified_current,
)

speed_rmse = rmse(
    true_speed,
    identified_speed,
)

current_r2 = r_squared(
    true_current,
    identified_current,
)

speed_r2 = r_squared(
    true_speed,
    identified_speed,
)

max_current_error = maximum_absolute_error(
    true_current,
    identified_current,
)

max_speed_error = maximum_absolute_error(
    true_speed,
    identified_speed,
)


# ============================================================
# Measurement-level errors
# ============================================================

measurement_current_rmse = rmse(
    true_current,
    measured_current,
)

measurement_speed_rmse = rmse(
    true_speed,
    measured_speed,
)


# ============================================================
# Parameter comparison
# ============================================================

parameter_names = [
    "R",
    "L",
    "Kt",
    "Ke",
    "J",
    "b",
]

true_values = np.array([
    true_parameters.R,
    true_parameters.L,
    true_parameters.Kt,
    true_parameters.Ke,
    true_parameters.J,
    true_parameters.b,
])

identified_values = np.array([
    identified_parameters.R,
    identified_parameters.L,
    identified_parameters.Kt,
    identified_parameters.Ke,
    identified_parameters.J,
    identified_parameters.b,
])

parameter_errors = (
    100.0
    * np.abs(
        identified_values - true_values
    )
    / np.abs(true_values)
)

mean_parameter_error = np.mean(
    parameter_errors
)


# ============================================================
# Reporting
# ============================================================

print()
print("=== DC MOTOR DIGITAL-TWIN VALIDATION ===")
print()

print(
    "Experiment: 08B"
)

print(
    "Purpose: Independent validation of "
    "the 08A identified model"
)

print()
print("Validation configuration:")
print(
    f"Simulation time: {simulation_time:.2f} s"
)
print(
    f"Sampling time:   {Ts:.4f} s"
)
print(
    f"Load torque:     {load_torque:.4f} N*m"
)

print()
print(
    "The identified parameters are frozen."
)
print(
    "No parameter optimization is performed."
)


# ============================================================
# Frozen parameters
# ============================================================

print()
print("=== FROZEN 08A PARAMETERS ===")
print()

print(
    f"{'Parameter':<12}"
    f"{'True':>14}"
    f"{'08A Identified':>18}"
    f"{'Error %':>14}"
)

print("-" * 62)

for name, true_value, identified_value, error in zip(
    parameter_names,
    true_values,
    identified_values,
    parameter_errors,
):
    print(
        f"{name:<12}"
        f"{true_value:>14.6f}"
        f"{identified_value:>18.6f}"
        f"{error:>13.3f}%"
    )

print()
print(
    f"Mean parameter error: "
    f"{mean_parameter_error:.3f}%"
)


# ============================================================
# State-space comparison
# ============================================================

print()
print("=== STATE-SPACE MODEL COMPARISON ===")

print()
print("A_true =")
print(A_true)

print()
print("A_identified =")
print(A_identified)

print()
print("B_true =")
print(B_true)

print()
print("B_identified =")
print(B_identified)

print()
print("E_true =")
print(E_true)

print()
print("E_identified =")
print(E_identified)


# ============================================================
# Validation performance
# ============================================================

print()
print("=== INDEPENDENT VALIDATION RESULTS ===")
print()

print(
    f"Current RMSE:           "
    f"{current_rmse:.6f} A"
)

print(
    f"Speed RMSE:             "
    f"{speed_rmse:.6f} rad/s"
)

print(
    f"Current R^2:            "
    f"{current_r2:.6f}"
)

print(
    f"Speed R^2:              "
    f"{speed_r2:.6f}"
)

print(
    f"Maximum current error:  "
    f"{max_current_error:.6f} A"
)

print(
    f"Maximum speed error:    "
    f"{max_speed_error:.6f} rad/s"
)


# ============================================================
# Measurement reference
# ============================================================

print()
print("=== MEASUREMENT NOISE REFERENCE ===")
print()

print(
    f"Measured current RMSE: "
    f"{measurement_current_rmse:.6f} A"
)

print(
    f"Measured speed RMSE:   "
    f"{measurement_speed_rmse:.6f} rad/s"
)


# ============================================================
# Generalization assessment
# ============================================================

print()
print("=== DIGITAL-TWIN GENERALIZATION ===")
print()

if speed_r2 >= 0.99:
    speed_assessment = (
        "Excellent speed-model generalization"
    )
elif speed_r2 >= 0.95:
    speed_assessment = (
        "Strong speed-model generalization"
    )
elif speed_r2 >= 0.90:
    speed_assessment = (
        "Acceptable speed-model generalization"
    )
else:
    speed_assessment = (
        "Weak speed-model generalization"
    )

print(
    f"Speed validation assessment: "
    f"{speed_assessment}"
)

print()
print(
    "The identified model was evaluated "
    "on an excitation trajectory that was "
    "not used during parameter estimation."
)


# ============================================================
# Validation plots
# ============================================================

os.makedirs(
    "results",
    exist_ok=True,
)


fig, axes = plt.subplots(
    4,
    1,
    figsize=(10, 12),
    sharex=True,
)


# ------------------------------------------------------------
# Voltage
# ------------------------------------------------------------

axes[0].plot(
    time,
    voltage,
    label="Independent validation input",
)

axes[0].set_ylabel(
    "Voltage [V]"
)

axes[0].set_title(
    "08B Independent Validation Excitation"
)

axes[0].grid(True)
axes[0].legend()


# ------------------------------------------------------------
# Current
# ------------------------------------------------------------

axes[1].plot(
    time,
    true_current,
    label="True motor",
)

axes[1].plot(
    time,
    measured_current,
    label="Noisy measurement",
    alpha=0.45,
)

axes[1].plot(
    time,
    identified_current,
    "--",
    label="Identified digital twin",
)

axes[1].set_ylabel(
    "Current [A]"
)

axes[1].set_title(
    "Independent Electrical-State Validation"
)

axes[1].grid(True)
axes[1].legend()


# ------------------------------------------------------------
# Speed
# ------------------------------------------------------------

axes[2].plot(
    time,
    true_speed,
    label="True motor",
)

axes[2].plot(
    time,
    measured_speed,
    label="Noisy measurement",
    alpha=0.45,
)

axes[2].plot(
    time,
    identified_speed,
    "--",
    label="Identified digital twin",
)

axes[2].set_ylabel(
    "Speed [rad/s]"
)

axes[2].set_title(
    "Independent Mechanical-State Validation"
)

axes[2].grid(True)
axes[2].legend()


# ------------------------------------------------------------
# Model errors
# ------------------------------------------------------------

axes[3].plot(
    time,
    speed_error,
    label="Speed model error",
)

axes[3].axhline(
    0.0,
    linestyle="--",
)

axes[3].set_xlabel(
    "Time [s]"
)

axes[3].set_ylabel(
    "Error [rad/s]"
)

axes[3].set_title(
    "Digital-Twin Speed Prediction Error"
)

axes[3].grid(True)
axes[3].legend()


plt.tight_layout()


output_path = (
    "results/08B_identified_model_validation.png"
)

plt.savefig(
    output_path,
    dpi=200,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# Final status
# ============================================================

print()
print(
    f"Saved plot: {output_path}"
)

print()
print("=== EXPERIMENT 08B COMPLETE ===")