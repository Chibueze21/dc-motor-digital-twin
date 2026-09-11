"""
Experiment 08A: DC Motor System Identification

Purpose
-------
Estimate the physical DC motor parameters from noisy
input/state measurements and validate the identified model.

The true motor parameters come from src.motor.DCMotorParameters.

Measured signals:
    voltage [V]
    current [A]
    angular speed [rad/s]

Estimated parameters:
    R  - armature resistance [Ohm]
    L  - armature inductance [H]
    Kt - torque constant [N*m/A]
    Ke - back-EMF constant [V*s/rad]
    J  - rotor inertia [kg*m^2]
    b  - viscous friction [N*m*s/rad]

The experiment intentionally adds measurement noise before
parameter estimation so that the identification is not based
on perfectly clean measurements.

The identified model is then simulated using the same input
trajectory and compared against the noise-free reference model.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

from src.motor import DCMotor, DCMotorParameters


# ============================================================
# Experiment configuration
# ============================================================

simulation_time = 10.0
Ts = 0.002

voltage_min = 0.0
voltage_max = 12.0

load_torque = 0.05

# Measurement noise standard deviations.
current_noise_std = 0.02       # A
speed_noise_std = 0.10         # rad/s

random_seed = 42


# ============================================================
# True physical model
# ============================================================

motor = DCMotor()
true_parameters = motor.p

A_true, B_true, E_true = motor.matrices()


# ============================================================
# Excitation signal
# ============================================================

def excitation_voltage(t):
    """
    Piecewise voltage excitation.

    The changing voltage levels provide excitation for both
    electrical and mechanical dynamics.
    """

    if t < 0.5:
        return 0.0
    elif t < 1.5:
        return 4.0
    elif t < 3.0:
        return 8.0
    elif t < 4.5:
        return 5.0
    elif t < 6.0:
        return 10.0
    elif t < 7.5:
        return 6.0
    elif t < 9.0:
        return 9.0
    else:
        return 3.0


# ============================================================
# Generate reference data
# ============================================================

time = np.arange(
    0.0,
    simulation_time + Ts,
    Ts,
)

initial_state = np.array([0.0, 0.0])


def true_motor_dynamics(t, x):
    voltage = excitation_voltage(t)

    return motor.derivatives(
        x,
        voltage=voltage,
        load_torque=load_torque,
    )


reference_solution = solve_ivp(
    true_motor_dynamics,
    [time[0], time[-1]],
    initial_state,
    t_eval=time,
    rtol=1e-9,
    atol=1e-11,
)

if not reference_solution.success:
    raise RuntimeError(
        "Reference motor simulation failed."
    )


true_current = reference_solution.y[0]
true_speed = reference_solution.y[1]

voltage = np.array([
    excitation_voltage(t)
    for t in time
])


# ============================================================
# Generate noisy measurements
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
# Parameter estimation
# ============================================================

def simulate_candidate(parameters):
    """
    Simulate a candidate motor parameter set.

    Parameters
    ----------
    parameters : array-like
        [R, L, Kt, Ke, J, b]

    Returns
    -------
    current : ndarray
    speed : ndarray
    """

    R, L, Kt, Ke, J, b = parameters

    candidate_parameters = DCMotorParameters(
        R=R,
        L=L,
        Kt=Kt,
        Ke=Ke,
        J=J,
        b=b,
    )

    candidate_motor = DCMotor(candidate_parameters)

    def candidate_dynamics(t, x):
        return candidate_motor.derivatives(
            x,
            voltage=excitation_voltage(t),
            load_torque=load_torque,
        )

    solution = solve_ivp(
        candidate_dynamics,
        [time[0], time[-1]],
        initial_state,
        t_eval=time,
        rtol=1e-6,
        atol=1e-8,
    )

    if not solution.success:
        raise RuntimeError(
            "Candidate motor simulation failed."
        )

    return solution.y[0], solution.y[1]


# Scale the current and speed residuals so that neither
# measurement dominates the optimization simply because
# of its numerical magnitude.

def residual_function(parameters):
    """
    Calculate normalized model-measurement residuals.
    """

    try:
        candidate_current, candidate_speed = (
            simulate_candidate(parameters)
        )
    except RuntimeError:
        return np.full(
            measured_current.size * 2,
            1e6,
        )

    current_residual = (
        candidate_current - measured_current
    ) / current_noise_std

    speed_residual = (
        candidate_speed - measured_speed
    ) / speed_noise_std

    return np.concatenate([
        current_residual,
        speed_residual,
    ])


# ============================================================
# Initial parameter guess
# ============================================================

initial_guess = np.array([
    1.5,     # R
    0.35,    # L
    0.08,    # Kt
    0.08,    # Ke
    0.03,    # J
    0.015,   # b
])


# Physical parameter bounds.

lower_bounds = np.array([
    0.1,      # R
    0.01,     # L
    0.01,     # Kt
    0.01,     # Ke
    0.001,    # J
    0.0001,   # b
])

upper_bounds = np.array([
    10.0,     # R
    5.0,      # L
    1.0,      # Kt
    1.0,      # Ke
    1.0,      # J
    0.5,      # b
])


print()
print("=== DC MOTOR SYSTEM IDENTIFICATION ===")
print()

print("True motor parameters:")
print(f"R  = {true_parameters.R:.6f} Ohm")
print(f"L  = {true_parameters.L:.6f} H")
print(f"Kt = {true_parameters.Kt:.6f} N*m/A")
print(f"Ke = {true_parameters.Ke:.6f} V*s/rad")
print(f"J  = {true_parameters.J:.6f} kg*m^2")
print(f"b  = {true_parameters.b:.6f} N*m*s/rad")

print()
print("Excitation:")
print(f"Simulation time: {simulation_time:.2f} s")
print(f"Sampling time:   {Ts:.4f} s")
print(f"Load torque:     {load_torque:.4f} N*m")

print()
print("Measurement noise:")
print(f"Current noise: {current_noise_std:.4f} A")
print(f"Speed noise:   {speed_noise_std:.4f} rad/s")


print()
print("Starting parameter estimation...")


# ============================================================
# Run least-squares identification
# ============================================================

identification_result = least_squares(
    residual_function,
    initial_guess,
    bounds=(lower_bounds, upper_bounds),
    method="trf",
    x_scale="jac",
    verbose=1,
    max_nfev=100,
)


identified_parameters = identification_result.x


# ============================================================
# Identified model simulation
# ============================================================

identified_current, identified_speed = (
    simulate_candidate(identified_parameters)
)


# ============================================================
# Parameter reporting
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

parameter_units = [
    "Ohm",
    "H",
    "N*m/A",
    "V*s/rad",
    "kg*m^2",
    "N*m*s/rad",
]


print()
print("=== IDENTIFICATION RESULT ===")
print()

print(f"Optimization success: {identification_result.success}")
print(
    f"Optimization message: "
    f"{identification_result.message}"
)
print(f"Function evaluations: {identification_result.nfev}")
print(f"Final cost:           {identification_result.cost:.6f}")

print()
print(
    f"{'Parameter':<12}"
    f"{'True':>14}"
    f"{'Identified':>16}"
    f"{'Error %':>14}"
)

print("-" * 58)

parameter_errors = []

for name, true_value, identified_value, unit in zip(
    parameter_names,
    true_values,
    identified_parameters,
    parameter_units,
):
    error_percent = (
        100.0
        * abs(identified_value - true_value)
        / abs(true_value)
    )

    parameter_errors.append(error_percent)

    print(
        f"{name:<12}"
        f"{true_value:>14.6f}"
        f"{identified_value:>16.6f}"
        f"{error_percent:>13.3f}%"
    )


# ============================================================
# Model validation metrics
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

    return 1.0 - residual_sum / total_sum


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


print()
print("=== IDENTIFIED MODEL VALIDATION ===")
print()

print(
    f"Current RMSE: {current_rmse:.6f} A"
)

print(
    f"Speed RMSE:   {speed_rmse:.6f} rad/s"
)

print(
    f"Current R^2:  {current_r2:.6f}"
)

print(
    f"Speed R^2:    {speed_r2:.6f}"
)


# ============================================================
# Build identified state-space matrices
# ============================================================

R_id, L_id, Kt_id, Ke_id, J_id, b_id = (
    identified_parameters
)

A_identified = np.array([
    [-R_id / L_id, -Ke_id / L_id],
    [ Kt_id / J_id, -b_id / J_id],
])

B_identified = np.array([
    [1.0 / L_id],
    [0.0],
])

E_identified = np.array([
    [0.0],
    [-1.0 / J_id],
])


print()
print("=== TRUE vs IDENTIFIED STATE-SPACE MODEL ===")

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
# Overall identification assessment
# ============================================================

mean_parameter_error = np.mean(
    parameter_errors
)

print()
print("=== IDENTIFICATION SUMMARY ===")
print()

print(
    f"Mean parameter error: "
    f"{mean_parameter_error:.3f}%"
)

print(
    f"Speed model RMSE:     "
    f"{speed_rmse:.6f} rad/s"
)

print(
    f"Speed model R^2:      "
    f"{speed_r2:.6f}"
)


# ============================================================
# Visualization
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


# Voltage excitation

axes[0].plot(
    time,
    voltage,
    label="Input voltage",
)

axes[0].set_ylabel("Voltage [V]")
axes[0].set_title(
    "System Identification Excitation"
)
axes[0].grid(True)
axes[0].legend()


# Current comparison

axes[1].plot(
    time,
    true_current,
    label="True current",
)

axes[1].plot(
    time,
    measured_current,
    label="Measured current",
    alpha=0.45,
)

axes[1].plot(
    time,
    identified_current,
    "--",
    label="Identified model",
)

axes[1].set_ylabel("Current [A]")
axes[1].set_title(
    "Electrical State Identification"
)
axes[1].grid(True)
axes[1].legend()


# Speed comparison

axes[2].plot(
    time,
    true_speed,
    label="True speed",
)

axes[2].plot(
    time,
    measured_speed,
    label="Measured speed",
    alpha=0.45,
)

axes[2].plot(
    time,
    identified_speed,
    "--",
    label="Identified model",
)

axes[2].set_ylabel("Speed [rad/s]")
axes[2].set_title(
    "Mechanical State Identification"
)
axes[2].grid(True)
axes[2].legend()


# Speed identification error

speed_error = (
    identified_speed - true_speed
)

axes[3].plot(
    time,
    speed_error,
    label="Identification error",
)

axes[3].axhline(
    0.0,
    linestyle="--",
)

axes[3].set_xlabel("Time [s]")
axes[3].set_ylabel("Error [rad/s]")
axes[3].set_title(
    "Identified Model Speed Error"
)
axes[3].grid(True)
axes[3].legend()


plt.tight_layout()

output_path = (
    "results/08_system_identification.png"
)

plt.savefig(
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
print("=== EXPERIMENT 08A COMPLETE ===")