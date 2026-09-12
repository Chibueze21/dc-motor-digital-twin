import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp
from scipy.signal import place_poles

from src.motor import DCMotor


# ============================================================
# Experiment configuration
# ============================================================

speed_reference = 30.0

load_torque = 0.05

voltage_min = 0.0
voltage_max = 12.0

simulation_time = 10.0

# Numerical integration step
Ts = 0.001


# ============================================================
# Create motor model
# ============================================================

motor = DCMotor()


# ============================================================
# Extract state-space model
# ============================================================

A, B, E = motor.matrices()

# Speed is the measured output.
#
# x = [current, speed]^T
#
# y = speed

C = np.array([
    [0.0, 1.0],
])

D = np.array([
    [0.0],
])


# ============================================================
# Plant analysis
# ============================================================

open_loop_poles = np.linalg.eigvals(A)

controllability_matrix = np.hstack([
    B,
    A @ B,
])

observability_matrix = np.vstack([
    C,
    C @ A,
])

controllability_rank = np.linalg.matrix_rank(
    controllability_matrix
)

observability_rank = np.linalg.matrix_rank(
    observability_matrix
)

number_of_states = A.shape[0]

is_controllable = (
    controllability_rank == number_of_states
)

is_observable = (
    observability_rank == number_of_states
)


# ============================================================
# State-feedback design
# ============================================================
#
# Control law:
#
#     u = -Kx + N*r
#
# Desired closed-loop poles are selected faster than the
# dominant open-loop dynamics while remaining reasonable
# for the physical motor.
# ============================================================

desired_poles = np.array([
    -5.0,
    -6.0,
])

pole_result = place_poles(
    A,
    B,
    desired_poles,
)

K = pole_result.gain_matrix


# ============================================================
# Reference precompensator
# ============================================================
#
# For:
#
#     u = -Kx + N*r
#
# choose N so that the nominal steady-state output tracks
# the reference.
#
# N = -1 / [C (A-BK)^(-1) B]
#
# for the SISO case.
# ============================================================

A_cl = A - B @ K

reference_gain = -1.0 / (
    C
    @ np.linalg.solve(A_cl, B)
)[0, 0]


# ============================================================
# Display model and controller information
# ============================================================

print("=== State-Space Control Experiment ===")

print("\nState-space model:")

print("\nA =")
print(A)

print("\nB =")
print(B)

print("\nE =")
print(E)

print("\nC =")
print(C)

print("\nD =")
print(D)

print("\nOpen-loop poles:")
print(open_loop_poles)

print("\nControllability matrix:")
print(controllability_matrix)

print(
    f"\nControllability rank: "
    f"{controllability_rank}/{number_of_states}"
)

print(
    f"Controllable:          "
    f"{is_controllable}"
)

print("\nObservability matrix:")
print(observability_matrix)

print(
    f"\nObservability rank: "
    f"{observability_rank}/{number_of_states}"
)

print(
    f"Observable:          "
    f"{is_observable}"
)

print("\nDesired closed-loop poles:")
print(desired_poles)

print("\nState-feedback gain K:")
print(K)

print(
    f"\nReference gain N: "
    f"{reference_gain:.6f}"
)


# ============================================================
# Simulation timeline
# ============================================================

time = np.arange(
    0.0,
    simulation_time + Ts,
    Ts,
)

n_steps = len(time)


# ============================================================
# Storage
# ============================================================

current = np.zeros(n_steps)
speed = np.zeros(n_steps)
voltage = np.zeros(n_steps)
error = np.zeros(n_steps)

state_feedback = np.zeros(n_steps)


# ============================================================
# Initial state
# ============================================================

x = np.array([
    0.0,  # current [A]
    0.0,  # speed [rad/s]
])


# ============================================================
# State-space simulation
# ============================================================

for k in range(n_steps - 1):

    # --------------------------------------------------------
    # Store current state
    # --------------------------------------------------------

    current[k] = x[0]
    speed[k] = x[1]

    # --------------------------------------------------------
    # Tracking error
    # --------------------------------------------------------

    error[k] = (
        speed_reference
        - speed[k]
    )

    # --------------------------------------------------------
    # State-feedback control
    # --------------------------------------------------------

    state_feedback[k] = (
        -(K @ x)[0]
    )

    voltage_unsaturated = (
        state_feedback[k]
        + reference_gain * speed_reference
    )

    # --------------------------------------------------------
    # Actuator saturation
    # --------------------------------------------------------

    voltage[k] = np.clip(
        voltage_unsaturated,
        voltage_min,
        voltage_max,
    )

    # --------------------------------------------------------
    # Integrate continuous motor dynamics
    # --------------------------------------------------------

    solution = solve_ivp(
        lambda t, state: motor.derivatives(
            state,
            voltage=voltage[k],
            load_torque=load_torque,
        ),
        [time[k], time[k + 1]],
        x,
        t_eval=[time[k + 1]],
    )

    x = solution.y[:, -1]


# ============================================================
# Final state
# ============================================================

current[-1] = x[0]
speed[-1] = x[1]

error[-1] = (
    speed_reference
    - speed[-1]
)

voltage[-1] = voltage[-2]

state_feedback[-1] = (
    -(K @ x)[0]
)


# ============================================================
# Performance metrics
# ============================================================

final_speed = speed[-1]

steady_state_error = (
    speed_reference
    - final_speed
)

peak_speed = np.max(speed)

overshoot = max(
    0.0,
    (peak_speed - speed_reference)
    / speed_reference
    * 100.0,
)


# 2% settling band

upper_band = (
    speed_reference * 1.02
)

lower_band = (
    speed_reference * 0.98
)

outside_band = np.where(
    (speed > upper_band)
    | (speed < lower_band)
)[0]


if len(outside_band) == 0:

    settling_time = 0.0

elif outside_band[-1] < len(time) - 1:

    settling_time = (
        time[outside_band[-1] + 1]
    )

else:

    settling_time = np.nan


# ============================================================
# Control effort metrics
# ============================================================

maximum_voltage = np.max(
    voltage
)

rms_voltage = np.sqrt(
    np.mean(voltage ** 2)
)

maximum_current = np.max(
    np.abs(current)
)

rms_current = np.sqrt(
    np.mean(current ** 2)
)


# ============================================================
# Results
# ============================================================

print("\n=== State-Space Performance ===")

print(
    f"Reference speed:      "
    f"{speed_reference:.4f} rad/s"
)

print(
    f"Final speed:          "
    f"{final_speed:.4f} rad/s"
)

print(
    f"Final current:        "
    f"{current[-1]:.4f} A"
)

print(
    f"Final voltage:        "
    f"{voltage[-1]:.4f} V"
)

print(
    f"Steady-state error:   "
    f"{steady_state_error:.4f} rad/s"
)

print(
    f"Peak speed:           "
    f"{peak_speed:.4f} rad/s"
)

print(
    f"Overshoot:            "
    f"{overshoot:.2f} %"
)

print(
    f"Settling time:        "
    f"{settling_time:.4f} s"
)

print(
    f"Maximum voltage:      "
    f"{maximum_voltage:.4f} V"
)

print(
    f"RMS voltage:          "
    f"{rms_voltage:.4f} V"
)

print(
    f"Maximum current:      "
    f"{maximum_current:.4f} A"
)

print(
    f"RMS current:          "
    f"{rms_current:.4f} A"
)


# ============================================================
# Plot results
# ============================================================

fig, axes = plt.subplots(
    3,
    1,
    figsize=(10, 9),
    sharex=True,
)


# ------------------------------------------------------------
# Speed
# ------------------------------------------------------------

axes[0].plot(
    time,
    speed,
    label="Motor speed",
)

axes[0].axhline(
    speed_reference,
    color="red",
    linestyle="--",
    label="Reference",
)

axes[0].set_ylabel(
    "Speed (rad/s)"
)

axes[0].set_title(
    "State-Space Speed Tracking"
)

axes[0].legend()
axes[0].grid(True)


# ------------------------------------------------------------
# Voltage
# ------------------------------------------------------------

axes[1].plot(
    time,
    voltage,
    label="Control voltage",
)

axes[1].axhline(
    voltage_max,
    color="red",
    linestyle="--",
    label="Voltage limit",
)

axes[1].set_ylabel(
    "Voltage (V)"
)

axes[1].legend()
axes[1].grid(True)


# ------------------------------------------------------------
# States
# ------------------------------------------------------------

axes[2].plot(
    time,
    current,
    label="Armature current",
)

axes[2].plot(
    time,
    speed,
    label="Angular speed",
)

axes[2].set_xlabel(
    "Time (s)"
)

axes[2].set_ylabel(
    "State value"
)

axes[2].set_title(
    "Motor State Evolution"
)

axes[2].legend()
axes[2].grid(True)


# ============================================================
# Save figure
# ============================================================

plt.tight_layout()

plt.savefig(
    "results/07_state_space_control.png",
    dpi=150,
)

plt.show()