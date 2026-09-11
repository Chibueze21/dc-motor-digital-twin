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

# Controller / numerical sampling time
Ts = 0.001


# ============================================================
# Create motor model
# ============================================================

motor = DCMotor()


# ============================================================
# Extract physical state-space model
# ============================================================

A, B, E = motor.matrices()

# State:
#
#     x = [current, speed]^T
#
# Measured output:
#
#     y = speed

C = np.array([
    [0.0, 1.0],
])

D = np.array([
    [0.0],
])


# ============================================================
# Basic plant analysis
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
# Augmented state-space model
# ============================================================
#
# Integral state:
#
#     z_dot = r - y
#
# Since y = Cx:
#
#     z_dot = r - Cx
#
# Therefore:
#
#     x_aug = [x, z]^T
#
#     x_aug_dot =
#
#         [ A   0 ] x_aug
#         [-C   0 ]
#
#       + [ B ] u
#         [ 0 ]
#
#       + [ 0 ] r
#         [ 1 ]
#
# ============================================================

A_aug = np.block([
    [
        A,
        np.zeros((2, 1)),
    ],
    [
        -C,
        np.zeros((1, 1)),
    ],
])

B_aug = np.vstack([
    B,
    [[0.0]],
])

E_aug = np.vstack([
    E,
    [[0.0]],
])

R_aug = np.array([
    [0.0],
    [0.0],
    [1.0],
])


# ============================================================
# Augmented controllability analysis
# ============================================================

augmented_controllability_matrix = np.hstack([
    B_aug,
    A_aug @ B_aug,
    A_aug @ A_aug @ B_aug,
])

augmented_controllability_rank = np.linalg.matrix_rank(
    augmented_controllability_matrix
)

augmented_state_count = A_aug.shape[0]

is_augmented_controllable = (
    augmented_controllability_rank
    == augmented_state_count
)


# ============================================================
# State-feedback design with integral action
# ============================================================
#
# Control law:
#
#     u = -Kx - Ki*z
#
# or:
#
#     u = -K_aug*x_aug
#
# ============================================================

desired_poles = np.array([
    -4.0,
    -5.0,
    -6.0,
])

pole_result = place_poles(
    A_aug,
    B_aug,
    desired_poles,
)

K_aug = pole_result.gain_matrix

K_state = K_aug[0, :2]

K_integral = K_aug[0, 2]


# ============================================================
# Closed-loop system
# ============================================================

A_aug_cl = (
    A_aug
    - B_aug @ K_aug
)

closed_loop_poles = np.linalg.eigvals(
    A_aug_cl
)


# ============================================================
# Display model and controller information
# ============================================================

print("=== Augmented State-Space Control Experiment ===")

print("\nPhysical state-space model:")

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


print("\n=== Augmented Model ===")

print("\nA_aug =")
print(A_aug)

print("\nB_aug =")
print(B_aug)

print("\nAugmented controllability matrix:")
print(augmented_controllability_matrix)

print(
    f"\nAugmented controllability rank: "
    f"{augmented_controllability_rank}"
    f"/{augmented_state_count}"
)

print(
    f"Augmented system controllable: "
    f"{is_augmented_controllable}"
)


print("\nDesired closed-loop poles:")
print(desired_poles)

print("\nAugmented state-feedback gain:")
print(K_aug)

print(
    f"\nCurrent feedback gain: "
    f"{K_state}"
)

print(
    f"Integral gain:         "
    f"{K_integral:.6f}"
)

print("\nActual closed-loop poles:")
print(closed_loop_poles)


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

integral_state = np.zeros(n_steps)

control_state_feedback = np.zeros(n_steps)
control_integral = np.zeros(n_steps)
voltage_unsaturated = np.zeros(n_steps)


# ============================================================
# Initial state
# ============================================================

x = np.array([
    0.0,  # current [A]
    0.0,  # speed [rad/s]
])

integral = 0.0


# ============================================================
# Simulation
# ============================================================

for k in range(n_steps - 1):

    # --------------------------------------------------------
    # Store physical states
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
    # Integral state
    # --------------------------------------------------------

    integral_state[k] = integral

    # --------------------------------------------------------
    # State-feedback contribution
    # --------------------------------------------------------

    control_state_feedback[k] = (
        -(K_state @ x)
    )

    # --------------------------------------------------------
    # Integral contribution
    # --------------------------------------------------------

    control_integral[k] = (
        -K_integral * integral
    )

    # --------------------------------------------------------
    # Unsaturated control command
    # --------------------------------------------------------

    voltage_unsaturated[k] = (
        control_state_feedback[k]
        + control_integral[k]
    )

    # --------------------------------------------------------
    # Actuator saturation
    # --------------------------------------------------------

    voltage[k] = np.clip(
        voltage_unsaturated[k],
        voltage_min,
        voltage_max,
    )

    # --------------------------------------------------------
    # Conditional integration anti-windup
    # --------------------------------------------------------
    #
    # If the actuator is saturated and the tracking error
    # would drive it further into saturation, freeze the
    # integral state.
    #
    # --------------------------------------------------------

    if (
        voltage_unsaturated[k] > voltage_max
        and error[k] > 0.0
    ):

        integral_dot = 0.0

    elif (
        voltage_unsaturated[k] < voltage_min
        and error[k] < 0.0
    ):

        integral_dot = 0.0

    else:

        integral_dot = error[k]

    integral += (
        integral_dot * Ts
    )

    # --------------------------------------------------------
    # Integrate physical motor dynamics
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

integral_state[-1] = integral

control_state_feedback[-1] = (
    -(K_state @ x)
)

control_integral[-1] = (
    -K_integral * integral
)

voltage_unsaturated[-1] = (
    control_state_feedback[-1]
    + control_integral[-1]
)

voltage[-1] = np.clip(
    voltage_unsaturated[-1],
    voltage_min,
    voltage_max,
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

print("\n=== Augmented State-Space Performance ===")

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

print(
    f"Final integral state: "
    f"{integral:.6f}"
)


# ============================================================
# Plot results
# ============================================================

fig, axes = plt.subplots(
    4,
    1,
    figsize=(10, 11),
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
    "Augmented State-Space Speed Tracking"
)

axes[0].legend()
axes[0].grid(True)


# ------------------------------------------------------------
# Voltage
# ------------------------------------------------------------

axes[1].plot(
    time,
    voltage,
    label="Applied voltage",
)

axes[1].plot(
    time,
    voltage_unsaturated,
    linestyle=":",
    label="Unsaturated command",
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
# Physical states
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

axes[2].set_ylabel(
    "State value"
)

axes[2].set_title(
    "Physical Motor States"
)

axes[2].legend()
axes[2].grid(True)


# ------------------------------------------------------------
# Integral state
# ------------------------------------------------------------

axes[3].plot(
    time,
    integral_state,
    label="Integral state",
)

axes[3].set_xlabel(
    "Time (s)"
)

axes[3].set_ylabel(
    "Integral state"
)

axes[3].set_title(
    "Integral State Evolution"
)

axes[3].legend()
axes[3].grid(True)


# ============================================================
# Save figure
# ============================================================

plt.tight_layout()

plt.savefig(
    "results/07_state_space_control.png",
    dpi=150,
)

plt.show()