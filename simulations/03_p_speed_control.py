import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from src.motor import DCMotor


# --------------------------------------------------
# Motor
# --------------------------------------------------

motor = DCMotor()


# --------------------------------------------------
# Experiment parameters
# --------------------------------------------------

speed_reference = 40.0       # rad/s
voltage_max = 12.0           # V

Kp = 1.0

disturbance_time = 5.0
load_torque = 0.05


# --------------------------------------------------
# Load disturbance
# --------------------------------------------------

def load_torque_profile(t):

    if t < disturbance_time:
        return 0.0

    return load_torque


# --------------------------------------------------
# P controller
# --------------------------------------------------

def controller(speed):

    error = speed_reference - speed

    voltage = Kp * error

    # Voltage saturation
    voltage = np.clip(
        voltage,
        0.0,
        voltage_max,
    )

    return voltage


# --------------------------------------------------
# Closed-loop dynamics
# --------------------------------------------------

def dynamics(t, x):

    speed = x[1]

    voltage = controller(speed)

    tau_L = load_torque_profile(t)

    return motor.derivatives(
        x,
        voltage=voltage,
        load_torque=tau_L,
    )


# --------------------------------------------------
# Initial conditions
# --------------------------------------------------

x0 = np.array([0.0, 0.0])


# --------------------------------------------------
# Simulation
# --------------------------------------------------

t_eval = np.linspace(
    0.0,
    15.0,
    1500,
)

solution = solve_ivp(
    dynamics,
    (0.0, 15.0),
    x0,
    t_eval=t_eval,
    rtol=1e-8,
    atol=1e-10,
)


# --------------------------------------------------
# Results
# --------------------------------------------------

time = solution.t
current = solution.y[0]
speed = solution.y[1]

voltage = np.array([
    controller(w)
    for w in speed
])

load = np.array([
    load_torque_profile(t)
    for t in time
])


# --------------------------------------------------
# Print results
# --------------------------------------------------

print()
print("=== P Speed Control Experiment ===")
print(f"Reference speed: {speed_reference:.4f} rad/s")
print(f"Kp:              {Kp:.4f}")
print(f"Final speed:     {speed[-1]:.4f} rad/s")
print(f"Final current:   {current[-1]:.4f} A")
print(f"Final voltage:   {voltage[-1]:.4f} V")
print(f"Final error:     {speed_reference - speed[-1]:.4f} rad/s")


# --------------------------------------------------
# Plot
# --------------------------------------------------

fig, axes = plt.subplots(
    4,
    1,
    figsize=(10, 11),
    sharex=True,
)


# Speed

axes[0].plot(
    time,
    speed,
    label="Motor speed",
)

axes[0].axhline(
    speed_reference,
    color="green",
    linestyle="--",
    label="Reference",
)

axes[0].axvline(
    disturbance_time,
    color="red",
    linestyle="--",
    label="Load applied",
)

axes[0].set_ylabel("Speed [rad/s]")
axes[0].set_title("Closed-Loop Speed Response — P Controller")
axes[0].grid(True)
axes[0].legend()


# Error

error = speed_reference - speed

axes[1].plot(
    time,
    error,
)

axes[1].axvline(
    disturbance_time,
    color="red",
    linestyle="--",
)

axes[1].set_ylabel("Error [rad/s]")
axes[1].set_title("Speed Tracking Error")
axes[1].grid(True)


# Voltage

axes[2].plot(
    time,
    voltage,
)

axes[2].axvline(
    disturbance_time,
    color="red",
    linestyle="--",
)

axes[2].set_ylabel("Voltage [V]")
axes[2].set_title("Control Voltage")
axes[2].grid(True)


# Current

axes[3].plot(
    time,
    current,
)

axes[3].axvline(
    disturbance_time,
    color="red",
    linestyle="--",
)

axes[3].set_xlabel("Time [s]")
axes[3].set_ylabel("Current [A]")
axes[3].set_title("Armature Current")
axes[3].grid(True)


plt.tight_layout()

plt.savefig(
    "results/03_p_speed_control.png",
    dpi=150,
)

plt.show()
