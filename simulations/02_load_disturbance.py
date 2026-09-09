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

voltage = 12.0

t_start = 0.0
t_end = 15.0

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
# Motor dynamics
# --------------------------------------------------

def dynamics(t, x):

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
# Numerical simulation
# --------------------------------------------------

t_eval = np.linspace(
    t_start,
    t_end,
    1500,
)

solution = solve_ivp(
    dynamics,
    (t_start, t_end),
    x0,
    t_eval=t_eval,
    rtol=1e-8,
    atol=1e-10,
)


# --------------------------------------------------
# Extract states
# --------------------------------------------------

time = solution.t
current = solution.y[0]
speed = solution.y[1]



print()
print("=== Load Disturbance Experiment ===")
print(f"Initial speed:        {speed[0]:.4f} rad/s")
print(f"Speed before load:    {speed[np.searchsorted(time, disturbance_time - 0.1)]:.4f} rad/s")
print(f"Final speed:          {speed[-1]:.4f} rad/s")
print(f"Final current:        {current[-1]:.4f} A")


# --------------------------------------------------
# Plot
# --------------------------------------------------

fig, axes = plt.subplots(
    3,
    1,
    figsize=(10, 9),
    sharex=True,
)


# Speed

axes[0].plot(
    time,
    speed,
    label="Motor speed",
)

axes[0].axvline(
    disturbance_time,
    color="red",
    linestyle="--",
    label="Load applied",
)

axes[0].set_ylabel("Speed [rad/s]")
axes[0].set_title("Open-Loop Speed Response")
axes[0].grid(True)
axes[0].legend()


# Current

axes[1].plot(
    time,
    current,
)

axes[1].axvline(
    disturbance_time,
    color="red",
    linestyle="--",
)

axes[1].set_ylabel("Current [A]")
axes[1].set_title("Armature Current")
axes[1].grid(True)


# Load torque

load = np.array([
    load_torque_profile(t)
    for t in time
])

axes[2].plot(
    time,
    load,
)

axes[2].axvline(
    disturbance_time,
    color="red",
    linestyle="--",
)

axes[2].set_xlabel("Time [s]")
axes[2].set_ylabel("Load [Nm]")
axes[2].set_title("Applied Load Torque")
axes[2].grid(True)


plt.tight_layout()
plt.savefig(
    "results/02_load_disturbance.png",
    dpi=150,
)

plt.show()
