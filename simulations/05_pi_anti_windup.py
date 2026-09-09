import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp

from src.motor import DCMotor
from src.controllers import PIController

# ============================================================
# Experiment configuration
# ============================================================

speed_reference = 40.0

Kp = 1.0
Ki = 0.5

voltage_min = 0.0
voltage_max = 12.0

load_torque = 0.05

simulation_time = 15.0
# Controller sampling time
Ts = 0.001
# ============================================================
# Create motor and controller
# ============================================================

motor = DCMotor()

controller = PIController(
    kp=Kp,
    ki=Ki,
    output_min=voltage_min,
    output_max=voltage_max,
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
integral = np.zeros(n_steps)

# ============================================================
# Initial state
# ============================================================

x = np.array([
    0.0,  # current (A)
    0.0,  # speed (rad/s)
])

# ============================================================
# Digital controller + continuous plant simulation
# ============================================================

for k in range(n_steps - 1):

    # Current measured motor state
    current[k] = x[0]
    speed[k] = x[1]

    # Tracking error
    error[k] = speed_reference - speed[k]

    # Controller update
    voltage[k] = controller.update(
        error=error[k],
        dt=Ts,
    )

    # Store controller internal state
    integral[k] = controller.integral

    # Integrate the continuous motor dynamics
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

    # Update motor state
    x = solution.y[:, -1]
# Store final state
current[-1] = x[0]
speed[-1] = x[1]

error[-1] = speed_reference - speed[-1]
voltage[-1] = voltage[-2]
integral[-1] = controller.integral


# ============================================================
# Results
# ============================================================

print("=== PI Anti-Windup Experiment ===")
print(f"Reference speed: {speed_reference:.4f} rad/s")
print(f"Kp:              {Kp:.4f}")
print(f"Ki:              {Ki:.4f}")
print(f"Final speed:     {speed[-1]:.4f} rad/s")
print(f"Final current:   {current[-1]:.4f} A")
print(f"Final voltage:   {voltage[-1]:.4f} V")
print(f"Final error:     {error[-1]:.4f} rad/s")
print(f"Final integral:  {integral[-1]:.4f}")


# ============================================================
# Plots
# ============================================================

fig, axes = plt.subplots(
    3,
    1,
    figsize=(10, 8),
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
    color="red",
    linestyle="--",
    label="Reference",
)

axes[0].set_ylabel("Speed (rad/s)")
axes[0].legend()
axes[0].grid(True)

# Voltage
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

axes[1].set_ylabel("Voltage (V)")
axes[1].legend()
axes[1].grid(True)

# Integral
axes[2].plot(
    time,
    integral,
    label="Integral state",
)

axes[2].set_ylabel("Integral")
axes[2].set_xlabel("Time (s)")
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()

plt.savefig(
    "results/05_pi_anti_windup.png",
    dpi=150,
)

plt.show()
