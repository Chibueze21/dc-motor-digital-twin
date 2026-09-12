import numpy as np
import matplotlib.pyplot as plt

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

dt = 0.001
simulation_time = 5.0


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
# Simulation arrays
# ============================================================

time = np.arange(
    0.0,
    simulation_time + dt,
    dt,
)

current = np.zeros_like(time)
speed = np.zeros_like(time)
voltage = np.zeros_like(time)
error = np.zeros_like(time)
integral = np.zeros_like(time)


# ============================================================
# Simulation loop
# ============================================================

for k in range(len(time) - 1):

    # Current motor state
    motor_state = [
        current[k],
        speed[k],
    ]

    # Tracking error
    error[k] = speed_reference - speed[k]

    # Controller
    voltage[k] = controller.update(
        error=error[k],
        dt=dt,
    )

    # Motor dynamics
    derivatives = motor.derivatives(
        motor_state,
        voltage=voltage[k],
        load_torque=load_torque,
    )

    # Euler integration
    current[k + 1] = (
        current[k]
        + derivatives[0] * dt
    )

    speed[k + 1] = (
        speed[k]
        + derivatives[1] * dt
    )

    integral[k] = controller.integral


# Final values
error[-1] = speed_reference - speed[-1]
voltage[-1] = controller.update(
    error=error[-1],
    dt=0.0,
)
integral[-1] = controller.integral


# ============================================================
# Results
# ============================================================

print("=== PI Speed Control Experiment ===")
print(f"Reference speed: {speed_reference:.4f} rad/s")
print(f"Kp:              {Kp:.4f}")
print(f"Ki:              {Ki:.4f}")
print(f"Final speed:     {speed[-1]:.4f} rad/s")
print(f"Final current:   {current[-1]:.4f} A")
print(f"Final voltage:   {voltage[-1]:.4f} V")
print(f"Final error:     {error[-1]:.4f} rad/s")
print(f"Final integral:  {integral[-1]:.4f}")


# ============================================================
# Plot
# ============================================================

fig, axes = plt.subplots(
    3,
    1,
    figsize=(10, 8),
    sharex=True,
)

axes[0].plot(time, speed, label="Motor speed")
axes[0].axhline(
    speed_reference,
    color="red",
    linestyle="--",
    label="Reference",
)
axes[0].set_ylabel("Speed (rad/s)")
axes[0].legend()
axes[0].grid(True)

axes[1].plot(time, voltage)
axes[1].axhline(
    voltage_max,
    color="red",
    linestyle="--",
    label="Voltage limit",
)
axes[1].set_ylabel("Voltage (V)")
axes[1].legend()
axes[1].grid(True)

axes[2].plot(time, integral)
axes[2].set_ylabel("Integral state")
axes[2].set_xlabel("Time (s)")
axes[2].grid(True)

plt.tight_layout()

plt.savefig(
    "results/05_pi_anti_windup.png",
    dpi=150,
)

plt.show()
