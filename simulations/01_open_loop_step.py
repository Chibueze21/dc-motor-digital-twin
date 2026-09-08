import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from src.motor import DCMotor


motor = DCMotor()

# Simulation settings
t_start = 0.0
t_end = 15.0

# Constant applied voltage
voltage = 12.0


def dynamics(t, x):
    return motor.derivatives(
        x,
        voltage=voltage,
        load_torque=0.0,
    )


# Initial state: motor at rest
x0 = np.array([0.0, 0.0])

t_eval = np.linspace(t_start, t_end, 1000)

solution = solve_ivp(
    dynamics,
    (t_start, t_end),
    x0,
    t_eval=t_eval,
    rtol=1e-8,
    atol=1e-10,
)

current = solution.y[0]
speed = solution.y[1]


# print results

print(f"Final current: {current[-1]:.4f} A")
print(f"Final speed:   {speed[-1]:.4f} rad/s")
print(f"Final speed:   {speed[-1] * 60 / (2 * np.pi):.2f} RPM")

# Plot results
fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

axes[0].plot(solution.t, current)
axes[0].set_ylabel("Current [A]")
axes[0].grid(True)

axes[1].plot(solution.t, speed)
axes[1].set_xlabel("Time [s]")
axes[1].set_ylabel("Speed [rad/s]")
axes[1].grid(True)

fig.suptitle("DC Motor Open-Loop Response to 12 V Step")


plt.tight_layout()

output_path = "results/01_open_loop_step.png"
plt.savefig(output_path, dpi=200, bbox_inches="tight")

print(f"Plot saved to: {output_path}")
