import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp

from src.motor import DCMotor
from src.controllers import PIDController


# ============================================================
# Experiment configuration
# ============================================================

speed_reference = 30.0

Kp = 1.0
Ki = 0.5
Kd = 0.01

voltage_min = 0.0
voltage_max = 12.0

load_torque = 0.05

simulation_time = 10.0

# Controller sampling time
Ts = 0.001


# ============================================================
# Create motor and controller
# ============================================================

motor = DCMotor()

controller = PIDController(
    kp=Kp,
    ki=Ki,
    kd=Kd,
    output_min=voltage_min,
    output_max=voltage_max,
    derivative_filter_tau=0.01,
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
derivative = np.zeros(n_steps)

# ============================================================
# Initial motor state
# ============================================================

x = np.array([
    0.0,  # current (A)
    0.0,  # speed (rad/s)
])

# ============================================================
# Digital PID controller + continuous motor
# ============================================================

for k in range(n_steps - 1):

    # --------------------------------------------------------
    # Measure motor state
    # --------------------------------------------------------

    current[k] = x[0]
    speed[k] = x[1]

    # --------------------------------------------------------
    # Calculate tracking error
    # --------------------------------------------------------

    error[k] = speed_reference - speed[k]

    # --------------------------------------------------------
    # PID controller
    # --------------------------------------------------------

    voltage[k] = controller.update(
        error=error[k],
        measurement=speed[k],
        dt=Ts,
    )

    # Store controller states
    integral[k] = controller.integral
    derivative[k] = controller.derivative

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

error[-1] = speed_reference - speed[-1]

voltage[-1] = voltage[-2]

integral[-1] = controller.integral
derivative[-1] = controller.derivative

# ============================================================
# Performance metrics
# ============================================================

final_speed = speed[-1]

steady_state_error = (
    speed_reference - final_speed
)

peak_speed = np.max(speed)

overshoot = max(
    0.0,
    (peak_speed - speed_reference)
    / speed_reference
    * 100.0,
)

# 2% settling band
upper_band = speed_reference * 1.02
lower_band = speed_reference * 0.98

outside_band = np.where(
    (speed > upper_band)
    | (speed < lower_band)
)[0]

if len(outside_band) == 0:
    settling_time = 0.0

elif outside_band[-1] < len(time) - 1:
    settling_time = time[outside_band[-1] + 1]

else:
    settling_time = np.nan

# ============================================================
# Results
# ============================================================

print("=== PID Speed Control Experiment ===")

print(f"Reference speed:      {speed_reference:.4f} rad/s")
print(f"Kp:                   {Kp:.4f}")
print(f"Ki:                   {Ki:.4f}")
print(f"Kd:                   {Kd:.4f}")

print(f"Final speed:          {final_speed:.4f} rad/s")
print(f"Final current:        {current[-1]:.4f} A")
print(f"Final voltage:        {voltage[-1]:.4f} V")
print(f"Steady-state error:   {steady_state_error:.4f} rad/s")
print(f"Peak speed:           {peak_speed:.4f} rad/s")
print(f"Overshoot:            {overshoot:.2f} %")
print(f"Settling time:        {settling_time:.4f} s")

# ============================================================
# Plots
# ============================================================

fig, axes = plt.subplots(
    4,
    1,
    figsize=(10, 10),
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
axes[0].set_title("PID Speed Tracking")
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
    label="Integral",
)

axes[2].set_ylabel("Integral")
axes[2].legend()
axes[2].grid(True)

# Derivative
axes[3].plot(
    time,
    derivative,
    label="Filtered derivative",
)

axes[3].set_ylabel("Derivative")
axes[3].set_xlabel("Time (s)")
axes[3].legend()
axes[3].grid(True)

plt.tight_layout()

plt.savefig(
    "results/06_pid_speed_control.png",
    dpi=150,
)

plt.show()
