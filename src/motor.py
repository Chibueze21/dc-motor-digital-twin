"""
DC motor state-space model.

States:
    x[0] = armature current [A]
    x[1] = angular velocity [rad/s]

Input:
    u = armature voltage [V]

Disturbance:
    tau_load = load torque [N*m]
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class DCMotorParameters:
    R: float = 2.0       # Armature resistance [Ohm]
    L: float = 0.5       # Armature inductance [H]
    Kt: float = 0.1      # Torque constant [N*m/A]
    Ke: float = 0.1      # Back-EMF constant [V*s/rad]
    J: float = 0.02      # Rotor inertia [kg*m^2]
    b: float = 0.01      # Viscous friction [N*m*s/rad]


class DCMotor:
    """Continuous-time state-space model of a permanent-magnet DC motor."""

    def __init__(self, parameters: DCMotorParameters | None = None):
        self.p = parameters or DCMotorParameters()

    def matrices(self):
        """Return the state-space matrices A, B, E."""

        p = self.p

        A = np.array([
            [-p.R / p.L, -p.Ke / p.L],
            [ p.Kt / p.J, -p.b / p.J],
        ])

        B = np.array([
            [1.0 / p.L],
            [0.0],
        ])

        E = np.array([
            [0.0],
            [-1.0 / p.J],
        ])

        return A, B, E

    def derivatives(self, x, voltage, load_torque=0.0):
        """Calculate state derivatives."""

        A, B, E = self.matrices()

        x = np.asarray(x, dtype=float).reshape(2, 1)

        dx = A @ x + B * voltage + E * load_torque

        return dx.ravel()
