import numpy as np
from src.motor import DCMotor


def test_state_space_dimensions():
    motor = DCMotor()

    A, B, E = motor.matrices()

    assert A.shape == (2, 2)
    assert B.shape == (2, 1)
    assert E.shape == (2, 1)


def test_motor_is_open_loop_stable():
    motor = DCMotor()

    A, _, _ = motor.matrices()

    eigenvalues = np.linalg.eigvals(A)

    assert np.all(np.real(eigenvalues) < 0)


def test_zero_voltage_zero_state_has_zero_derivative():
    motor = DCMotor()

    x = np.array([0.0, 0.0])

    dx = motor.derivatives(
        x,
        voltage=0.0,
        load_torque=0.0,
    )

    np.testing.assert_allclose(dx, [0.0, 0.0])


def test_positive_voltage_produces_positive_initial_current_derivative():
    motor = DCMotor()

    x = np.array([0.0, 0.0])

    dx = motor.derivatives(
        x,
        voltage=12.0,
        load_torque=0.0,
    )

    assert dx[0] > 0


def test_load_torque_opposes_acceleration():
    motor = DCMotor()

    x = np.array([0.0, 0.0])

    dx = motor.derivatives(
        x,
        voltage=0.0,
        load_torque=1.0,
    )

    assert dx[1] < 0

def test_analytical_steady_state():
    motor = DCMotor()

    p = motor.p

    voltage = 12.0

    expected_speed = voltage / (
        p.Ke + p.R * p.b / p.Kt
    )

    expected_current = (
        p.b / p.Kt
    ) * expected_speed

    np.testing.assert_allclose(
        expected_speed,
        40.0,
        rtol=1e-12,
    )

    np.testing.assert_allclose(
        expected_current,
        4.0,
        rtol=1e-12,
    )
