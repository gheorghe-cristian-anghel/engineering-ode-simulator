import numpy as np
import pytest

from models.pendulum import (
    natural_frequency,
    pendulum_energy,
    simulate_pendulum,
    small_angle_period,
)


def test_natural_frequency():
    """For L = 1 m, natural frequency should be sqrt(g)."""
    assert natural_frequency(1.0, 9.81) == pytest.approx(np.sqrt(9.81))


def test_small_angle_period():
    """Small-angle period should be 2*pi*sqrt(L/g)."""
    assert small_angle_period(1.0, 9.81) == pytest.approx(
        2 * np.pi * np.sqrt(1 / 9.81)
    )


def test_initial_theta_matches_theta0():
    """The simulated angle should start at theta0."""
    theta0 = np.radians(30)

    t, theta, _ = simulate_pendulum(1.0, theta0, 0, (0, 10), 200)

    assert t[0] == pytest.approx(0)
    assert theta[0] == pytest.approx(theta0)


def test_initial_omega_matches_omega0():
    """The simulated angular velocity should start at omega0."""
    omega0 = 0.5

    _, _, omega = simulate_pendulum(1.0, 0.1, omega0, (0, 10), 200)

    assert omega[0] == pytest.approx(omega0)


def test_nonlinear_pendulum_conserves_energy():
    """An undamped nonlinear pendulum should approximately conserve energy."""
    theta0 = np.radians(30)

    _, theta, omega = simulate_pendulum(1.0, theta0, 0, (0, 10), 2000)
    energy = pendulum_energy(theta, omega, 1.0)

    assert energy[-1] == pytest.approx(energy[0], rel=1e-6)


def test_small_angle_nonlinear_and_linear_stay_close():
    """For a 5 degree initial angle, nonlinear and linear models should agree."""
    theta0 = np.radians(5)

    _, nonlinear_theta, _ = simulate_pendulum(1.0, theta0, 0, (0, 10), 2000)
    _, linear_theta, _ = simulate_pendulum(
        1.0,
        theta0,
        0,
        (0, 10),
        2000,
        linear=True,
    )

    assert np.max(np.abs(nonlinear_theta - linear_theta)) < 0.002


def test_invalid_l_raises_value_error():
    """Pendulum length must be positive."""
    with pytest.raises(ValueError):
        simulate_pendulum(0, 0.1, 0, (0, 10), 200)


def test_invalid_g_raises_value_error():
    """Gravity must be positive."""
    with pytest.raises(ValueError):
        simulate_pendulum(1.0, 0.1, 0, (0, 10), 200, g=0)


def test_invalid_num_points_raises_value_error():
    """Number of samples must be positive."""
    with pytest.raises(ValueError):
        simulate_pendulum(1.0, 0.1, 0, (0, 10), 0)


def test_invalid_t_span_raises_value_error():
    """The final simulation time must be greater than the initial time."""
    with pytest.raises(ValueError):
        simulate_pendulum(1.0, 0.1, 0, (10, 0), 200)
