import numpy as np
import pytest

from models.dc_motor import (
    rad_per_sec_to_rpm,
    simulate_dc_motor,
    steady_state_current,
    steady_state_speed,
)


def test_rad_per_sec_to_rpm():
    """One revolution per second should be 60 rpm."""
    assert rad_per_sec_to_rpm(2 * np.pi) == pytest.approx(60)


def test_steady_state_speed():
    """Steady-state speed should match the DC motor algebraic solution."""
    omega_ss = steady_state_speed(
        R=1,
        b=0.001,
        Kt=0.01,
        Ke=0.01,
        voltage=12,
        load_torque=0,
    )

    assert omega_ss == pytest.approx(109.09, rel=0.001)


def test_steady_state_current():
    """Steady-state current should balance viscous damping torque."""
    current_ss = steady_state_current(
        b=0.001,
        Kt=0.01,
        omega_ss=109.09,
        load_torque=0,
    )

    assert current_ss == pytest.approx(10.909, rel=0.001)


def test_initial_current_equals_i0():
    """The simulated armature current should start at i0."""
    t, current, _ = simulate_dc_motor(1, 0.5, 0.01, 0.001, 0.01, 0.01, 2, 0, (0, 5), 200)

    assert t[0] == pytest.approx(0)
    assert current[0] == pytest.approx(2)


def test_initial_speed_equals_omega0():
    """The simulated angular speed should start at omega0."""
    _, _, omega = simulate_dc_motor(1, 0.5, 0.01, 0.001, 0.01, 0.01, 0, 3, (0, 5), 200)

    assert omega[0] == pytest.approx(3)


def test_final_speed_approaches_steady_state_speed():
    """For a long simulation, motor speed should approach steady state."""
    R = 1
    L = 0.5
    J = 0.01
    b = 0.001
    Kt = 0.01
    Ke = 0.01
    voltage = 12

    _, _, omega = simulate_dc_motor(R, L, J, b, Kt, Ke, 0, 0, (0, 80), 4000)
    omega_ss = steady_state_speed(R, b, Kt, Ke, voltage)

    assert omega[-1] == pytest.approx(omega_ss, rel=0.01)


def test_final_current_approaches_steady_state_current():
    """For a long simulation, armature current should approach steady state."""
    R = 1
    L = 0.5
    J = 0.01
    b = 0.001
    Kt = 0.01
    Ke = 0.01
    voltage = 12

    _, current, _ = simulate_dc_motor(R, L, J, b, Kt, Ke, 0, 0, (0, 80), 4000)
    omega_ss = steady_state_speed(R, b, Kt, Ke, voltage)
    current_ss = steady_state_current(b, Kt, omega_ss)

    assert current[-1] == pytest.approx(current_ss, rel=0.01)


def test_invalid_r_raises_value_error():
    """Resistance must be positive."""
    with pytest.raises(ValueError):
        simulate_dc_motor(0, 0.5, 0.01, 0.001, 0.01, 0.01, 0, 0, (0, 5), 200)


def test_invalid_l_raises_value_error():
    """Inductance must be positive."""
    with pytest.raises(ValueError):
        simulate_dc_motor(1, 0, 0.01, 0.001, 0.01, 0.01, 0, 0, (0, 5), 200)


def test_invalid_j_raises_value_error():
    """Rotor inertia must be positive."""
    with pytest.raises(ValueError):
        simulate_dc_motor(1, 0.5, 0, 0.001, 0.01, 0.01, 0, 0, (0, 5), 200)


def test_invalid_b_raises_value_error():
    """Viscous damping must be nonnegative."""
    with pytest.raises(ValueError):
        simulate_dc_motor(1, 0.5, 0.01, -0.001, 0.01, 0.01, 0, 0, (0, 5), 200)


def test_invalid_kt_raises_value_error():
    """Torque constant must be positive."""
    with pytest.raises(ValueError):
        simulate_dc_motor(1, 0.5, 0.01, 0.001, 0, 0.01, 0, 0, (0, 5), 200)


def test_invalid_ke_raises_value_error():
    """Back-emf constant must be positive."""
    with pytest.raises(ValueError):
        simulate_dc_motor(1, 0.5, 0.01, 0.001, 0.01, 0, 0, 0, (0, 5), 200)


def test_invalid_num_points_raises_value_error():
    """Number of samples must be positive."""
    with pytest.raises(ValueError):
        simulate_dc_motor(1, 0.5, 0.01, 0.001, 0.01, 0.01, 0, 0, (0, 5), 0)


def test_invalid_t_span_raises_value_error():
    """The final simulation time must be greater than the initial time."""
    with pytest.raises(ValueError):
        simulate_dc_motor(1, 0.5, 0.01, 0.001, 0.01, 0.01, 0, 0, (5, 0), 200)
