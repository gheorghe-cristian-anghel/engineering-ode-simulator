import numpy as np
import pytest
from numpy.testing import assert_allclose

from models.quadcopter_attitude import (
    constant_torque,
    linearized_attitude_state_space,
    quadcopter_attitude_dynamics,
    simulate_quadcopter_attitude,
    torque_step,
)


def test_constant_torque_returns_expected_torque_vector():
    """Constant torque helper should return the same vector for any time."""
    torque_func = constant_torque(0.01, -0.02, 0.03)

    assert_allclose(torque_func(0.0), [0.01, -0.02, 0.03])
    assert_allclose(torque_func(5.0), [0.01, -0.02, 0.03])


def test_torque_step_returns_before_and_after_vectors():
    """Torque step should switch at and after the step time."""
    torque_func = torque_step(
        t_step=1.0,
        before=(0.0, 0.0, 0.0),
        after=(0.01, -0.008, 0.005),
    )

    assert_allclose(torque_func(0.99), [0.0, 0.0, 0.0])
    assert_allclose(torque_func(1.0), [0.01, -0.008, 0.005])
    assert_allclose(torque_func(2.0), [0.01, -0.008, 0.005])


def test_zero_torque_zero_state_gives_zero_derivative():
    """Zero torque at zero attitude should produce zero derivative."""
    derivative = quadcopter_attitude_dynamics(0.0, np.zeros(6))

    assert_allclose(derivative, np.zeros(6), atol=1e-12)


def test_positive_roll_torque_gives_positive_roll_acceleration():
    """Positive roll torque should create positive roll angular acceleration."""
    derivative = quadcopter_attitude_dynamics(
        0.0,
        np.zeros(6),
        Ixx=0.02,
        torque_func=constant_torque(tau_phi=0.01),
    )

    assert derivative[3] > 0.0
    assert derivative[3] == pytest.approx(0.5)


def test_simulate_quadcopter_attitude_returns_expected_shapes():
    """Attitude simulation should return time, state, and torque arrays."""
    t, states, torques = simulate_quadcopter_attitude(num_points=50)

    assert t.shape == (50,)
    assert states.shape == (50, 6)
    assert torques.shape == (50, 3)


def test_initial_state_is_preserved():
    """The first simulated state sample should match the initial state."""
    initial_state = [0.1, -0.2, 0.3, 0.4, -0.5, 0.6]

    t, states, _ = simulate_quadcopter_attitude(
        initial_state=initial_state,
        t_span=(0.0, 1.0),
        num_points=20,
    )

    assert t[0] == pytest.approx(0.0)
    assert_allclose(states[0], initial_state)


def test_invalid_ixx_raises_value_error():
    """Roll inertia must be positive."""
    with pytest.raises(ValueError):
        simulate_quadcopter_attitude(Ixx=0.0)


def test_invalid_iyy_raises_value_error():
    """Pitch inertia must be positive."""
    with pytest.raises(ValueError):
        simulate_quadcopter_attitude(Iyy=0.0)


def test_invalid_izz_raises_value_error():
    """Yaw inertia must be positive."""
    with pytest.raises(ValueError):
        simulate_quadcopter_attitude(Izz=0.0)


def test_invalid_initial_state_length_raises_value_error():
    """Initial attitude state must contain six values."""
    with pytest.raises(ValueError):
        simulate_quadcopter_attitude(initial_state=[0.0, 0.0, 0.0])


def test_invalid_t_span_raises_value_error():
    """Final time must be greater than initial time."""
    with pytest.raises(ValueError):
        simulate_quadcopter_attitude(t_span=(1.0, 0.0))


def test_invalid_num_points_raises_value_error():
    """Number of samples must be positive."""
    with pytest.raises(ValueError):
        simulate_quadcopter_attitude(num_points=0)


def test_invalid_torque_vector_length_raises_value_error():
    """Torque function must return three torque values."""
    with pytest.raises(ValueError):
        quadcopter_attitude_dynamics(
            0.0,
            np.zeros(6),
            torque_func=lambda t: [0.0, 0.0],
        )


def test_linearized_attitude_state_space_returns_correct_matrix_shapes():
    """Linearized attitude matrices should have expected dimensions."""
    A, B, C, D = linearized_attitude_state_space()

    assert A.shape == (6, 6)
    assert B.shape == (6, 3)
    assert C.shape == (6, 6)
    assert D.shape == (6, 3)


def test_linearized_attitude_state_space_returns_expected_values():
    """Linearized attitude matrices should match the decoupled model."""
    A, B, C, D = linearized_attitude_state_space(Ixx=0.02, Iyy=0.04, Izz=0.05)

    expected_A = np.array(
        [
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    expected_B = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [50.0, 0.0, 0.0],
            [0.0, 25.0, 0.0],
            [0.0, 0.0, 20.0],
        ]
    )

    assert_allclose(A, expected_A)
    assert_allclose(B, expected_B)
    assert_allclose(C, np.eye(6))
    assert_allclose(D, np.zeros((6, 3)))
