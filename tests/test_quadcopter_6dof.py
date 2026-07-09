import numpy as np
import pytest
from numpy.testing import assert_allclose

from models.quadcopter_6dof import (
    constant_control,
    control_step,
    euler_rate_matrix,
    hover_control,
    quadcopter_6dof_dynamics,
    rotation_matrix_body_to_inertial,
    simulate_quadcopter_6dof,
)


def test_rotation_matrix_body_to_inertial_returns_expected_shape():
    """Body-to-inertial rotation matrix should be 3 by 3."""
    rotation = rotation_matrix_body_to_inertial(0.1, -0.2, 0.3)

    assert rotation.shape == (3, 3)


def test_rotation_matrix_zero_angles_is_identity():
    """Zero roll, pitch, and yaw should produce the identity matrix."""
    rotation = rotation_matrix_body_to_inertial(0.0, 0.0, 0.0)

    assert_allclose(rotation, np.eye(3), atol=1e-12)


def test_rotation_matrix_is_orthonormal():
    """Body-to-inertial rotation should preserve vector lengths."""
    rotation = rotation_matrix_body_to_inertial(0.3, -0.2, 0.4)

    assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
    assert np.linalg.det(rotation) == pytest.approx(1.0)


def test_euler_rate_matrix_returns_expected_shape():
    """Euler rate matrix should map three body rates to three angle rates."""
    rate_matrix = euler_rate_matrix(0.1, 0.2)

    assert rate_matrix.shape == (3, 3)


def test_hover_control_returns_weight_and_zero_torques():
    """Hover control should balance weight with no body torque."""
    control = hover_control(m=1.5, g=9.81)

    assert_allclose(control, [1.5 * 9.81, 0.0, 0.0, 0.0])


def test_constant_control_returns_expected_control_vector():
    """Constant control helper should return the same vector for any time."""
    control_func = constant_control(10.0, 0.1, -0.2, 0.3)

    assert_allclose(control_func(0.0), [10.0, 0.1, -0.2, 0.3])
    assert_allclose(control_func(5.0), [10.0, 0.1, -0.2, 0.3])


def test_control_step_returns_before_and_after_vectors():
    """Control step should switch at and after the step time."""
    before = [9.81, 0.0, 0.0, 0.0]
    after = [10.0, 0.01, -0.02, 0.03]
    control_func = control_step(2.0, before=before, after=after)

    assert_allclose(control_func(1.99), before)
    assert_allclose(control_func(2.0), after)
    assert_allclose(control_func(3.0), after)


def test_dynamics_returns_derivative_length_12():
    """The 6-DOF dynamics function should return one derivative per state."""
    derivative = quadcopter_6dof_dynamics(0.0, np.zeros(12))

    assert derivative.shape == (12,)


def test_hover_at_zero_state_gives_zero_acceleration():
    """Hover thrust at level attitude should cancel gravity."""
    derivative = quadcopter_6dof_dynamics(
        0.0,
        np.zeros(12),
        m=1.2,
        g=9.81,
        control_func=constant_control(1.2 * 9.81),
    )

    assert_allclose(derivative[3:6], [0.0, 0.0, 0.0], atol=1e-12)
    assert_allclose(derivative[9:12], [0.0, 0.0, 0.0], atol=1e-12)


def test_below_hover_thrust_gives_negative_vertical_acceleration():
    """Below-hover thrust should accelerate the vehicle downward."""
    derivative = quadcopter_6dof_dynamics(
        0.0,
        np.zeros(12),
        control_func=constant_control(0.9 * 9.81),
    )

    assert derivative[5] < 0.0


def test_initial_tilt_with_hover_thrust_gives_horizontal_acceleration():
    """A pitched vehicle should redirect thrust into horizontal acceleration."""
    state = np.zeros(12)
    state[7] = np.radians(10.0)

    derivative = quadcopter_6dof_dynamics(
        0.0,
        state,
        control_func=constant_control(9.81),
    )

    assert derivative[3] > 0.0
    assert derivative[5] < 0.0


def test_positive_roll_with_hover_thrust_gives_negative_y_acceleration():
    """Positive roll redirects thrust toward negative inertial y."""
    state = np.zeros(12)
    state[6] = np.radians(10.0)

    derivative = quadcopter_6dof_dynamics(
        0.0,
        state,
        control_func=constant_control(9.81),
    )

    assert derivative[4] < 0.0


def test_positive_roll_torque_gives_positive_roll_acceleration():
    """Positive roll torque should create positive p_dot."""
    derivative = quadcopter_6dof_dynamics(
        0.0,
        np.zeros(12),
        Ixx=0.02,
        control_func=constant_control(9.81, tau_phi=0.01),
    )

    assert derivative[9] == pytest.approx(0.5)


def test_simulate_quadcopter_6dof_returns_expected_shapes():
    """6-DOF simulation should return time, state, and control arrays."""
    t, states, controls = simulate_quadcopter_6dof(num_points=50)

    assert t.shape == (50,)
    assert states.shape == (50, 12)
    assert controls.shape == (50, 4)


def test_default_hover_simulation_remains_finite_and_near_hover():
    """Default 6-DOF hover control should keep the vehicle near rest."""
    _, states, controls = simulate_quadcopter_6dof(
        t_span=(0.0, 1.0),
        num_points=50,
    )

    assert np.all(np.isfinite(states))
    assert np.all(np.isfinite(controls))
    assert_allclose(states[-1, 0:6], np.zeros(6), atol=1e-9)
    assert_allclose(states[-1, 6:12], np.zeros(6), atol=1e-9)
    assert controls[-1, 0] == pytest.approx(9.81)
    assert_allclose(controls[-1, 1:4], np.zeros(3), atol=1e-12)


def test_initial_state_is_preserved():
    """The first simulated state sample should match the initial state."""
    initial_state = np.array(
        [1.0, -2.0, 3.0, 0.1, -0.2, 0.3, 0.01, -0.02, 0.03, 0.4, -0.5, 0.6]
    )

    t, states, _ = simulate_quadcopter_6dof(
        initial_state=initial_state,
        t_span=(0.0, 1.0),
        num_points=20,
    )

    assert t[0] == pytest.approx(0.0)
    assert_allclose(states[0], initial_state)


def test_invalid_mass_raises_value_error():
    """Vehicle mass must be positive."""
    with pytest.raises(ValueError):
        simulate_quadcopter_6dof(m=0.0)


def test_invalid_gravity_raises_value_error():
    """Gravity must be positive."""
    with pytest.raises(ValueError):
        simulate_quadcopter_6dof(g=0.0)


def test_invalid_inertia_raises_value_error():
    """Principal inertias must be positive."""
    with pytest.raises(ValueError):
        simulate_quadcopter_6dof(Ixx=0.0)

    with pytest.raises(ValueError):
        simulate_quadcopter_6dof(Iyy=0.0)

    with pytest.raises(ValueError):
        simulate_quadcopter_6dof(Izz=0.0)


def test_invalid_drag_raises_value_error():
    """Linear drag coefficient must be nonnegative."""
    with pytest.raises(ValueError):
        simulate_quadcopter_6dof(c_drag=-0.1)


def test_invalid_control_vector_length_raises_value_error():
    """Control function must return four values."""
    with pytest.raises(ValueError):
        quadcopter_6dof_dynamics(
            0.0,
            np.zeros(12),
            control_func=lambda t: [9.81, 0.0, 0.0],
        )


def test_negative_thrust_raises_value_error():
    """Total thrust must be nonnegative."""
    with pytest.raises(ValueError):
        constant_control(-1.0)


def test_invalid_state_length_raises_value_error():
    """State vector must contain twelve values."""
    with pytest.raises(ValueError):
        quadcopter_6dof_dynamics(0.0, np.zeros(11))


def test_invalid_t_span_raises_value_error():
    """Final time must be greater than initial time."""
    with pytest.raises(ValueError):
        simulate_quadcopter_6dof(t_span=(1.0, 0.0))


def test_invalid_num_points_raises_value_error():
    """Number of samples must be positive."""
    with pytest.raises(ValueError):
        simulate_quadcopter_6dof(num_points=0)
