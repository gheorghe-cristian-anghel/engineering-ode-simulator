import numpy as np
import pytest
from numpy.testing import assert_allclose

from models.quadcopter_altitude import (
    constant_thrust,
    hover_thrust,
    linearized_altitude_state_space,
    quadcopter_altitude_dynamics,
    simulate_quadcopter_altitude,
    thrust_step,
)


def test_hover_thrust_returns_m_times_g():
    """Hover thrust should balance vehicle weight."""
    assert hover_thrust(m=1.5, g=9.81) == pytest.approx(1.5 * 9.81)


def test_constant_thrust_returns_expected_value():
    """Constant thrust helper should return the same value for any time."""
    thrust_func = constant_thrust(12.0)

    assert thrust_func(0.0) == pytest.approx(12.0)
    assert thrust_func(10.0) == pytest.approx(12.0)


def test_thrust_step_returns_before_and_after_values():
    """Thrust step should switch at and after the step time."""
    thrust_func = thrust_step(t_step=2.0, thrust_before=9.0, thrust_after=11.0)

    assert thrust_func(1.99) == pytest.approx(9.0)
    assert thrust_func(2.0) == pytest.approx(11.0)
    assert thrust_func(3.0) == pytest.approx(11.0)


def test_dynamics_at_hover_with_zero_velocity_gives_zero_acceleration():
    """Hover thrust should produce zero vertical acceleration at rest."""
    m = 1.2
    g = 9.81
    derivative = quadcopter_altitude_dynamics(
        0.0,
        [0.0, 0.0],
        m=m,
        g=g,
        thrust_func=constant_thrust(hover_thrust(m, g)),
    )

    assert_allclose(derivative, [0.0, 0.0], atol=1e-12)


def test_below_hover_thrust_gives_negative_acceleration():
    """Thrust below hover should accelerate downward."""
    m = 1.0
    thrust = 0.9 * hover_thrust(m)

    derivative = quadcopter_altitude_dynamics(
        0.0,
        [0.0, 0.0],
        m=m,
        thrust_func=constant_thrust(thrust),
    )

    assert derivative[1] < 0.0


def test_above_hover_thrust_gives_positive_acceleration():
    """Thrust above hover should accelerate upward."""
    m = 1.0
    thrust = 1.1 * hover_thrust(m)

    derivative = quadcopter_altitude_dynamics(
        0.0,
        [0.0, 0.0],
        m=m,
        thrust_func=constant_thrust(thrust),
    )

    assert derivative[1] > 0.0


def test_simulation_returns_arrays_of_equal_length():
    """Altitude simulation should return time, state, and thrust arrays."""
    t, z, v, thrust = simulate_quadcopter_altitude(num_points=50)

    assert t.shape == (50,)
    assert z.shape == (50,)
    assert v.shape == (50,)
    assert thrust.shape == (50,)


def test_default_hover_simulation_remains_finite_and_physically_plausible():
    """Default hover thrust should hold altitude and keep states finite."""
    _, z, v, thrust = simulate_quadcopter_altitude(
        t_span=(0.0, 1.0),
        num_points=50,
    )

    assert np.all(np.isfinite(z))
    assert np.all(np.isfinite(v))
    assert np.all(np.isfinite(thrust))
    assert z[-1] == pytest.approx(z[0], abs=1e-9)
    assert v[-1] == pytest.approx(0.0, abs=1e-9)


def test_initial_altitude_and_velocity_match_inputs():
    """First simulated sample should preserve initial altitude and velocity."""
    t, z, v, _ = simulate_quadcopter_altitude(
        z0=2.0,
        v0=-0.5,
        t_span=(0.0, 1.0),
        num_points=20,
    )

    assert t[0] == pytest.approx(0.0)
    assert z[0] == pytest.approx(2.0)
    assert v[0] == pytest.approx(-0.5)


def test_invalid_mass_raises_value_error():
    """Vehicle mass must be positive."""
    with pytest.raises(ValueError):
        simulate_quadcopter_altitude(m=0.0)


def test_invalid_gravity_raises_value_error():
    """Gravity must be positive."""
    with pytest.raises(ValueError):
        simulate_quadcopter_altitude(g=0.0)


def test_invalid_drag_raises_value_error():
    """Linear drag coefficient must be nonnegative."""
    with pytest.raises(ValueError):
        simulate_quadcopter_altitude(c_drag=-0.1)


def test_invalid_t_span_raises_value_error():
    """Final time must be greater than initial time."""
    with pytest.raises(ValueError):
        simulate_quadcopter_altitude(t_span=(1.0, 0.0))


def test_invalid_num_points_raises_value_error():
    """Number of samples must be positive."""
    with pytest.raises(ValueError):
        simulate_quadcopter_altitude(num_points=0)


def test_linearized_altitude_state_space_returns_correct_shapes():
    """Hover-linearized state-space matrices should have expected dimensions."""
    A, B, C, D = linearized_altitude_state_space()

    assert A.shape == (2, 2)
    assert B.shape == (2, 1)
    assert C.shape == (2, 2)
    assert D.shape == (2, 1)


def test_linearized_altitude_state_space_returns_expected_values():
    """Hover-linearized matrices should match the altitude model."""
    A, B, C, D = linearized_altitude_state_space(m=2.0, c_drag=0.4)

    assert_allclose(A, [[0.0, 1.0], [0.0, -0.2]])
    assert_allclose(B, [[0.0], [0.5]])
    assert_allclose(C, [[1.0, 0.0], [0.0, 1.0]])
    assert_allclose(D, [[0.0], [0.0]])
