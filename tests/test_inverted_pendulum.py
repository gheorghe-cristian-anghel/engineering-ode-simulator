import numpy as np
import pytest
from numpy.testing import assert_allclose

from models.inverted_pendulum import (
    inverted_pendulum_dynamics,
    linearized_inverted_pendulum_state_space,
    simulate_inverted_pendulum,
)


def test_dynamics_returns_derivative_with_length_four():
    """The nonlinear dynamics should return one derivative per state."""
    derivative = inverted_pendulum_dynamics(0.0, [0.0, 0.0, 0.1, 0.0])

    assert len(derivative) == 4


def test_zero_state_zero_force_is_upright_equilibrium():
    """The upright state should remain at rest with zero force."""
    derivative = inverted_pendulum_dynamics(0.0, [0.0, 0.0, 0.0, 0.0])

    assert_allclose(derivative, [0.0, 0.0, 0.0, 0.0], atol=1e-12)


def test_simulate_inverted_pendulum_returns_expected_shapes():
    """Simulation should return time and state arrays with expected shapes."""
    t, states = simulate_inverted_pendulum([0.0, 0.0, 0.1, 0.0], num_points=50)

    assert t.shape == (50,)
    assert states.shape == (50, 4)


def test_initial_state_is_preserved():
    """The first simulated sample should equal the initial state."""
    x0 = [1.0, -0.2, 0.1, 0.3]

    t, states = simulate_inverted_pendulum(x0, t_span=(0.0, 1.0), num_points=25)

    assert t[0] == pytest.approx(0.0)
    assert_allclose(states[0], x0)


def test_invalid_m_cart_raises_value_error():
    """Cart mass must be positive."""
    with pytest.raises(ValueError):
        simulate_inverted_pendulum([0.0, 0.0, 0.0, 0.0], M=0.0)


def test_invalid_m_pendulum_raises_value_error():
    """Pendulum mass must be positive."""
    with pytest.raises(ValueError):
        simulate_inverted_pendulum([0.0, 0.0, 0.0, 0.0], m=0.0)


def test_invalid_l_raises_value_error():
    """Pendulum length must be positive."""
    with pytest.raises(ValueError):
        simulate_inverted_pendulum([0.0, 0.0, 0.0, 0.0], l=0.0)


def test_invalid_g_raises_value_error():
    """Gravity must be positive."""
    with pytest.raises(ValueError):
        simulate_inverted_pendulum([0.0, 0.0, 0.0, 0.0], g=0.0)


def test_invalid_b_raises_value_error():
    """Cart damping must be nonnegative."""
    with pytest.raises(ValueError):
        simulate_inverted_pendulum([0.0, 0.0, 0.0, 0.0], b=-0.1)


def test_invalid_x0_length_raises_value_error():
    """Initial state must contain four values."""
    with pytest.raises(ValueError):
        simulate_inverted_pendulum([0.0, 0.0, 0.0])


def test_invalid_num_points_raises_value_error():
    """Number of samples must be positive."""
    with pytest.raises(ValueError):
        simulate_inverted_pendulum([0.0, 0.0, 0.0, 0.0], num_points=0)


def test_invalid_t_span_raises_value_error():
    """Final time must be greater than initial time."""
    with pytest.raises(ValueError):
        simulate_inverted_pendulum([0.0, 0.0, 0.0, 0.0], t_span=(1.0, 0.0))


def test_linearized_state_space_matrix_shapes():
    """Linearized helper should return matrices with expected dimensions."""
    A, B, C, D = linearized_inverted_pendulum_state_space()

    assert A.shape == (4, 4)
    assert B.shape == (4, 1)
    assert C.shape == (2, 4)
    assert D.shape == (2, 1)


def test_linearized_upright_model_has_unstable_eigenvalue():
    """The upright open-loop linearized model should have an unstable mode."""
    A, _, _, _ = linearized_inverted_pendulum_state_space()
    eigenvalues = np.linalg.eigvals(A)

    assert np.max(np.real(eigenvalues)) > 0.0
