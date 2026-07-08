import numpy as np
import pytest

from analysis.lqr import lqr
from models.inverted_pendulum_lqr import (
    design_inverted_pendulum_lqr,
    lqr_control_force,
    simulate_inverted_pendulum_lqr,
)


def test_lqr_returns_gain_with_expected_shape():
    """LQR gain should have shape (inputs, states)."""
    A = np.array([[0.0, 1.0], [-2.0, -0.5]])
    B = np.array([[0.0], [1.0]])
    Q = np.eye(2)
    R = np.array([[1.0]])

    K, _, _ = lqr(A, B, Q, R)

    assert K.shape == (1, 2)


def test_lqr_returns_riccati_solution_with_expected_shape():
    """Riccati solution should have shape (states, states)."""
    A = np.array([[0.0, 1.0], [-2.0, -0.5]])
    B = np.array([[0.0], [1.0]])
    Q = np.eye(2)
    R = np.array([[1.0]])

    _, P, _ = lqr(A, B, Q, R)

    assert P.shape == (2, 2)


def test_lqr_closed_loop_eigenvalues_are_stable_for_simple_system():
    """A controllable unstable system should be stabilized by LQR."""
    A = np.array([[0.0, 1.0], [1.0, 0.0]])
    B = np.array([[0.0], [1.0]])
    Q = np.eye(2)
    R = np.array([[1.0]])

    _, _, closed_loop_eigenvalues = lqr(A, B, Q, R)

    assert np.all(np.real(closed_loop_eigenvalues) < 0.0)


def test_valid_lqr_weight_matrices_are_accepted():
    """Symmetric PSD Q and positive definite R should produce finite gains."""
    A = np.array([[0.0, 1.0], [-2.0, -0.5]])
    B = np.array([[0.0], [1.0]])
    Q = np.diag([4.0, 0.5])
    R = np.array([[0.25]])

    K, P, closed_loop_eigenvalues = lqr(A, B, Q, R)

    assert np.all(np.isfinite(K))
    assert np.all(np.isfinite(P))
    assert np.all(np.isfinite(closed_loop_eigenvalues))


def test_invalid_a_dimensions_raise_value_error():
    """A must be square."""
    with pytest.raises(ValueError):
        lqr([[0.0, 1.0]], [[0.0], [1.0]], np.eye(2), [[1.0]])


def test_invalid_b_dimensions_raise_value_error():
    """B row count must match the number of states."""
    with pytest.raises(ValueError):
        lqr(np.eye(2), [[1.0]], np.eye(2), [[1.0]])


def test_invalid_q_dimensions_raise_value_error():
    """Q shape must match A shape."""
    with pytest.raises(ValueError):
        lqr(np.eye(2), [[0.0], [1.0]], np.eye(3), [[1.0]])


def test_invalid_r_dimensions_raise_value_error():
    """R shape must match the number of inputs."""
    with pytest.raises(ValueError):
        lqr(np.eye(2), [[0.0], [1.0]], np.eye(2), np.eye(2))


def test_nonsymmetric_q_raises_value_error():
    """Q must be symmetric."""
    with pytest.raises(ValueError):
        lqr(np.eye(2), [[0.0], [1.0]], [[1.0, 2.0], [0.0, 1.0]], [[1.0]])


def test_negative_semidefinite_q_raises_value_error():
    """Q must be positive semidefinite."""
    with pytest.raises(ValueError):
        lqr(np.eye(2), [[0.0], [1.0]], [[1.0, 0.0], [0.0, -1.0]], [[1.0]])


def test_nonsymmetric_r_raises_value_error():
    """R must be symmetric."""
    with pytest.raises(ValueError):
        lqr(
            np.eye(2),
            np.eye(2),
            np.eye(2),
            [[1.0, 1.0], [0.0, 1.0]],
        )


def test_nonpositive_definite_r_raises_value_error():
    """R must be symmetric positive definite."""
    with pytest.raises(ValueError):
        lqr(np.eye(2), [[0.0], [1.0]], np.eye(2), [[0.0]])


def test_design_inverted_pendulum_lqr_gain_shape():
    """Inverted pendulum LQR gain should have one input and four states."""
    K, _, _, _, _, _ = design_inverted_pendulum_lqr()

    assert K.shape == (1, 4)


def test_inverted_pendulum_lqr_closed_loop_eigenvalues_are_stable():
    """The linearized upright model should be stable under LQR feedback."""
    _, closed_loop_eigenvalues, _, _, _, _ = design_inverted_pendulum_lqr()

    assert np.all(np.real(closed_loop_eigenvalues) < 0.0)


def test_simulate_inverted_pendulum_lqr_returns_expected_shapes():
    """LQR simulation should return time, state, and force arrays."""
    K, _, _, _, _, _ = design_inverted_pendulum_lqr()

    t, states, control_force = simulate_inverted_pendulum_lqr(
        [0.0, 0.0, np.radians(5.0), 0.0],
        K,
        t_span=(0.0, 2.0),
        num_points=200,
        force_limit=20.0,
    )

    assert t.shape == (200,)
    assert states.shape == (200, 4)
    assert control_force.shape == (200,)


def test_small_initial_angle_is_reduced_by_lqr():
    """For a small perturbation, LQR should reduce angle magnitude."""
    initial_angle = np.radians(5.0)
    K, _, _, _, _, _ = design_inverted_pendulum_lqr()

    _, states, _ = simulate_inverted_pendulum_lqr(
        [0.0, 0.0, initial_angle, 0.0],
        K,
        t_span=(0.0, 5.0),
        num_points=1000,
        force_limit=20.0,
    )

    assert abs(states[-1, 2]) < abs(initial_angle)


def test_lqr_control_force_is_finite():
    """Returned LQR control force should be finite."""
    K, _, _, _, _, _ = design_inverted_pendulum_lqr()
    force = lqr_control_force([0.0, 0.0, np.radians(5.0), 0.0], K)

    assert np.isfinite(force)


def test_lqr_control_force_respects_force_limit():
    """Optional force limit should clip the LQR force."""
    K, _, _, _, _, _ = design_inverted_pendulum_lqr()
    force = lqr_control_force([0.0, 0.0, np.radians(20.0), 0.0], K, force_limit=1.0)

    assert abs(force) <= 1.0
