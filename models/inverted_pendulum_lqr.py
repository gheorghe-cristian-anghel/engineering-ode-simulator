"""LQR control helpers for the inverted pendulum / cart-pole model."""

import numpy as np
from scipy.integrate import solve_ivp

from analysis.lqr import lqr
from models.inverted_pendulum import (
    inverted_pendulum_dynamics,
    linearized_inverted_pendulum_state_space,
    validate_inverted_pendulum_parameters,
)


def _validate_state_vector(state, name):
    """Return a validated four-element state vector."""
    state_array = np.asarray(state, dtype=float)

    if state_array.ndim != 1 or len(state_array) != 4:
        raise ValueError(f"{name} must be a one-dimensional vector of length 4")

    if not np.all(np.isfinite(state_array)):
        raise ValueError(f"{name} must contain only finite values")

    return state_array


def _validate_lqr_gain(K):
    """Return a validated single-input LQR gain matrix."""
    gain = np.asarray(K, dtype=float)

    if gain.shape != (1, 4):
        raise ValueError("K must have shape (1, 4)")

    if not np.all(np.isfinite(gain)):
        raise ValueError("K must contain only finite values")

    return gain


def _validate_force_limit(force_limit):
    """Validate and return an optional symmetric force limit."""
    if force_limit is None:
        return None

    force_limit = float(force_limit)

    if force_limit <= 0:
        raise ValueError("force_limit must be positive")

    return force_limit


def design_inverted_pendulum_lqr(
    M=1.0,
    m=0.1,
    l=0.5,
    g=9.81,
    b=0.0,
    Q=None,
    R=None,
):
    """Design an LQR gain for the upright linearized inverted pendulum.

    The gain uses the state convention ``[x, x_dot, theta, theta_dot]`` from
    ``models.inverted_pendulum`` and the continuous-time feedback law
    ``u = -K*x``. The resulting controller is intended for small deviations
    near the upright equilibrium.
    """
    validate_inverted_pendulum_parameters(M, m, l, g, b)

    if Q is None:
        Q = np.diag([1.0, 1.0, 100.0, 10.0])

    if R is None:
        R = np.array([[0.1]])

    A, B, C, D = linearized_inverted_pendulum_state_space(M, m, l, g, b)
    K, _, closed_loop_eigenvalues = lqr(A, B, Q, R)

    return K, closed_loop_eigenvalues, A, B, C, D


def lqr_control_force(state, K, reference_state=None, force_limit=None):
    """Return the LQR cart force for ``u = -K*(state - reference_state)``."""
    state = _validate_state_vector(state, "state")
    K = _validate_lqr_gain(K)
    force_limit = _validate_force_limit(force_limit)

    if reference_state is None:
        reference_state = np.zeros(4)

    reference_state = _validate_state_vector(reference_state, "reference_state")
    state_error = state - reference_state
    force = float(-(K @ state_error)[0])

    if force_limit is not None:
        force = float(np.clip(force, -force_limit, force_limit))

    return force


def inverted_pendulum_lqr_dynamics(
    t,
    state,
    K,
    reference_state=None,
    force_limit=None,
    M=1.0,
    m=0.1,
    l=0.5,
    g=9.81,
    b=0.0,
):
    """Return nonlinear inverted pendulum derivatives under LQR feedback."""
    force = lqr_control_force(state, K, reference_state, force_limit)
    force_func = lambda sample_time: force

    return inverted_pendulum_dynamics(
        t,
        state,
        M=M,
        m=m,
        l=l,
        g=g,
        b=b,
        force_func=force_func,
    )


def simulate_inverted_pendulum_lqr(
    x0,
    K,
    t_span=(0.0, 5.0),
    num_points=1000,
    M=1.0,
    m=0.1,
    l=0.5,
    g=9.81,
    b=0.0,
    force_limit=None,
    reference_state=None,
):
    """Simulate nonlinear inverted pendulum stabilization with LQR.

    Returns
    -------
    tuple
        ``(t, states, control_force)`` where ``states`` has columns
        ``[x, x_dot, theta, theta_dot]`` and ``control_force`` is the LQR cart
        force reconstructed at each returned time sample.
    """
    validate_inverted_pendulum_parameters(M, m, l, g, b)
    initial_state = _validate_state_vector(x0, "x0")
    K = _validate_lqr_gain(K)
    force_limit = _validate_force_limit(force_limit)

    if reference_state is None:
        reference_state = np.zeros(4)

    reference_state = _validate_state_vector(reference_state, "reference_state")

    if num_points <= 0:
        raise ValueError("num_points must be positive")

    if t_span[1] <= t_span[0]:
        raise ValueError("t_span final time must be greater than initial time")

    t_eval = np.linspace(t_span[0], t_span[1], num_points)

    solution = solve_ivp(
        inverted_pendulum_lqr_dynamics,
        t_span,
        initial_state,
        args=(K, reference_state, force_limit, M, m, l, g, b),
        t_eval=t_eval,
        rtol=1e-9,
        atol=1e-11,
    )

    states = solution.y.T
    control_force = np.array(
        [
            lqr_control_force(sample_state, K, reference_state, force_limit)
            for sample_state in states
        ]
    )

    return solution.t, states, control_force
