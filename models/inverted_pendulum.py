"""Inverted pendulum / cart-pole model.

This module models a pendulum mounted on a moving cart. The pendulum angle
``theta`` is measured from the upright vertical equilibrium:

- ``theta = 0`` means the pendulum is perfectly upright.
- Positive ``theta`` means the pendulum leans to the right.
- Positive cart force pushes the cart to the right.

The open-loop upright equilibrium is unstable without feedback control.
"""

import numpy as np
from scipy.integrate import solve_ivp


def validate_inverted_pendulum_parameters(M, m, l, g, b):
    """Validate cart-pole physical parameters."""
    if M <= 0:
        raise ValueError("M must be positive")

    if m <= 0:
        raise ValueError("m must be positive")

    if l <= 0:
        raise ValueError("l must be positive")

    if g <= 0:
        raise ValueError("g must be positive")

    if b < 0:
        raise ValueError("b must be nonnegative")


def _validate_initial_state(x0):
    """Return a validated four-element initial state vector."""
    initial_state = np.asarray(x0, dtype=float)

    if initial_state.ndim != 1 or len(initial_state) != 4:
        raise ValueError("x0 must be a one-dimensional state vector of length 4")

    if not np.all(np.isfinite(initial_state)):
        raise ValueError("x0 must contain only finite values")

    return initial_state


def zero_force(t):
    """Return zero cart force."""
    return 0.0


def inverted_pendulum_dynamics(
    t,
    state,
    M=1.0,
    m=0.1,
    l=0.5,
    g=9.81,
    b=0.0,
    force_func=None,
):
    """Return derivatives for the nonlinear inverted pendulum model.

    Parameters
    ----------
    t : float
        Time in seconds.
    state : array-like
        Current state ``[x, x_dot, theta, theta_dot]``. ``x`` is cart position
        in meters, and ``theta`` is pendulum angle in radians measured from the
        upright vertical. Positive ``theta`` leans to the right.
    M : float, optional
        Cart mass in kilograms.
    m : float, optional
        Pendulum mass in kilograms.
    l : float, optional
        Pendulum length to the center of mass in meters.
    g : float, optional
        Gravitational acceleration in meters per second squared.
    b : float, optional
        Cart viscous damping/friction in newton-seconds per meter.
    force_func : callable, optional
        Function returning horizontal cart force ``F(t)`` in newtons. Positive
        force pushes the cart to the right. If None, zero force is used.
    """
    validate_inverted_pendulum_parameters(M, m, l, g, b)

    if force_func is None:
        force_func = zero_force

    if not callable(force_func):
        raise ValueError("force_func must be callable")

    x, x_dot, theta, theta_dot = np.asarray(state, dtype=float)
    force = float(force_func(t))

    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    denominator = M + m * sin_theta**2

    x_ddot = (
        force
        - b * x_dot
        + m * l * theta_dot**2 * sin_theta
        - m * g * sin_theta * cos_theta
    ) / denominator
    theta_ddot = (g * sin_theta - x_ddot * cos_theta) / l

    return np.array([x_dot, x_ddot, theta_dot, theta_ddot])


def simulate_inverted_pendulum(
    x0,
    t_span=(0.0, 5.0),
    num_points=1000,
    M=1.0,
    m=0.1,
    l=0.5,
    g=9.81,
    b=0.0,
    force_func=None,
):
    """Simulate nonlinear inverted pendulum motion with ``solve_ivp``.

    Parameters
    ----------
    x0 : array-like
        Initial state ``[x, x_dot, theta, theta_dot]``.
    t_span : tuple, optional
        Start and end time in seconds.
    num_points : int, optional
        Number of time samples to evaluate.
    M, m, l, g, b : float, optional
        Cart-pole physical parameters.
    force_func : callable, optional
        Function returning horizontal cart force ``F(t)`` in newtons.

    Returns
    -------
    tuple
        ``(t, states)`` where ``states`` has shape ``(num_points, 4)`` and
        columns ``[x, x_dot, theta, theta_dot]``.
    """
    validate_inverted_pendulum_parameters(M, m, l, g, b)
    initial_state = _validate_initial_state(x0)

    if num_points <= 0:
        raise ValueError("num_points must be positive")

    if t_span[1] <= t_span[0]:
        raise ValueError("t_span final time must be greater than initial time")

    if force_func is None:
        force_func = zero_force

    if not callable(force_func):
        raise ValueError("force_func must be callable")

    t_eval = np.linspace(t_span[0], t_span[1], num_points)

    solution = solve_ivp(
        inverted_pendulum_dynamics,
        t_span,
        initial_state,
        args=(M, m, l, g, b, force_func),
        t_eval=t_eval,
        rtol=1e-9,
        atol=1e-11,
    )

    if not solution.success:
        raise RuntimeError(f"inverted pendulum integration failed: {solution.message}")

    return solution.t, solution.y.T


def linearized_inverted_pendulum_state_space(
    M=1.0,
    m=0.1,
    l=0.5,
    g=9.81,
    b=0.0,
):
    """Return the upright linearized state-space model.

    The model is linearized around the open-loop upright equilibrium:
    ``theta = 0``, cart velocity ``x_dot = 0``, pendulum angular velocity
    ``theta_dot = 0``, and cart force ``F = 0``.

    State is ``[x, x_dot, theta, theta_dot]`` and input is cart force ``F``.
    Outputs are cart position and pendulum angle. The upright equilibrium has
    an unstable mode without feedback control. This helper does not implement
    LQR or any other stabilizing controller.
    """
    validate_inverted_pendulum_parameters(M, m, l, g, b)

    A = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, -b / M, -m * g / M, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, b / (M * l), (M + m) * g / (M * l), 0.0],
        ]
    )
    B = np.array([[0.0], [1.0 / M], [0.0], [-1.0 / (M * l)]])
    C = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
    D = np.array([[0.0], [0.0]])

    return A, B, C, D
