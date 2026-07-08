"""One-dimensional quadcopter altitude dynamics model.

This module models simplified vertical motion for a quadcopter. The state is
``[z, v]`` where ``z`` is altitude in meters and ``v`` is vertical velocity in
meters per second. Positive altitude, velocity, and thrust are upward.
"""

import numpy as np
from scipy.integrate import solve_ivp


def validate_quadcopter_altitude_parameters(m=1.0, g=9.81, c_drag=0.0):
    """Validate quadcopter altitude model parameters."""
    if m <= 0:
        raise ValueError("m must be positive")

    if g <= 0:
        raise ValueError("g must be positive")

    if c_drag < 0:
        raise ValueError("c_drag must be nonnegative")


def _validate_state(state):
    """Return a validated two-element altitude state vector."""
    state_array = np.asarray(state, dtype=float)

    if state_array.ndim != 1 or len(state_array) != 2:
        raise ValueError("state must be a one-dimensional vector of length 2")

    if not np.all(np.isfinite(state_array)):
        raise ValueError("state must contain only finite values")

    return state_array


def _validate_thrust(thrust, name="thrust"):
    """Return a validated nonnegative thrust value."""
    thrust = float(thrust)

    if not np.isfinite(thrust):
        raise ValueError(f"{name} must be finite")

    if thrust < 0:
        raise ValueError(f"{name} must be nonnegative")

    return thrust


def hover_thrust(m=1.0, g=9.81):
    """Return the thrust required to hover in newtons."""
    validate_quadcopter_altitude_parameters(m=m, g=g)

    return m * g


def constant_thrust(thrust):
    """Return a callable constant thrust command ``T(t)``."""
    thrust = _validate_thrust(thrust)

    return lambda t: thrust


def thrust_step(t_step, thrust_before, thrust_after):
    """Return a callable thrust step command.

    The command returns ``thrust_before`` before ``t_step`` and
    ``thrust_after`` at and after ``t_step``.
    """
    t_step = float(t_step)

    if not np.isfinite(t_step):
        raise ValueError("t_step must be finite")

    thrust_before = _validate_thrust(thrust_before, "thrust_before")
    thrust_after = _validate_thrust(thrust_after, "thrust_after")

    return lambda t: thrust_before if t < t_step else thrust_after


def quadcopter_altitude_dynamics(
    t,
    state,
    m=1.0,
    g=9.81,
    c_drag=0.0,
    thrust_func=None,
):
    """Return derivatives for the 1D quadcopter altitude model.

    Parameters
    ----------
    t : float
        Time in seconds.
    state : array-like
        Current state ``[z, v]`` where ``z`` is altitude and ``v`` is vertical
        velocity. Positive values are upward.
    m : float, optional
        Vehicle mass in kilograms.
    g : float, optional
        Gravitational acceleration in meters per second squared.
    c_drag : float, optional
        Linear vertical drag coefficient in newton-seconds per meter.
    thrust_func : callable, optional
        Function returning total thrust ``T(t)`` in newtons. If None, hover
        thrust is used.
    """
    validate_quadcopter_altitude_parameters(m=m, g=g, c_drag=c_drag)
    _, v = _validate_state(state)

    if thrust_func is None:
        thrust_func = constant_thrust(hover_thrust(m, g))

    if not callable(thrust_func):
        raise ValueError("thrust_func must be callable")

    thrust = _validate_thrust(thrust_func(t))

    dz_dt = v
    dv_dt = (thrust - m * g - c_drag * v) / m

    return np.array([dz_dt, dv_dt])


def simulate_quadcopter_altitude(
    z0=0.0,
    v0=0.0,
    t_span=(0.0, 5.0),
    num_points=1000,
    m=1.0,
    g=9.81,
    c_drag=0.0,
    thrust_func=None,
):
    """Simulate simplified 1D quadcopter altitude motion.

    Returns
    -------
    tuple
        ``(t, z, v, thrust)`` where ``z`` is altitude, ``v`` is vertical
        velocity, and ``thrust`` is the thrust command evaluated over time.
    """
    validate_quadcopter_altitude_parameters(m=m, g=g, c_drag=c_drag)

    if num_points <= 0:
        raise ValueError("num_points must be positive")

    if t_span[1] <= t_span[0]:
        raise ValueError("t_span final time must be greater than initial time")

    if thrust_func is None:
        thrust_func = constant_thrust(hover_thrust(m, g))

    if not callable(thrust_func):
        raise ValueError("thrust_func must be callable")

    initial_state = _validate_state([z0, v0])

    t_eval = np.linspace(t_span[0], t_span[1], num_points)

    solution = solve_ivp(
        quadcopter_altitude_dynamics,
        t_span,
        initial_state,
        args=(m, g, c_drag, thrust_func),
        t_eval=t_eval,
    )

    if not solution.success:
        raise RuntimeError(f"quadcopter altitude integration failed: {solution.message}")

    thrust = np.array(
        [_validate_thrust(thrust_func(sample_time)) for sample_time in solution.t]
    )

    return solution.t, solution.y[0], solution.y[1], thrust


def linearized_altitude_state_space(m=1.0, g=9.81, c_drag=0.0):
    """Return the hover-linearized altitude state-space model.

    The state is ``[z, v]`` and the input is thrust deviation from hover,
    ``delta_T = T - m*g``. Outputs are altitude and vertical velocity.
    """
    validate_quadcopter_altitude_parameters(m=m, g=g, c_drag=c_drag)

    A = np.array([[0.0, 1.0], [0.0, -c_drag / m]])
    B = np.array([[0.0], [1.0 / m]])
    C = np.array([[1.0, 0.0], [0.0, 1.0]])
    D = np.array([[0.0], [0.0]])

    return A, B, C, D
