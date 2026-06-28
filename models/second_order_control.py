"""Second-order control system step response model.

This module models the standard second-order transfer function response using
a first-order state-space form suitable for scipy.integrate.solve_ivp.
"""

import numpy as np
from scipy.integrate import solve_ivp


def validate_second_order_parameters(omega_n, zeta):
    """Validate natural frequency and damping ratio."""
    if omega_n <= 0:
        raise ValueError("omega_n must be positive")

    if zeta < 0:
        raise ValueError("zeta must be nonnegative")


def step_input(t, amplitude=1.0):
    """Return a constant step input amplitude."""
    return amplitude


def response_type(zeta):
    """Return a readable response type from damping ratio."""
    if np.isclose(zeta, 0):
        return "undamped"

    if zeta < 1:
        return "underdamped"

    if np.isclose(zeta, 1):
        return "critically damped"

    return "overdamped"


def damped_natural_frequency(omega_n, zeta):
    """Return damped natural frequency for an underdamped response."""
    validate_second_order_parameters(omega_n, zeta)

    if zeta >= 1:
        return None

    return omega_n * np.sqrt(1 - zeta**2)


def theoretical_overshoot_percent(zeta):
    """Return theoretical percent overshoot for an underdamped response."""
    if zeta < 0:
        raise ValueError("zeta must be nonnegative")

    if zeta <= 0 or zeta >= 1:
        return 0

    return np.exp(-zeta * np.pi / np.sqrt(1 - zeta**2)) * 100


def theoretical_peak_time(omega_n, zeta):
    """Return theoretical peak time for an underdamped response."""
    omega_d = damped_natural_frequency(omega_n, zeta)

    if omega_d is None or zeta == 0:
        return None

    return np.pi / omega_d


def approx_settling_time(omega_n, zeta, tolerance=0.02):
    """Return approximate settling time for a second-order response."""
    validate_second_order_parameters(omega_n, zeta)

    if tolerance <= 0:
        raise ValueError("tolerance must be positive")

    if zeta == 0:
        return None

    return -np.log(tolerance) / (zeta * omega_n)


def second_order_ode(t, state, omega_n, zeta, input_func):
    """Return derivatives for the second-order control system.

    Parameters
    ----------
    t : float
        Time in seconds.
    state : list or array
        Current state. ``state[0]`` is output y and ``state[1]`` is output
        velocity v.
    omega_n : float
        Undamped natural frequency in radians per second.
    zeta : float
        Damping ratio.
    input_func : callable
        Function that returns the input u(t).
    """
    y = state[0]
    v = state[1]
    input_value = input_func(t)

    dydt = v
    dvdt = omega_n**2 * (input_value - y) - 2 * zeta * omega_n * v

    return [dydt, dvdt]


def simulate_second_order_system(
    omega_n,
    zeta,
    y0,
    v0,
    t_span,
    num_points,
    input_func=None,
):
    """Simulate second-order system output with scipy.integrate.solve_ivp.

    Parameters
    ----------
    omega_n : float
        Undamped natural frequency in radians per second.
    zeta : float
        Damping ratio.
    y0 : float
        Initial output value.
    v0 : float
        Initial output velocity.
    t_span : tuple
        Start and end time in seconds, for example ``(0, 8)``.
    num_points : int
        Number of time points to evaluate.
    input_func : callable, optional
        Function that returns the input u(t). If None, a unit step is used.

    Returns
    -------
    tuple
        ``(t, y, v)`` where ``t`` is time, ``y`` is output, and ``v`` is output
        velocity.
    """
    validate_second_order_parameters(omega_n, zeta)

    if num_points <= 0:
        raise ValueError("num_points must be positive")

    if t_span[1] <= t_span[0]:
        raise ValueError("t_span final time must be greater than initial time")

    if input_func is None:
        input_func = step_input

    t_eval = np.linspace(t_span[0], t_span[1], num_points)

    solution = solve_ivp(
        second_order_ode,
        t_span,
        [y0, v0],
        args=(omega_n, zeta, input_func),
        t_eval=t_eval,
    )

    return solution.t, solution.y[0], solution.y[1]
