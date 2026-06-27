"""First-order control system step response model.

This module models the response of a standard first-order system to an input
signal. With a constant step input, the output moves smoothly toward its
steady-state value.
"""

import numpy as np
from scipy.integrate import solve_ivp


def step_input(t, amplitude=1.0):
    """Return a constant step input amplitude."""
    return amplitude


def _validate_tau(tau):
    """Raise ValueError if the time constant is not positive."""
    if tau <= 0:
        raise ValueError("tau must be positive")


def _validate_simulation_inputs(tau, t_span, num_points):
    """Validate the basic simulation parameters."""
    _validate_tau(tau)

    if num_points <= 0:
        raise ValueError("num_points must be positive")

    if t_span[1] <= t_span[0]:
        raise ValueError("t_span final time must be greater than initial time")


def first_order_ode(t, y, tau, K, input_func):
    """Return dy/dt for a first-order control system.

    Parameters
    ----------
    t : float
        Time in seconds.
    y : list or array
        Current state. ``y[0]`` is the system output.
    tau : float
        Time constant in seconds.
    K : float
        System gain.
    input_func : callable
        Function that returns the input u(t).
    """
    output = y[0]
    input_value = input_func(t)
    return [(K * input_value - output) / tau]


def analytical_step_response(t, tau, K, amplitude, y0):
    """Return the analytical response for a constant step input."""
    _validate_tau(tau)
    final_value = K * amplitude
    return final_value + (y0 - final_value) * np.exp(-np.asarray(t) / tau)


def simulate_first_order_system(
    tau,
    K,
    y0,
    t_span,
    num_points,
    input_func=None,
):
    """Simulate first-order system output with scipy.integrate.solve_ivp.

    Parameters
    ----------
    tau : float
        Time constant in seconds.
    K : float
        System gain.
    y0 : float
        Initial output value.
    t_span : tuple
        Start and end time in seconds, for example ``(0, 10)``.
    num_points : int
        Number of time points to evaluate.
    input_func : callable, optional
        Function that returns the input u(t). If None, a unit step is used.

    Returns
    -------
    tuple
        ``(t, y)`` where ``t`` is the time array and ``y`` is the numerical
        output array.
    """
    _validate_simulation_inputs(tau, t_span, num_points)

    if input_func is None:
        input_func = step_input

    t_eval = np.linspace(t_span[0], t_span[1], num_points)

    solution = solve_ivp(
        first_order_ode,
        t_span,
        [y0],
        args=(tau, K, input_func),
        t_eval=t_eval,
    )

    return solution.t, solution.y[0]


def steady_state_value(K, amplitude):
    """Return the steady-state output for a constant input amplitude."""
    return K * amplitude


def settling_time(tau, tolerance=0.02):
    """Return the approximate settling time for a first-order system."""
    _validate_tau(tau)
    return -tau * np.log(tolerance)


def rise_time(tau):
    """Return the 10% to 90% rise time for a first-order system."""
    _validate_tau(tau)
    return tau * np.log(9)
