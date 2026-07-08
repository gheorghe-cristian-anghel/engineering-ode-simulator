"""Newton's Law of Cooling model.

This module models the temperature of an object cooling or warming toward the
temperature of its environment.
"""

import numpy as np
from scipy.integrate import solve_ivp


def cooling_ode(t, y, k, T_env):
    """Return dT/dt for Newton's Law of Cooling.

    Parameters
    ----------
    t : float
        Time in minutes. The equation does not depend directly on time, but
        solve_ivp expects the argument.
    y : list or array
        Current state. ``y[0]`` is the object temperature T.
    k : float
        Cooling constant per minute.
    T_env : float
        Environment temperature in degrees Celsius.
    """
    T = y[0]
    return [-k * (T - T_env)]


def analytical_cooling(t, k, T_env, T0):
    """Return the analytical temperature for Newton's Law of Cooling."""
    return T_env + (T0 - T_env) * np.exp(-k * np.asarray(t))


def simulate_cooling(k, T_env, T0, t_span, num_points):
    """Simulate temperature with scipy.integrate.solve_ivp.

    Parameters
    ----------
    k : float
        Cooling constant per minute.
    T_env : float
        Environment temperature in degrees Celsius.
    T0 : float
        Initial object temperature in degrees Celsius.
    t_span : tuple
        Start and end time in minutes, for example ``(0, 60)``.
    num_points : int
        Number of time points to evaluate.

    Returns
    -------
    tuple
        ``(t, T)`` where ``t`` is the time array and ``T`` is the numerical
        temperature array.
    """
    t_eval = np.linspace(t_span[0], t_span[1], num_points)

    solution = solve_ivp(
        cooling_ode,
        t_span,
        [T0],
        args=(k, T_env),
        t_eval=t_eval,
    )

    if not solution.success:
        raise RuntimeError(f"cooling integration failed: {solution.message}")

    return solution.t, solution.y[0]
