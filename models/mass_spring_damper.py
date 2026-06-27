"""Mass-spring-damper vibration model.

This module models the displacement and velocity of a mass attached to a
spring and damper. With no external force, the model describes free vibration.
"""

import numpy as np
from scipy.integrate import solve_ivp


def natural_frequency(m, k):
    """Return the undamped natural frequency in radians per second."""
    return np.sqrt(k / m)


def damping_ratio(m, c, k):
    """Return the damping ratio for a mass-spring-damper system."""
    return c / (2 * np.sqrt(m * k))


def mechanical_energy(x, v, m, k):
    """Return total mechanical energy from displacement and velocity."""
    kinetic_energy = 0.5 * m * np.asarray(v) ** 2
    potential_energy = 0.5 * k * np.asarray(x) ** 2
    return kinetic_energy + potential_energy


def _zero_force(t):
    """Return zero external force for free vibration."""
    return 0


def mass_spring_damper_ode(t, y, m, c, k, force_func):
    """Return derivatives for the mass-spring-damper first-order system.

    Parameters
    ----------
    t : float
        Time in seconds.
    y : list or array
        Current state. ``y[0]`` is displacement x and ``y[1]`` is velocity v.
    m : float
        Mass in kilograms.
    c : float
        Damping coefficient in newton-seconds per meter.
    k : float
        Spring stiffness in newtons per meter.
    force_func : callable
        Function that returns the external force F(t) in newtons.
    """
    x = y[0]
    v = y[1]
    force = force_func(t)

    dxdt = v
    dvdt = (force - c * v - k * x) / m

    return [dxdt, dvdt]


def simulate_mass_spring_damper(
    m,
    c,
    k,
    x0,
    v0,
    t_span,
    num_points,
    force_func=None,
):
    """Simulate displacement and velocity with scipy.integrate.solve_ivp.

    Parameters
    ----------
    m : float
        Mass in kilograms.
    c : float
        Damping coefficient in newton-seconds per meter.
    k : float
        Spring stiffness in newtons per meter.
    x0 : float
        Initial displacement in meters.
    v0 : float
        Initial velocity in meters per second.
    t_span : tuple
        Start and end time in seconds, for example ``(0, 20)``.
    num_points : int
        Number of time points to evaluate.
    force_func : callable, optional
        Function that returns the external force F(t). If None, F(t) = 0.

    Returns
    -------
    tuple
        ``(t, x, v)`` where ``t`` is time, ``x`` is displacement, and ``v`` is
        velocity.
    """
    if force_func is None:
        force_func = _zero_force

    t_eval = np.linspace(t_span[0], t_span[1], num_points)

    solution = solve_ivp(
        mass_spring_damper_ode,
        t_span,
        [x0, v0],
        args=(m, c, k, force_func),
        t_eval=t_eval,
    )

    return solution.t, solution.y[0], solution.y[1]
