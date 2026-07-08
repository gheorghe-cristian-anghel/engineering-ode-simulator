"""Simple pendulum model.

This module models an undamped simple pendulum with both the full nonlinear
equation and the small-angle linear approximation.
"""

import numpy as np
from scipy.integrate import solve_ivp


def validate_pendulum_parameters(L, g):
    """Validate pendulum length and gravitational acceleration."""
    if L <= 0:
        raise ValueError("L must be positive")

    if g <= 0:
        raise ValueError("g must be positive")


def natural_frequency(L, g=9.81):
    """Return the small-angle natural frequency in radians per second."""
    validate_pendulum_parameters(L, g)
    return np.sqrt(g / L)


def small_angle_period(L, g=9.81):
    """Return the small-angle oscillation period in seconds."""
    validate_pendulum_parameters(L, g)
    return 2 * np.pi * np.sqrt(L / g)


def pendulum_energy(theta, omega, L, g=9.81, m=1.0):
    """Return total mechanical energy for a simple pendulum.

    The zero potential energy reference is the lowest point of the swing.
    """
    validate_pendulum_parameters(L, g)

    if m <= 0:
        raise ValueError("m must be positive")

    theta = np.asarray(theta)
    omega = np.asarray(omega)

    kinetic_energy = 0.5 * m * (L * omega) ** 2
    potential_energy = m * g * L * (1 - np.cos(theta))
    return kinetic_energy + potential_energy


def pendulum_ode(t, y, L, g):
    """Return derivatives for the nonlinear simple pendulum.

    Parameters
    ----------
    t : float
        Time in seconds.
    y : list or array
        Current state. ``y[0]`` is angle theta and ``y[1]`` is angular velocity
        omega.
    L : float
        Pendulum length in meters.
    g : float
        Gravitational acceleration in meters per second squared.
    """
    theta = y[0]
    omega = y[1]

    dtheta_dt = omega
    domega_dt = -(g / L) * np.sin(theta)

    return [dtheta_dt, domega_dt]


def linear_pendulum_ode(t, y, L, g):
    """Return derivatives for the small-angle linear pendulum approximation."""
    theta = y[0]
    omega = y[1]

    dtheta_dt = omega
    domega_dt = -(g / L) * theta

    return [dtheta_dt, domega_dt]


def simulate_pendulum(
    L,
    theta0,
    omega0,
    t_span,
    num_points,
    g=9.81,
    linear=False,
):
    """Simulate pendulum angle and angular velocity with solve_ivp.

    Parameters
    ----------
    L : float
        Pendulum length in meters.
    theta0 : float
        Initial angle in radians.
    omega0 : float
        Initial angular velocity in radians per second.
    t_span : tuple
        Start and end time in seconds, for example ``(0, 10)``.
    num_points : int
        Number of time points to evaluate.
    g : float, optional
        Gravitational acceleration in meters per second squared.
    linear : bool, optional
        If True, use the small-angle linear approximation.

    Returns
    -------
    tuple
        ``(t, theta, omega)`` where ``theta`` is angle in radians and ``omega``
        is angular velocity in radians per second.
    """
    validate_pendulum_parameters(L, g)

    if num_points <= 0:
        raise ValueError("num_points must be positive")

    if t_span[1] <= t_span[0]:
        raise ValueError("t_span final time must be greater than initial time")

    ode_function = linear_pendulum_ode if linear else pendulum_ode
    t_eval = np.linspace(t_span[0], t_span[1], num_points)

    solution = solve_ivp(
        ode_function,
        t_span,
        [theta0, omega0],
        args=(L, g),
        t_eval=t_eval,
        rtol=1e-9,
        atol=1e-11,
    )

    if not solution.success:
        raise RuntimeError(f"pendulum integration failed: {solution.message}")

    return solution.t, solution.y[0], solution.y[1]
