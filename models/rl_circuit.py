"""RL circuit step response model.

This module models the current in a series resistor-inductor circuit connected
to a constant input voltage.
"""

import numpy as np
from scipy.integrate import solve_ivp


def validate_rl_parameters(R, L):
    """Validate resistor and inductor values for an RL circuit."""
    if R <= 0:
        raise ValueError("R must be positive")

    if L <= 0:
        raise ValueError("L must be positive")


def time_constant(R, L):
    """Return the RL circuit time constant tau = L / R."""
    validate_rl_parameters(R, L)
    return L / R


def steady_state_current(R, Vin):
    """Return the steady-state current i_ss = Vin / R."""
    if R <= 0:
        raise ValueError("R must be positive")

    return Vin / R


def rl_ode(t, y, R, L, Vin):
    """Return di/dt for a series RL circuit.

    Parameters
    ----------
    t : float
        Time in seconds. The equation does not depend directly on time, but
        solve_ivp expects the argument.
    y : list or array
        Current state. ``y[0]`` is the circuit current i.
    R : float
        Resistance in ohms.
    L : float
        Inductance in henries.
    Vin : float
        Input voltage in volts.
    """
    current = y[0]
    return [(Vin - R * current) / L]


def analytical_rl(t, R, L, Vin, i0):
    """Return the analytical current for an RL circuit step response."""
    tau = time_constant(R, L)
    current_ss = steady_state_current(R, Vin)
    return current_ss + (i0 - current_ss) * np.exp(-np.asarray(t) / tau)


def simulate_rl(R, L, Vin, i0, t_span, num_points):
    """Simulate RL circuit current with scipy.integrate.solve_ivp.

    Parameters
    ----------
    R : float
        Resistance in ohms.
    L : float
        Inductance in henries.
    Vin : float
        Input voltage in volts.
    i0 : float
        Initial current in amps.
    t_span : tuple
        Start and end time in seconds, for example ``(0, 1.5)``.
    num_points : int
        Number of time points to evaluate.

    Returns
    -------
    tuple
        ``(t, i)`` where ``t`` is the time array and ``i`` is the numerical
        current array.
    """
    validate_rl_parameters(R, L)

    if num_points <= 0:
        raise ValueError("num_points must be positive")

    if t_span[1] <= t_span[0]:
        raise ValueError("t_span final time must be greater than initial time")

    t_eval = np.linspace(t_span[0], t_span[1], num_points)

    solution = solve_ivp(
        rl_ode,
        t_span,
        [i0],
        args=(R, L, Vin),
        t_eval=t_eval,
    )

    return solution.t, solution.y[0]
