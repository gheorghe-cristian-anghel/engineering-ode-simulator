"""RC circuit charging model.

This module models the capacitor voltage in a simple series RC circuit
connected to a constant input voltage.
"""

import numpy as np
from scipy.integrate import solve_ivp


def rc_ode(t, y, R, C, Vin):
    """Return dVc/dt for a charging RC circuit.

    Parameters
    ----------
    t : float
        Time in seconds. The equation does not depend directly on time, but
        solve_ivp expects the argument.
    y : list or array
        Current state. ``y[0]`` is the capacitor voltage Vc.
    R : float
        Resistance in ohms.
    C : float
        Capacitance in farads.
    Vin : float
        Input voltage in volts.
    """
    Vc = y[0]
    return [(Vin - Vc) / (R * C)]


def analytical_rc(t, R, C, Vin, V0):
    """Return the analytical capacitor voltage for an RC charging circuit."""
    return Vin + (V0 - Vin) * np.exp(-np.asarray(t) / (R * C))


def simulate_rc(R, C, Vin, V0, t_span, num_points):
    """Simulate capacitor voltage with scipy.integrate.solve_ivp.

    Parameters
    ----------
    R : float
        Resistance in ohms.
    C : float
        Capacitance in farads.
    Vin : float
        Input voltage in volts.
    V0 : float
        Initial capacitor voltage in volts.
    t_span : tuple
        Start and end time in seconds, for example ``(0, 5)``.
    num_points : int
        Number of time points to evaluate.

    Returns
    -------
    tuple
        ``(t, Vc)`` where ``t`` is the time array and ``Vc`` is the numerical
        capacitor voltage array.
    """
    t_eval = np.linspace(t_span[0], t_span[1], num_points)

    solution = solve_ivp(
        rc_ode,
        t_span,
        [V0],
        args=(R, C, Vin),
        t_eval=t_eval,
    )

    return solution.t, solution.y[0]
