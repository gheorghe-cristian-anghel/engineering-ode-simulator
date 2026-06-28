"""Series RLC circuit step response model.

This module models capacitor voltage and current in a series resistor-inductor-
capacitor circuit connected to a constant input voltage.
"""

import numpy as np
from scipy.integrate import solve_ivp


def validate_rlc_parameters(R, L, C):
    """Validate resistor, inductor, and capacitor values for an RLC circuit."""
    if R < 0:
        raise ValueError("R must be nonnegative")

    if L <= 0:
        raise ValueError("L must be positive")

    if C <= 0:
        raise ValueError("C must be positive")


def natural_frequency(L, C):
    """Return the undamped natural frequency in radians per second."""
    validate_rlc_parameters(0, L, C)
    return 1 / np.sqrt(L * C)


def damping_ratio(R, L, C):
    """Return the damping ratio for a series RLC circuit."""
    validate_rlc_parameters(R, L, C)
    return (R / 2) * np.sqrt(C / L)


def steady_state_voltage(Vin):
    """Return the DC steady-state capacitor voltage."""
    return Vin


def rlc_ode(t, y, R, L, C, Vin):
    """Return derivatives for the series RLC first-order system.

    Parameters
    ----------
    t : float
        Time in seconds.
    y : list or array
        Current state. ``y[0]`` is capacitor voltage Vc and ``y[1]`` is current i.
    R : float
        Resistance in ohms.
    L : float
        Inductance in henries.
    C : float
        Capacitance in farads.
    Vin : float
        Input voltage in volts.
    """
    capacitor_voltage = y[0]
    current = y[1]

    dvc_dt = current / C
    di_dt = (Vin - R * current - capacitor_voltage) / L

    return [dvc_dt, di_dt]


def simulate_rlc(R, L, C, Vin, Vc0, i0, t_span, num_points):
    """Simulate capacitor voltage and current with scipy.integrate.solve_ivp.

    Parameters
    ----------
    R : float
        Resistance in ohms.
    L : float
        Inductance in henries.
    C : float
        Capacitance in farads.
    Vin : float
        Input voltage in volts.
    Vc0 : float
        Initial capacitor voltage in volts.
    i0 : float
        Initial current in amps.
    t_span : tuple
        Start and end time in seconds, for example ``(0, 10)``.
    num_points : int
        Number of time points to evaluate.

    Returns
    -------
    tuple
        ``(t, Vc, i)`` where ``t`` is time, ``Vc`` is capacitor voltage, and
        ``i`` is current.
    """
    validate_rlc_parameters(R, L, C)

    if num_points <= 0:
        raise ValueError("num_points must be positive")

    if t_span[1] <= t_span[0]:
        raise ValueError("t_span final time must be greater than initial time")

    t_eval = np.linspace(t_span[0], t_span[1], num_points)

    solution = solve_ivp(
        rlc_ode,
        t_span,
        [Vc0, i0],
        args=(R, L, C, Vin),
        t_eval=t_eval,
    )

    return solution.t, solution.y[0], solution.y[1]
