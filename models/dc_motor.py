"""Permanent-magnet DC motor speed response model.

This module models armature current and angular speed for a DC motor with
electrical armature dynamics and rotor mechanical dynamics.
"""

import numpy as np
from scipy.integrate import solve_ivp


def validate_dc_motor_parameters(R, L, J, b, Kt, Ke):
    """Validate DC motor electrical and mechanical parameters."""
    if R <= 0:
        raise ValueError("R must be positive")

    if L <= 0:
        raise ValueError("L must be positive")

    if J <= 0:
        raise ValueError("J must be positive")

    if b < 0:
        raise ValueError("b must be nonnegative")

    if Kt <= 0:
        raise ValueError("Kt must be positive")

    if Ke <= 0:
        raise ValueError("Ke must be positive")


def voltage_step(t, voltage=12.0):
    """Return a constant voltage step."""
    return voltage


def zero_load_torque(t):
    """Return zero load torque."""
    return 0.0


def dc_motor_ode(t, state, R, L, J, b, Kt, Ke, voltage_func, load_torque_func):
    """Return derivatives for the DC motor first-order system.

    Parameters
    ----------
    t : float
        Time in seconds.
    state : list or array
        Current state. ``state[0]`` is armature current i and ``state[1]`` is
        angular speed omega.
    R : float
        Armature resistance in ohms.
    L : float
        Armature inductance in henries.
    J : float
        Rotor inertia in kilogram-meter squared.
    b : float
        Viscous damping coefficient in newton-meter-seconds per radian.
    Kt : float
        Torque constant in newton-meters per amp.
    Ke : float
        Back-emf constant in volt-seconds per radian.
    voltage_func : callable
        Function that returns applied voltage V(t).
    load_torque_func : callable
        Function that returns load torque TL(t).
    """
    current = state[0]
    omega = state[1]
    voltage = voltage_func(t)
    load_torque = load_torque_func(t)

    di_dt = (voltage - R * current - Ke * omega) / L
    domega_dt = (Kt * current - b * omega - load_torque) / J

    return [di_dt, domega_dt]


def simulate_dc_motor(
    R,
    L,
    J,
    b,
    Kt,
    Ke,
    i0,
    omega0,
    t_span,
    num_points,
    voltage_func=None,
    load_torque_func=None,
):
    """Simulate DC motor current and speed with scipy.integrate.solve_ivp.

    Parameters
    ----------
    R : float
        Armature resistance in ohms.
    L : float
        Armature inductance in henries.
    J : float
        Rotor inertia in kilogram-meter squared.
    b : float
        Viscous damping coefficient in newton-meter-seconds per radian.
    Kt : float
        Torque constant in newton-meters per amp.
    Ke : float
        Back-emf constant in volt-seconds per radian.
    i0 : float
        Initial armature current in amps.
    omega0 : float
        Initial angular speed in radians per second.
    t_span : tuple
        Start and end time in seconds, for example ``(0, 5)``.
    num_points : int
        Number of time points to evaluate.
    voltage_func : callable, optional
        Function that returns applied voltage V(t). If None, a 12 V step is used.
    load_torque_func : callable, optional
        Function that returns load torque TL(t). If None, zero load torque is used.

    Returns
    -------
    tuple
        ``(t, i, omega)`` where ``t`` is time, ``i`` is armature current, and
        ``omega`` is angular speed.
    """
    validate_dc_motor_parameters(R, L, J, b, Kt, Ke)

    if num_points <= 0:
        raise ValueError("num_points must be positive")

    if t_span[1] <= t_span[0]:
        raise ValueError("t_span final time must be greater than initial time")

    if voltage_func is None:
        voltage_func = voltage_step

    if load_torque_func is None:
        load_torque_func = zero_load_torque

    t_eval = np.linspace(t_span[0], t_span[1], num_points)

    solution = solve_ivp(
        dc_motor_ode,
        t_span,
        [i0, omega0],
        args=(R, L, J, b, Kt, Ke, voltage_func, load_torque_func),
        t_eval=t_eval,
    )

    if not solution.success:
        raise RuntimeError(f"DC motor integration failed: {solution.message}")

    return solution.t, solution.y[0], solution.y[1]


def steady_state_speed(R, b, Kt, Ke, voltage, load_torque=0.0):
    """Return steady-state angular speed for constant voltage and load torque."""
    if R <= 0:
        raise ValueError("R must be positive")

    if b < 0:
        raise ValueError("b must be nonnegative")

    if Kt <= 0:
        raise ValueError("Kt must be positive")

    if Ke <= 0:
        raise ValueError("Ke must be positive")

    return (voltage - R * load_torque / Kt) / (Ke + R * b / Kt)


def steady_state_current(b, Kt, omega_ss, load_torque=0.0):
    """Return steady-state armature current for speed and load torque."""
    if b < 0:
        raise ValueError("b must be nonnegative")

    if Kt <= 0:
        raise ValueError("Kt must be positive")

    return (b * omega_ss + load_torque) / Kt


def rad_per_sec_to_rpm(omega):
    """Convert angular speed from radians per second to revolutions per minute."""
    return np.asarray(omega) * 60 / (2 * np.pi)
