"""Closed-loop DC motor speed control with a PI controller.

This module uses the DC motor plant model with a PID-compatible controller
API. For this first closed-loop version, only PI control is implemented.
"""

import numpy as np
from scipy.integrate import solve_ivp

from models.dc_motor import validate_dc_motor_parameters


def validate_pi_parameters(Kp, Ki, Kd=0.0):
    """Validate PI controller gains.

    Kd is included for future PID extension, but derivative action is not
    implemented in this continuous PI model yet.
    """
    if Kp < 0:
        raise ValueError("Kp must be nonnegative")

    if Ki < 0:
        raise ValueError("Ki must be nonnegative")

    if not np.isclose(Kd, 0.0):
        raise NotImplementedError("Derivative action is not implemented yet")


def validate_voltage_limits(voltage_min, voltage_max):
    """Validate controller voltage saturation limits."""
    if voltage_max <= voltage_min:
        raise ValueError("voltage_max must be greater than voltage_min")


def speed_reference_step(t, target_speed=100.0):
    """Return a constant speed reference in radians per second."""
    return target_speed


def zero_load_torque(t):
    """Return zero load torque."""
    return 0.0


def pi_control_voltage(error, integral_error, Kp, Ki, voltage_min, voltage_max):
    """Return saturated PI control voltage."""
    validate_pi_parameters(Kp, Ki)
    validate_voltage_limits(voltage_min, voltage_max)

    raw_voltage = Kp * error + Ki * integral_error
    return np.clip(raw_voltage, voltage_min, voltage_max)


def closed_loop_motor_ode(
    t,
    state,
    R,
    L,
    J,
    b,
    Kt,
    Ke,
    Kp,
    Ki,
    Kd,
    reference_func,
    load_torque_func,
    voltage_min,
    voltage_max,
):
    """Return derivatives for closed-loop PI motor speed control."""
    validate_pi_parameters(Kp, Ki, Kd)

    current = state[0]
    omega = state[1]
    integral_error = state[2]

    error = reference_func(t) - omega
    voltage = pi_control_voltage(
        error,
        integral_error,
        Kp,
        Ki,
        voltage_min,
        voltage_max,
    )
    load_torque = load_torque_func(t)

    di_dt = (voltage - R * current - Ke * omega) / L
    domega_dt = (Kt * current - b * omega - load_torque) / J
    dintegral_error_dt = error

    return [di_dt, domega_dt, dintegral_error_dt]


def simulate_pi_motor_control(
    R,
    L,
    J,
    b,
    Kt,
    Ke,
    Kp,
    Ki,
    i0,
    omega0,
    integral_error0,
    t_span,
    num_points,
    reference_func=None,
    load_torque_func=None,
    voltage_min=0.0,
    voltage_max=24.0,
    Kd=0.0,
):
    """Simulate closed-loop DC motor speed control with a PI controller.

    Returns
    -------
    tuple
        ``(t, i, omega, integral_error, voltage)`` where ``voltage`` is the
        saturated controller output reconstructed at each returned time sample.
    """
    validate_dc_motor_parameters(R, L, J, b, Kt, Ke)
    validate_pi_parameters(Kp, Ki, Kd)
    validate_voltage_limits(voltage_min, voltage_max)

    if num_points <= 0:
        raise ValueError("num_points must be positive")

    if t_span[1] <= t_span[0]:
        raise ValueError("t_span final time must be greater than initial time")

    if reference_func is None:
        reference_func = speed_reference_step

    if load_torque_func is None:
        load_torque_func = zero_load_torque

    t_eval = np.linspace(t_span[0], t_span[1], num_points)

    solution = solve_ivp(
        closed_loop_motor_ode,
        t_span,
        [i0, omega0, integral_error0],
        args=(
            R,
            L,
            J,
            b,
            Kt,
            Ke,
            Kp,
            Ki,
            Kd,
            reference_func,
            load_torque_func,
            voltage_min,
            voltage_max,
        ),
        t_eval=t_eval,
    )

    time = solution.t
    current = solution.y[0]
    omega = solution.y[1]
    integral_error = solution.y[2]
    voltage = np.array(
        [
            pi_control_voltage(
                reference_func(sample_time) - sample_speed,
                sample_integral_error,
                Kp,
                Ki,
                voltage_min,
                voltage_max,
            )
            for sample_time, sample_speed, sample_integral_error in zip(
                time,
                omega,
                integral_error,
            )
        ]
    )

    return time, current, omega, integral_error, voltage
