"""Full 6-DOF rigid-body quadcopter dynamics model.

This module models open-loop translational and rotational quadcopter motion.
The 12-state vector is::

    [x, y, z, vx, vy, vz, phi, theta, psi, p, q, r]

``x`` and ``y`` are horizontal inertial-frame positions, ``z`` is altitude
with positive upward, ``vx``, ``vy``, and ``vz`` are inertial-frame velocities,
``phi``, ``theta``, and ``psi`` are roll, pitch, and yaw Euler angles, and
``p``, ``q``, and ``r`` are body angular rates.

The control vector is ``[T, tau_phi, tau_theta, tau_psi]``. Total thrust ``T``
acts along the body upward axis. A standard ZYX Euler convention is used:
``R = Rz(psi) @ Ry(theta) @ Rx(phi)`` maps body-frame vectors to inertial-frame
vectors. With zero yaw, positive pitch redirects thrust toward positive
inertial ``x`` and positive roll redirects thrust toward negative inertial
``y``.
"""

import numpy as np
from scipy.integrate import solve_ivp


def validate_quadcopter_6dof_parameters(
    m=1.0,
    g=9.81,
    Ixx=0.02,
    Iyy=0.02,
    Izz=0.04,
    c_drag=0.0,
):
    """Validate full 6-DOF quadcopter model parameters."""
    if m <= 0:
        raise ValueError("m must be positive")

    if g <= 0:
        raise ValueError("g must be positive")

    if Ixx <= 0:
        raise ValueError("Ixx must be positive")

    if Iyy <= 0:
        raise ValueError("Iyy must be positive")

    if Izz <= 0:
        raise ValueError("Izz must be positive")

    if c_drag < 0:
        raise ValueError("c_drag must be nonnegative")


def _validate_state(state, name="state"):
    """Return a validated 12-element state vector."""
    state_array = np.asarray(state, dtype=float)

    if state_array.ndim != 1 or len(state_array) != 12:
        raise ValueError(f"{name} must be a one-dimensional vector of length 12")

    if not np.all(np.isfinite(state_array)):
        raise ValueError(f"{name} must contain only finite values")

    return state_array


def _validate_control(control, name="control"):
    """Return a validated control vector ``[T, tau_phi, tau_theta, tau_psi]``."""
    control_array = np.asarray(control, dtype=float)

    if control_array.ndim != 1 or len(control_array) != 4:
        raise ValueError(f"{name} must be a one-dimensional vector of length 4")

    if not np.all(np.isfinite(control_array)):
        raise ValueError(f"{name} must contain only finite values")

    if control_array[0] < 0:
        raise ValueError("thrust must be nonnegative")

    return control_array


def _validate_t_span(t_span):
    """Validate and return simulation start and end times."""
    if len(t_span) != 2:
        raise ValueError("t_span must contain start and end times")

    start_time = float(t_span[0])
    end_time = float(t_span[1])

    if not np.isfinite(start_time) or not np.isfinite(end_time):
        raise ValueError("t_span values must be finite")

    if end_time <= start_time:
        raise ValueError("t_span final time must be greater than initial time")

    return start_time, end_time


def rotation_matrix_body_to_inertial(phi, theta, psi):
    """Return the ZYX body-to-inertial rotation matrix.

    The returned matrix maps body-frame vectors into the inertial frame using
    ``R = Rz(psi) @ Ry(theta) @ Rx(phi)``.
    """
    phi = float(phi)
    theta = float(theta)
    psi = float(psi)

    if not np.all(np.isfinite([phi, theta, psi])):
        raise ValueError("Euler angles must be finite")

    c_phi = np.cos(phi)
    s_phi = np.sin(phi)
    c_theta = np.cos(theta)
    s_theta = np.sin(theta)
    c_psi = np.cos(psi)
    s_psi = np.sin(psi)

    rotation_x = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, c_phi, -s_phi],
            [0.0, s_phi, c_phi],
        ]
    )
    rotation_y = np.array(
        [
            [c_theta, 0.0, s_theta],
            [0.0, 1.0, 0.0],
            [-s_theta, 0.0, c_theta],
        ]
    )
    rotation_z = np.array(
        [
            [c_psi, -s_psi, 0.0],
            [s_psi, c_psi, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    return rotation_z @ rotation_y @ rotation_x


def euler_rate_matrix(phi, theta):
    """Return the matrix mapping body rates to Euler angle rates.

    The returned matrix ``E`` satisfies::

        [phi_dot, theta_dot, psi_dot] = E @ [p, q, r]

    The ZYX Euler representation is singular when ``cos(theta)`` is zero.
    """
    phi = float(phi)
    theta = float(theta)

    if not np.all(np.isfinite([phi, theta])):
        raise ValueError("Euler angles must be finite")

    c_theta = np.cos(theta)
    if np.isclose(c_theta, 0.0, atol=1e-8):
        raise ValueError("Euler angle singularity: cos(theta) is too close to zero")

    s_phi = np.sin(phi)
    c_phi = np.cos(phi)
    t_theta = np.tan(theta)

    return np.array(
        [
            [1.0, s_phi * t_theta, c_phi * t_theta],
            [0.0, c_phi, -s_phi],
            [0.0, s_phi / c_theta, c_phi / c_theta],
        ]
    )


def hover_control(m=1.0, g=9.81):
    """Return hover thrust with zero body torques."""
    validate_quadcopter_6dof_parameters(m=m, g=g)

    return np.array([m * g, 0.0, 0.0, 0.0])


def constant_control(T, tau_phi=0.0, tau_theta=0.0, tau_psi=0.0):
    """Return a callable constant control command."""
    control = _validate_control([T, tau_phi, tau_theta, tau_psi])

    return lambda t: control.copy()


def control_step(t_step, before, after):
    """Return a callable control step command.

    The command returns ``before`` before ``t_step`` and ``after`` at and after
    ``t_step``.
    """
    t_step = float(t_step)

    if not np.isfinite(t_step):
        raise ValueError("t_step must be finite")

    before = _validate_control(before, "before")
    after = _validate_control(after, "after")

    return lambda t: before.copy() if t < t_step else after.copy()


def quadcopter_6dof_dynamics(
    t,
    state,
    m=1.0,
    g=9.81,
    Ixx=0.02,
    Iyy=0.02,
    Izz=0.04,
    c_drag=0.0,
    control_func=None,
):
    """Return derivatives for the full 6-DOF quadcopter model."""
    validate_quadcopter_6dof_parameters(
        m=m,
        g=g,
        Ixx=Ixx,
        Iyy=Iyy,
        Izz=Izz,
        c_drag=c_drag,
    )
    state = _validate_state(state)
    _, _, _, vx, vy, vz, phi, theta, psi, p, q, r = state

    if control_func is None:
        control_func = constant_control(m * g)

    if not callable(control_func):
        raise ValueError("control_func must be callable")

    T, tau_phi, tau_theta, tau_psi = _validate_control(control_func(t))

    velocity = np.array([vx, vy, vz])
    rotation = rotation_matrix_body_to_inertial(phi, theta, psi)
    thrust_body = np.array([0.0, 0.0, T])
    thrust_inertial = rotation @ thrust_body
    gravity = np.array([0.0, 0.0, -g])
    acceleration = thrust_inertial / m + gravity - c_drag * velocity / m

    euler_rates = euler_rate_matrix(phi, theta) @ np.array([p, q, r])

    p_dot = tau_phi / Ixx
    q_dot = tau_theta / Iyy
    r_dot = tau_psi / Izz

    return np.array(
        [
            vx,
            vy,
            vz,
            acceleration[0],
            acceleration[1],
            acceleration[2],
            euler_rates[0],
            euler_rates[1],
            euler_rates[2],
            p_dot,
            q_dot,
            r_dot,
        ]
    )


def simulate_quadcopter_6dof(
    initial_state=None,
    t_span=(0.0, 5.0),
    num_points=1000,
    m=1.0,
    g=9.81,
    Ixx=0.02,
    Iyy=0.02,
    Izz=0.04,
    c_drag=0.0,
    control_func=None,
):
    """Simulate open-loop full 6-DOF quadcopter motion.

    Returns
    -------
    tuple
        ``(t, states, controls)`` where ``states`` has shape
        ``(num_points, 12)`` and ``controls`` has shape ``(num_points, 4)``.
    """
    validate_quadcopter_6dof_parameters(
        m=m,
        g=g,
        Ixx=Ixx,
        Iyy=Iyy,
        Izz=Izz,
        c_drag=c_drag,
    )
    start_time, end_time = _validate_t_span(t_span)

    if num_points <= 0:
        raise ValueError("num_points must be positive")

    if initial_state is None:
        initial_state = np.zeros(12)

    initial_state = _validate_state(initial_state, "initial_state")

    if control_func is None:
        control_func = constant_control(m * g)

    if not callable(control_func):
        raise ValueError("control_func must be callable")

    t_eval = np.linspace(start_time, end_time, num_points)

    solution = solve_ivp(
        quadcopter_6dof_dynamics,
        (start_time, end_time),
        initial_state,
        args=(m, g, Ixx, Iyy, Izz, c_drag, control_func),
        t_eval=t_eval,
    )

    if not solution.success:
        raise RuntimeError(f"quadcopter 6-DOF integration failed: {solution.message}")

    controls = np.array(
        [_validate_control(control_func(sample_time)) for sample_time in solution.t]
    )

    return solution.t, solution.y.T, controls
