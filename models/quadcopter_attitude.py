"""Simplified quadcopter rotational attitude dynamics model.

This module models open-loop quadcopter attitude motion using decoupled
rigid-body rotational dynamics. The state is ``[phi, theta, psi, p, q, r]``:

- ``phi`` is roll angle in radians.
- ``theta`` is pitch angle in radians.
- ``psi`` is yaw angle in radians.
- ``p`` is roll rate in radians per second.
- ``q`` is pitch rate in radians per second.
- ``r`` is yaw rate in radians per second.

Small-angle kinematics are used for this first attitude model:
``phi_dot = p``, ``theta_dot = q``, and ``psi_dot = r``. Body torque inputs are
``[tau_phi, tau_theta, tau_psi]`` in newton-meters.
"""

import numpy as np
from scipy.integrate import solve_ivp


def validate_quadcopter_attitude_parameters(Ixx=0.02, Iyy=0.02, Izz=0.04):
    """Validate quadcopter rotational inertia parameters."""
    if Ixx <= 0:
        raise ValueError("Ixx must be positive")

    if Iyy <= 0:
        raise ValueError("Iyy must be positive")

    if Izz <= 0:
        raise ValueError("Izz must be positive")


def _validate_attitude_state(state, name="state"):
    """Return a validated six-element attitude state vector."""
    state_array = np.asarray(state, dtype=float)

    if state_array.ndim != 1 or len(state_array) != 6:
        raise ValueError(f"{name} must be a one-dimensional vector of length 6")

    if not np.all(np.isfinite(state_array)):
        raise ValueError(f"{name} must contain only finite values")

    return state_array


def _validate_torque_vector(torque, name="torque"):
    """Return a validated three-element body torque vector."""
    torque_array = np.asarray(torque, dtype=float)

    if torque_array.ndim != 1 or len(torque_array) != 3:
        raise ValueError(f"{name} must be a one-dimensional vector of length 3")

    if not np.all(np.isfinite(torque_array)):
        raise ValueError(f"{name} must contain only finite values")

    return torque_array


def zero_torque(t):
    """Return zero body torque."""
    return np.array([0.0, 0.0, 0.0])


def constant_torque(tau_phi=0.0, tau_theta=0.0, tau_psi=0.0):
    """Return a callable constant body torque command."""
    torque = _validate_torque_vector([tau_phi, tau_theta, tau_psi])

    return lambda t: torque.copy()


def torque_step(t_step, before=(0.0, 0.0, 0.0), after=(0.0, 0.0, 0.0)):
    """Return a callable body torque step command.

    The command returns ``before`` before ``t_step`` and ``after`` at and after
    ``t_step``.
    """
    t_step = float(t_step)

    if not np.isfinite(t_step):
        raise ValueError("t_step must be finite")

    before = _validate_torque_vector(before, "before")
    after = _validate_torque_vector(after, "after")

    return lambda t: before.copy() if t < t_step else after.copy()


def quadcopter_attitude_dynamics(
    t,
    state,
    Ixx=0.02,
    Iyy=0.02,
    Izz=0.04,
    torque_func=None,
):
    """Return derivatives for the simplified attitude model.

    Parameters
    ----------
    t : float
        Time in seconds.
    state : array-like
        Current state ``[phi, theta, psi, p, q, r]``.
    Ixx, Iyy, Izz : float, optional
        Principal moments of inertia in kilogram-meter squared.
    torque_func : callable, optional
        Function returning body torques ``[tau_phi, tau_theta, tau_psi]`` in
        newton-meters. If None, zero torque is used.
    """
    validate_quadcopter_attitude_parameters(Ixx=Ixx, Iyy=Iyy, Izz=Izz)
    phi, theta, psi, p, q, r = _validate_attitude_state(state)

    if torque_func is None:
        torque_func = zero_torque

    if not callable(torque_func):
        raise ValueError("torque_func must be callable")

    tau_phi, tau_theta, tau_psi = _validate_torque_vector(torque_func(t))

    phi_dot = p
    theta_dot = q
    psi_dot = r
    p_dot = tau_phi / Ixx
    q_dot = tau_theta / Iyy
    r_dot = tau_psi / Izz

    return np.array([phi_dot, theta_dot, psi_dot, p_dot, q_dot, r_dot])


def simulate_quadcopter_attitude(
    initial_state=None,
    t_span=(0.0, 5.0),
    num_points=1000,
    Ixx=0.02,
    Iyy=0.02,
    Izz=0.04,
    torque_func=None,
):
    """Simulate simplified open-loop quadcopter attitude motion.

    Returns
    -------
    tuple
        ``(t, states, torques)`` where ``states`` has shape
        ``(num_points, 6)`` with columns ``[phi, theta, psi, p, q, r]`` and
        ``torques`` has shape ``(num_points, 3)``.
    """
    validate_quadcopter_attitude_parameters(Ixx=Ixx, Iyy=Iyy, Izz=Izz)

    if initial_state is None:
        initial_state = np.zeros(6)

    initial_state = _validate_attitude_state(initial_state, "initial_state")

    if num_points <= 0:
        raise ValueError("num_points must be positive")

    if t_span[1] <= t_span[0]:
        raise ValueError("t_span final time must be greater than initial time")

    if torque_func is None:
        torque_func = zero_torque

    if not callable(torque_func):
        raise ValueError("torque_func must be callable")

    t_eval = np.linspace(t_span[0], t_span[1], num_points)

    solution = solve_ivp(
        quadcopter_attitude_dynamics,
        t_span,
        initial_state,
        args=(Ixx, Iyy, Izz, torque_func),
        t_eval=t_eval,
    )

    if not solution.success:
        raise RuntimeError(f"quadcopter attitude integration failed: {solution.message}")

    torques = np.array(
        [_validate_torque_vector(torque_func(sample_time)) for sample_time in solution.t]
    )

    return solution.t, solution.y.T, torques


def linearized_attitude_state_space(Ixx=0.02, Iyy=0.02, Izz=0.04):
    """Return the linear attitude state-space model.

    The state is ``[phi, theta, psi, p, q, r]`` and the input is body torque
    ``[tau_phi, tau_theta, tau_psi]``. Outputs are all six states.
    """
    validate_quadcopter_attitude_parameters(Ixx=Ixx, Iyy=Iyy, Izz=Izz)

    A = np.array(
        [
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    B = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0 / Ixx, 0.0, 0.0],
            [0.0, 1.0 / Iyy, 0.0],
            [0.0, 0.0, 1.0 / Izz],
        ]
    )
    C = np.eye(6)
    D = np.zeros((6, 3))

    return A, B, C, D
