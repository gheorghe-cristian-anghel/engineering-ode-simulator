"""Educational trajectory tracking for the full 6-DOF quadcopter model.

This module builds a simple cascaded PD trajectory controller around the
open-loop plant in ``models.quadcopter_6dof``. It is intended for learning and
portfolio demonstrations, not as a production drone autopilot. The controller
uses inertial-frame position error to command acceleration, converts horizontal
acceleration into small roll and pitch commands, then uses attitude PD control
to command body torques.
"""

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from models.quadcopter_6dof import (
    quadcopter_6dof_dynamics,
    validate_quadcopter_6dof_parameters,
)


@dataclass
class TrajectoryReference:
    """Position, velocity, and acceleration reference at one time sample."""

    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray


def _validate_three_vector(values, name):
    """Return a validated three-element finite vector."""
    value_array = np.asarray(values, dtype=float)

    if value_array.ndim != 1 or len(value_array) != 3:
        raise ValueError(f"{name} must be a one-dimensional vector of length 3")

    if not np.all(np.isfinite(value_array)):
        raise ValueError(f"{name} must contain only finite values")

    return value_array


def _validate_nonnegative_three_vector(values, name):
    """Return a validated three-element vector with nonnegative entries."""
    value_array = _validate_three_vector(values, name)

    if np.any(value_array < 0.0):
        raise ValueError(f"{name} values must be nonnegative")

    return value_array


def _validate_state(state, name="state"):
    """Return a validated 12-element 6-DOF state vector."""
    state_array = np.asarray(state, dtype=float)

    if state_array.ndim != 1 or len(state_array) != 12:
        raise ValueError(f"{name} must be a one-dimensional vector of length 12")

    if not np.all(np.isfinite(state_array)):
        raise ValueError(f"{name} must contain only finite values")

    return state_array


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


def _validate_limits(limits, name, require_nonnegative_min=False):
    """Validate and return a two-element saturation limit tuple."""
    if len(limits) != 2:
        raise ValueError(f"{name} must contain minimum and maximum values")

    lower = float(limits[0])
    upper = float(limits[1])

    if not np.isfinite(lower) or not np.isfinite(upper):
        raise ValueError(f"{name} values must be finite")

    if require_nonnegative_min and lower < 0.0:
        raise ValueError(f"{name} minimum must be nonnegative")

    if upper <= lower:
        raise ValueError(f"{name} maximum must be greater than minimum")

    return lower, upper


def _reference_from_value(reference):
    """Return a TrajectoryReference from a dataclass or dictionary."""
    if isinstance(reference, TrajectoryReference):
        position = reference.position
        velocity = reference.velocity
        acceleration = reference.acceleration
    elif isinstance(reference, dict):
        position = reference["position"]
        velocity = reference["velocity"]
        acceleration = reference["acceleration"]
    else:
        raise ValueError("trajectory reference must be a TrajectoryReference or dict")

    return TrajectoryReference(
        position=_validate_three_vector(position, "reference position"),
        velocity=_validate_three_vector(velocity, "reference velocity"),
        acceleration=_validate_three_vector(acceleration, "reference acceleration"),
    )


def hover_trajectory(position=(0.0, 0.0, 2.0)):
    """Return a fixed hover-point trajectory function."""
    position = _validate_three_vector(position, "position")

    def trajectory_func(t):
        return TrajectoryReference(
            position=position.copy(),
            velocity=np.zeros(3),
            acceleration=np.zeros(3),
        )

    return trajectory_func


def circular_trajectory(
    radius=1.0,
    altitude=2.0,
    angular_speed=0.5,
    center=(0.0, 0.0),
):
    """Return a horizontal circular trajectory at constant altitude."""
    radius = float(radius)
    altitude = float(altitude)
    angular_speed = float(angular_speed)
    center_array = np.asarray(center, dtype=float)

    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("radius must be positive and finite")

    if not np.isfinite(altitude):
        raise ValueError("altitude must be finite")

    if not np.isfinite(angular_speed) or angular_speed <= 0.0:
        raise ValueError("angular_speed must be positive and finite")

    if center_array.ndim != 1 or len(center_array) != 2:
        raise ValueError("center must be a one-dimensional vector of length 2")

    if not np.all(np.isfinite(center_array)):
        raise ValueError("center must contain only finite values")

    def trajectory_func(t):
        angle = angular_speed * t
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)

        position = np.array(
            [
                center_array[0] + radius * cos_angle,
                center_array[1] + radius * sin_angle,
                altitude,
            ]
        )
        velocity = np.array(
            [
                -radius * angular_speed * sin_angle,
                radius * angular_speed * cos_angle,
                0.0,
            ]
        )
        acceleration = np.array(
            [
                -radius * angular_speed**2 * cos_angle,
                -radius * angular_speed**2 * sin_angle,
                0.0,
            ]
        )

        return TrajectoryReference(position, velocity, acceleration)

    return trajectory_func


def compute_trajectory_tracking_control(
    state,
    reference,
    m=1.0,
    g=9.81,
    Kp_pos=(0.8, 0.8, 3.0),
    Kd_pos=(1.4, 1.4, 2.6),
    Kp_att=(0.25, 0.25, 0.16),
    Kd_att=(0.12, 0.12, 0.08),
    thrust_limits=(0.0, 25.0),
    torque_limits=(-0.2, 0.2),
    max_tilt=np.radians(20.0),
):
    """Compute one cascaded PD trajectory tracking control command.

    Returns
    -------
    tuple
        ``(control, attitude_command, acceleration_command)``. ``control`` is
        ``[T, tau_phi, tau_theta, tau_psi]`` and ``attitude_command`` is
        ``[phi_cmd, theta_cmd, psi_cmd]`` in radians.
    """
    validate_quadcopter_6dof_parameters(m=m, g=g)
    state = _validate_state(state)
    reference = _reference_from_value(reference)
    Kp_pos = _validate_nonnegative_three_vector(Kp_pos, "Kp_pos")
    Kd_pos = _validate_nonnegative_three_vector(Kd_pos, "Kd_pos")
    Kp_att = _validate_nonnegative_three_vector(Kp_att, "Kp_att")
    Kd_att = _validate_nonnegative_three_vector(Kd_att, "Kd_att")
    thrust_min, thrust_max = _validate_limits(
        thrust_limits,
        "thrust_limits",
        require_nonnegative_min=True,
    )
    torque_min, torque_max = _validate_limits(torque_limits, "torque_limits")
    max_tilt = float(max_tilt)

    if not np.isfinite(max_tilt) or max_tilt <= 0.0:
        raise ValueError("max_tilt must be positive and finite")

    position = state[0:3]
    velocity = state[3:6]
    phi, theta, psi = state[6:9]
    p, q, r = state[9:12]

    position_error = reference.position - position
    velocity_error = reference.velocity - velocity
    acceleration_command = (
        reference.acceleration + Kp_pos * position_error + Kd_pos * velocity_error
    )

    thrust = m * (g + acceleration_command[2])
    thrust = np.clip(thrust, thrust_min, thrust_max)

    phi_cmd = -acceleration_command[1] / g
    theta_cmd = acceleration_command[0] / g
    psi_cmd = 0.0
    attitude_command = np.array(
        [
            np.clip(phi_cmd, -max_tilt, max_tilt),
            np.clip(theta_cmd, -max_tilt, max_tilt),
            psi_cmd,
        ]
    )

    angle_error = attitude_command - np.array([phi, theta, psi])
    rate_error = -np.array([p, q, r])
    torques = Kp_att * angle_error + Kd_att * rate_error
    torques = np.clip(torques, torque_min, torque_max)

    control = np.array([thrust, torques[0], torques[1], torques[2]])

    return control, attitude_command, acceleration_command


def simulate_quadcopter_trajectory_tracking(
    trajectory_func,
    initial_state=None,
    t_span=(0.0, 10.0),
    dt=0.01,
    m=1.0,
    g=9.81,
    Ixx=0.02,
    Iyy=0.02,
    Izz=0.04,
    c_drag=0.05,
    Kp_pos=(0.8, 0.8, 3.0),
    Kd_pos=(1.4, 1.4, 2.6),
    Kp_att=(0.25, 0.25, 0.16),
    Kd_att=(0.12, 0.12, 0.08),
    thrust_limits=(0.0, 25.0),
    torque_limits=(-0.2, 0.2),
    max_tilt=np.radians(20.0),
):
    """Simulate sampled trajectory tracking using the full 6-DOF model."""
    if not callable(trajectory_func):
        raise ValueError("trajectory_func must be callable")

    validate_quadcopter_6dof_parameters(
        m=m,
        g=g,
        Ixx=Ixx,
        Iyy=Iyy,
        Izz=Izz,
        c_drag=c_drag,
    )
    start_time, end_time = _validate_t_span(t_span)

    if dt <= 0:
        raise ValueError("dt must be positive")

    if initial_state is None:
        initial_state = np.zeros(12)

    initial_state = _validate_state(initial_state, "initial_state")

    duration = end_time - start_time
    steps = int(np.ceil(duration / dt - 1e-12))
    time = start_time + np.arange(steps + 1) * dt
    time[-1] = end_time

    states = np.zeros((steps + 1, 12))
    controls = np.zeros((steps + 1, 4))
    reference_positions = np.zeros((steps + 1, 3))
    reference_velocities = np.zeros((steps + 1, 3))
    reference_accelerations = np.zeros((steps + 1, 3))
    position_errors = np.zeros((steps + 1, 3))
    attitude_commands = np.zeros((steps + 1, 3))
    commanded_accelerations = np.zeros((steps + 1, 3))

    states[0] = initial_state

    for index in range(steps):
        sample_dt = time[index + 1] - time[index]
        reference = _reference_from_value(trajectory_func(time[index]))
        control, attitude_command, acceleration_command = (
            compute_trajectory_tracking_control(
                states[index],
                reference,
                m=m,
                g=g,
                Kp_pos=Kp_pos,
                Kd_pos=Kd_pos,
                Kp_att=Kp_att,
                Kd_att=Kd_att,
                thrust_limits=thrust_limits,
                torque_limits=torque_limits,
                max_tilt=max_tilt,
            )
        )

        controls[index] = control
        reference_positions[index] = reference.position
        reference_velocities[index] = reference.velocity
        reference_accelerations[index] = reference.acceleration
        position_errors[index] = reference.position - states[index, 0:3]
        attitude_commands[index] = attitude_command
        commanded_accelerations[index] = acceleration_command

        solution = solve_ivp(
            quadcopter_6dof_dynamics,
            (time[index], time[index + 1]),
            states[index],
            args=(
                m,
                g,
                Ixx,
                Iyy,
                Izz,
                c_drag,
                lambda sample_time, held_control=control: held_control.copy(),
            ),
            t_eval=[time[index + 1]],
        )

        if not solution.success:
            raise RuntimeError(
                f"quadcopter trajectory integration failed: {solution.message}"
            )

        states[index + 1] = solution.y[:, -1]

    final_reference = _reference_from_value(trajectory_func(time[-1]))
    final_control, final_attitude_command, final_acceleration_command = (
        compute_trajectory_tracking_control(
            states[-1],
            final_reference,
            m=m,
            g=g,
            Kp_pos=Kp_pos,
            Kd_pos=Kd_pos,
            Kp_att=Kp_att,
            Kd_att=Kd_att,
            thrust_limits=thrust_limits,
            torque_limits=torque_limits,
            max_tilt=max_tilt,
        )
    )

    controls[-1] = final_control
    reference_positions[-1] = final_reference.position
    reference_velocities[-1] = final_reference.velocity
    reference_accelerations[-1] = final_reference.acceleration
    position_errors[-1] = final_reference.position - states[-1, 0:3]
    attitude_commands[-1] = final_attitude_command
    commanded_accelerations[-1] = final_acceleration_command

    result = {
        "time": time,
        "states": states,
        "controls": controls,
        "reference_positions": reference_positions,
        "reference_velocities": reference_velocities,
        "reference_accelerations": reference_accelerations,
        "position_errors": position_errors,
        "tracking_error_norm": np.linalg.norm(position_errors, axis=1),
        "attitude_commands": attitude_commands,
        "commanded_accelerations": commanded_accelerations,
    }
    result["tracking_metrics"] = summarize_trajectory_tracking(result)

    return result


def summarize_trajectory_tracking(result):
    """Return practical trajectory-tracking summary metrics."""
    states = np.asarray(result["states"], dtype=float)
    reference_positions = np.asarray(result["reference_positions"], dtype=float)
    position_errors = np.asarray(result["position_errors"], dtype=float)
    controls = np.asarray(result["controls"], dtype=float)

    if states.ndim != 2 or states.shape[1] != 12:
        raise ValueError("states must have shape (n, 12)")

    if reference_positions.shape != (len(states), 3):
        raise ValueError("reference_positions must have shape (len(states), 3)")

    if position_errors.shape != (len(states), 3):
        raise ValueError("position_errors must have shape (len(states), 3)")

    if controls.shape != (len(states), 4):
        raise ValueError("controls must have shape (len(states), 4)")

    error_norm = np.linalg.norm(position_errors, axis=1)

    return {
        "final_position_error_norm": float(error_norm[-1]),
        "rms_position_error": float(np.sqrt(np.mean(error_norm**2))),
        "max_position_error": float(np.max(error_norm)),
        "max_thrust": float(np.max(controls[:, 0])),
        "max_abs_torque": float(np.max(np.abs(controls[:, 1:4]))),
        "final_position": states[-1, 0:3].copy(),
        "final_reference_position": reference_positions[-1].copy(),
    }
