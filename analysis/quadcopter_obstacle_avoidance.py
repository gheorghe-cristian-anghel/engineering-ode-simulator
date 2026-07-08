"""Static obstacle avoidance helpers for 6-DOF quadcopter tracking.

This module adds a simple repulsive potential-field term to the existing
quadcopter trajectory tracking controller. It is intentionally educational:
obstacles are static spheres, and the method is local reactive avoidance, not
global path planning.
"""

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from analysis.quadcopter_trajectory_tracking import (
    TrajectoryReference,
    _reference_from_value,
    _validate_limits,
    _validate_nonnegative_three_vector,
    _validate_state,
    _validate_t_span,
    _validate_three_vector,
    compute_trajectory_tracking_control,
    summarize_trajectory_tracking,
)
from models.quadcopter_6dof import (
    quadcopter_6dof_dynamics,
    validate_quadcopter_6dof_parameters,
)


@dataclass
class SphericalObstacle:
    """Static spherical obstacle used by the avoidance helper."""

    center: np.ndarray
    radius: float
    influence_radius: float

    def __post_init__(self):
        """Validate obstacle geometry."""
        self.center = _validate_three_vector(self.center, "center")
        self.radius = float(self.radius)
        self.influence_radius = float(self.influence_radius)

        if not np.isfinite(self.radius) or self.radius <= 0.0:
            raise ValueError("radius must be positive and finite")

        if (
            not np.isfinite(self.influence_radius)
            or self.influence_radius <= self.radius
        ):
            raise ValueError("influence_radius must be greater than radius")


def _as_obstacle(obstacle):
    """Return a validated SphericalObstacle."""
    if not isinstance(obstacle, SphericalObstacle):
        raise ValueError("obstacle must be a SphericalObstacle")

    return obstacle


def _validate_obstacles(obstacles):
    """Return validated static spherical obstacles."""
    if obstacles is None:
        raise ValueError("obstacles must not be None")

    if isinstance(obstacles, SphericalObstacle):
        obstacle_list = [obstacles]
    else:
        obstacle_list = list(obstacles)

    if len(obstacle_list) == 0:
        raise ValueError("at least one obstacle is required")

    return tuple(_as_obstacle(obstacle) for obstacle in obstacle_list)


def distance_to_obstacle(position, obstacle):
    """Return signed clearance from a position to a spherical obstacle."""
    position = _validate_three_vector(position, "position")
    obstacle = _as_obstacle(obstacle)

    return float(np.linalg.norm(position - obstacle.center) - obstacle.radius)


def nearest_obstacle_clearance(position, obstacles):
    """Return the minimum signed clearance to a list of obstacles."""
    obstacle_list = _validate_obstacles(obstacles)
    clearances = [
        distance_to_obstacle(position, obstacle) for obstacle in obstacle_list
    ]

    return float(np.min(clearances))


def repulsive_avoidance_acceleration(
    position,
    obstacles,
    gain=1.0,
    max_acceleration=3.0,
):
    """Return a clipped repulsive acceleration away from nearby obstacles."""
    position = _validate_three_vector(position, "position")
    obstacle_list = _validate_obstacles(obstacles)
    gain = float(gain)
    max_acceleration = float(max_acceleration)
    epsilon = 1e-3

    if not np.isfinite(gain) or gain < 0.0:
        raise ValueError("gain must be nonnegative and finite")

    if not np.isfinite(max_acceleration) or max_acceleration < 0.0:
        raise ValueError("max_acceleration must be nonnegative and finite")

    total_acceleration = np.zeros(3)

    for obstacle in obstacle_list:
        offset = position - obstacle.center
        distance_from_center = float(np.linalg.norm(offset))

        if distance_from_center <= epsilon:
            direction = np.array([1.0, 0.0, 0.0])
        else:
            direction = offset / distance_from_center

        clearance = distance_from_center - obstacle.radius
        influence_clearance = obstacle.influence_radius - obstacle.radius

        if clearance >= influence_clearance:
            continue

        effective_clearance = max(clearance, epsilon)
        strength = (
            gain
            * (1.0 / effective_clearance - 1.0 / influence_clearance)
            / effective_clearance**2
        )
        total_acceleration += strength * direction

    acceleration_norm = float(np.linalg.norm(total_acceleration))
    if acceleration_norm > max_acceleration > 0.0:
        total_acceleration *= max_acceleration / acceleration_norm
    elif max_acceleration == 0.0:
        total_acceleration[:] = 0.0

    return total_acceleration


def _obstacle_aware_reference(reference, avoidance_acceleration):
    """Return a trajectory reference with added avoidance acceleration."""
    return TrajectoryReference(
        position=reference.position,
        velocity=reference.velocity,
        acceleration=reference.acceleration + avoidance_acceleration,
    )


def simulate_quadcopter_obstacle_avoidance(
    trajectory_func,
    obstacles,
    initial_state=None,
    t_span=(0.0, 12.0),
    dt=0.01,
    m=1.0,
    g=9.81,
    Ixx=0.02,
    Iyy=0.02,
    Izz=0.04,
    c_drag=0.05,
    Kp_pos=(2.0, 2.0, 4.0),
    Kd_pos=(1.5, 1.5, 3.0),
    Kp_att=(0.08, 0.08, 0.08),
    Kd_att=(0.03, 0.03, 0.03),
    avoidance_gain=0.8,
    max_avoidance_acceleration=2.0,
    thrust_limits=(0.0, 25.0),
    torque_limits=(-0.2, 0.2),
    max_tilt=np.radians(20.0),
):
    """Simulate quadcopter trajectory tracking with static obstacle avoidance."""
    if not callable(trajectory_func):
        raise ValueError("trajectory_func must be callable")

    obstacle_list = _validate_obstacles(obstacles)
    validate_quadcopter_6dof_parameters(
        m=m,
        g=g,
        Ixx=Ixx,
        Iyy=Iyy,
        Izz=Izz,
        c_drag=c_drag,
    )
    start_time, end_time = _validate_t_span(t_span)
    Kp_pos = _validate_nonnegative_three_vector(Kp_pos, "Kp_pos")
    Kd_pos = _validate_nonnegative_three_vector(Kd_pos, "Kd_pos")
    Kp_att = _validate_nonnegative_three_vector(Kp_att, "Kp_att")
    Kd_att = _validate_nonnegative_three_vector(Kd_att, "Kd_att")
    _validate_limits(thrust_limits, "thrust_limits", require_nonnegative_min=True)
    _validate_limits(torque_limits, "torque_limits")

    if dt <= 0:
        raise ValueError("dt must be positive")

    avoidance_gain = float(avoidance_gain)
    max_avoidance_acceleration = float(max_avoidance_acceleration)

    if not np.isfinite(avoidance_gain) or avoidance_gain < 0.0:
        raise ValueError("avoidance_gain must be nonnegative and finite")

    if (
        not np.isfinite(max_avoidance_acceleration)
        or max_avoidance_acceleration < 0.0
    ):
        raise ValueError(
            "max_avoidance_acceleration must be nonnegative and finite"
        )

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
    avoidance_accelerations = np.zeros((steps + 1, 3))
    nearest_clearances = np.zeros(steps + 1)

    states[0] = initial_state

    for index in range(steps):
        reference = _reference_from_value(trajectory_func(time[index]))
        avoidance_acceleration = repulsive_avoidance_acceleration(
            states[index, 0:3],
            obstacle_list,
            gain=avoidance_gain,
            max_acceleration=max_avoidance_acceleration,
        )
        adjusted_reference = _obstacle_aware_reference(
            reference,
            avoidance_acceleration,
        )
        control, attitude_command, acceleration_command = (
            compute_trajectory_tracking_control(
                states[index],
                adjusted_reference,
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
        avoidance_accelerations[index] = avoidance_acceleration
        nearest_clearances[index] = nearest_obstacle_clearance(
            states[index, 0:3],
            obstacle_list,
        )

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
                f"quadcopter obstacle-avoidance integration failed: {solution.message}"
            )

        states[index + 1] = solution.y[:, -1]

    final_reference = _reference_from_value(trajectory_func(time[-1]))
    final_avoidance_acceleration = repulsive_avoidance_acceleration(
        states[-1, 0:3],
        obstacle_list,
        gain=avoidance_gain,
        max_acceleration=max_avoidance_acceleration,
    )
    final_adjusted_reference = _obstacle_aware_reference(
        final_reference,
        final_avoidance_acceleration,
    )
    final_control, final_attitude_command, final_acceleration_command = (
        compute_trajectory_tracking_control(
            states[-1],
            final_adjusted_reference,
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
    avoidance_accelerations[-1] = final_avoidance_acceleration
    nearest_clearances[-1] = nearest_obstacle_clearance(
        states[-1, 0:3],
        obstacle_list,
    )

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
        "avoidance_accelerations": avoidance_accelerations,
        "avoidance_acceleration_norm": np.linalg.norm(
            avoidance_accelerations,
            axis=1,
        ),
        "nearest_clearances": nearest_clearances,
        "obstacles": obstacle_list,
    }
    result["tracking_metrics"] = summarize_trajectory_tracking(result)
    result["obstacle_metrics"] = summarize_obstacle_avoidance(result)

    return result


def summarize_obstacle_avoidance(result):
    """Return practical metrics for a static-obstacle avoidance response."""
    states = np.asarray(result["states"], dtype=float)
    reference_positions = np.asarray(result["reference_positions"], dtype=float)
    position_errors = np.asarray(result["position_errors"], dtype=float)
    controls = np.asarray(result["controls"], dtype=float)
    avoidance_accelerations = np.asarray(
        result["avoidance_accelerations"],
        dtype=float,
    )
    nearest_clearances = np.asarray(result["nearest_clearances"], dtype=float)

    if states.ndim != 2 or states.shape[1] != 12:
        raise ValueError("states must have shape (n, 12)")

    if reference_positions.shape != (len(states), 3):
        raise ValueError("reference_positions must have shape (len(states), 3)")

    if position_errors.shape != (len(states), 3):
        raise ValueError("position_errors must have shape (len(states), 3)")

    if controls.shape != (len(states), 4):
        raise ValueError("controls must have shape (len(states), 4)")

    if avoidance_accelerations.shape != (len(states), 3):
        raise ValueError("avoidance_accelerations must have shape (len(states), 3)")

    if nearest_clearances.shape != (len(states),):
        raise ValueError("nearest_clearances must have shape (len(states),)")

    error_norm = np.linalg.norm(position_errors, axis=1)
    avoidance_norm = np.linalg.norm(avoidance_accelerations, axis=1)

    return {
        "min_clearance": float(np.min(nearest_clearances)),
        "final_position_error_norm": float(error_norm[-1]),
        "rms_position_error": float(np.sqrt(np.mean(error_norm**2))),
        "max_position_error": float(np.max(error_norm)),
        "max_avoidance_acceleration": float(np.max(avoidance_norm)),
        "max_thrust": float(np.max(controls[:, 0])),
        "max_abs_torque": float(np.max(np.abs(controls[:, 1:4]))),
    }
