"""Educational waypoint following for the full 6-DOF quadcopter model.

This module converts discrete 3D waypoint lists into time-based reference
trajectories, then reuses the simplified cascaded trajectory tracker from
``analysis.quadcopter_trajectory_tracking``. It is waypoint following, not
path planning: there is no obstacle avoidance, MPC, or rotor-level motor
mixing.
"""

import numpy as np

from analysis.quadcopter_trajectory_tracking import (
    TrajectoryReference,
    simulate_quadcopter_trajectory_tracking,
)


VALID_SMOOTHING_MODES = ("linear", "smoothstep")


def _validate_waypoints(waypoints):
    """Return validated waypoints with shape ``(n, 3)``."""
    waypoint_array = np.asarray(waypoints, dtype=float)

    if waypoint_array.ndim != 2 or waypoint_array.shape[1] != 3:
        raise ValueError("waypoints must be a two-dimensional array with shape (n, 3)")

    if len(waypoint_array) < 2:
        raise ValueError("at least two waypoints are required")

    if not np.all(np.isfinite(waypoint_array)):
        raise ValueError("waypoints must contain only finite values")

    return waypoint_array


def _validate_segment_time(segment_time):
    """Return a validated positive segment time."""
    segment_time = float(segment_time)

    if not np.isfinite(segment_time) or segment_time <= 0.0:
        raise ValueError("segment_time must be positive and finite")

    return segment_time


def _validate_smoothing(smoothing):
    """Return a validated waypoint smoothing mode."""
    if smoothing not in VALID_SMOOTHING_MODES:
        raise ValueError("smoothing must be 'linear' or 'smoothstep'")

    return smoothing


def total_waypoint_time(waypoints, segment_time=4.0):
    """Return total reference-trajectory time for a waypoint list."""
    waypoint_array = _validate_waypoints(waypoints)
    segment_time = _validate_segment_time(segment_time)

    return float((len(waypoint_array) - 1) * segment_time)


def waypoint_trajectory(
    waypoints,
    segment_time=4.0,
    smoothing="smoothstep",
):
    """Return a trajectory function that moves through 3D waypoints.

    Parameters
    ----------
    waypoints : array-like
        Sequence of ``(x, y, z)`` waypoint positions in meters.
    segment_time : float
        Time in seconds spent moving between each adjacent waypoint pair.
    smoothing : {"linear", "smoothstep"}
        Interpolation mode. ``"linear"`` uses constant velocity per segment.
        ``"smoothstep"`` uses ``s = 3u^2 - 2u^3`` with analytical velocity and
        acceleration.
    """
    waypoint_array = _validate_waypoints(waypoints)
    segment_time = _validate_segment_time(segment_time)
    smoothing = _validate_smoothing(smoothing)
    final_time = total_waypoint_time(waypoint_array, segment_time)

    def trajectory_func(t):
        t = float(t)

        if not np.isfinite(t):
            raise ValueError("trajectory time must be finite")

        if t <= 0.0:
            segment_index = 0
            u = 0.0
        elif t >= final_time:
            return TrajectoryReference(
                position=waypoint_array[-1].copy(),
                velocity=np.zeros(3),
                acceleration=np.zeros(3),
            )
        else:
            segment_index = int(t // segment_time)
            u = (t - segment_index * segment_time) / segment_time

        start = waypoint_array[segment_index]
        end = waypoint_array[segment_index + 1]
        delta = end - start

        if smoothing == "linear":
            position = start + u * delta
            velocity = delta / segment_time
            acceleration = np.zeros(3)
        else:
            blend = 3.0 * u**2 - 2.0 * u**3
            blend_rate = (6.0 * u - 6.0 * u**2) / segment_time
            blend_acceleration = (6.0 - 12.0 * u) / segment_time**2

            position = start + blend * delta
            velocity = blend_rate * delta
            acceleration = blend_acceleration * delta

        return TrajectoryReference(
            position=position,
            velocity=velocity,
            acceleration=acceleration,
        )

    return trajectory_func


def simulate_quadcopter_waypoint_following(
    waypoints,
    segment_time=4.0,
    initial_state=None,
    dt=0.01,
    m=1.0,
    g=9.81,
    Ixx=0.02,
    Iyy=0.02,
    Izz=0.04,
    c_drag=0.05,
    Kp_pos=(2.0, 2.0, 4.0),
    Kd_pos=(1.5, 1.5, 3.0),
    Kp_att=(0.25, 0.25, 0.16),
    Kd_att=(0.12, 0.12, 0.08),
    thrust_limits=(0.0, 25.0),
    torque_limits=(-0.2, 0.2),
    smoothing="smoothstep",
    hold_time=2.0,
):
    """Simulate waypoint following with the existing 6-DOF trajectory tracker."""
    waypoint_array = _validate_waypoints(waypoints)
    segment_time = _validate_segment_time(segment_time)
    smoothing = _validate_smoothing(smoothing)
    hold_time = float(hold_time)

    if not np.isfinite(hold_time) or hold_time < 0.0:
        raise ValueError("hold_time must be nonnegative and finite")

    if initial_state is None:
        initial_state = np.zeros(12)
        initial_state[0:3] = waypoint_array[0]

    trajectory_func = waypoint_trajectory(
        waypoint_array,
        segment_time=segment_time,
        smoothing=smoothing,
    )
    trajectory_time = total_waypoint_time(waypoint_array, segment_time)

    result = simulate_quadcopter_trajectory_tracking(
        trajectory_func,
        initial_state=initial_state,
        t_span=(0.0, trajectory_time + hold_time),
        dt=dt,
        m=m,
        g=g,
        Ixx=Ixx,
        Iyy=Iyy,
        Izz=Izz,
        c_drag=c_drag,
        Kp_pos=Kp_pos,
        Kd_pos=Kd_pos,
        Kp_att=Kp_att,
        Kd_att=Kd_att,
        thrust_limits=thrust_limits,
        torque_limits=torque_limits,
    )
    result["waypoints"] = waypoint_array.copy()
    result["segment_time"] = segment_time
    result["smoothing"] = smoothing
    result["trajectory_time"] = trajectory_time
    result["hold_time"] = hold_time
    result["waypoint_metrics"] = summarize_waypoint_following(result, waypoint_array)

    return result


def summarize_waypoint_following(result, waypoints):
    """Return practical waypoint-following metrics."""
    waypoints = _validate_waypoints(waypoints)
    time = np.asarray(result["time"], dtype=float)
    states = np.asarray(result["states"], dtype=float)
    controls = np.asarray(result["controls"], dtype=float)
    position_errors = np.asarray(result["position_errors"], dtype=float)

    if time.ndim != 1 or len(time) < 2:
        raise ValueError("time must be a one-dimensional array with at least two samples")

    if states.shape != (len(time), 12):
        raise ValueError("states must have shape (len(time), 12)")

    if controls.shape != (len(time), 4):
        raise ValueError("controls must have shape (len(time), 4)")

    if position_errors.shape != (len(time), 3):
        raise ValueError("position_errors must have shape (len(time), 3)")

    positions = states[:, 0:3]
    error_norm = np.linalg.norm(position_errors, axis=1)
    distances_to_waypoints = np.linalg.norm(
        positions[:, np.newaxis, :] - waypoints[np.newaxis, :, :],
        axis=2,
    )
    closest_indices = np.argmin(distances_to_waypoints, axis=0)

    return {
        "final_waypoint_error": float(np.linalg.norm(positions[-1] - waypoints[-1])),
        "rms_position_error": float(np.sqrt(np.mean(error_norm**2))),
        "max_position_error": float(np.max(error_norm)),
        "max_thrust": float(np.max(controls[:, 0])),
        "max_abs_torque": float(np.max(np.abs(controls[:, 1:4]))),
        "total_time": float(time[-1] - time[0]),
        "number_of_waypoints": int(len(waypoints)),
        "nearest_distance_to_each_waypoint": np.min(distances_to_waypoints, axis=0),
        "closest_approach_time_to_each_waypoint": time[closest_indices],
        "final_position": positions[-1].copy(),
        "final_waypoint": waypoints[-1].copy(),
    }
