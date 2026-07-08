"""PID attitude-control helpers for the simplified quadcopter model."""

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from models.discrete_pid import DiscretePID
from models.quadcopter_attitude import (
    quadcopter_attitude_dynamics,
    validate_quadcopter_attitude_parameters,
)


AXIS_NAMES = ("roll", "pitch", "yaw")


@dataclass
class AttitudeResponseMetrics:
    """Summary metrics for a controlled attitude response."""

    final_angle_deg: dict
    final_error_deg: dict
    peak_angle_deg: dict
    overshoot_percent: dict
    settling_time: dict
    max_abs_torque: dict


def zero_disturbance_torque(t):
    """Return zero external body torque disturbance."""
    return np.array([0.0, 0.0, 0.0])


def disturbance_torque_step(
    t_step,
    before=(0.0, 0.0, 0.0),
    after=(0.0, 0.0, 0.0),
):
    """Return a callable external body torque disturbance step."""
    t_step = float(t_step)

    if not np.isfinite(t_step):
        raise ValueError("t_step must be finite")

    before = _validate_torque_vector(before, "before")
    after = _validate_torque_vector(after, "after")

    return lambda t: before.copy() if t < t_step else after.copy()


def _validate_t_span(t_span):
    """Validate and return simulation start and end times."""
    if len(t_span) != 2:
        raise ValueError("t_span must contain start and end times")

    start_time = float(t_span[0])
    end_time = float(t_span[1])

    if end_time <= start_time:
        raise ValueError("t_span final time must be greater than initial time")

    return start_time, end_time


def _validate_three_element_vector(values, name):
    """Return a validated three-element finite vector."""
    value_array = np.asarray(values, dtype=float)

    if value_array.ndim != 1 or len(value_array) != 3:
        raise ValueError(f"{name} must be a one-dimensional vector of length 3")

    if not np.all(np.isfinite(value_array)):
        raise ValueError(f"{name} must contain only finite values")

    return value_array


def _validate_attitude_state(state, name):
    """Return a validated six-element attitude state vector."""
    state_array = np.asarray(state, dtype=float)

    if state_array.ndim != 1 or len(state_array) != 6:
        raise ValueError(f"{name} must be a one-dimensional vector of length 6")

    if not np.all(np.isfinite(state_array)):
        raise ValueError(f"{name} must contain only finite values")

    return state_array


def _validate_torque_vector(torque, name):
    """Return a validated three-element body torque vector."""
    torque_array = np.asarray(torque, dtype=float)

    if torque_array.ndim != 1 or len(torque_array) != 3:
        raise ValueError(f"{name} must be a one-dimensional vector of length 3")

    if not np.all(np.isfinite(torque_array)):
        raise ValueError(f"{name} must contain only finite values")

    return torque_array


def _validate_nonnegative_gain_vector(values, name):
    """Return a validated nonnegative PID gain vector."""
    value_array = _validate_three_element_vector(values, name)

    if np.any(value_array < 0.0):
        raise ValueError(f"{name} gains must be nonnegative")

    return value_array


def _validate_torque_limits(torque_limits):
    """Return per-axis torque saturation limits."""
    limits_array = np.asarray(torque_limits, dtype=float)

    if limits_array.shape == (2,):
        limits_array = np.tile(limits_array, (3, 1))
    elif limits_array.shape != (3, 2):
        raise ValueError(
            "torque_limits must be (min, max) or three (min, max) pairs"
        )

    if not np.all(np.isfinite(limits_array)):
        raise ValueError("torque_limits must contain only finite values")

    if np.any(limits_array[:, 1] <= limits_array[:, 0]):
        raise ValueError("each maximum torque limit must be greater than minimum")

    return limits_array


def _attitude_ode_with_held_torque(
    t,
    state,
    Ixx,
    Iyy,
    Izz,
    held_control_torque,
    disturbance_torque_func,
):
    """Return attitude derivatives for held control torque plus disturbance."""
    disturbance_torque = _validate_torque_vector(
        disturbance_torque_func(t),
        "disturbance_torque",
    )
    total_torque = held_control_torque + disturbance_torque

    return quadcopter_attitude_dynamics(
        t,
        state,
        Ixx=Ixx,
        Iyy=Iyy,
        Izz=Izz,
        torque_func=lambda sample_time: total_torque,
    )


def simulate_attitude_pid_control(
    target_angles=(10.0, -5.0, 15.0),
    initial_state=None,
    t_span=(0.0, 5.0),
    dt=0.01,
    Ixx=0.02,
    Iyy=0.02,
    Izz=0.04,
    Kp=(0.10, 0.10, 0.14),
    Ki=(0.001, 0.001, 0.0015),
    Kd=(0.09, 0.09, 0.14),
    torque_limits=(-0.2, 0.2),
    disturbance_torque_func=None,
    anti_windup=True,
):
    """Simulate PID-controlled roll, pitch, and yaw attitude tracking.

    ``target_angles`` are supplied in degrees for readability. Internally the
    attitude states, errors, body rates, and controller calculations use
    radians and radians per second.
    """
    validate_quadcopter_attitude_parameters(Ixx=Ixx, Iyy=Iyy, Izz=Izz)
    start_time, end_time = _validate_t_span(t_span)

    if dt <= 0:
        raise ValueError("dt must be positive")

    target_angles_deg = _validate_three_element_vector(target_angles, "target_angles")
    target_angles_rad = np.radians(target_angles_deg)

    if initial_state is None:
        initial_state = np.zeros(6)

    initial_state = _validate_attitude_state(initial_state, "initial_state")

    Kp = _validate_nonnegative_gain_vector(Kp, "Kp")
    Ki = _validate_nonnegative_gain_vector(Ki, "Ki")
    Kd = _validate_nonnegative_gain_vector(Kd, "Kd")
    torque_limits = _validate_torque_limits(torque_limits)

    if disturbance_torque_func is None:
        disturbance_torque_func = zero_disturbance_torque

    if not callable(disturbance_torque_func):
        raise ValueError("disturbance_torque_func must be callable")

    controllers = [
        DiscretePID(
            Kp=Kp[index],
            Ki=Ki[index],
            Kd=Kd[index],
            output_min=torque_limits[index, 0],
            output_max=torque_limits[index, 1],
            anti_windup=anti_windup,
        )
        for index in range(3)
    ]

    duration = end_time - start_time
    steps = int(np.ceil(duration / dt))
    time = start_time + np.arange(steps + 1) * dt
    time[-1] = end_time

    states = np.zeros((steps + 1, 6))
    control_torques = np.zeros((steps + 1, 3))
    disturbance_torques = np.zeros((steps + 1, 3))
    total_torques = np.zeros((steps + 1, 3))
    errors_rad = np.zeros((steps + 1, 3))

    states[0] = initial_state

    for index in range(steps):
        sample_dt = time[index + 1] - time[index]
        angles = states[index, 0:3]
        errors_rad[index] = target_angles_rad - angles

        for axis in range(3):
            control_torques[index, axis] = controllers[axis].update(
                target_angles_rad[axis],
                angles[axis],
                sample_dt,
            )

        disturbance_torques[index] = _validate_torque_vector(
            disturbance_torque_func(time[index]),
            "disturbance_torque",
        )
        total_torques[index] = control_torques[index] + disturbance_torques[index]

        solution = solve_ivp(
            _attitude_ode_with_held_torque,
            (time[index], time[index + 1]),
            states[index],
            args=(
                Ixx,
                Iyy,
                Izz,
                control_torques[index],
                disturbance_torque_func,
            ),
            t_eval=[time[index + 1]],
        )

        if not solution.success:
            raise RuntimeError(
                f"attitude PID integration failed: {solution.message}"
            )

        states[index + 1] = solution.y[:, -1]

    errors_rad[-1] = target_angles_rad - states[-1, 0:3]
    disturbance_torques[-1] = _validate_torque_vector(
        disturbance_torque_func(time[-1]),
        "disturbance_torque",
    )

    if steps > 0:
        control_torques[-1] = control_torques[-2]

    total_torques[-1] = control_torques[-1] + disturbance_torques[-1]

    angles_rad = states[:, 0:3]
    body_rates = states[:, 3:6]

    return {
        "time": time,
        "states": states,
        "angles_rad": angles_rad,
        "angles_deg": np.degrees(angles_rad),
        "body_rates": body_rates,
        "torques": control_torques,
        "control_torques": control_torques,
        "disturbance_torques": disturbance_torques,
        "total_torques": total_torques,
        "errors_rad": errors_rad,
        "errors_deg": np.degrees(errors_rad),
        "target_angles_rad": target_angles_rad,
        "target_angles_deg": target_angles_deg,
    }


def _settling_time_to_target(time, values, target_value, settling_threshold):
    """Return earliest time after which an angle stays near target."""
    absolute_band = np.radians(0.5)

    if np.isclose(target_value, 0.0):
        band = absolute_band
    else:
        band = max(settling_threshold * abs(target_value), absolute_band)

    inside_band = np.abs(values - target_value) <= band

    for index in range(len(time)):
        if np.all(inside_band[index:]):
            return float(time[index])

    return None


def _axis_overshoot_percent(values, initial_value, target_value):
    """Return overshoot percentage for one attitude axis."""
    response_change = target_value - initial_value

    if np.isclose(response_change, 0.0):
        return 0.0

    increasing = response_change >= 0.0

    if increasing:
        peak_value = np.max(values)
        overshoot = (peak_value - target_value) / abs(response_change) * 100.0
    else:
        peak_value = np.min(values)
        overshoot = (target_value - peak_value) / abs(response_change) * 100.0

    return float(max(0.0, overshoot))


def summarize_attitude_response(result, settling_threshold=0.02):
    """Return practical attitude-tracking response metrics per axis."""
    if settling_threshold <= 0:
        raise ValueError("settling_threshold must be positive")

    time = np.asarray(result["time"], dtype=float)
    angles_rad = np.asarray(result["angles_rad"], dtype=float)
    torques = np.asarray(result["torques"], dtype=float)
    target_angles_rad = np.asarray(result["target_angles_rad"], dtype=float)

    if len(time) < 2:
        raise ValueError("response must contain at least two samples")

    if angles_rad.shape != (len(time), 3):
        raise ValueError("angles_rad must have shape (len(time), 3)")

    if torques.shape != (len(time), 3):
        raise ValueError("torques must have shape (len(time), 3)")

    target_angles_rad = _validate_three_element_vector(
        target_angles_rad,
        "target_angles_rad",
    )

    final_angle_deg = {}
    final_error_deg = {}
    peak_angle_deg = {}
    overshoot_percent = {}
    settling_time = {}
    max_abs_torque = {}

    for axis, axis_name in enumerate(AXIS_NAMES):
        values = angles_rad[:, axis]
        target_value = target_angles_rad[axis]
        initial_value = values[0]
        response_change = target_value - initial_value

        if response_change >= 0.0:
            peak_angle = float(np.max(values))
        else:
            peak_angle = float(np.min(values))

        final_angle_deg[axis_name] = float(np.degrees(values[-1]))
        final_error_deg[axis_name] = float(np.degrees(target_value - values[-1]))
        peak_angle_deg[axis_name] = float(np.degrees(peak_angle))
        overshoot_percent[axis_name] = _axis_overshoot_percent(
            values,
            initial_value,
            target_value,
        )
        settling_time[axis_name] = _settling_time_to_target(
            time,
            values,
            target_value,
            settling_threshold,
        )
        max_abs_torque[axis_name] = float(np.max(np.abs(torques[:, axis])))

    return AttitudeResponseMetrics(
        final_angle_deg=final_angle_deg,
        final_error_deg=final_error_deg,
        peak_angle_deg=peak_angle_deg,
        overshoot_percent=overshoot_percent,
        settling_time=settling_time,
        max_abs_torque=max_abs_torque,
    )
