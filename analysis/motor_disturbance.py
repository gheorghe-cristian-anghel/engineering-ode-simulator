"""Reusable helpers for DC motor load-disturbance studies."""

import numpy as np

from models.pid_motor_control import load_torque_step as _pi_load_torque_step


def load_torque_step(t, step_time=12.0, initial=0.0, final=0.03):
    """Return a step load torque disturbance."""
    return _pi_load_torque_step(
        t,
        step_time=step_time,
        initial_torque=initial,
        final_torque=final,
    )


def summarize_motor_disturbance_response(
    time,
    speed,
    target_speed,
    disturbance_time,
    voltage=None,
    current=None,
    recovery_tolerance=0.02,
):
    """Return practical speed-disturbance response metrics.

    Recovery time is measured from the disturbance until the speed returns
    within ``recovery_tolerance`` of ``target_speed`` and stays there.
    """
    if recovery_tolerance <= 0:
        raise ValueError("recovery_tolerance must be positive")

    time = _as_response_array(time, "time")
    speed = _as_response_array(speed, "speed")
    optional_arrays = _validate_optional_arrays(time, voltage, current)
    _validate_equal_lengths(time, speed)

    disturbance_index = int(np.argmin(np.abs(time - disturbance_time)))
    before_index = max(0, disturbance_index - 1)
    recovery_index = _recovery_index(
        time,
        speed,
        disturbance_index,
        target_speed,
        recovery_tolerance,
    )

    speed_before = float(speed[before_index])
    minimum_speed_after = float(np.min(speed[disturbance_index:]))

    metrics = {
        "speed_before_disturbance": speed_before,
        "minimum_speed_after_disturbance": minimum_speed_after,
        "speed_drop": float(speed_before - minimum_speed_after),
        "final_speed": float(speed[-1]),
        "final_error": float(target_speed - speed[-1]),
        "recovery_time": (
            None
            if recovery_index is None
            else float(time[recovery_index] - time[disturbance_index])
        ),
    }

    if "voltage" in optional_arrays:
        voltage = optional_arrays["voltage"]
        metrics["voltage_before_disturbance"] = float(voltage[before_index])
        metrics["voltage_final"] = float(voltage[-1])

    if "current" in optional_arrays:
        current = optional_arrays["current"]
        metrics["current_before_disturbance"] = float(current[before_index])
        metrics["current_final"] = float(current[-1])

    return metrics


def _as_response_array(values, name):
    """Return a validated one-dimensional response array."""
    array = np.asarray(values, dtype=float)

    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")

    if len(array) < 2:
        raise ValueError(f"{name} must contain at least 2 samples")

    return array


def _validate_equal_lengths(time, speed):
    """Validate required response array lengths."""
    if len(time) != len(speed):
        raise ValueError("time and speed must have the same length")


def _validate_optional_arrays(time, voltage, current):
    """Validate optional voltage and current arrays."""
    optional_arrays = {}

    if voltage is not None:
        voltage_array = _as_response_array(voltage, "voltage")
        if len(voltage_array) != len(time):
            raise ValueError("voltage must have the same length as time")
        optional_arrays["voltage"] = voltage_array

    if current is not None:
        current_array = _as_response_array(current, "current")
        if len(current_array) != len(time):
            raise ValueError("current must have the same length as time")
        optional_arrays["current"] = current_array

    return optional_arrays


def _recovery_index(time, speed, disturbance_index, target_speed, tolerance):
    """Return the first post-disturbance index that stays within tolerance."""
    if target_speed == 0:
        band = tolerance
    else:
        band = tolerance * abs(target_speed)

    lower = target_speed - band
    upper = target_speed + band
    inside_band = (speed >= lower) & (speed <= upper)

    for index in range(disturbance_index, len(time)):
        if np.all(inside_band[index:]):
            return index

    return None
