"""Educational PID tuning helpers for DC motor speed control."""

from dataclasses import dataclass

import numpy as np

from models.discrete_pid import DiscretePID, simulate_discrete_pid_motor_control


DEFAULT_MOTOR_PARAMS = {
    "R": 1.0,
    "L": 0.5,
    "J": 0.01,
    "b": 0.001,
    "Kt": 0.01,
    "Ke": 0.01,
}


@dataclass
class PIDTuningResult:
    """Summary metrics for one PID tuning simulation."""

    label: str
    Kp: float
    Ki: float
    Kd: float
    final_speed: float
    final_error: float
    peak_speed: float
    overshoot_percent: float
    settling_time: float | None
    max_voltage: float
    max_current: float


def _validate_t_span(t_span):
    """Validate and return a simulation start and end time."""
    if len(t_span) != 2:
        raise ValueError("t_span must contain start and end times")

    start_time = float(t_span[0])
    end_time = float(t_span[1])

    if end_time <= start_time:
        raise ValueError("t_span final time must be greater than initial time")

    return start_time, end_time


def _settling_time_to_target(t, y, target_value, tolerance=0.02):
    """Return earliest time after which speed stays near the target."""
    if target_value == 0:
        band = tolerance
    else:
        band = tolerance * abs(target_value)

    lower = target_value - band
    upper = target_value + band
    inside_band = (y >= lower) & (y <= upper)

    for index in range(len(y)):
        if np.all(inside_band[index:]):
            return float(t[index])

    return None


def _calculate_pid_metrics(label, Kp, Ki, Kd, t, current, speed, voltage, target_speed):
    """Calculate educational PID tuning metrics."""
    final_speed = float(speed[-1])
    final_error = float(target_speed - final_speed)
    peak_speed = float(np.max(speed))

    if target_speed == 0:
        overshoot_percent = 0.0
    else:
        overshoot = (peak_speed - target_speed) / abs(target_speed) * 100
        overshoot_percent = float(max(0.0, overshoot))

    return PIDTuningResult(
        label=label,
        Kp=Kp,
        Ki=Ki,
        Kd=Kd,
        final_speed=final_speed,
        final_error=final_error,
        peak_speed=peak_speed,
        overshoot_percent=overshoot_percent,
        settling_time=_settling_time_to_target(t, speed, target_speed),
        max_voltage=float(np.max(np.abs(voltage))),
        max_current=float(np.max(np.abs(current))),
    )


def run_pid_tuning_case(
    label,
    Kp,
    Ki,
    Kd,
    target_speed=80.0,
    t_span=(0.0, 25.0),
    dt=0.01,
    motor_params=None,
    output_min=0.0,
    output_max=24.0,
    anti_windup=True,
):
    """Run one discrete PID motor speed-control tuning case.

    Returns
    -------
    tuple
        ``(result, simulation)`` where ``result`` is a ``PIDTuningResult`` and
        ``simulation`` contains time, current, speed, voltage, and error arrays.
    """
    start_time, end_time = _validate_t_span(t_span)

    if dt <= 0:
        raise ValueError("dt must be positive")

    if motor_params is None:
        motor_params = DEFAULT_MOTOR_PARAMS

    pid = DiscretePID(
        Kp,
        Ki,
        Kd,
        output_min=output_min,
        output_max=output_max,
        anti_windup=anti_windup,
    )

    t, current, speed, voltage, error = simulate_discrete_pid_motor_control(
        motor_params["R"],
        motor_params["L"],
        motor_params["J"],
        motor_params["b"],
        motor_params["Kt"],
        motor_params["Ke"],
        pid,
        target_speed,
        i0=0.0,
        omega0=0.0,
        t_final=end_time - start_time,
        dt=dt,
    )
    t = t + start_time

    result = _calculate_pid_metrics(
        label,
        Kp,
        Ki,
        Kd,
        t,
        current,
        speed,
        voltage,
        target_speed,
    )
    simulation = {
        "t": t,
        "current": current,
        "speed": speed,
        "voltage": voltage,
        "error": error,
    }

    return result, simulation


def compare_pid_cases(
    cases,
    target_speed=80.0,
    t_span=(0.0, 25.0),
    dt=0.01,
    motor_params=None,
    output_min=0.0,
    output_max=24.0,
    anti_windup=True,
):
    """Run several PID tuning cases and return results and simulations."""
    results = []
    simulations = {}

    for case in cases:
        result, simulation = run_pid_tuning_case(
            case["label"],
            case["Kp"],
            case["Ki"],
            case["Kd"],
            target_speed=target_speed,
            t_span=t_span,
            dt=dt,
            motor_params=motor_params,
            output_min=output_min,
            output_max=output_max,
            anti_windup=anti_windup,
        )
        results.append(result)
        simulations[result.label] = simulation

    return results, simulations
