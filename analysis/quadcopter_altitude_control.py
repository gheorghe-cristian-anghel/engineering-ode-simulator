"""PID altitude-control helpers for the 1D quadcopter altitude model."""

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from models.discrete_pid import DiscretePID
from models.quadcopter_altitude import (
    hover_thrust,
    validate_quadcopter_altitude_parameters,
)


@dataclass
class AltitudeResponseMetrics:
    """Summary metrics for a controlled altitude response."""

    final_altitude: float
    final_error: float
    peak_altitude: float
    overshoot_percent: float
    settling_time: float | None
    max_thrust: float
    min_thrust: float


def downward_force_step(t_step, force_before=0.0, force_after=0.0):
    """Return a callable downward disturbance force step in newtons."""
    t_step = float(t_step)
    force_before = float(force_before)
    force_after = float(force_after)

    if not np.isfinite(t_step):
        raise ValueError("t_step must be finite")

    if not np.isfinite(force_before) or not np.isfinite(force_after):
        raise ValueError("disturbance forces must be finite")

    if force_before < 0 or force_after < 0:
        raise ValueError("disturbance forces must be nonnegative")

    return lambda t: force_before if t < t_step else force_after


def zero_downward_force(t):
    """Return zero downward disturbance force."""
    return 0.0


def _validate_t_span(t_span):
    """Validate and return simulation start and end times."""
    if len(t_span) != 2:
        raise ValueError("t_span must contain start and end times")

    start_time = float(t_span[0])
    end_time = float(t_span[1])

    if end_time <= start_time:
        raise ValueError("t_span final time must be greater than initial time")

    return start_time, end_time


def _validate_thrust_limits(thrust_limits):
    """Validate and return thrust saturation limits."""
    if len(thrust_limits) != 2:
        raise ValueError("thrust_limits must contain minimum and maximum thrust")

    thrust_min = float(thrust_limits[0])
    thrust_max = float(thrust_limits[1])

    if not np.isfinite(thrust_min) or not np.isfinite(thrust_max):
        raise ValueError("thrust limits must be finite")

    if thrust_min < 0:
        raise ValueError("minimum thrust must be nonnegative")

    if thrust_max <= thrust_min:
        raise ValueError("maximum thrust must be greater than minimum thrust")

    return thrust_min, thrust_max


def _altitude_ode_with_held_thrust(
    t,
    state,
    m,
    g,
    c_drag,
    held_thrust,
    disturbance_force_func,
):
    """Return altitude derivatives for a held thrust command."""
    z = state[0]
    v = state[1]
    disturbance_force = float(disturbance_force_func(t))

    if not np.isfinite(disturbance_force):
        raise ValueError("disturbance_force_func must return finite values")

    if disturbance_force < 0:
        raise ValueError("disturbance force must be nonnegative")

    dz_dt = v
    dv_dt = (held_thrust - m * g - c_drag * v - disturbance_force) / m

    return [dz_dt, dv_dt]


def simulate_altitude_pid_control(
    target_altitude=5.0,
    z0=0.0,
    v0=0.0,
    t_span=(0.0, 10.0),
    dt=0.01,
    m=1.0,
    g=9.81,
    c_drag=0.2,
    Kp=4.0,
    Ki=1.0,
    Kd=3.0,
    thrust_limits=(0.0, 25.0),
    disturbance_force_func=None,
    anti_windup=True,
):
    """Simulate PID-controlled altitude tracking for a 1D quadcopter.

    The PID controller computes a thrust correction around hover thrust once
    per sample time ``dt``. The total thrust command is held constant while
    the continuous altitude plant is integrated over that sample interval.
    """
    validate_quadcopter_altitude_parameters(m=m, g=g, c_drag=c_drag)
    start_time, end_time = _validate_t_span(t_span)
    thrust_min, thrust_max = _validate_thrust_limits(thrust_limits)

    target_altitude = float(target_altitude)
    z0 = float(z0)
    v0 = float(v0)

    if not np.isfinite(target_altitude):
        raise ValueError("target_altitude must be finite")

    if not np.isfinite(z0) or not np.isfinite(v0):
        raise ValueError("initial altitude and velocity must be finite")

    if dt <= 0:
        raise ValueError("dt must be positive")

    if disturbance_force_func is None:
        disturbance_force_func = zero_downward_force

    if not callable(disturbance_force_func):
        raise ValueError("disturbance_force_func must be callable")

    hover = hover_thrust(m=m, g=g)
    pid = DiscretePID(
        Kp=Kp,
        Ki=Ki,
        Kd=Kd,
        output_min=thrust_min - hover,
        output_max=thrust_max - hover,
        anti_windup=anti_windup,
    )

    duration = end_time - start_time
    steps = int(np.ceil(duration / dt))
    time = start_time + np.arange(steps + 1) * dt
    time[-1] = end_time

    altitude = np.zeros(steps + 1)
    velocity = np.zeros(steps + 1)
    thrust = np.zeros(steps + 1)
    error = np.zeros(steps + 1)
    disturbance_force = np.zeros(steps + 1)

    altitude[0] = z0
    velocity[0] = v0

    for index in range(steps):
        sample_dt = time[index + 1] - time[index]
        error[index] = target_altitude - altitude[index]
        disturbance_force[index] = float(disturbance_force_func(time[index]))

        thrust_correction = pid.update(target_altitude, altitude[index], sample_dt)
        thrust[index] = np.clip(hover + thrust_correction, thrust_min, thrust_max)

        solution = solve_ivp(
            _altitude_ode_with_held_thrust,
            (time[index], time[index + 1]),
            [altitude[index], velocity[index]],
            args=(
                m,
                g,
                c_drag,
                thrust[index],
                disturbance_force_func,
            ),
            t_eval=[time[index + 1]],
        )

        altitude[index + 1] = solution.y[0, -1]
        velocity[index + 1] = solution.y[1, -1]

    error[-1] = target_altitude - altitude[-1]
    disturbance_force[-1] = float(disturbance_force_func(time[-1]))

    if steps > 0:
        thrust[-1] = thrust[-2]

    return {
        "time": time,
        "altitude": altitude,
        "velocity": velocity,
        "thrust": thrust,
        "error": error,
        "target_altitude": target_altitude,
        "hover_thrust": hover,
        "disturbance_force": disturbance_force,
    }


def summarize_altitude_response(result, settling_threshold=0.02):
    """Return practical altitude-tracking response metrics."""
    if settling_threshold <= 0:
        raise ValueError("settling_threshold must be positive")

    time = np.asarray(result["time"], dtype=float)
    altitude = np.asarray(result["altitude"], dtype=float)
    thrust = np.asarray(result["thrust"], dtype=float)
    target_altitude = float(result["target_altitude"])

    if len(time) < 2:
        raise ValueError("response must contain at least two samples")

    if len({len(time), len(altitude), len(thrust)}) != 1:
        raise ValueError("response arrays must have equal lengths")

    initial_altitude = float(altitude[0])
    final_altitude = float(altitude[-1])
    final_error = float(target_altitude - final_altitude)
    increasing = target_altitude >= initial_altitude
    peak_altitude = float(np.max(altitude) if increasing else np.min(altitude))
    response_change = target_altitude - initial_altitude

    if np.isclose(response_change, 0.0):
        overshoot_percent = 0.0
    elif increasing:
        overshoot_percent = max(
            0.0,
            (peak_altitude - target_altitude) / abs(response_change) * 100.0,
        )
    else:
        overshoot_percent = max(
            0.0,
            (target_altitude - peak_altitude) / abs(response_change) * 100.0,
        )

    if np.isclose(target_altitude, 0.0):
        band = settling_threshold
    else:
        band = settling_threshold * abs(target_altitude)

    inside_band = np.abs(altitude - target_altitude) <= band
    settling_time = None
    for index in range(len(time)):
        if np.all(inside_band[index:]):
            settling_time = float(time[index])
            break

    return AltitudeResponseMetrics(
        final_altitude=final_altitude,
        final_error=final_error,
        peak_altitude=peak_altitude,
        overshoot_percent=float(overshoot_percent),
        settling_time=settling_time,
        max_thrust=float(np.max(thrust)),
        min_thrust=float(np.min(thrust)),
    )
