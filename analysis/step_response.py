"""Reusable step response metrics.

This module provides a small analysis helper for simulated step responses.
It works with time and output arrays from any model, not just the first-order
control example.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class StepResponseInfo:
    """Summary metrics for a step response."""

    initial_value: float
    final_value: float
    steady_state_value: float
    rise_time: float | None
    settling_time: float | None
    peak_value: float
    peak_time: float
    overshoot_percent: float


def _validate_inputs(t, y, settling_threshold, rise_limits):
    """Validate step response inputs."""
    if t.ndim != 1 or y.ndim != 1:
        raise ValueError("t and y must be one-dimensional arrays")

    if len(t) == 0 or len(y) == 0:
        raise ValueError("t and y must be non-empty")

    if len(t) != len(y):
        raise ValueError("t and y must have the same length")

    if len(t) < 2:
        raise ValueError("t and y must have at least 2 samples")

    if settling_threshold <= 0:
        raise ValueError("settling_threshold must be positive")

    low, high = rise_limits
    if low < 0 or high > 1 or low >= high:
        raise ValueError("rise_limits must be between 0 and 1 with low < high")


def _crossing_time(t, y, level, increasing):
    """Return the first interpolated time where y crosses a level."""
    for index in range(len(y)):
        reached = y[index] >= level if increasing else y[index] <= level

        if not reached:
            continue

        if index == 0:
            return float(t[0])

        previous_y = y[index - 1]
        current_y = y[index]
        previous_t = t[index - 1]
        current_t = t[index]

        if np.isclose(current_y, previous_y):
            return float(current_t)

        fraction = (level - previous_y) / (current_y - previous_y)
        return float(previous_t + fraction * (current_t - previous_t))

    return None


def _settling_time(t, y, steady_state_value, band):
    """Return the earliest time after which all samples stay inside the band."""
    lower = steady_state_value - band
    upper = steady_state_value + band
    inside_band = (y >= lower) & (y <= upper)

    for index in range(len(y)):
        if np.all(inside_band[index:]):
            return float(t[index])

    return None


def _is_monotonic_response(y, increasing):
    """Return True when the response moves in one direction only."""
    differences = np.diff(y)

    if increasing:
        return bool(np.all(differences >= -1e-12))

    return bool(np.all(differences <= 1e-12))


def calculate_step_info(t, y, settling_threshold=0.02, rise_limits=(0.1, 0.9)):
    """Calculate standard step response metrics from time and output arrays.

    Parameters
    ----------
    t : array-like
        Time samples.
    y : array-like
        Output samples.
    settling_threshold : float, optional
        Fraction of the response change used for the settling band.
    rise_limits : tuple, optional
        Low and high fractions used for rise time, usually 10% and 90%.

    Returns
    -------
    StepResponseInfo
        Dataclass containing the calculated response metrics.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    _validate_inputs(t, y, settling_threshold, rise_limits)

    initial_value = float(y[0])
    final_value = float(y[-1])
    tail_count = max(1, int(np.ceil(0.05 * len(y))))
    steady_state_value = float(np.mean(y[-tail_count:]))

    delta = steady_state_value - initial_value
    increasing = delta >= 0

    if np.isclose(delta, 0):
        rise_time = None
        settling = float(t[0]) if np.allclose(y, steady_state_value) else None
        overshoot_percent = 0.0
    else:
        low_fraction, high_fraction = rise_limits
        low_level = initial_value + low_fraction * delta
        high_level = initial_value + high_fraction * delta

        t_low = _crossing_time(t, y, low_level, increasing)
        t_high = _crossing_time(t, y, high_level, increasing)
        rise_time = None if t_low is None or t_high is None else t_high - t_low

        band = settling_threshold * abs(delta)
        settling = _settling_time(t, y, steady_state_value, band)

        if _is_monotonic_response(y, increasing):
            overshoot_percent = 0.0
        elif increasing:
            peak_value = float(np.max(y))
            overshoot = (peak_value - steady_state_value) / abs(delta) * 100
            overshoot_percent = float(max(0, overshoot))
        else:
            peak_value = float(np.min(y))
            overshoot = (steady_state_value - peak_value) / abs(delta) * 100
            overshoot_percent = float(max(0, overshoot))

    if increasing:
        peak_index = int(np.argmax(y))
    else:
        peak_index = int(np.argmin(y))

    peak_value = float(y[peak_index])
    peak_time = float(t[peak_index])

    return StepResponseInfo(
        initial_value=initial_value,
        final_value=final_value,
        steady_state_value=steady_state_value,
        rise_time=rise_time,
        settling_time=settling,
        peak_value=peak_value,
        peak_time=peak_time,
        overshoot_percent=overshoot_percent,
    )
