import numpy as np
import pytest

from analysis.step_response import calculate_step_info
from models.first_order_control import analytical_step_response


def test_invalid_input_lengths_raise_value_error():
    """Time and response arrays must have the same length."""
    with pytest.raises(ValueError):
        calculate_step_info([0, 1, 2], [0, 1])


def test_empty_arrays_raise_value_error():
    """Empty arrays do not contain enough information for metrics."""
    with pytest.raises(ValueError):
        calculate_step_info([], [])


def test_first_order_response_rise_time_matches_expected_value():
    """A first-order response should have 10% to 90% rise time tau*log(9)."""
    tau = 1.5
    t = np.linspace(0, 20 * tau, 5000)
    y = analytical_step_response(t, tau, K=2.0, amplitude=1.0, y0=0.0)

    info = calculate_step_info(t, y)

    assert info.rise_time == pytest.approx(tau * np.log(9), rel=0.01)


def test_first_order_response_settling_time_matches_expected_value():
    """A first-order response should settle near -tau*log(0.02)."""
    tau = 1.5
    t = np.linspace(0, 20 * tau, 5000)
    y = analytical_step_response(t, tau, K=2.0, amplitude=1.0, y0=0.0)

    info = calculate_step_info(t, y)

    assert info.settling_time == pytest.approx(-tau * np.log(0.02), rel=0.01)


def test_first_order_response_has_no_overshoot():
    """A monotonic first-order response should have no overshoot."""
    tau = 1.5
    t = np.linspace(0, 20 * tau, 5000)
    y = analytical_step_response(t, tau, K=2.0, amplitude=1.0, y0=0.0)

    info = calculate_step_info(t, y)

    assert info.overshoot_percent == pytest.approx(0)


def test_artificial_overshoot_response_has_20_percent_overshoot():
    """A response peaking at 1.2 with steady state 1.0 has 20% overshoot."""
    t = np.arange(10)
    y = np.array([0.0, 0.5, 1.2, 1.05, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    info = calculate_step_info(t, y)

    assert info.overshoot_percent == pytest.approx(20.0)


def test_response_that_never_settles_returns_none():
    """If the tail keeps leaving the settling band, settling time is None."""
    t = np.linspace(0, 10, 100)
    y = np.ones_like(t)
    y[0] = 0.0
    y[-5:] = [0.8, 1.2, 0.8, 1.2, 0.8]

    info = calculate_step_info(t, y)

    assert info.settling_time is None
