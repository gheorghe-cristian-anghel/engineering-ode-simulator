import numpy as np
import pytest

from models.first_order_control import (
    analytical_step_response,
    rise_time,
    settling_time,
    simulate_first_order_system,
    steady_state_value,
)


def test_initial_output_equals_y0():
    """The simulated output should start at y0."""
    tau = 1.5
    K = 2.0
    y0 = 0.5

    t, output = simulate_first_order_system(tau, K, y0, (0, 10), 100)

    assert t[0] == pytest.approx(0)
    assert output[0] == pytest.approx(y0)


def test_steady_state_value():
    """The steady-state value should be gain times input amplitude."""
    assert steady_state_value(2.0, 1.0) == pytest.approx(2.0)


def test_analytical_response_at_one_time_constant():
    """At t = tau, output should reach about 1 - exp(-1) of final value."""
    tau = 1.5
    K = 2.0
    amplitude = 1.0
    y0 = 0.0
    final_value = steady_state_value(K, amplitude)

    output_at_tau = analytical_step_response(tau, tau, K, amplitude, y0)

    assert output_at_tau == pytest.approx((1 - np.exp(-1)) * final_value)


def test_numerical_and_analytical_solutions_match_for_step_input():
    """Numerical and analytical step responses should closely agree."""
    tau = 1.5
    K = 2.0
    amplitude = 1.0
    y0 = 0.0

    t, numerical_output = simulate_first_order_system(tau, K, y0, (0, 10), 200)
    analytical_output = analytical_step_response(t, tau, K, amplitude, y0)

    assert numerical_output == pytest.approx(analytical_output, rel=0.01)


def test_settling_time():
    """Settling time should follow -tau * log(tolerance)."""
    tau = 1.5

    assert settling_time(tau, 0.02) == pytest.approx(-tau * np.log(0.02))


def test_rise_time():
    """Rise time should follow tau * log(9)."""
    tau = 1.5

    assert rise_time(tau) == pytest.approx(tau * np.log(9))


def test_invalid_tau_raises_value_error():
    """A non-positive time constant should raise ValueError."""
    with pytest.raises(ValueError):
        simulate_first_order_system(0, 2.0, 0.0, (0, 10), 100)
