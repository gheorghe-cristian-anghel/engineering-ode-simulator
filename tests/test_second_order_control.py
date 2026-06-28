import numpy as np
import pytest

from models.second_order_control import (
    approx_settling_time,
    damped_natural_frequency,
    response_type,
    simulate_second_order_system,
    theoretical_overshoot_percent,
    theoretical_peak_time,
)


def test_response_type_labels():
    """Response type should match standard damping ratio categories."""
    assert response_type(0) == "undamped"
    assert response_type(0.3) == "underdamped"
    assert response_type(1.0) == "critically damped"
    assert response_type(2.0) == "overdamped"


def test_damped_natural_frequency():
    """Underdamped frequency should be omega_n*sqrt(1-zeta^2)."""
    assert damped_natural_frequency(4, 0.3) == pytest.approx(
        4 * np.sqrt(1 - 0.3**2)
    )


def test_damped_natural_frequency_none_for_zeta_at_least_one():
    """Damped natural frequency is only defined for underdamped response."""
    assert damped_natural_frequency(4, 1.0) is None
    assert damped_natural_frequency(4, 2.0) is None


def test_theoretical_overshoot_percent():
    """For zeta = 0.3, theoretical overshoot should be about 37.2%."""
    assert theoretical_overshoot_percent(0.3) == pytest.approx(37.2, rel=0.01)


def test_theoretical_overshoot_percent_zero_for_critical_damping():
    """Critical damping should have no overshoot."""
    assert theoretical_overshoot_percent(1.0) == pytest.approx(0)


def test_theoretical_peak_time():
    """Peak time should be pi divided by damped natural frequency."""
    omega_d = damped_natural_frequency(4, 0.3)

    assert theoretical_peak_time(4, 0.3) == pytest.approx(np.pi / omega_d)


def test_theoretical_peak_time_none_for_critical_damping():
    """Critical damping does not have an underdamped peak time."""
    assert theoretical_peak_time(4, 1.0) is None


def test_approx_settling_time():
    """Settling time approximation should follow -log(tol)/(zeta*omega_n)."""
    assert approx_settling_time(4, 0.3, 0.02) == pytest.approx(
        -np.log(0.02) / (0.3 * 4)
    )


def test_initial_output_equals_y0():
    """The simulated output should start at y0."""
    t, output, _ = simulate_second_order_system(4, 0.3, 0.2, 0, (0, 8), 200)

    assert t[0] == pytest.approx(0)
    assert output[0] == pytest.approx(0.2)


def test_initial_velocity_equals_v0():
    """The simulated output velocity should start at v0."""
    _, _, velocity = simulate_second_order_system(4, 0.3, 0, 0.5, (0, 8), 200)

    assert velocity[0] == pytest.approx(0.5)


def test_final_output_approaches_unit_step():
    """For a unit step and enough time, final output should approach 1."""
    _, output, _ = simulate_second_order_system(4, 0.3, 0, 0, (0, 12), 3000)

    assert output[-1] == pytest.approx(1, rel=0.01)


def test_final_velocity_approaches_zero():
    """For a stable unit-step response, final velocity should approach zero."""
    _, _, velocity = simulate_second_order_system(4, 0.3, 0, 0, (0, 12), 3000)

    assert velocity[-1] == pytest.approx(0, abs=0.01)


def test_underdamped_response_overshoots_unit_step():
    """An underdamped second-order step response should overshoot."""
    _, output, _ = simulate_second_order_system(4, 0.3, 0, 0, (0, 8), 2000)

    assert np.max(output) > 1


def test_invalid_omega_n_raises_value_error():
    """Natural frequency must be positive."""
    with pytest.raises(ValueError):
        simulate_second_order_system(0, 0.3, 0, 0, (0, 8), 200)


def test_invalid_zeta_raises_value_error():
    """Damping ratio must be nonnegative."""
    with pytest.raises(ValueError):
        simulate_second_order_system(4, -0.1, 0, 0, (0, 8), 200)


def test_invalid_num_points_raises_value_error():
    """Number of samples must be positive."""
    with pytest.raises(ValueError):
        simulate_second_order_system(4, 0.3, 0, 0, (0, 8), 0)


def test_invalid_t_span_raises_value_error():
    """The final simulation time must be greater than the initial time."""
    with pytest.raises(ValueError):
        simulate_second_order_system(4, 0.3, 0, 0, (8, 0), 200)
