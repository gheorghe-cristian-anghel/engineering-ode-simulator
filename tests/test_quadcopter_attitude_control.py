import numpy as np
import pytest
from numpy.testing import assert_allclose

from analysis.quadcopter_attitude_control import (
    disturbance_torque_step,
    simulate_attitude_pid_control,
    summarize_attitude_response,
)


def test_disturbance_torque_step_returns_before_and_after_vectors():
    """Disturbance torque step should switch at and after the step time."""
    disturbance_func = disturbance_torque_step(
        2.0,
        before=(0.0, 0.0, 0.0),
        after=(0.02, 0.0, -0.01),
    )

    assert_allclose(disturbance_func(1.99), [0.0, 0.0, 0.0])
    assert_allclose(disturbance_func(2.0), [0.02, 0.0, -0.01])
    assert_allclose(disturbance_func(3.0), [0.02, 0.0, -0.01])


def test_simulation_returns_expected_shapes():
    """Attitude PID simulation should return aligned time-series arrays."""
    result = simulate_attitude_pid_control(t_span=(0.0, 1.0), dt=0.02)
    sample_count = len(result["time"])

    assert result["states"].shape == (sample_count, 6)
    assert result["angles_rad"].shape == (sample_count, 3)
    assert result["angles_deg"].shape == (sample_count, 3)
    assert result["body_rates"].shape == (sample_count, 3)
    assert result["torques"].shape == (sample_count, 3)
    assert result["errors_rad"].shape == (sample_count, 3)
    assert result["errors_deg"].shape == (sample_count, 3)


def test_initial_state_is_preserved():
    """The response should start from the configured initial attitude state."""
    initial_state = [0.1, -0.2, 0.3, 0.4, -0.5, 0.6]

    result = simulate_attitude_pid_control(
        initial_state=initial_state,
        t_span=(0.0, 1.0),
        dt=0.02,
    )

    assert_allclose(result["states"][0], initial_state)


def test_torques_remain_within_limits():
    """PID torque commands should respect configured saturation limits."""
    torque_limits = (-0.05, 0.05)

    result = simulate_attitude_pid_control(
        t_span=(0.0, 2.0),
        dt=0.02,
        torque_limits=torque_limits,
    )

    assert np.min(result["torques"]) >= torque_limits[0]
    assert np.max(result["torques"]) <= torque_limits[1]


def test_default_response_moves_toward_target_angles():
    """Default gains should move all attitude axes toward their targets."""
    result = simulate_attitude_pid_control(t_span=(0.0, 3.0), dt=0.02)

    initial_error = np.abs(result["errors_deg"][0])
    final_error = np.abs(result["errors_deg"][-1])

    assert np.all(final_error < initial_error)


def test_final_errors_are_reasonably_small_for_default_gains():
    """Default gains should track close to the target over a longer run."""
    result = simulate_attitude_pid_control(t_span=(0.0, 7.0), dt=0.02)

    assert np.all(np.abs(result["errors_deg"][-1]) < 1.0)


def test_zero_target_from_zero_state_stays_near_zero():
    """Zero attitude command from rest should remain near zero attitude."""
    result = simulate_attitude_pid_control(
        target_angles=(0.0, 0.0, 0.0),
        t_span=(0.0, 2.0),
        dt=0.02,
    )

    assert_allclose(result["angles_deg"][-1], [0.0, 0.0, 0.0], atol=1e-9)
    assert_allclose(result["body_rates"][-1], [0.0, 0.0, 0.0], atol=1e-9)


def test_invalid_dt_raises_value_error():
    """Sample time must be positive."""
    with pytest.raises(ValueError):
        simulate_attitude_pid_control(dt=0.0)


def test_invalid_torque_limits_raise_value_error():
    """Maximum torque limit must be greater than minimum torque limit."""
    with pytest.raises(ValueError):
        simulate_attitude_pid_control(torque_limits=(0.2, 0.2))


def test_invalid_gain_length_raises_value_error():
    """Each gain input must contain one value per attitude axis."""
    with pytest.raises(ValueError):
        simulate_attitude_pid_control(Kp=(0.08, 0.08))


def test_negative_gain_raises_value_error():
    """PID gains must be nonnegative."""
    with pytest.raises(ValueError):
        simulate_attitude_pid_control(Ki=(0.01, -0.01, 0.01))


def test_metrics_helper_returns_expected_fields():
    """Attitude metrics should include the expected summary dictionaries."""
    result = simulate_attitude_pid_control(t_span=(0.0, 3.0), dt=0.02)
    metrics = summarize_attitude_response(result)

    assert set(metrics.__dict__) == {
        "final_angle_deg",
        "final_error_deg",
        "peak_angle_deg",
        "overshoot_percent",
        "settling_time",
        "max_abs_torque",
    }
    assert set(metrics.final_angle_deg) == {"roll", "pitch", "yaw"}
