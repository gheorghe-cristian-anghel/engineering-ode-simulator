import pytest

from analysis.quadcopter_altitude_control import (
    downward_force_step,
    simulate_altitude_pid_control,
    summarize_altitude_response,
)


def test_downward_force_step_returns_before_and_after_values():
    """Downward force step should switch at and after the step time."""
    force_func = downward_force_step(2.0, force_before=0.0, force_after=1.5)

    assert force_func(1.99) == pytest.approx(0.0)
    assert force_func(2.0) == pytest.approx(1.5)
    assert force_func(3.0) == pytest.approx(1.5)


def test_simulation_returns_arrays_of_equal_length():
    """Altitude PID simulation should return aligned time-series arrays."""
    result = simulate_altitude_pid_control(t_span=(0.0, 2.0), dt=0.02)
    lengths = {
        len(result["time"]),
        len(result["altitude"]),
        len(result["velocity"]),
        len(result["thrust"]),
        len(result["error"]),
        len(result["disturbance_force"]),
    }

    assert len(lengths) == 1


def test_initial_altitude_and_velocity_are_preserved():
    """The returned response should start from z0 and v0."""
    result = simulate_altitude_pid_control(
        target_altitude=5.0,
        z0=1.5,
        v0=-0.2,
        t_span=(0.0, 1.0),
        dt=0.02,
    )

    assert result["altitude"][0] == pytest.approx(1.5)
    assert result["velocity"][0] == pytest.approx(-0.2)


def test_thrust_remains_within_limits():
    """PID thrust command should respect configured saturation limits."""
    thrust_limits = (2.0, 18.0)

    result = simulate_altitude_pid_control(
        t_span=(0.0, 3.0),
        dt=0.02,
        thrust_limits=thrust_limits,
    )

    assert result["thrust"].min() >= thrust_limits[0]
    assert result["thrust"].max() <= thrust_limits[1]


def test_altitude_increases_toward_target_for_default_gains():
    """Default gains should move altitude upward toward a positive target."""
    result = simulate_altitude_pid_control(t_span=(0.0, 4.0), dt=0.02)

    assert result["altitude"][-1] > result["altitude"][0]
    assert result["altitude"][-1] > 0.5 * result["target_altitude"]


def test_final_altitude_is_reasonably_close_to_target():
    """Default gains should track close to the target over a longer run."""
    result = simulate_altitude_pid_control(t_span=(0.0, 10.0), dt=0.02)
    metrics = summarize_altitude_response(result)

    assert metrics.final_altitude == pytest.approx(result["target_altitude"], abs=0.2)


def test_hover_target_stays_near_hover_altitude():
    """When starting at the target, the controller should hold altitude."""
    result = simulate_altitude_pid_control(
        target_altitude=2.0,
        z0=2.0,
        v0=0.0,
        t_span=(0.0, 2.0),
        dt=0.02,
    )

    assert result["altitude"][-1] == pytest.approx(2.0, abs=0.05)


def test_invalid_gains_raise_value_error():
    """PID gains must be nonnegative."""
    with pytest.raises(ValueError):
        simulate_altitude_pid_control(Kp=-1.0)


def test_invalid_thrust_limits_raise_value_error():
    """Maximum thrust must be greater than minimum thrust."""
    with pytest.raises(ValueError):
        simulate_altitude_pid_control(thrust_limits=(10.0, 10.0))


def test_invalid_dt_raises_value_error():
    """Sample time must be positive."""
    with pytest.raises(ValueError):
        simulate_altitude_pid_control(dt=0.0)


def test_invalid_t_span_raises_value_error():
    """Final time must be greater than initial time."""
    with pytest.raises(ValueError):
        simulate_altitude_pid_control(t_span=(1.0, 0.0))


def test_metrics_helper_returns_expected_fields():
    """Altitude metrics should include the expected summary values."""
    result = simulate_altitude_pid_control(t_span=(0.0, 4.0), dt=0.02)
    metrics = summarize_altitude_response(result)

    assert set(metrics.__dict__) == {
        "final_altitude",
        "final_error",
        "peak_altitude",
        "overshoot_percent",
        "settling_time",
        "max_thrust",
        "min_thrust",
    }
