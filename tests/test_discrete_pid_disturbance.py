import numpy as np
import pytest

from models.discrete_pid import (
    load_torque_step,
    simulate_discrete_pid_motor_with_disturbance,
    summarize_disturbance_response,
)


def _simulate_disturbance_response():
    """Run a discrete PID disturbance response with example-like defaults."""
    return simulate_discrete_pid_motor_with_disturbance()


def test_load_torque_is_initial_before_disturbance_time():
    """Load torque should equal the initial value before the disturbance."""
    torque = load_torque_step(
        11.99,
        disturbance_time=12.0,
        initial_torque=0.0,
        final_torque=0.03,
    )

    assert torque == pytest.approx(0.0)


def test_load_torque_is_final_at_disturbance_time():
    """Load torque should switch at the disturbance time."""
    torque = load_torque_step(
        12.0,
        disturbance_time=12.0,
        initial_torque=0.0,
        final_torque=0.03,
    )

    assert torque == pytest.approx(0.03)


def test_load_torque_is_final_after_disturbance_time():
    """Load torque should stay at the final value after the disturbance."""
    torque = load_torque_step(
        13.0,
        disturbance_time=12.0,
        initial_torque=0.0,
        final_torque=0.03,
    )

    assert torque == pytest.approx(0.03)


def test_disturbance_simulation_arrays_have_equal_lengths():
    """Returned time-series arrays should stay aligned sample-by-sample."""
    result = _simulate_disturbance_response()
    lengths = {
        len(result["time"]),
        len(result["speed"]),
        len(result["current"]),
        len(result["voltage"]),
        len(result["error"]),
        len(result["load_torque"]),
    }

    assert len(lengths) == 1


def test_voltage_remains_within_limits():
    """Discrete PID voltage should respect configured saturation limits."""
    result = _simulate_disturbance_response()

    assert result["voltage"].min() >= 0.0
    assert result["voltage"].max() <= 24.0


def test_speed_drops_after_disturbance():
    """A load torque step should cause a temporary speed drop."""
    result = _simulate_disturbance_response()
    time = result["time"]
    speed = result["speed"]
    disturbance_index = int(np.argmin(np.abs(time - 12.0)))
    before_index = max(0, disturbance_index - 1)

    speed_before = speed[before_index]
    minimum_speed_after = np.min(speed[disturbance_index:])

    assert minimum_speed_after < speed_before


def test_final_error_is_smaller_than_largest_post_disturbance_error():
    """The controller should recover toward the target after the speed drop."""
    result = _simulate_disturbance_response()
    time = result["time"]
    speed = result["speed"]
    target_speed = result["target_speed"]
    disturbance_index = int(np.argmin(np.abs(time - 12.0)))

    final_error = abs(target_speed - speed[-1])
    largest_post_disturbance_error = np.max(np.abs(target_speed - speed[disturbance_index:]))

    assert final_error < largest_post_disturbance_error


def test_default_disturbance_response_recovers_near_target():
    """Default disturbance-response gains should recover close to target speed."""
    result = _simulate_disturbance_response()

    assert result["speed"][-1] == pytest.approx(result["target_speed"], abs=2.0)


def test_metrics_helper_returns_expected_keys():
    """Disturbance metrics should include the expected summary values."""
    result = _simulate_disturbance_response()
    metrics = summarize_disturbance_response(
        result,
        disturbance_time=12.0,
        target_speed=result["target_speed"],
    )

    assert set(metrics) == {
        "speed_before_disturbance",
        "minimum_speed_after_disturbance",
        "speed_drop",
        "final_speed",
        "final_error",
        "voltage_before_disturbance",
        "voltage_after_recovery_or_final",
        "current_before_disturbance",
        "current_after_recovery_or_final",
        "recovery_time",
    }


def test_recovery_time_is_none_or_nonnegative():
    """Recovery time should be None when unrecovered or a nonnegative duration."""
    result = _simulate_disturbance_response()
    metrics = summarize_disturbance_response(
        result,
        disturbance_time=12.0,
        target_speed=result["target_speed"],
    )

    assert metrics["recovery_time"] is None or metrics["recovery_time"] >= 0.0
