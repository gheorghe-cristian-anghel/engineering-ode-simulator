import numpy as np
import pytest

from analysis.motor_disturbance import (
    load_torque_step,
    summarize_motor_disturbance_response,
)


EXPECTED_BASE_KEYS = {
    "speed_before_disturbance",
    "minimum_speed_after_disturbance",
    "speed_drop",
    "final_speed",
    "final_error",
    "recovery_time",
}


def test_load_torque_step_returns_initial_value_before_step():
    """The load torque should equal the initial value before the step."""
    torque = load_torque_step(11.0, step_time=12.0, initial=0.0, final=0.03)

    assert torque == pytest.approx(0.0)


def test_load_torque_step_returns_final_value_at_and_after_step():
    """The load torque should switch to the final value at the step time."""
    assert load_torque_step(12.0, step_time=12.0, initial=0.0, final=0.03) == pytest.approx(0.03)
    assert load_torque_step(13.0, step_time=12.0, initial=0.0, final=0.03) == pytest.approx(0.03)


def test_metrics_helper_returns_expected_keys():
    """The base metrics should be present without optional voltage/current data."""
    time = np.arange(6, dtype=float)
    speed = np.array([0.0, 10.0, 10.0, 8.0, 9.0, 9.5])

    metrics = summarize_motor_disturbance_response(
        time,
        speed,
        target_speed=10.0,
        disturbance_time=2.0,
    )

    assert set(metrics) == EXPECTED_BASE_KEYS


def test_recovery_time_is_none_for_response_that_never_recovers():
    """A response that stays outside the target band should not report recovery."""
    time = np.arange(6, dtype=float)
    speed = np.array([0.0, 10.0, 10.0, 8.0, 8.5, 8.8])

    metrics = summarize_motor_disturbance_response(
        time,
        speed,
        target_speed=10.0,
        disturbance_time=2.0,
    )

    assert metrics["recovery_time"] is None


def test_recovery_time_is_nonnegative_for_response_that_recovers():
    """Recovery time should be nonnegative when the response returns to target."""
    time = np.arange(6, dtype=float)
    speed = np.array([0.0, 10.0, 10.0, 8.0, 9.9, 10.0])

    metrics = summarize_motor_disturbance_response(
        time,
        speed,
        target_speed=10.0,
        disturbance_time=2.0,
    )

    assert metrics["recovery_time"] is not None
    assert metrics["recovery_time"] >= 0.0


def test_speed_drop_is_positive_when_response_dips_after_disturbance():
    """A post-disturbance speed dip should produce a positive speed drop."""
    time = np.arange(6, dtype=float)
    speed = np.array([0.0, 10.0, 10.0, 8.0, 9.0, 10.0])

    metrics = summarize_motor_disturbance_response(
        time,
        speed,
        target_speed=10.0,
        disturbance_time=2.0,
    )

    assert metrics["speed_drop"] > 0.0


def test_helper_handles_optional_voltage_and_current_fields():
    """Optional voltage and current arrays should add before/final metrics."""
    time = np.arange(6, dtype=float)
    speed = np.array([0.0, 10.0, 10.0, 8.0, 9.9, 10.0])
    voltage = np.array([0.0, 5.0, 5.0, 7.0, 7.5, 7.2])
    current = np.array([0.0, 1.0, 1.0, 1.5, 1.4, 1.3])

    metrics = summarize_motor_disturbance_response(
        time,
        speed,
        target_speed=10.0,
        disturbance_time=2.0,
        voltage=voltage,
        current=current,
    )

    assert metrics["voltage_before_disturbance"] == pytest.approx(5.0)
    assert metrics["voltage_final"] == pytest.approx(7.2)
    assert metrics["current_before_disturbance"] == pytest.approx(1.0)
    assert metrics["current_final"] == pytest.approx(1.3)


def test_mismatched_optional_array_length_raises_value_error():
    """Optional arrays must align with the time samples."""
    with pytest.raises(ValueError):
        summarize_motor_disturbance_response(
            [0.0, 1.0, 2.0],
            [10.0, 9.0, 10.0],
            target_speed=10.0,
            disturbance_time=1.0,
            voltage=[5.0, 6.0],
        )
