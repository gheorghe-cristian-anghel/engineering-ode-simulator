import math

import pytest

from analysis.pid_tuning import compare_pid_cases, run_pid_tuning_case


def test_pid_tuning_case_returns_result_and_simulation():
    """A tuning case should return summary metrics and simulation arrays."""
    result, simulation = run_pid_tuning_case(
        "Baseline PID",
        Kp=0.16,
        Ki=0.018,
        Kd=0.12,
        t_span=(0.0, 5.0),
        dt=0.02,
    )

    assert result.label == "Baseline PID"
    assert set(simulation) == {"t", "current", "speed", "voltage", "error"}


def test_pid_tuning_metrics_are_finite():
    """Scalar PID tuning metrics should be finite values."""
    result, _ = run_pid_tuning_case(
        "Baseline PID",
        Kp=0.16,
        Ki=0.018,
        Kd=0.12,
        t_span=(0.0, 5.0),
        dt=0.02,
    )

    assert math.isfinite(result.final_speed)
    assert math.isfinite(result.final_error)
    assert math.isfinite(result.peak_speed)
    assert math.isfinite(result.overshoot_percent)
    assert math.isfinite(result.max_voltage)
    assert math.isfinite(result.max_current)


def test_pid_tuning_simulation_arrays_have_equal_lengths():
    """Returned simulation arrays should be aligned sample-by-sample."""
    _, simulation = run_pid_tuning_case(
        "Baseline PID",
        Kp=0.16,
        Ki=0.018,
        Kd=0.12,
        t_span=(0.0, 5.0),
        dt=0.02,
    )

    lengths = {len(values) for values in simulation.values()}

    assert len(lengths) == 1


def test_baseline_pid_final_speed_approaches_target():
    """The baseline PID gains should approach the target over a long run."""
    result, _ = run_pid_tuning_case(
        "Baseline PID",
        Kp=0.16,
        Ki=0.018,
        Kd=0.12,
        target_speed=80.0,
        t_span=(0.0, 25.0),
        dt=0.01,
    )

    assert result.final_speed == pytest.approx(80.0, abs=2.0)


def test_negative_gain_raises_value_error():
    """Discrete PID validation should reject negative gains."""
    with pytest.raises(ValueError):
        run_pid_tuning_case(
            "Invalid",
            Kp=-0.1,
            Ki=0.018,
            Kd=0.12,
        )


def test_compare_pid_cases_returns_one_result_per_case():
    """Comparison helper should run every supplied tuning case."""
    cases = [
        {"label": "P", "Kp": 0.16, "Ki": 0.0, "Kd": 0.0},
        {"label": "PI", "Kp": 0.16, "Ki": 0.018, "Kd": 0.0},
        {"label": "PID", "Kp": 0.16, "Ki": 0.018, "Kd": 0.12},
    ]

    results, simulations = compare_pid_cases(cases, t_span=(0.0, 5.0), dt=0.02)

    assert len(results) == len(cases)
    assert set(simulations) == {"P", "PI", "PID"}
