import pytest

from analysis.rlc_sweep import run_rlc_sweep


EXPECTED_SUMMARY_KEYS = {
    "parameter_name",
    "parameter_value",
    "natural_frequency",
    "damping_ratio",
    "final_voltage",
    "peak_voltage",
    "overshoot_percent",
    "settling_time",
}


def test_helper_returns_one_result_per_parameter_value():
    """Each sweep value should produce one summary result."""
    parameter_values = [0.5, 1.0, 2.0]

    results, _ = run_rlc_sweep("R", parameter_values)

    assert len(results) == len(parameter_values)


def test_invalid_parameter_name_raises_value_error():
    """Only R, L, and C sweeps are supported."""
    with pytest.raises(ValueError):
        run_rlc_sweep("Vin", [5.0])


def test_empty_parameter_list_raises_value_error():
    """A sweep must include at least one parameter value."""
    with pytest.raises(ValueError):
        run_rlc_sweep("R", [])


def test_zero_inductance_raises_value_error_through_model_validation():
    """Invalid inductance should be rejected by the existing RLC validation."""
    with pytest.raises(ValueError):
        run_rlc_sweep("L", [0.0])


def test_negative_capacitance_raises_value_error_through_model_validation():
    """Invalid capacitance should be rejected by the existing RLC validation."""
    with pytest.raises(ValueError):
        run_rlc_sweep("C", [-0.25])


def test_lower_resistance_has_higher_or_equal_overshoot_than_higher_resistance():
    """Lower R should reduce damping and increase overshoot for the chosen cases."""
    results, _ = run_rlc_sweep(
        "R",
        [0.5, 5.0],
        L=1.0,
        C=0.25,
        Vin=5.0,
        t_span=(0.0, 12.0),
        num_points=1200,
    )

    low_resistance_result = results[0]
    high_resistance_result = results[1]

    assert (
        low_resistance_result["overshoot_percent"]
        >= high_resistance_result["overshoot_percent"]
    )


def test_returned_simulation_arrays_have_equal_lengths():
    """Time, voltage, and current arrays should align for each simulation."""
    _, simulations = run_rlc_sweep("C", [0.1, 0.25], num_points=300)

    for simulation in simulations:
        assert len(simulation["t"]) == 300
        assert len(simulation["capacitor_voltage"]) == len(simulation["t"])
        assert len(simulation["current"]) == len(simulation["t"])


def test_summary_metrics_contain_expected_keys():
    """Sweep summaries should expose the expected educational metrics."""
    results, _ = run_rlc_sweep("L", [1.0])

    assert EXPECTED_SUMMARY_KEYS.issubset(results[0].keys())
