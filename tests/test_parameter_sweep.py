import pytest

from analysis.parameter_sweep import calculate_tracking_error, run_pi_motor_gain_sweep


def _default_sweep_inputs():
    """Return practical default inputs for PI motor gain sweep tests."""
    motor_params = {
        "R": 1.0,
        "L": 0.5,
        "J": 0.01,
        "b": 0.001,
        "Kt": 0.01,
        "Ke": 0.01,
    }
    controller_params = {
        "Kp": 0.8,
        "Ki": 2.0,
        "Kd": 0.0,
        "voltage_min": 0.0,
        "voltage_max": 24.0,
        "target_speed": 80.0,
    }
    simulation_params = {
        "i0": 0.0,
        "omega0": 0.0,
        "integral_error0": 0.0,
        "t_span": (0.0, 8.0),
        "num_points": 1000,
    }

    return motor_params, controller_params, simulation_params


def test_tracking_error_is_positive_when_final_value_is_below_reference():
    """Tracking error should be reference minus final value."""
    assert calculate_tracking_error(80, 78) == pytest.approx(2)


def test_tracking_error_is_negative_when_final_value_is_above_reference():
    """Tracking error should be negative when the response ends above reference."""
    assert calculate_tracking_error(80, 82) == pytest.approx(-2)


def test_invalid_parameter_name_raises_value_error():
    """Only Kp and Ki sweeps are supported."""
    motor_params, controller_params, simulation_params = _default_sweep_inputs()

    with pytest.raises(ValueError):
        run_pi_motor_gain_sweep(
            "Kd",
            [0.8],
            motor_params,
            controller_params,
            simulation_params,
        )


def test_empty_parameter_values_raise_value_error():
    """A sweep must include at least one parameter value."""
    motor_params, controller_params, simulation_params = _default_sweep_inputs()

    with pytest.raises(ValueError):
        run_pi_motor_gain_sweep(
            "Kp",
            [],
            motor_params,
            controller_params,
            simulation_params,
        )


def test_negative_parameter_value_raises_value_error():
    """PI controller gains must be nonnegative."""
    motor_params, controller_params, simulation_params = _default_sweep_inputs()

    with pytest.raises(ValueError):
        run_pi_motor_gain_sweep(
            "Kp",
            [0.2, -0.5],
            motor_params,
            controller_params,
            simulation_params,
        )


def test_sweep_returns_one_result_per_parameter_value():
    """Each sweep value should produce one SweepResult."""
    motor_params, controller_params, simulation_params = _default_sweep_inputs()
    parameter_values = [0.2, 0.8, 1.2]

    results, _ = run_pi_motor_gain_sweep(
        "Kp",
        parameter_values,
        motor_params,
        controller_params,
        simulation_params,
    )

    assert len(results) == len(parameter_values)


def test_sweep_returns_one_simulation_per_parameter_value():
    """Each sweep value should produce one simulation dictionary."""
    motor_params, controller_params, simulation_params = _default_sweep_inputs()
    parameter_values = [0.2, 0.8, 1.2]

    _, simulations = run_pi_motor_gain_sweep(
        "Kp",
        parameter_values,
        motor_params,
        controller_params,
        simulation_params,
    )

    assert len(simulations) == len(parameter_values)


def test_every_result_has_correct_parameter_name():
    """Sweep results should record the swept parameter name."""
    motor_params, controller_params, simulation_params = _default_sweep_inputs()

    results, _ = run_pi_motor_gain_sweep(
        "Ki",
        [0.5, 2.0],
        motor_params,
        controller_params,
        simulation_params,
    )

    assert all(result.parameter_name == "Ki" for result in results)


def test_voltage_values_remain_within_limits():
    """All returned voltage samples should respect saturation limits."""
    motor_params, controller_params, simulation_params = _default_sweep_inputs()

    _, simulations = run_pi_motor_gain_sweep(
        "Kp",
        [0.2, 0.8, 1.2],
        motor_params,
        controller_params,
        simulation_params,
    )

    voltage_min = controller_params["voltage_min"]
    voltage_max = controller_params["voltage_max"]

    for simulation in simulations.values():
        assert simulation["voltage"].min() >= voltage_min
        assert simulation["voltage"].max() <= voltage_max


def test_reasonable_kp_sweep_tracks_target_speed():
    """With Kp=0.8 and Ki=2.0, final speed error should be small."""
    motor_params, controller_params, simulation_params = _default_sweep_inputs()

    results, _ = run_pi_motor_gain_sweep(
        "Kp",
        [0.8],
        motor_params,
        controller_params,
        simulation_params,
    )

    assert abs(results[0].final_error) < 5
