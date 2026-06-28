import pytest

from models.discrete_pid import DiscretePID, simulate_discrete_pid_motor_control


def _simulate_example_response():
    """Run an example-like discrete PID motor simulation."""
    pid = DiscretePID(
        Kp=0.35,
        Ki=0.8,
        Kd=0.02,
        output_min=0.0,
        output_max=24.0,
        anti_windup=True,
    )

    return simulate_discrete_pid_motor_control(
        R=1.0,
        L=0.5,
        J=0.01,
        b=0.001,
        Kt=0.01,
        Ke=0.01,
        pid=pid,
        target_speed=80.0,
        i0=0.0,
        omega0=0.0,
        t_final=25.0,
        dt=0.01,
    )


def test_negative_kp_raises_value_error():
    """Kp must be nonnegative."""
    with pytest.raises(ValueError):
        DiscretePID(-1.0, 0.0, 0.0)


def test_negative_ki_raises_value_error():
    """Ki must be nonnegative."""
    with pytest.raises(ValueError):
        DiscretePID(0.0, -1.0, 0.0)


def test_negative_kd_raises_value_error():
    """Kd must be nonnegative."""
    with pytest.raises(ValueError):
        DiscretePID(0.0, 0.0, -1.0)


def test_invalid_output_limits_raise_value_error():
    """Maximum output must be greater than minimum output."""
    with pytest.raises(ValueError):
        DiscretePID(1.0, 0.0, 0.0, output_min=5.0, output_max=5.0)


def test_update_with_nonpositive_dt_raises_value_error():
    """Controller update sample time must be positive."""
    pid = DiscretePID(1.0, 0.0, 0.0)

    with pytest.raises(ValueError):
        pid.update(setpoint=1.0, measurement=0.0, dt=0.0)


def test_pure_proportional_output():
    """A pure proportional controller should return Kp times error."""
    pid = DiscretePID(Kp=2.0, Ki=0.0, Kd=0.0)
    output = pid.update(setpoint=10.0, measurement=7.0, dt=0.1)

    assert output == pytest.approx(6.0)


def test_output_saturation_high():
    """Controller output should saturate at output_max."""
    pid = DiscretePID(Kp=2.0, Ki=0.0, Kd=0.0, output_max=5.0)
    output = pid.update(setpoint=10.0, measurement=0.0, dt=0.1)

    assert output == pytest.approx(5.0)


def test_output_saturation_low():
    """Controller output should saturate at output_min."""
    pid = DiscretePID(Kp=2.0, Ki=0.0, Kd=0.0, output_min=0.0)
    output = pid.update(setpoint=0.0, measurement=10.0, dt=0.1)

    assert output == pytest.approx(0.0)


def test_integral_accumulates_when_not_saturated():
    """Integral state should accumulate over repeated unsaturated updates."""
    pid = DiscretePID(Kp=0.0, Ki=1.0, Kd=0.0)

    pid.update(setpoint=1.0, measurement=0.0, dt=0.5)
    pid.update(setpoint=1.0, measurement=0.0, dt=0.5)

    assert pid.integral == pytest.approx(1.0)


def test_reset_clears_internal_state():
    """Reset should clear integral, previous values, and last output."""
    pid = DiscretePID(Kp=1.0, Ki=1.0, Kd=1.0)
    pid.update(setpoint=1.0, measurement=0.0, dt=0.1)

    pid.reset()

    assert pid.integral == pytest.approx(0.0)
    assert pid.previous_error is None
    assert pid.previous_measurement is None
    assert pid.last_output is None


def test_derivative_on_measurement_avoids_derivative_kick():
    """A setpoint-only change should not create derivative output."""
    pid = DiscretePID(Kp=0.0, Ki=0.0, Kd=1.0)

    pid.update(setpoint=0.0, measurement=0.0, dt=0.1)
    output = pid.update(setpoint=10.0, measurement=0.0, dt=0.1)

    assert output == pytest.approx(0.0)


def test_anti_windup_limits_integral_growth_when_saturated_high():
    """Integral should not grow farther into upper saturation."""
    pid = DiscretePID(Kp=0.0, Ki=1.0, Kd=0.0, output_max=1.0)

    pid.update(setpoint=10.0, measurement=0.0, dt=1.0)
    pid.update(setpoint=10.0, measurement=0.0, dt=1.0)

    assert pid.integral == pytest.approx(0.0)


def test_simulation_returns_arrays_of_equal_length():
    """Discrete PID motor simulation should return aligned arrays."""
    t, current, speed, voltage, error = _simulate_example_response()

    assert len(current) == len(t)
    assert len(speed) == len(t)
    assert len(voltage) == len(t)
    assert len(error) == len(t)


def test_initial_current_and_speed_match_inputs():
    """The returned simulation should start from i0 and omega0."""
    _, current, speed, _, _ = _simulate_example_response()

    assert current[0] == pytest.approx(0.0)
    assert speed[0] == pytest.approx(0.0)


def test_simulation_voltage_values_stay_within_limits():
    """Returned voltage samples should respect PID output limits."""
    _, _, _, voltage, _ = _simulate_example_response()

    assert voltage.min() >= 0.0
    assert voltage.max() <= 24.0


def test_final_speed_approaches_target():
    """Discrete PID speed control should approach the target speed."""
    _, _, speed, _, _ = _simulate_example_response()

    assert abs(80.0 - speed[-1]) < 5.0
