import pytest

from models.pid_motor_control import (
    anti_windup_integral_derivative,
    pi_control_voltage,
    simulate_pi_motor_control,
)


def test_pi_control_voltage_unsaturated():
    """PI voltage should equal Kp*error + Ki*integral_error inside limits."""
    voltage = pi_control_voltage(10, 2, Kp=0.5, Ki=1.0, voltage_min=0, voltage_max=24)

    assert voltage == pytest.approx(7)


def test_pi_control_voltage_saturates_at_maximum():
    """PI voltage should saturate at voltage_max."""
    voltage = pi_control_voltage(100, 10, Kp=1.0, Ki=1.0, voltage_min=0, voltage_max=24)

    assert voltage == pytest.approx(24)


def test_pi_control_voltage_saturates_at_minimum():
    """PI voltage should saturate at voltage_min."""
    voltage = pi_control_voltage(-100, 0, Kp=1.0, Ki=1.0, voltage_min=0, voltage_max=24)

    assert voltage == pytest.approx(0)


def test_anti_windup_stops_integrating_farther_into_upper_saturation():
    """Integral error should stop growing when positive error causes saturation."""
    derivative = anti_windup_integral_derivative(
        error=10,
        raw_voltage=30,
        voltage_min=0,
        voltage_max=24,
    )

    assert derivative == pytest.approx(0)


def test_anti_windup_integrates_when_error_moves_out_of_upper_saturation():
    """Integral error should be allowed to unwind from upper saturation."""
    derivative = anti_windup_integral_derivative(
        error=-10,
        raw_voltage=30,
        voltage_min=0,
        voltage_max=24,
    )

    assert derivative == pytest.approx(-10)


def test_simulation_initial_current_equals_i0():
    """The simulated current should start at i0."""
    t, current, _, _, _ = simulate_pi_motor_control(
        1,
        0.5,
        0.01,
        0.001,
        0.01,
        0.01,
        0.8,
        2.0,
        2,
        0,
        0,
        (0, 5),
        200,
    )

    assert t[0] == pytest.approx(0)
    assert current[0] == pytest.approx(2)


def test_simulation_initial_speed_equals_omega0():
    """The simulated speed should start at omega0."""
    _, _, omega, _, _ = simulate_pi_motor_control(
        1,
        0.5,
        0.01,
        0.001,
        0.01,
        0.01,
        0.8,
        2.0,
        0,
        3,
        0,
        (0, 5),
        200,
    )

    assert omega[0] == pytest.approx(3)


def test_simulation_initial_integral_error_equals_input():
    """The simulated integral error should start at integral_error0."""
    _, _, _, integral_error, _ = simulate_pi_motor_control(
        1,
        0.5,
        0.01,
        0.001,
        0.01,
        0.01,
        0.8,
        2.0,
        0,
        0,
        4,
        (0, 5),
        200,
    )

    assert integral_error[0] == pytest.approx(4)


def test_speed_approaches_reference():
    """Closed-loop motor speed should approach the 100 rad/s reference."""
    _, _, omega, _, _ = simulate_pi_motor_control(
        1,
        0.5,
        0.01,
        0.001,
        0.01,
        0.01,
        0.5,
        0.05,
        0,
        0,
        0,
        (0, 30),
        3000,
    )

    assert omega[-1] == pytest.approx(100, rel=0.05)


def test_final_speed_error_is_much_smaller_than_initial_error():
    """The controller should greatly reduce speed error."""
    target_speed = 100
    _, _, omega, _, _ = simulate_pi_motor_control(
        1,
        0.5,
        0.01,
        0.001,
        0.01,
        0.01,
        0.5,
        0.05,
        0,
        0,
        0,
        (0, 30),
        3000,
    )

    initial_error = target_speed - omega[0]
    final_error = target_speed - omega[-1]

    assert abs(final_error) < 0.1 * abs(initial_error)


def test_voltage_values_stay_within_limits():
    """All returned controller voltage samples should respect saturation limits."""
    voltage_min = 0
    voltage_max = 24

    _, _, _, _, voltage = simulate_pi_motor_control(
        1,
        0.5,
        0.01,
        0.001,
        0.01,
        0.01,
        0.8,
        2.0,
        0,
        0,
        0,
        (0, 5),
        1000,
        voltage_min=voltage_min,
        voltage_max=voltage_max,
    )

    assert voltage.min() >= voltage_min
    assert voltage.max() <= voltage_max


def test_invalid_negative_kp_raises_value_error():
    """Kp must be nonnegative."""
    with pytest.raises(ValueError):
        simulate_pi_motor_control(1, 0.5, 0.01, 0.001, 0.01, 0.01, -1, 2, 0, 0, 0, (0, 5), 200)


def test_invalid_negative_ki_raises_value_error():
    """Ki must be nonnegative."""
    with pytest.raises(ValueError):
        simulate_pi_motor_control(1, 0.5, 0.01, 0.001, 0.01, 0.01, 1, -2, 0, 0, 0, (0, 5), 200)


def test_nonzero_kd_raises_error():
    """Derivative action is reserved for a future PID implementation."""
    with pytest.raises((ValueError, NotImplementedError)):
        simulate_pi_motor_control(
            1,
            0.5,
            0.01,
            0.001,
            0.01,
            0.01,
            1,
            2,
            0,
            0,
            0,
            (0, 5),
            200,
            Kd=0.1,
        )


def test_invalid_voltage_limits_raise_value_error():
    """Maximum voltage must be greater than minimum voltage."""
    with pytest.raises(ValueError):
        simulate_pi_motor_control(
            1,
            0.5,
            0.01,
            0.001,
            0.01,
            0.01,
            1,
            2,
            0,
            0,
            0,
            (0, 5),
            200,
            voltage_min=24,
            voltage_max=24,
        )
