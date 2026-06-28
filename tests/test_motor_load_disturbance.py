import numpy as np
import pytest

from models.pid_motor_control import (
    load_torque_step,
    simulate_pi_motor_control,
    speed_reference_step,
)


def _simulate_disturbance_response():
    """Run an example-like motor load disturbance simulation."""
    R = 1.0
    L = 0.5
    J = 0.01
    b = 0.001
    Kt = 0.01
    Ke = 0.01
    target_speed = 80.0
    Kp = 0.3
    Ki = 0.04
    Kd = 0.0
    voltage_min = 0.0
    voltage_max = 24.0
    i0 = 0.0
    omega0 = 0.0
    integral_error0 = 0.0
    t_span = (0.0, 25.0)
    num_points = 3000
    disturbance_time = 12.0
    load_before = 0.0
    load_after = 0.03

    reference_func = lambda t: speed_reference_step(t, target_speed)
    load_torque_func = lambda t: load_torque_step(
        t,
        disturbance_time,
        load_before,
        load_after,
    )

    t, current, omega, integral_error, voltage = simulate_pi_motor_control(
        R,
        L,
        J,
        b,
        Kt,
        Ke,
        Kp,
        Ki,
        i0,
        omega0,
        integral_error0,
        t_span,
        num_points,
        reference_func=reference_func,
        load_torque_func=load_torque_func,
        voltage_min=voltage_min,
        voltage_max=voltage_max,
        Kd=Kd,
    )

    return {
        "t": t,
        "current": current,
        "omega": omega,
        "integral_error": integral_error,
        "voltage": voltage,
        "target_speed": target_speed,
        "voltage_min": voltage_min,
        "voltage_max": voltage_max,
        "disturbance_time": disturbance_time,
    }


def _disturbance_indices(t, disturbance_time):
    """Return indices near and before the disturbance."""
    disturbance_index = int(np.argmin(np.abs(t - disturbance_time)))
    before_index = max(0, disturbance_index - 1)

    return disturbance_index, before_index


def test_load_torque_step_returns_initial_torque_before_step_time():
    """The load torque should equal initial_torque before the step."""
    torque = load_torque_step(5.0, step_time=12.0, initial_torque=0.0, final_torque=0.03)

    assert torque == pytest.approx(0.0)


def test_load_torque_step_returns_final_torque_at_step_time():
    """The load torque should switch at step_time."""
    torque = load_torque_step(12.0, step_time=12.0, initial_torque=0.0, final_torque=0.03)

    assert torque == pytest.approx(0.03)


def test_load_torque_step_returns_final_torque_after_step_time():
    """The load torque should equal final_torque after the step."""
    torque = load_torque_step(13.0, step_time=12.0, initial_torque=0.0, final_torque=0.03)

    assert torque == pytest.approx(0.03)


def test_disturbance_simulation_returns_arrays_of_equal_length():
    """The closed-loop disturbance simulation should return aligned arrays."""
    response = _simulate_disturbance_response()
    expected_length = len(response["t"])

    assert len(response["current"]) == expected_length
    assert len(response["omega"]) == expected_length
    assert len(response["integral_error"]) == expected_length
    assert len(response["voltage"]) == expected_length


def test_speed_drops_after_load_disturbance():
    """A load torque step should cause a temporary speed drop."""
    response = _simulate_disturbance_response()
    disturbance_index, before_index = _disturbance_indices(
        response["t"],
        response["disturbance_time"],
    )

    speed_before = response["omega"][before_index]
    minimum_speed_after = np.min(response["omega"][disturbance_index:])

    assert minimum_speed_after < speed_before


def test_final_speed_recovers_near_target():
    """The PI controller should recover speed close to the target."""
    response = _simulate_disturbance_response()
    final_speed = response["omega"][-1]

    assert abs(response["target_speed"] - final_speed) < 5.0


def test_voltage_remains_within_limits():
    """Controller voltage should respect saturation limits."""
    response = _simulate_disturbance_response()

    assert response["voltage"].min() >= response["voltage_min"]
    assert response["voltage"].max() <= response["voltage_max"]


def test_current_increases_after_disturbance_recovery():
    """Final current should be higher because the motor carries load torque."""
    response = _simulate_disturbance_response()
    _, before_index = _disturbance_indices(
        response["t"],
        response["disturbance_time"],
    )

    current_before = response["current"][before_index]
    current_near_final_recovery = response["current"][-1]

    assert current_near_final_recovery > current_before
