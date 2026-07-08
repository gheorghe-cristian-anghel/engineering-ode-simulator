import numpy as np
import pytest
from numpy.testing import assert_allclose

from analysis.quadcopter_trajectory_tracking import (
    circular_trajectory,
    compute_trajectory_tracking_control,
    hover_trajectory,
    simulate_quadcopter_trajectory_tracking,
    summarize_trajectory_tracking,
)


def test_hover_trajectory_returns_expected_constant_position():
    """Hover trajectory should hold a fixed position."""
    trajectory = hover_trajectory(position=(1.0, -2.0, 3.0))

    reference_0 = trajectory(0.0)
    reference_5 = trajectory(5.0)

    assert_allclose(reference_0.position, [1.0, -2.0, 3.0])
    assert_allclose(reference_5.position, [1.0, -2.0, 3.0])
    assert_allclose(reference_0.velocity, np.zeros(3))
    assert_allclose(reference_0.acceleration, np.zeros(3))


def test_circular_trajectory_returns_expected_position_at_zero():
    """Circular trajectory should start on the positive x side of the circle."""
    trajectory = circular_trajectory(
        radius=2.0,
        altitude=3.0,
        angular_speed=0.5,
        center=(1.0, -1.0),
    )

    reference = trajectory(0.0)

    assert_allclose(reference.position, [3.0, -1.0, 3.0])


def test_circular_trajectory_velocity_has_expected_shape():
    """Circular trajectory velocity should be a three-element vector."""
    trajectory = circular_trajectory(radius=1.0, altitude=2.0, angular_speed=0.3)
    reference = trajectory(1.0)

    assert reference.velocity.shape == (3,)
    assert reference.acceleration.shape == (3,)


def test_compute_control_returns_expected_shapes():
    """Controller should return control, attitude command, and acceleration."""
    state = np.zeros(12)
    reference = hover_trajectory((0.0, 0.0, 2.0))(0.0)

    control, attitude_command, acceleration_command = (
        compute_trajectory_tracking_control(state, reference)
    )

    assert control.shape == (4,)
    assert attitude_command.shape == (3,)
    assert acceleration_command.shape == (3,)


def test_simulation_returns_expected_shapes_for_short_hover_case():
    """Trajectory simulation should return aligned time-series arrays."""
    initial_state = np.zeros(12)
    result = simulate_quadcopter_trajectory_tracking(
        hover_trajectory((0.0, 0.0, 1.0)),
        initial_state=initial_state,
        t_span=(0.0, 0.2),
        dt=0.05,
    )
    sample_count = len(result["time"])

    assert result["states"].shape == (sample_count, 12)
    assert result["controls"].shape == (sample_count, 4)
    assert result["reference_positions"].shape == (sample_count, 3)
    assert result["reference_velocities"].shape == (sample_count, 3)
    assert result["position_errors"].shape == (sample_count, 3)
    assert result["attitude_commands"].shape == (sample_count, 3)
    assert result["tracking_error_norm"].shape == (sample_count,)


def test_thrust_remains_within_limits():
    """Controller thrust commands should respect saturation limits."""
    thrust_limits = (0.0, 15.0)
    result = simulate_quadcopter_trajectory_tracking(
        hover_trajectory((0.0, 0.0, 2.0)),
        t_span=(0.0, 0.5),
        dt=0.05,
        thrust_limits=thrust_limits,
    )

    assert np.min(result["controls"][:, 0]) >= thrust_limits[0]
    assert np.max(result["controls"][:, 0]) <= thrust_limits[1]


def test_torques_remain_within_limits():
    """Controller torque commands should respect saturation limits."""
    torque_limits = (-0.05, 0.05)
    initial_state = np.zeros(12)
    initial_state[0:3] = [0.5, -0.5, 0.0]

    result = simulate_quadcopter_trajectory_tracking(
        hover_trajectory((0.0, 0.0, 2.0)),
        initial_state=initial_state,
        t_span=(0.0, 0.5),
        dt=0.05,
        torque_limits=torque_limits,
    )

    assert np.min(result["controls"][:, 1:4]) >= torque_limits[0]
    assert np.max(result["controls"][:, 1:4]) <= torque_limits[1]


def test_hover_tracking_final_error_is_smaller_than_initial_error():
    """Hover tracking should reduce position error for the default gains."""
    initial_state = np.zeros(12)
    initial_state[0:3] = [0.5, -0.5, 0.0]

    result = simulate_quadcopter_trajectory_tracking(
        hover_trajectory((0.0, 0.0, 2.0)),
        initial_state=initial_state,
        t_span=(0.0, 5.0),
        dt=0.02,
    )

    assert result["tracking_error_norm"][-1] < result["tracking_error_norm"][0]


def test_default_hover_tracking_errors_and_command_metrics_are_plausible():
    """Default hover tracking should remain finite and settle near target."""
    result = simulate_quadcopter_trajectory_tracking(
        hover_trajectory((0.0, 0.0, 2.0)),
        t_span=(0.0, 3.0),
        dt=0.05,
    )
    metrics = summarize_trajectory_tracking(result)

    assert np.all(np.isfinite(result["states"]))
    assert np.all(np.isfinite(result["controls"]))
    assert metrics["final_position_error_norm"] < 0.1
    assert metrics["rms_position_error"] < 1.0
    assert np.isfinite(metrics["max_thrust"])
    assert np.isfinite(metrics["max_abs_torque"])


def test_invalid_dt_raises_value_error():
    """Sample time must be positive."""
    with pytest.raises(ValueError):
        simulate_quadcopter_trajectory_tracking(
            hover_trajectory(),
            dt=0.0,
        )


def test_invalid_thrust_limits_raise_value_error():
    """Maximum thrust must be greater than minimum thrust."""
    with pytest.raises(ValueError):
        simulate_quadcopter_trajectory_tracking(
            hover_trajectory(),
            thrust_limits=(10.0, 10.0),
        )


def test_invalid_torque_limits_raise_value_error():
    """Maximum torque must be greater than minimum torque."""
    with pytest.raises(ValueError):
        simulate_quadcopter_trajectory_tracking(
            hover_trajectory(),
            torque_limits=(0.2, 0.2),
        )


def test_invalid_circular_trajectory_parameters_raise_value_error():
    """Circular trajectory parameters should be validated."""
    with pytest.raises(ValueError):
        circular_trajectory(radius=0.0)

    with pytest.raises(ValueError):
        circular_trajectory(angular_speed=0.0)


def test_metrics_helper_returns_expected_keys():
    """Trajectory metrics helper should return the expected summary keys."""
    result = simulate_quadcopter_trajectory_tracking(
        hover_trajectory((0.0, 0.0, 1.0)),
        t_span=(0.0, 0.2),
        dt=0.05,
    )
    metrics = summarize_trajectory_tracking(result)

    assert set(metrics) == {
        "final_position_error_norm",
        "rms_position_error",
        "max_position_error",
        "max_thrust",
        "max_abs_torque",
        "final_position",
        "final_reference_position",
    }
