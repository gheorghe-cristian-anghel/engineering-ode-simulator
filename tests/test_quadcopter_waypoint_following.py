import numpy as np
import pytest
from numpy.testing import assert_allclose

from analysis.quadcopter_waypoint_following import (
    simulate_quadcopter_waypoint_following,
    summarize_waypoint_following,
    total_waypoint_time,
    waypoint_trajectory,
)


def test_waypoint_trajectory_returns_first_waypoint_at_zero():
    """Waypoint trajectory should start at the first waypoint."""
    waypoints = [(0.0, 0.0, 1.0), (1.0, 0.0, 2.0)]
    trajectory = waypoint_trajectory(waypoints, segment_time=2.0)

    reference = trajectory(0.0)

    assert_allclose(reference.position, [0.0, 0.0, 1.0])


def test_waypoint_trajectory_returns_final_waypoint_after_total_time():
    """Waypoint trajectory should hold the final waypoint after the last segment."""
    waypoints = [(0.0, 0.0, 1.0), (1.0, 0.0, 2.0), (1.0, 1.0, 2.0)]
    trajectory = waypoint_trajectory(waypoints, segment_time=2.0)

    reference = trajectory(10.0)

    assert_allclose(reference.position, [1.0, 1.0, 2.0])
    assert_allclose(reference.velocity, np.zeros(3))
    assert_allclose(reference.acceleration, np.zeros(3))


def test_linear_interpolation_gives_midpoint_at_half_segment_time():
    """Linear waypoint smoothing should interpolate halfway at half time."""
    waypoints = [(0.0, 0.0, 1.0), (2.0, 0.0, 3.0)]
    trajectory = waypoint_trajectory(
        waypoints,
        segment_time=4.0,
        smoothing="linear",
    )

    reference = trajectory(2.0)

    assert_allclose(reference.position, [1.0, 0.0, 2.0])
    assert_allclose(reference.velocity, [0.5, 0.0, 0.5])
    assert_allclose(reference.acceleration, np.zeros(3))


def test_smoothstep_interpolation_gives_correct_endpoints():
    """Smoothstep waypoint smoothing should pass through segment endpoints."""
    waypoints = [(0.0, 0.0, 1.0), (2.0, 0.0, 3.0)]
    trajectory = waypoint_trajectory(
        waypoints,
        segment_time=4.0,
        smoothing="smoothstep",
    )

    assert_allclose(trajectory(0.0).position, [0.0, 0.0, 1.0])
    assert_allclose(trajectory(4.0).position, [2.0, 0.0, 3.0])


def test_trajectory_velocity_and_acceleration_have_shape_three():
    """Waypoint references should include three-axis velocity and acceleration."""
    trajectory = waypoint_trajectory(
        [(0.0, 0.0, 1.0), (1.0, 1.0, 2.0)],
        segment_time=2.0,
    )

    reference = trajectory(1.0)

    assert reference.velocity.shape == (3,)
    assert reference.acceleration.shape == (3,)


def test_invalid_waypoint_count_raises_value_error():
    """At least two 3D waypoints are required."""
    with pytest.raises(ValueError):
        waypoint_trajectory([(0.0, 0.0, 1.0)])


def test_invalid_waypoint_dimension_raises_value_error():
    """Each waypoint must have three coordinates."""
    with pytest.raises(ValueError):
        waypoint_trajectory([(0.0, 0.0), (1.0, 1.0)])


def test_invalid_segment_time_raises_value_error():
    """Segment time must be positive."""
    with pytest.raises(ValueError):
        waypoint_trajectory(
            [(0.0, 0.0, 1.0), (1.0, 0.0, 2.0)],
            segment_time=0.0,
        )


def test_invalid_smoothing_raises_value_error():
    """Only supported smoothing modes should be accepted."""
    with pytest.raises(ValueError):
        waypoint_trajectory(
            [(0.0, 0.0, 1.0), (1.0, 0.0, 2.0)],
            smoothing="cubic",
        )


def test_total_waypoint_time_returns_expected_value():
    """Total time should be one segment time per adjacent waypoint pair."""
    waypoints = [
        (0.0, 0.0, 1.0),
        (1.0, 0.0, 1.5),
        (1.0, 1.0, 2.0),
    ]

    assert total_waypoint_time(waypoints, segment_time=3.0) == pytest.approx(6.0)


def test_simulate_quadcopter_waypoint_following_returns_expected_shapes():
    """Waypoint-following simulation should return aligned time-series arrays."""
    result = simulate_quadcopter_waypoint_following(
        [(0.0, 0.0, 1.0), (0.2, 0.0, 1.1)],
        segment_time=0.2,
        dt=0.1,
        hold_time=0.1,
    )
    sample_count = len(result["time"])

    assert result["states"].shape == (sample_count, 12)
    assert result["controls"].shape == (sample_count, 4)
    assert result["reference_positions"].shape == (sample_count, 3)
    assert result["position_errors"].shape == (sample_count, 3)
    assert result["waypoints"].shape == (2, 3)


def test_final_waypoint_error_is_smaller_than_initial_error_for_basic_case():
    """A simple hover-like waypoint case should reduce error from an offset start."""
    initial_state = np.zeros(12)
    result = simulate_quadcopter_waypoint_following(
        [(0.0, 0.0, 1.0), (0.0, 0.0, 1.0)],
        segment_time=0.5,
        initial_state=initial_state,
        dt=0.05,
        hold_time=1.5,
    )
    initial_error = np.linalg.norm(initial_state[0:3] - result["waypoints"][-1])

    assert result["waypoint_metrics"]["final_waypoint_error"] < initial_error


def test_metrics_helper_returns_expected_keys():
    """Waypoint metrics should include practical tracking and command summaries."""
    result = simulate_quadcopter_waypoint_following(
        [(0.0, 0.0, 1.0), (0.1, 0.0, 1.0)],
        segment_time=0.2,
        dt=0.1,
        hold_time=0.1,
    )
    metrics = summarize_waypoint_following(result, result["waypoints"])

    assert set(metrics) == {
        "final_waypoint_error",
        "rms_position_error",
        "max_position_error",
        "max_thrust",
        "max_abs_torque",
        "total_time",
        "number_of_waypoints",
        "nearest_distance_to_each_waypoint",
        "closest_approach_time_to_each_waypoint",
        "final_position",
        "final_waypoint",
    }
