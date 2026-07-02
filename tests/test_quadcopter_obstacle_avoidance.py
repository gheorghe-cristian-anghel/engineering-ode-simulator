import numpy as np
import pytest

from analysis.quadcopter_obstacle_avoidance import (
    SphericalObstacle,
    distance_to_obstacle,
    nearest_obstacle_clearance,
    repulsive_avoidance_acceleration,
    simulate_quadcopter_obstacle_avoidance,
    summarize_obstacle_avoidance,
)
from analysis.quadcopter_trajectory_tracking import hover_trajectory


def _basic_obstacle():
    """Return a simple obstacle for geometry tests."""
    return SphericalObstacle(
        center=np.array([0.0, 0.0, 0.0]),
        radius=1.0,
        influence_radius=2.0,
    )


def test_spherical_obstacle_validates_and_stores_geometry():
    """SphericalObstacle should store center, radius, and influence radius."""
    obstacle = SphericalObstacle(center=[1.0, 2.0, 3.0], radius=0.5, influence_radius=1.5)

    assert obstacle.center.shape == (3,)
    assert obstacle.radius == pytest.approx(0.5)
    assert obstacle.influence_radius == pytest.approx(1.5)


def test_invalid_obstacle_center_raises_value_error():
    """Obstacle center must have three coordinates."""
    with pytest.raises(ValueError):
        SphericalObstacle(center=[1.0, 2.0], radius=0.5, influence_radius=1.5)


def test_distance_to_obstacle_positive_outside_obstacle():
    """Signed clearance should be positive outside the sphere."""
    obstacle = _basic_obstacle()

    clearance = distance_to_obstacle([2.5, 0.0, 0.0], obstacle)

    assert clearance == pytest.approx(1.5)


def test_distance_to_obstacle_zero_at_surface():
    """Signed clearance should be zero on the obstacle surface."""
    obstacle = _basic_obstacle()

    clearance = distance_to_obstacle([1.0, 0.0, 0.0], obstacle)

    assert clearance == pytest.approx(0.0)


def test_distance_to_obstacle_negative_inside_obstacle():
    """Signed clearance should be negative inside the sphere."""
    obstacle = _basic_obstacle()

    clearance = distance_to_obstacle([0.5, 0.0, 0.0], obstacle)

    assert clearance == pytest.approx(-0.5)


def test_nearest_obstacle_clearance_returns_minimum_clearance():
    """Nearest clearance should be the minimum signed clearance."""
    obstacles = [
        SphericalObstacle(center=[0.0, 0.0, 0.0], radius=0.5, influence_radius=1.5),
        SphericalObstacle(center=[2.0, 0.0, 0.0], radius=0.5, influence_radius=1.5),
    ]

    clearance = nearest_obstacle_clearance([1.7, 0.0, 0.0], obstacles)

    assert clearance == pytest.approx(-0.2)


def test_repulsive_avoidance_is_zero_outside_influence_radius():
    """Repulsion should be zero when outside every influence region."""
    obstacle = _basic_obstacle()

    acceleration = repulsive_avoidance_acceleration([3.0, 0.0, 0.0], [obstacle])

    assert np.linalg.norm(acceleration) == pytest.approx(0.0)


def test_repulsive_avoidance_points_away_from_obstacle():
    """Repulsion should push outward from the obstacle center."""
    obstacle = _basic_obstacle()

    acceleration = repulsive_avoidance_acceleration(
        [1.5, 0.0, 0.0],
        [obstacle],
        gain=0.1,
    )

    assert acceleration[0] > 0.0
    assert abs(acceleration[1]) < 1e-12
    assert abs(acceleration[2]) < 1e-12


def test_repulsive_avoidance_magnitude_is_clipped():
    """Repulsive acceleration should respect the configured maximum."""
    obstacle = _basic_obstacle()

    acceleration = repulsive_avoidance_acceleration(
        [1.01, 0.0, 0.0],
        [obstacle],
        gain=10.0,
        max_acceleration=0.25,
    )

    assert np.linalg.norm(acceleration) <= 0.25 + 1e-12


def test_invalid_obstacle_radius_raises_value_error():
    """Obstacle radius must be positive."""
    with pytest.raises(ValueError):
        SphericalObstacle(center=[0.0, 0.0, 0.0], radius=0.0, influence_radius=1.0)


def test_invalid_influence_radius_raises_value_error():
    """Influence radius must be greater than physical radius."""
    with pytest.raises(ValueError):
        SphericalObstacle(center=[0.0, 0.0, 0.0], radius=1.0, influence_radius=1.0)


def test_simulation_returns_expected_shapes_for_short_case():
    """Obstacle-aware simulation should return aligned time-series arrays."""
    obstacle = SphericalObstacle(center=[5.0, 0.0, 1.0], radius=0.2, influence_radius=1.0)
    initial_state = np.zeros(12)

    result = simulate_quadcopter_obstacle_avoidance(
        hover_trajectory((0.0, 0.0, 1.0)),
        [obstacle],
        initial_state=initial_state,
        t_span=(0.0, 0.1),
        dt=0.05,
    )
    sample_count = len(result["time"])

    assert result["states"].shape == (sample_count, 12)
    assert result["controls"].shape == (sample_count, 4)
    assert result["reference_positions"].shape == (sample_count, 3)
    assert result["reference_velocities"].shape == (sample_count, 3)
    assert result["position_errors"].shape == (sample_count, 3)
    assert result["avoidance_accelerations"].shape == (sample_count, 3)
    assert result["nearest_clearances"].shape == (sample_count,)


def test_nearest_clearances_are_stored():
    """Simulation result should include finite nearest-obstacle clearances."""
    obstacle = SphericalObstacle(center=[5.0, 0.0, 1.0], radius=0.2, influence_radius=1.0)

    result = simulate_quadcopter_obstacle_avoidance(
        hover_trajectory((0.0, 0.0, 1.0)),
        [obstacle],
        t_span=(0.0, 0.1),
        dt=0.05,
    )

    assert np.all(np.isfinite(result["nearest_clearances"]))


def test_summarize_obstacle_avoidance_returns_expected_keys():
    """Obstacle avoidance metrics should include expected summary values."""
    obstacle = SphericalObstacle(center=[5.0, 0.0, 1.0], radius=0.2, influence_radius=1.0)
    result = simulate_quadcopter_obstacle_avoidance(
        hover_trajectory((0.0, 0.0, 1.0)),
        [obstacle],
        t_span=(0.0, 0.1),
        dt=0.05,
    )
    metrics = summarize_obstacle_avoidance(result)

    assert set(metrics) == {
        "min_clearance",
        "final_position_error_norm",
        "rms_position_error",
        "max_position_error",
        "max_avoidance_acceleration",
        "max_thrust",
        "max_abs_torque",
    }
