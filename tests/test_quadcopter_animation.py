import numpy as np
import pytest
from numpy.testing import assert_allclose

from visualization.quadcopter_animation import (
    animate_quadcopter_6dof,
    compute_quadcopter_body_points,
)


def test_compute_quadcopter_body_points_returns_expected_keys():
    """Geometry helper should expose hub, rotor points, and arm endpoints."""
    points = compute_quadcopter_body_points(
        position=[1.0, 2.0, 3.0],
        phi=0.0,
        theta=0.0,
        psi=0.0,
    )

    assert set(points) == {
        "center",
        "front",
        "back",
        "right",
        "left",
        "arm_1",
        "arm_2",
    }


def test_zero_attitude_places_rotors_at_body_offsets():
    """Zero attitude should leave body-frame rotor offsets unchanged."""
    position = np.array([1.0, 2.0, 3.0])
    arm_length = 0.5

    points = compute_quadcopter_body_points(
        position=position,
        phi=0.0,
        theta=0.0,
        psi=0.0,
        arm_length=arm_length,
    )

    assert_allclose(points["center"], position)
    assert_allclose(points["front"], position + [arm_length, 0.0, 0.0])
    assert_allclose(points["back"], position + [-arm_length, 0.0, 0.0])
    assert_allclose(points["right"], position + [0.0, arm_length, 0.0])
    assert_allclose(points["left"], position + [0.0, -arm_length, 0.0])


def test_arm_length_sets_rotor_distance_from_center():
    """Each rotor should be one arm length away from the body center."""
    arm_length = 0.75
    points = compute_quadcopter_body_points(
        position=[0.0, 0.0, 0.0],
        phi=0.2,
        theta=-0.1,
        psi=0.3,
        arm_length=arm_length,
    )

    for key in ("front", "back", "right", "left"):
        distance = np.linalg.norm(points[key] - points["center"])
        assert distance == pytest.approx(arm_length)


def test_invalid_position_length_raises_value_error():
    """Position must contain exactly three values."""
    with pytest.raises(ValueError):
        compute_quadcopter_body_points(
            position=[0.0, 0.0],
            phi=0.0,
            theta=0.0,
            psi=0.0,
        )


def test_invalid_arm_length_raises_value_error():
    """Arm length must be positive."""
    with pytest.raises(ValueError):
        compute_quadcopter_body_points(
            position=[0.0, 0.0, 0.0],
            phi=0.0,
            theta=0.0,
            psi=0.0,
            arm_length=0.0,
        )


def test_animate_quadcopter_validates_time_shape():
    """Animation helper should require one-dimensional time samples."""
    time = np.zeros((2, 2))
    states = np.zeros((4, 12))

    with pytest.raises(ValueError):
        animate_quadcopter_6dof(time, states, show=False)


def test_invalid_states_shape_raises_value_error():
    """Animation helper should require twelve 6-DOF state columns."""
    time = np.linspace(0.0, 1.0, 5)
    states = np.zeros((5, 11))

    with pytest.raises(ValueError):
        animate_quadcopter_6dof(time, states, show=False)


def test_invalid_reference_positions_shape_raises_value_error():
    """Reference positions should match the trajectory shape convention."""
    time = np.linspace(0.0, 1.0, 5)
    states = np.zeros((5, 12))
    reference_positions = np.zeros((5, 2))

    with pytest.raises(ValueError):
        animate_quadcopter_6dof(
            time,
            states,
            reference_positions=reference_positions,
            show=False,
        )


def test_invalid_reference_positions_length_raises_value_error():
    """Reference positions should have the same sample count as time."""
    time = np.linspace(0.0, 1.0, 5)
    states = np.zeros((5, 12))
    reference_positions = np.zeros((4, 3))

    with pytest.raises(ValueError):
        animate_quadcopter_6dof(
            time,
            states,
            reference_positions=reference_positions,
            show=False,
        )


def test_invalid_waypoints_shape_raises_value_error():
    """Waypoint markers should be provided as three-column positions."""
    time = np.linspace(0.0, 1.0, 5)
    states = np.zeros((5, 12))
    waypoints = np.zeros((3, 2))

    with pytest.raises(ValueError):
        animate_quadcopter_6dof(time, states, waypoints=waypoints, show=False)
