"""Integration tests for the FastAPI backend foundation."""

import math

from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def _all_finite(value):
    """Return whether nested numeric response data contains only finite values."""
    if isinstance(value, dict):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_all_finite(item) for item in value)
    return math.isfinite(value)


def test_health_endpoint_reports_project_metadata():
    """The API health endpoint should identify the service and its version."""
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {
        "project_name": "Engineering Simulation Toolkit",
        "status": "ok",
        "api_version": "v1",
    }


def test_heat_endpoint_returns_small_finite_deterministic_response():
    """A small heat request should return coordinates and a finite final field."""
    request = {
        "alpha": 0.01,
        "width": 1.0,
        "height": 1.0,
        "nx": 5,
        "ny": 5,
        "dt": 0.01,
        "t_final": 0.02,
        "boundary_value": 0.0,
        "initial_condition": {"kind": "sine", "amplitude": 1.0},
        "include_snapshots": True,
        "max_snapshots": 3,
    }

    response = client.post("/api/pde/heat-2d", json=request)

    assert response.status_code == 200
    data = response.json()
    assert data["coordinates"]["x"] == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert len(data["final_field"]) == 5
    assert len(data["final_field"][0]) == 5
    assert data["final_field"][0] == [0.0] * 5
    assert len(data["snapshots"]) == 3
    assert data["stability"]["is_stable"] is True
    assert _all_finite(data["final_field"])


def test_heat_endpoint_rejects_invalid_and_oversized_requests():
    """Schema limits must reject invalid physics and oversized payloads."""
    invalid_response = client.post(
        "/api/pde/heat-2d",
        json={"alpha": 0.0},
    )
    oversized_response = client.post(
        "/api/pde/heat-2d",
        json={"nx": 257, "ny": 257},
    )

    assert invalid_response.status_code == 422
    assert oversized_response.status_code == 422
    assert "alpha" in invalid_response.text
    assert "nx" in oversized_response.text


def test_heat_endpoint_rejects_unknown_initial_condition_options():
    """Nested heat configuration must be just as strict as the outer request."""
    response = client.post(
        "/api/pde/heat-2d",
        json={"initial_condition": {"kind": "sine", "unexpected": 1}},
    )

    assert response.status_code == 422
    assert "unexpected" in response.text


def test_heat_endpoint_enforces_work_budget_and_downsamples_snapshots():
    """Heat requests must bound compute work and snapshot payload dimensions."""
    budget_response = client.post(
        "/api/pde/heat-2d",
        json={"nx": 256, "ny": 256, "t_final": 1.0},
    )
    snapshot_response = client.post(
        "/api/pde/heat-2d",
        json={
            "nx": 65,
            "ny": 65,
            "t_final": 0.01,
            "include_snapshots": True,
            "max_snapshots": 2,
        },
    )

    assert budget_response.status_code == 400
    assert "cell-step limit" in budget_response.text
    assert snapshot_response.status_code == 200
    snapshot = snapshot_response.json()["snapshots"][0]
    assert len(snapshot["x"]) * len(snapshot["y"]) <= 4_096
    assert snapshot["downsample_stride"] > 1


def test_uav_endpoint_returns_finite_paths_and_metrics():
    """A bounded UAV simulation should expose aligned paths and finite metrics."""
    response = client.post(
        "/api/uav/obstacle-avoidance",
        json={
            "waypoints": [[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]],
            "segment_time": 0.1,
            "t_final": 0.1,
            "dt": 0.05,
            "obstacles": [
                {
                    "center": [5.0, 0.0, 1.0],
                    "radius": 0.2,
                    "influence_radius": 1.0,
                }
            ],
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert len(data["time"]) == len(data["actual_path"])
    assert len(data["reference_path"]) == len(data["actual_path"])
    assert len(data["waypoints"]) == 2
    assert data["metrics"]["minimum_clearance"] > 0.0
    assert _all_finite(data["actual_path"])
    assert _all_finite(data["metrics"])


def test_kalman_endpoint_returns_finite_deterministic_metrics():
    """A seeded DC motor scenario should be repeatable and numerically finite."""
    request = {"t_final": 0.04, "dt": 0.01, "random_seed": 7}

    first = client.post("/api/estimation/kalman-filter", json=request)
    second = client.post("/api/estimation/kalman-filter", json=request)

    assert first.status_code == 200
    assert first.json() == second.json()
    assert _all_finite(first.json()["estimates"])
    assert _all_finite(first.json()["metrics"])
