"""Version-ready API routes that delegate numerical work to services."""

from fastapi import APIRouter, HTTPException

from backend.schemas.simulations import (
    HeatRequest,
    HeatResponse,
    KalmanRequest,
    KalmanResponse,
    UavRequest,
    UavResponse,
)
from backend.services.simulations import (
    run_heat_simulation,
    run_kalman_simulation,
    run_uav_simulation,
)

router = APIRouter(prefix="/api", tags=["simulations"])


@router.get("/health", summary="Check API health")
def health():
    """Return stable project metadata without running a simulation."""
    return {
        "project_name": "Engineering Simulation Toolkit",
        "status": "ok",
        "api_version": "v1",
    }


@router.post(
    "/pde/heat-2d",
    response_model=HeatResponse,
    summary="Run a bounded 2D heat-equation simulation",
)
def heat_2d(request: HeatRequest) -> HeatResponse:
    """Solve the existing explicit 2D diffusion model and return JSON data."""
    return _simulation_response(run_heat_simulation, request)


@router.post(
    "/uav/obstacle-avoidance",
    response_model=UavResponse,
    summary="Run UAV waypoint obstacle avoidance",
)
def obstacle_avoidance(request: UavRequest) -> UavResponse:
    """Run the existing 6-DOF UAV obstacle-avoidance simulation."""
    return _simulation_response(run_uav_simulation, request)


@router.post(
    "/estimation/kalman-filter",
    response_model=KalmanResponse,
    summary="Run a DC motor linear Kalman filter",
)
def kalman_filter(request: KalmanRequest) -> KalmanResponse:
    """Estimate DC motor current and speed with the existing Kalman filter."""
    return _simulation_response(run_kalman_simulation, request)


def _simulation_response(service, request):
    """Translate known simulation input/runtime errors to useful HTTP responses."""
    try:
        return service(request)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(
            status_code=500, detail=f"simulation failed: {error}"
        ) from error
