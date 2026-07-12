"""Validated request and response models for simulation endpoints."""

import math
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]
Vector3 = Annotated[list[FiniteFloat], Field(min_length=3, max_length=3)]
Vector2 = Annotated[list[FiniteFloat], Field(min_length=2, max_length=2)]


class HeatInitialCondition(BaseModel):
    """Supported analytic initial-temperature configurations."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["gaussian", "rectangle", "sine"] = "gaussian"
    amplitude: FiniteFloat = Field(default=1.0, ge=-1_000_000, le=1_000_000)
    center: Vector2 | None = None
    width: FiniteFloat | None = Field(default=None, gt=0)
    x_range: Annotated[list[FiniteFloat], Field(min_length=2, max_length=2)] | None = (
        None
    )
    y_range: Annotated[list[FiniteFloat], Field(min_length=2, max_length=2)] | None = (
        None
    )
    mode_x: int = Field(default=1, ge=1, le=10)
    mode_y: int = Field(default=1, ge=1, le=10)

    @model_validator(mode="after")
    def validate_kind_specific_values(self):
        """Reject malformed rectangular configuration ranges."""
        for name, value in (("x_range", self.x_range), ("y_range", self.y_range)):
            if value is not None and value[1] <= value[0]:
                raise ValueError(f"{name} must be strictly increasing")
        return self


class HeatRequest(BaseModel):
    """Bounded request for the explicit 2D heat solver."""

    model_config = ConfigDict(extra="forbid")

    alpha: FiniteFloat = Field(default=0.01, gt=0, le=1.0)
    width: FiniteFloat = Field(default=1.0, gt=0, le=1000.0)
    height: FiniteFloat = Field(default=1.0, gt=0, le=1000.0)
    nx: int = Field(default=51, ge=3, le=256)
    ny: int = Field(default=51, ge=3, le=256)
    dt: FiniteFloat | None = Field(default=None, gt=0)
    t_final: FiniteFloat = Field(default=1.0, gt=0, le=5.0)
    boundary_type: Literal["dirichlet", "neumann"] = "dirichlet"
    boundary_value: (
        FiniteFloat | Annotated[list[FiniteFloat], Field(min_length=4, max_length=4)]
    ) = 0.0
    initial_condition: HeatInitialCondition = Field(
        default_factory=HeatInitialCondition
    )
    include_snapshots: bool = False
    max_snapshots: int = Field(default=10, ge=1, le=25)

    @model_validator(mode="after")
    def validate_grid_size(self):
        """Keep the final temperature field below the JSON payload limit."""
        if self.nx * self.ny > 65_536:
            raise ValueError("nx * ny must not exceed 65536")
        return self


class ObstacleRequest(BaseModel):
    """A static spherical obstacle in meters."""

    model_config = ConfigDict(extra="forbid")

    center: Vector3
    radius: FiniteFloat = Field(gt=0, le=1000)
    influence_radius: FiniteFloat = Field(gt=0, le=2000)

    @model_validator(mode="after")
    def validate_influence_radius(self):
        if self.influence_radius <= self.radius:
            raise ValueError("influence_radius must be greater than radius")
        return self


class UavRequest(BaseModel):
    """Bounded waypoint-following obstacle avoidance scenario."""

    model_config = ConfigDict(extra="forbid")

    waypoints: Annotated[list[Vector3], Field(min_length=2, max_length=8)]
    obstacles: Annotated[list[ObstacleRequest], Field(min_length=1, max_length=6)]
    segment_time: FiniteFloat = Field(default=4.0, gt=0, le=10)
    t_final: FiniteFloat = Field(default=8.0, gt=0, le=10)
    dt: FiniteFloat = Field(default=0.02, ge=0.01, le=0.2)
    smoothing: Literal["linear", "smoothstep"] = "smoothstep"
    avoidance_gain: FiniteFloat = Field(default=0.03, ge=0, le=10)
    max_avoidance_acceleration: FiniteFloat = Field(default=1.5, ge=0, le=10)

    @model_validator(mode="after")
    def validate_sample_count(self):
        if math.ceil(self.t_final / self.dt) + 1 > 1_001:
            raise ValueError("t_final / dt must produce at most 1001 samples")
        return self


class KalmanRequest(BaseModel):
    """Bounded, deterministic-capable DC motor Kalman filter scenario."""

    model_config = ConfigDict(extra="forbid")

    voltage: FiniteFloat = Field(default=12.0, ge=-1000, le=1000)
    t_final: FiniteFloat = Field(default=5.0, gt=0, le=10)
    dt: FiniteFloat = Field(default=0.01, ge=0.001, le=0.1)
    measurement_noise_std: FiniteFloat = Field(default=2.0, ge=0, le=1000)
    random_seed: int = Field(default=0, ge=0, le=2_147_483_647)

    @model_validator(mode="after")
    def validate_sample_count(self):
        if math.ceil(self.t_final / self.dt) + 1 > 2_001:
            raise ValueError("t_final / dt must produce at most 2001 samples")
        return self


class HeatSnapshot(BaseModel):
    """A spatially downsampled heat field at one stored time."""

    time: FiniteFloat
    x: list[FiniteFloat]
    y: list[FiniteFloat]
    field: list[list[FiniteFloat]]
    downsample_stride: int


class HeatResponse(BaseModel):
    """JSON contract returned by the 2D heat-equation endpoint."""

    coordinates: dict[str, list[FiniteFloat]]
    final_field: list[list[FiniteFloat]]
    snapshots: list[HeatSnapshot]
    stability: dict[str, FiniteFloat | bool]
    thermal_metrics: dict[str, FiniteFloat]
    method: dict[str, str | FiniteFloat]
    units: dict[str, str]


class UavMetrics(BaseModel):
    """Summary metrics for a UAV obstacle-avoidance response."""

    final_error: FiniteFloat
    rms_error: FiniteFloat
    minimum_clearance: FiniteFloat
    max_thrust: FiniteFloat
    max_abs_torque: FiniteFloat


class UavResponse(BaseModel):
    """JSON contract returned by the UAV obstacle-avoidance endpoint."""

    time: list[FiniteFloat]
    reference_path: list[Vector3]
    actual_path: list[Vector3]
    waypoints: list[Vector3]
    obstacles: list[ObstacleRequest]
    metrics: UavMetrics
    units: dict[str, str]


class KalmanMetrics(BaseModel):
    """Summary errors for the DC motor Kalman-filter response."""

    rms_current_error: FiniteFloat
    rms_speed_error: FiniteFloat
    final_current_error: FiniteFloat
    final_speed_error: FiniteFloat


class KalmanResponse(BaseModel):
    """JSON contract returned by the linear Kalman-filter endpoint."""

    time: list[FiniteFloat]
    true_state: list[list[FiniteFloat]]
    measurements: list[FiniteFloat]
    estimates: list[list[FiniteFloat]]
    errors: list[list[FiniteFloat]]
    metrics: KalmanMetrics
    method: dict[str, str]
    units: dict[str, str]
