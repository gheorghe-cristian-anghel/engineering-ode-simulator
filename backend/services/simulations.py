"""Thin adapters from validated schemas to existing scientific modules."""

import math

import numpy as np

from analysis.kalman_filter import KalmanFilter, discretize_state_space
from analysis.quadcopter_obstacle_avoidance import (
    SphericalObstacle,
    simulate_quadcopter_obstacle_avoidance,
)
from analysis.quadcopter_waypoint_following import waypoint_trajectory
from analysis.state_space import dc_motor_state_space
from backend.core.serialization import to_jsonable
from backend.schemas.simulations import HeatRequest, KalmanRequest, UavRequest
from models.heat_equation_2d import (
    gaussian_hotspot_2d,
    rectangular_hot_region_2d,
    simulate_heat_equation_2d,
    sine_initial_condition_2d,
)

MAX_HEAT_TIME_STEPS = 50_000
MAX_HEAT_CELL_STEPS = 5_000_000
MAX_HEAT_SNAPSHOT_POINTS = 4_096


def run_heat_simulation(request: HeatRequest):
    """Run the existing heat solver with bounded storage and JSON-ready output."""
    dx = request.width / (request.nx - 1)
    dy = request.height / (request.ny - 1)
    requested_dt = request.dt or 0.4 / (request.alpha * (1.0 / dx**2 + 1.0 / dy**2))
    time_steps = math.ceil(request.t_final / requested_dt)
    if time_steps > MAX_HEAT_TIME_STEPS:
        raise ValueError(
            f"requested heat simulation exceeds the {MAX_HEAT_TIME_STEPS} step limit"
        )
    if request.nx * request.ny * time_steps > MAX_HEAT_CELL_STEPS:
        raise ValueError(
            "requested heat simulation exceeds the 5000000 cell-step limit"
        )

    stored_target = request.max_snapshots if request.include_snapshots else 1
    store_every = max(1, math.ceil(time_steps / stored_target))
    result = simulate_heat_equation_2d(
        width=request.width,
        height=request.height,
        alpha=request.alpha,
        t_final=request.t_final,
        nx=request.nx,
        ny=request.ny,
        dt=request.dt,
        initial_condition=_heat_initial_condition(request),
        boundary_type=request.boundary_type,
        boundary_values=request.boundary_value,
        store_every=store_every,
    )
    temperatures = result["temperature"]
    snapshots = (
        _heat_snapshots(result, request.max_snapshots)
        if request.include_snapshots
        else []
    )

    return to_jsonable(
        {
            "coordinates": {"x": result["x"], "y": result["y"]},
            "final_field": temperatures[-1],
            "snapshots": snapshots,
            "stability": {
                "rx": result["rx"],
                "ry": result["ry"],
                "sum": result["stability_sum"],
                "limit": 0.5,
                "is_stable": result["stability_sum"] <= 0.5,
            },
            "thermal_metrics": {
                "initial_min": np.min(temperatures[0]),
                "initial_max": np.max(temperatures[0]),
                "final_min": np.min(temperatures[-1]),
                "final_max": np.max(temperatures[-1]),
                "final_mean": np.mean(temperatures[-1]),
            },
            "method": {
                "name": "explicit finite difference",
                "equation": "u_t = alpha * (u_xx + u_yy)",
                "boundary_type": result["boundary_type"],
                "actual_dt": result["dt"],
            },
            "units": {"coordinates": "m", "time": "s", "temperature": "arbitrary"},
        }
    )


def _heat_initial_condition(request: HeatRequest):
    """Build one supported analytic initial condition for the existing solver."""
    config = request.initial_condition
    if config.kind == "gaussian":
        center = config.center or (request.width / 2, request.height / 2)
        width = config.width or 0.1 * min(request.width, request.height)
        return lambda x, y: gaussian_hotspot_2d(
            x, y, center=center, width=width, amplitude=config.amplitude
        )
    if config.kind == "rectangle":
        x_range = config.x_range or [0.4 * request.width, 0.6 * request.width]
        y_range = config.y_range or [0.4 * request.height, 0.6 * request.height]
        return lambda x, y: rectangular_hot_region_2d(
            x, y, x_range=x_range, y_range=y_range, value=config.amplitude
        )
    return lambda x, y: sine_initial_condition_2d(
        x,
        y,
        amplitude=config.amplitude,
        mode_x=config.mode_x,
        mode_y=config.mode_y,
        width=request.width,
        height=request.height,
    )


def _heat_snapshots(result, maximum):
    """Select at most the requested number of stored heat snapshots."""
    indices = np.linspace(
        0, len(result["t"]) - 1, min(maximum, len(result["t"])), dtype=int
    )
    field = result["temperature"]
    stride = max(
        1,
        math.ceil(
            math.sqrt(field.shape[1] * field.shape[2] / MAX_HEAT_SNAPSHOT_POINTS)
        ),
    )
    return [
        {
            "time": result["t"][index],
            "x": result["x"][::stride],
            "y": result["y"][::stride],
            "field": field[index, ::stride, ::stride],
            "downsample_stride": stride,
        }
        for index in indices
    ]


def run_uav_simulation(request: UavRequest):
    """Run the existing waypoint obstacle-avoidance simulation."""
    waypoints = np.asarray(request.waypoints, dtype=float)
    obstacles = [
        SphericalObstacle(**obstacle.model_dump()) for obstacle in request.obstacles
    ]
    initial_state = np.zeros(12)
    initial_state[:3] = waypoints[0]
    result = simulate_quadcopter_obstacle_avoidance(
        waypoint_trajectory(
            waypoints, segment_time=request.segment_time, smoothing=request.smoothing
        ),
        obstacles,
        initial_state=initial_state,
        t_span=(0.0, request.t_final),
        dt=request.dt,
        Kd_pos=(2.0, 2.0, 3.2),
        Kp_att=(0.25, 0.25, 0.16),
        Kd_att=(0.12, 0.12, 0.08),
        avoidance_gain=request.avoidance_gain,
        max_avoidance_acceleration=request.max_avoidance_acceleration,
    )
    metrics = result["obstacle_metrics"]
    response = {
            "time": result["time"],
            "reference_path": result["reference_positions"],
            "actual_path": result["states"][:, :3],
            "waypoints": waypoints,
            "obstacles": [obstacle.model_dump() for obstacle in request.obstacles],
            "metrics": {
                "final_error": metrics["final_position_error_norm"],
                "rms_error": metrics["rms_position_error"],
                "minimum_clearance": metrics["min_clearance"],
                "max_thrust": metrics["max_thrust"],
                "max_abs_torque": metrics["max_abs_torque"],
            },
            "units": {"position": "m", "time": "s", "thrust": "N", "torque": "N*m"},
    }
    if request.include_series:
        response["series"] = {
            "reference_altitude": result["reference_positions"][:, 2],
            "actual_altitude": result["states"][:, 2],
            "tracking_error": result["tracking_error_norm"],
            "clearance": result["nearest_clearances"],
            "thrust": result["controls"][:, 0],
            "torque": np.max(np.abs(result["controls"][:, 1:4]), axis=1),
        }
    return to_jsonable(response)


def run_kalman_simulation(request: KalmanRequest):
    """Run the existing linear Kalman filter on the DC motor state-space model."""
    continuous_a, continuous_b, measurement_matrix, _ = dc_motor_state_space()
    discrete_a, discrete_b = discretize_state_space(
        continuous_a, continuous_b, request.dt
    )
    samples = math.ceil(request.t_final / request.dt) + 1
    time = np.linspace(0.0, request.t_final, samples)
    true_states = np.zeros((samples, 2))
    input_vector = np.asarray([request.voltage])
    for index in range(samples - 1):
        true_states[index + 1] = (
            discrete_a @ true_states[index] + discrete_b @ input_vector
        )

    generator = np.random.default_rng(request.random_seed)
    measurements = true_states[:, 1] + generator.normal(
        0.0, request.measurement_noise_std, samples
    )
    kalman_filter = KalmanFilter(
        A=discrete_a,
        B=discrete_b,
        C=measurement_matrix,
        Q=np.diag([1e-4, 1e-3]),
        R=np.array([[request.measurement_noise_std**2]]),
        x_hat=np.zeros(2),
        P=np.diag([10.0, 100.0]),
        name="DC motor speed Kalman filter",
    )
    estimates = np.zeros_like(true_states)
    estimates[0], _ = kalman_filter.update(measurements[0], input_vector)
    for index in range(1, samples):
        estimates[index], _ = kalman_filter.step(measurements[index], input_vector)
    errors = true_states - estimates

    return to_jsonable(
        {
            "time": time,
            "true_state": true_states,
            "measurements": measurements,
            "estimates": estimates,
            "errors": errors,
            "metrics": {
                "rms_current_error": np.sqrt(np.mean(errors[:, 0] ** 2)),
                "rms_speed_error": np.sqrt(np.mean(errors[:, 1] ** 2)),
                "final_current_error": errors[-1, 0],
                "final_speed_error": errors[-1, 1],
            },
            "method": {
                "name": "linear discrete-time Kalman filter",
                "model": "DC motor",
            },
            "units": {"time": "s", "current": "A", "speed": "rad/s", "voltage": "V"},
        }
    )
