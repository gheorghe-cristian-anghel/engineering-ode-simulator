# UAV Models

The UAV examples demonstrate simplified educational/prototype quadcopter
dynamics, control, tracking, waypoint following, and local obstacle avoidance.

## Governing Equations

The simplified 6-DOF quadcopter model uses the state:

```text
[x, y, z, vx, vy, vz, phi, theta, psi, p, q, r]
```

The control input is:

```text
[T, tau_phi, tau_theta, tau_psi]
```

Translation is driven by thrust rotated into the inertial frame, gravity, and
optional linear drag:

```text
position_dot = velocity
velocity_dot = R(phi, theta, psi) [0, 0, T]^T / m + [0, 0, -g]^T - drag/m
```

Euler-angle rates are mapped from body rates, and rotational acceleration is
approximated from commanded body torques and diagonal inertias.

## Assumptions

- Rigid-body dynamics with ZYX Euler angles.
- Total thrust and body torques are direct commands.
- Inertias are diagonal and aerodynamic effects are simplified.
- High-fidelity rotor/motor allocation, rotor dynamics, motor mixing, propeller
  aerodynamics, sensor fusion, and autopilot firmware are outside scope.

## Numerical Method

- Open-loop 6-DOF dynamics are integrated with `solve_ivp`.
- Altitude and attitude PID examples use sampled controllers.
- Trajectory tracking uses a simplified cascaded controller around the 6-DOF
  plant.
- Waypoint following generates smooth reference paths and reuses the tracking
  controller.
- Obstacle avoidance adds a local repulsive acceleration term near static
  spherical obstacles.

## What the Repository Demonstrates

- Hover and tilted-thrust behavior in a 6-DOF rigid-body model.
- 1D altitude control and simplified attitude control.
- Trajectory tracking for hover and circular references.
- Waypoint following through multiple 3D goals.
- Reactive obstacle avoidance around static spherical obstacles.
- 3D plotting and animation helpers for simulated trajectories.

## Relevant Files and Examples

- `models/quadcopter_altitude.py`
- `models/quadcopter_attitude.py`
- `models/quadcopter_6dof.py`
- `analysis/quadcopter_altitude_control.py`
- `analysis/quadcopter_attitude_control.py`
- `analysis/quadcopter_trajectory_tracking.py`
- `analysis/quadcopter_waypoint_following.py`
- `analysis/quadcopter_obstacle_avoidance.py`
- `visualization/quadcopter_animation.py`
- `examples/run_quadcopter_6dof_hover.py`
- `examples/run_quadcopter_trajectory_circle_tracking.py`
- `examples/run_quadcopter_waypoint_following.py`
- `examples/run_quadcopter_obstacle_avoidance.py`

## Limitations

- The simulations are simplified educational/prototype models and are not
  flight-ready or a flight stack.
- Euler angles have singularities near extreme pitch.
- The obstacle avoidance method is local and reactive, so it can fail in
  cluttered or adversarial environments.
- There is no wind field, battery model, onboard estimator, collision model,
  SLAM, global planner, or hardware validation.
