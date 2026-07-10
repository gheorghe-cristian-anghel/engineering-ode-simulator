---
title: UAV Quadcopter Module
tags:
  - engineering-simulation-toolkit
  - uav
  - quadcopter
---

# UAV Quadcopter Module

The UAV/quadcopter work is an educational set of simplified dynamics, control, tracking, waypoint, obstacle-avoidance, and animation examples.

## Simplified 6-DOF Model

`models/quadcopter_6dof.py` models open-loop rigid-body dynamics:

- inertial-frame translation
- ZYX Euler angles
- total thrust
- body torques
- gravity
- optional simple linear drag

It does not include rotor mixing, motor dynamics, sensor fusion, flight firmware, SLAM, or high-fidelity aerodynamics.

## Altitude Control

Altitude examples use simplified 1D vertical dynamics plus sampled PID control around hover thrust.

Relevant areas:

- `models/quadcopter_altitude.py`
- `analysis/quadcopter_altitude_control.py`
- Streamlit UAV altitude tab

## Trajectory Tracking

Trajectory tracking uses simplified cascaded control on top of the 6-DOF model. It supports hover and circular references.

Relevant area:

- `analysis/quadcopter_trajectory_tracking.py`

## Waypoint Following

Waypoint following converts discrete 3D waypoints into linear or smoothstep reference trajectories, then reuses the simplified 6-DOF trajectory controller.

Relevant area:

- `analysis/quadcopter_waypoint_following.py`

## Obstacle Avoidance

Obstacle avoidance uses local repulsive acceleration around static spherical obstacles.

Relevant area:

- `analysis/quadcopter_obstacle_avoidance.py`

## Plots and Animations

Examples generate trajectory plots and 3D animations. See [[Visualization and Animations]].

## Limitations

> [!warning]
> No flight-ready claims. These are educational/prototype simulations only.

Do not claim:

- certified flight control
- real autopilot behavior
- real-time embedded readiness
- complete collision avoidance
- high-fidelity aircraft modeling

Related: [[Scientific Correctness]], [[Streamlit App]].
