# Architecture

## Overview

Engineering ODE Simulator is organized into models, analysis tools, examples,
tests, and documentation. Model modules focus on equations and numerical
simulation. Examples demonstrate usage and plotting. Tests verify numerical and
engineering behavior. The Streamlit app provides a small browser-based UI over
selected existing simulations without moving equations into the presentation
layer.

## Folder Responsibilities

### models/

Physical and control-system models.

Each model usually contains:

- validation functions
- ODE function
- simulation function
- helper formulas

`models/discrete_pid.py` contains reusable controller logic, not only
motor-specific code. Its `DiscretePID` class can be used for embedded-style
digital control with fixed sample time, output saturation, anti-windup, and
derivative-on-measurement. The module also includes a focused disturbance
response helper that reuses the sampled controller and DC motor plant with a
step load torque input.

`models/inverted_pendulum.py` contains an advanced nonlinear cart-pole model
and a linearized upright state-space helper. It demonstrates open-loop
instability and provides the plant model used by the LQR stabilization helper.

`models/inverted_pendulum_lqr.py` contains inverted-pendulum-specific LQR
design and nonlinear closed-loop simulation helpers. It reuses the nonlinear
cart-pole dynamics from `models/inverted_pendulum.py` and applies state
feedback as an external cart force.

`models/quadcopter_altitude.py` contains a simplified one-dimensional UAV
vertical dynamics model. It provides hover thrust, constant and step thrust
commands, open-loop altitude simulation, and a hover-linearized state-space
helper without introducing attitude or full 6-DOF dynamics.

`models/quadcopter_attitude.py` contains a simplified UAV rotational dynamics
model. It provides constant and step body torque commands, open-loop roll,
pitch, and yaw attitude simulation, and a linear attitude state-space helper
without introducing full 6-DOF translation or attitude PID control.

`models/quadcopter_6dof.py` contains the full open-loop UAV rigid-body
dynamics model. It combines inertial-frame translation, ZYX Euler-angle
rotation, total thrust, body torques, gravity, and optional simple linear drag
without introducing trajectory tracking, rotor mixing, or autopilot control.

### analysis/

Reusable analysis and output tools independent of any one model.

Current example:

- step response metrics
- parameter sweep utilities
- motor load disturbance metrics
- RLC sweep helpers for educational component studies
- PID tuning helpers for educational discrete-control examples
- frequency response and Bode plot utilities
- transfer function step and impulse response utilities
- state-space simulation utilities
- LQR optimal-control utilities
- Kalman filter state-estimation utilities
- Extended Kalman Filter nonlinear state-estimation utilities
- Unscented Kalman Filter nonlinear state-estimation utilities
- CSV export utilities

`analysis/transfer_function.py` provides reusable continuous-time transfer
function infrastructure, including validated coefficient storage, SciPy signal
conversion, common low-pass helpers, and step/impulse response simulation.

`analysis/pid_tuning.py` provides reusable PID tuning infrastructure for
educational examples built on the discrete PID motor simulation.

`analysis/quadcopter_altitude_control.py` provides sampled PID altitude
control logic built on `models/quadcopter_altitude.py`. It computes thrust
commands around hover thrust with saturation and anti-windup, then integrates
the vertical altitude plant over each held-thrust sample interval.

`analysis/quadcopter_attitude_control.py` provides sampled PID attitude
control logic built on `models/quadcopter_attitude.py`. It uses independent
roll, pitch, and yaw PID loops to command body torques with saturation and
anti-windup, then integrates the rotational plant over each held-torque sample
interval. Optional external disturbance torque can be added without changing
the attitude dynamics model.

`analysis/quadcopter_trajectory_tracking.py` provides UAV trajectory-control
infrastructure built on `models/quadcopter_6dof.py`. It includes hover-point
and circular reference generators plus a simplified cascaded PD controller
that commands total thrust and body torques during sampled closed-loop
integration.

`analysis/quadcopter_waypoint_following.py` provides educational UAV waypoint
following logic built on the trajectory tracking infrastructure. It converts
discrete 3D waypoint goals into linear or smoothstep reference trajectories,
then reuses the existing cascaded 6-DOF controller without adding obstacle
avoidance, MPC, rotor mixing, or autopilot complexity.

`analysis/lqr.py` provides reusable continuous-time Linear Quadratic
Regulator infrastructure based on the continuous algebraic Riccati equation.

`analysis/kalman_filter.py` provides reusable discrete-time linear Kalman
filter infrastructure for estimating hidden states from noisy measurements.

`analysis/extended_kalman_filter.py` provides reusable Extended Kalman Filter
infrastructure for nonlinear state estimation. The pendulum EKF example uses
nonlinear prediction plus local Jacobians to estimate angle and hidden angular
velocity from noisy angle measurements.

`analysis/unscented_kalman_filter.py` provides reusable Unscented Kalman Filter
infrastructure for nonlinear state estimation. The pendulum UKF example uses
sigma points to estimate angle and hidden angular velocity from noisy angle
measurements without manually deriving Jacobians.

### visualization/

Reusable visualization helpers for plotting and animation code that should not
live in the physics model modules. `visualization/inverted_pendulum_animation.py`
animates cart-pole trajectories using Matplotlib while preserving the state and
angle convention from `models/inverted_pendulum.py`.
`visualization/quadcopter_animation.py` animates full 6-DOF quadcopter
trajectories in 3D using the state convention and body-to-inertial rotation
matrix from `models/quadcopter_6dof.py`.

### examples/

Runnable scripts that demonstrate models, print important parameters, and
generate plots or animations. Selected examples can also export simulation
arrays to CSV files under `outputs/`.

The PID tuning examples compare P, PI, and PID behavior and show how `Kp`,
`Ki`, and `Kd` affect DC motor speed tracking, overshoot, settling time, and
control voltage.

The RLC sweep examples compare resistance, capacitance, and inductance effects
on capacitor-voltage transient response while reusing the RLC model and step
response metrics.

The DC motor disturbance rejection examples compare fixed-voltage open-loop
behavior with continuous PI and discrete PID feedback control under the same
load torque step. They reuse the DC motor plant, existing controller models,
and shared disturbance metrics.

The discrete PID disturbance response example demonstrates practical
closed-loop rejection of a load torque step while keeping the motor equations
and sampled controller logic in the model layer.

The quadcopter attitude PID examples demonstrate roll, pitch, and yaw target
tracking plus simple disturbance torque rejection while keeping the UAV scope
limited to rotational attitude dynamics.

The full 6-DOF quadcopter examples demonstrate hover, tilted thrust causing
horizontal acceleration, and torque-driven attitude response using the
standalone rigid-body plant model.

The quadcopter trajectory tracking examples demonstrate controlled 6-DOF UAV
motion by tracking a fixed hover point and a slow circular path with a simple
cascaded controller.

The quadcopter waypoint-following example demonstrates discrete UAV navigation
goals by tracking a smooth reference path through several 3D waypoints with
the same simplified cascaded controller.

The quadcopter animation examples reuse the waypoint-following and circular
trajectory results, then visualize the 6-DOF vehicle position, attitude,
reference path, trail, and waypoint markers without changing controller logic.

### streamlit_app.py

Interactive browser UI for selected simulations. It imports model and analysis
functions, builds Streamlit controls, and displays Matplotlib figures with
`st.pyplot`.

### tests/

pytest tests for numerical correctness and engineering behavior.

### docs/

Longer documentation and project notes.

`docs/screenshots/` stores selected generated plot images used for
documentation and portfolio presentation.
`docs/animations/` is reserved for selected generated animations when they are
explicitly kept as portfolio assets.

## Model Pattern

The usual model pattern is:

1. Validate parameters.
2. Define the ODE.
3. Simulate with `solve_ivp`.
4. Return NumPy arrays.
5. Add an example script that plots results.
6. Add tests that verify physical behavior.

## Testing Philosophy

Tests should check:

- correct initial conditions
- expected steady-state behavior
- known formulas
- numerical and analytical agreement where available
- conservation laws where applicable
- physically meaningful behavior such as overshoot, damping, or convergence

## Future Architecture Improvements

Possible improvements:

- package installation with `pyproject.toml`
- shared plotting utilities
- shared validation utilities
- exporting results to CSV
