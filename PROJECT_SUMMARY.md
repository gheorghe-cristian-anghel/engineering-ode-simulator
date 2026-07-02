# Engineering ODE Simulator - Project Summary

## Purpose

Engineering ODE Simulator is a Python engineering simulation library for
modeling physical and control systems governed by ordinary differential
equations.

The project goals are to:

- learn scientific computing
- learn differential equation modeling
- learn numerical simulation
- build a portfolio project
- prepare for control systems, robotics, electrical engineering, and automation work

## Technology Stack

- Python
- NumPy
- SciPy
- Matplotlib
- Streamlit
- pytest
- Git/GitHub
- VS Code

Simulations use `scipy.integrate.solve_ivp` for numerical integration.

## Project Structure

- `models/`: engineering models and simulation functions
- `analysis/`: reusable analysis tools independent of one specific model
- `visualization/`: reusable plotting and animation helpers
- `examples/`: runnable scripts that print parameters and plot results
- `streamlit_app.py`: browser UI for selected interactive simulations
- `tests/`: pytest coverage for model behavior and helper formulas
- `docs/`: longer documentation, equations, architecture notes, and future ideas
- `docs/screenshots/`: selected generated plots for documentation and portfolio presentation
- `docs/animations/`: selected generated animations for portfolio presentation

## Coding Conventions

- One model per module.
- Each model has validation functions where appropriate.
- Model modules usually include an ODE function and a `simulate_*` function.
- Each model should have an example script and pytest tests.
- Use docstrings and beginner-readable code.
- Prefer clear engineering names over overly abstract names.
- Use SI units unless stated otherwise.
- Examples should print important parameters and plot results.
- Tests should check physics and engineering behavior, not only syntax.

## Implemented Models

| Model | Domain | Type / Order | Main File | Example | Test File | Status |
|---|---|---:|---|---|---|---|
| RC Circuit | Electrical | First-order | `models/rc_circuit.py` | `examples/run_rc_circuit.py` | `tests/test_rc_circuit.py` | Implemented |
| RL Circuit | Electrical | First-order | `models/rl_circuit.py` | `examples/run_rl_circuit.py` | `tests/test_rl_circuit.py` | Implemented |
| RLC Circuit | Electrical | Second-order | `models/rlc_circuit.py` | `examples/run_rlc_circuit.py` | `tests/test_rlc_circuit.py` | Implemented |
| Newton Cooling | Thermal | First-order | `models/cooling.py` | `examples/run_cooling.py` | `tests/test_cooling.py` | Implemented |
| Mass-Spring-Damper | Mechanical | Second-order | `models/mass_spring_damper.py` | `examples/run_mass_spring_damper.py` | `tests/test_mass_spring_damper.py` | Implemented |
| Pendulum | Mechanical | Nonlinear second-order | `models/pendulum.py` | `examples/run_pendulum.py` | `tests/test_pendulum.py` | Implemented |
| Inverted Pendulum / Cart-Pole | Control / Mechanical | Nonlinear fourth-order plus linearized state-space | `models/inverted_pendulum.py` | `examples/run_inverted_pendulum_open_loop.py` | `tests/test_inverted_pendulum.py` | Implemented |
| Inverted Pendulum LQR Control | Control / Mechanical | State-feedback control | `models/inverted_pendulum_lqr.py` | `examples/run_inverted_pendulum_lqr.py` | `tests/test_lqr.py` | Implemented |
| Quadcopter Altitude | UAV / Mechanical | Nonlinear second-order vertical dynamics | `models/quadcopter_altitude.py` | `examples/run_quadcopter_altitude_open_loop.py` | `tests/test_quadcopter_altitude.py` | Implemented |
| Quadcopter Altitude PID Control | UAV / Control | Sampled-data PID altitude tracking | `analysis/quadcopter_altitude_control.py` | `examples/run_quadcopter_altitude_pid.py` | `tests/test_quadcopter_altitude_control.py` | Implemented |
| Quadcopter Attitude | UAV / Rotational Dynamics | Linear sixth-order rotational dynamics | `models/quadcopter_attitude.py` | `examples/run_quadcopter_attitude_roll_torque.py` | `tests/test_quadcopter_attitude.py` | Implemented |
| Quadcopter Attitude PID Control | UAV / Control | Sampled-data PID attitude tracking | `analysis/quadcopter_attitude_control.py` | `examples/run_quadcopter_attitude_pid.py` | `tests/test_quadcopter_attitude_control.py` | Implemented |
| Full 6-DOF Quadcopter | UAV / Rigid-Body Dynamics | Nonlinear twelfth-order dynamics | `models/quadcopter_6dof.py` | `examples/run_quadcopter_6dof_hover.py` | `tests/test_quadcopter_6dof.py` | Implemented |
| Quadcopter Trajectory Tracking | UAV / Control | Cascaded PD trajectory tracking | `analysis/quadcopter_trajectory_tracking.py` | `examples/run_quadcopter_trajectory_circle_tracking.py` | `tests/test_quadcopter_trajectory_tracking.py` | Implemented |
| Quadcopter Waypoint Following | UAV / Control | Smooth waypoint reference tracking | `analysis/quadcopter_waypoint_following.py` | `examples/run_quadcopter_waypoint_following.py` | `tests/test_quadcopter_waypoint_following.py` | Implemented |
| First-Order Control System | Control | First-order | `models/first_order_control.py` | `examples/run_first_order_control.py` | `tests/test_first_order_control.py` | Implemented |
| Second-Order Control System | Control | Second-order | `models/second_order_control.py` | `examples/run_second_order_control.py` | `tests/test_second_order_control.py` | Implemented |
| DC Motor | Electromechanical | Coupled first-order system | `models/dc_motor.py` | `examples/run_dc_motor.py` | `tests/test_dc_motor.py` | Implemented |
| PI Motor Speed Control | Control / Electromechanical | Closed-loop system | `models/pid_motor_control.py` | `examples/run_pid_motor_control.py` | `tests/test_pid_motor_control.py` | Implemented |
| PI Motor Load Disturbance | Control / Electromechanical | Disturbance response | `models/pid_motor_control.py` | `examples/run_motor_load_disturbance.py` | `tests/test_motor_load_disturbance.py` | Implemented |
| Discrete PID Motor Speed Control | Control / Electromechanical | Sampled-data control | `models/discrete_pid.py` | `examples/run_discrete_pid_motor.py` | `tests/test_discrete_pid.py` | Implemented |
| Discrete PID Load Disturbance | Control / Electromechanical | Sampled-data disturbance response | `models/discrete_pid.py` | `examples/run_discrete_pid_disturbance_response.py` | `tests/test_discrete_pid_disturbance.py` | Implemented |

Implemented control features include a reusable `DiscretePID` controller in
`models/discrete_pid.py`. It models embedded-style digital control with fixed
sample time `dt`, output saturation, anti-windup, and
derivative-on-measurement. The DC motor example is
`examples/run_discrete_pid_motor.py`, with tests in `tests/test_discrete_pid.py`.
The discrete PID disturbance response example uses the same sampled controller
and DC motor model with a step load torque disturbance.
Educational PID tuning examples compare P, PI, and PID behavior and show how
changing `Kp`, `Ki`, and `Kd` affects DC motor speed tracking.
DC motor load disturbance rejection examples compare fixed-voltage open-loop
behavior against continuous PI and discrete PID feedback control.
The quadcopter altitude model provides a one-dimensional UAV vertical dynamics
foundation with hover thrust, thrust commands, and a hover-linearized
state-space helper for later altitude-control examples.
The altitude PID control helper builds on that plant with a sampled discrete
PID loop, thrust saturation, anti-windup, target-altitude tracking metrics,
and a downward-force disturbance example.
The quadcopter attitude model extends the UAV track with simplified rotational
dynamics for roll, pitch, yaw, and body rates under open-loop body torques.
The attitude PID control helper builds on the attitude plant with independent
sampled PID controllers for roll, pitch, and yaw torque commands, including
target-angle tracking metrics and a simple disturbance torque rejection
example.
The full 6-DOF quadcopter model combines inertial-frame translation and
rigid-body Euler-angle rotation using total thrust and body torque inputs. Its
open-loop examples demonstrate hover, tilted-thrust horizontal acceleration,
and torque-driven attitude response without adding trajectory tracking or
autopilot control.
The quadcopter trajectory tracking helper adds hover-point and circular
reference generators plus a simplified cascaded PD controller for the full
6-DOF plant. It demonstrates controlled UAV motion without adding MPC,
waypoint planning, rotor mixing, or individual motor dynamics.
The quadcopter waypoint-following helper converts discrete 3D waypoints into
linear or smoothstep reference trajectories, then reuses the full 6-DOF
trajectory tracking controller to demonstrate educational waypoint following
without obstacle avoidance, MPC, or rotor-level motor mixing.

## Analysis Tools

Reusable step response metrics are implemented in `analysis/step_response.py`.
PI motor controller gain sweeps are implemented in
`analysis/parameter_sweep.py`.
DC motor load disturbance metrics are implemented in
`analysis/motor_disturbance.py`.
RLC parameter sweep helpers are implemented in `analysis/rlc_sweep.py` for
educational resistance, capacitance, and inductance transient-response studies.
PID tuning helpers for discrete motor control examples are implemented in
`analysis/pid_tuning.py`.
Quadcopter altitude PID control helpers are implemented in
`analysis/quadcopter_altitude_control.py`.
Quadcopter attitude PID control helpers are implemented in
`analysis/quadcopter_attitude_control.py`.
Quadcopter trajectory tracking helpers are implemented in
`analysis/quadcopter_trajectory_tracking.py`.
Quadcopter waypoint-following helpers are implemented in
`analysis/quadcopter_waypoint_following.py`.
CSV export helpers are implemented in `analysis/export_utils.py`.
Frequency response and Bode plot helpers are implemented in
`analysis/frequency_response.py`.
Transfer function helpers for continuous-time step and impulse response
simulation are implemented in `analysis/transfer_function.py`.
Continuous-time state-space simulation helpers are implemented in
`analysis/state_space.py`.
Continuous-time LQR design helpers are implemented in `analysis/lqr.py`.
Discrete-time Kalman filter helpers are implemented in
`analysis/kalman_filter.py`.
Extended Kalman Filter helpers for nonlinear state estimation are implemented
in `analysis/extended_kalman_filter.py`, with a nonlinear pendulum EKF example
that estimates angle and hidden angular velocity from noisy angle
measurements.
Unscented Kalman Filter helpers for nonlinear state estimation are implemented
in `analysis/unscented_kalman_filter.py`, with a nonlinear pendulum UKF example
that estimates angle and hidden angular velocity from noisy angle measurements
without manually deriving Jacobians.
Bootstrap Particle Filter helpers for nonlinear and non-Gaussian state
estimation are implemented in `analysis/particle_filter.py`, with a nonlinear
pendulum particle-filter example that estimates angle and hidden angular
velocity using weighted particles and resampling.
Reusable cart-pole animation helpers are implemented in
`visualization/inverted_pendulum_animation.py`.
Reusable 3D quadcopter animation helpers are implemented in
`visualization/quadcopter_animation.py`.

The Streamlit MVP in `streamlit_app.py` provides an interactive browser UI for
RC circuit charging, RLC circuit step response, and discrete PID motor speed
control while reusing the existing model and analysis modules.

The tool estimates:

- initial value
- final value
- steady-state value
- rise time
- settling time
- peak value
- peak time
- overshoot

The parameter sweep tool compares PI controller gain choices using:

- final value
- final tracking error
- peak value
- overshoot
- settling time
- maximum control effort
- maximum current

The RLC sweep helper compares component values using:

- natural frequency
- damping ratio
- final capacitor voltage
- peak voltage
- overshoot
- settling time

The PID tuning helper compares discrete PID controller gain choices using:

- final speed
- final tracking error
- peak speed
- overshoot
- settling time
- maximum voltage
- maximum current

The discrete PID disturbance helper summarizes:

- speed before disturbance
- minimum speed after disturbance
- speed drop
- recovery time when the response returns to the target band
- final speed and final error
- voltage and current before and after recovery or at the end

The motor disturbance helper summarizes open-loop and feedback disturbance
responses using:

- speed before disturbance
- minimum speed after disturbance
- speed drop
- final speed and final error
- recovery time
- optional voltage and current before/final values

The export utility validates column names, dimensions, and column lengths,
creates missing output directories automatically, and writes simulation arrays
to CSV files for later analysis in Excel, MATLAB, Python, or reports.

The frequency response utility computes continuous-time transfer-function
magnitude and phase data and provides a reusable Matplotlib Bode plot helper.
The transfer function utility provides a reusable model representation plus
step response, impulse response, and common low-pass transfer function helpers.
The state-space utility simulates linear systems in the form
`x_dot = A*x + B*u` and `y = C*x + D*u`.

Selected examples also save plot screenshots in `docs/screenshots/` as
documentation and portfolio presentation assets.

## Current Development Workflow

1. Study the engineering concept.
2. Derive the differential equation.
3. Convert to a first-order system for `solve_ivp`.
4. Implement the model module.
5. Add an example script.
6. Add tests.
7. Run pytest.
8. Run the example and inspect plots.
9. Update README/docs.
10. Commit and push to GitHub.

## Current Status

The project can simulate first-order systems, second-order systems, nonlinear
pendulum motion, open-loop inverted pendulum/cart-pole instability, LQR
stabilization of the nonlinear inverted pendulum near upright, open-loop DC
motor dynamics, and closed-loop PI motor speed control with disturbance
rejection. It also includes embedded-style discrete PID motor speed control
and discrete PID disturbance response, Kalman filter examples for noisy
state estimation, a Streamlit GUI MVP for selected simulations,
Extended Kalman Filter nonlinear observer examples,
Unscented Kalman Filter nonlinear observer examples,
Particle Filter nonlinear observer examples,
quadcopter altitude dynamics,
quadcopter altitude PID control,
quadcopter attitude dynamics,
quadcopter attitude PID control,
full 6-DOF quadcopter rigid-body dynamics,
quadcopter trajectory tracking,
quadcopter waypoint following,
Matplotlib animation examples for inverted pendulum trajectories,
3D Matplotlib animation examples for full 6-DOF quadcopter trajectories,
reusable step response metrics, frequency response analysis, transfer function
utilities, state-space simulation utilities, PI gain sweep analysis, RLC
parameter sweep examples, DC motor disturbance rejection comparison examples,
PID tuning examples, and pytest coverage for implemented models.

## How Future AI Chats Should Use This File

Future ChatGPT/Codex chats should read `PROJECT_SUMMARY.md` and `ROADMAP.md`
before implementing new features. These files provide the current project
scope, coding style, implemented models, and planned direction.
