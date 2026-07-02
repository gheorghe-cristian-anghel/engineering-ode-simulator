# Portfolio Notes

## Project Pitch

Engineering ODE Simulator is a Python-based simulation toolkit for modeling
electrical, mechanical, thermal, and control systems using ordinary
differential equations. It demonstrates scientific computing, numerical
simulation, control-system analysis, and clean software engineering practices.

## Skills Demonstrated

- Python scientific computing
- NumPy
- SciPy `solve_ivp`
- Matplotlib visualization
- Streamlit browser UI for selected simulations
- frequency response and Bode plot analysis
- transfer function step and impulse response analysis
- state-space modeling for linear systems
- constrained optimal control with Model Predictive Control
- receding-horizon simulation
- CSV export for Excel, MATLAB, LibreOffice Calc, or Python analysis
- saved plot screenshots for GitHub and README presentation
- saved or interactive Matplotlib animations for demos and social posts
- pytest testing
- Git/GitHub workflow
- differential equation modeling
- PDE-based scientific computing with finite differences
- finite difference derivative accuracy and convergence analysis
- control systems
- optimal state-space feedback with LQR
- Kalman filtering and noisy-measurement state estimation
- Extended Kalman filtering for nonlinear observer design
- Unscented Kalman filtering without manually derived Jacobians
- particle filtering with weighted particles and resampling
- nonlinear instability and linearization
- UAV altitude dynamics
- UAV altitude PID control
- UAV attitude dynamics
- UAV attitude PID control
- UAV full 6-DOF rigid-body dynamics
- UAV trajectory tracking
- UAV waypoint following
- UAV static obstacle avoidance
- electromechanical modeling
- embedded-style digital PID control
- discrete disturbance rejection for motor speed control
- practical PID tuning intuition
- parameter validation
- documentation

## Best Models to Showcase

### RLC Circuit

Shows second-order electrical dynamics, damping, overshoot, and oscillation.
The RLC parameter sweep examples demonstrate how resistance, capacitance, and
inductance shape damping ratio, natural frequency, overshoot, resonance-like
ringing, and settling behavior.

### Mass-Spring-Damper

Shows mechanical vibration and energy/damping concepts.

### 1D Heat Equation

Broadens the project from ODE and control simulations into PDE-based
scientific computing. The example uses an explicit finite-difference method to
show how a hot Gaussian temperature pulse diffuses along a rod while the peak
temperature decreases.

### 1D Wave Equation

Complements the heat equation by showing propagation instead of diffusion. The
example uses explicit second-order finite differences to show a Gaussian
displacement splitting into traveling waves that reflect from fixed
boundaries.

### Finite Difference Methods

Connects the heat and wave PDE solvers to general numerical differentiation.
The examples compare forward, backward, and central derivative formulas
against analytical derivatives and show first- and second-order convergence as
the grid spacing decreases.

### Second-Order Control System

Shows damping ratio, natural frequency, overshoot, peak time, and settling time.

### Inverted Pendulum / Cart-Pole

Shows nonlinear dynamics, open-loop upright instability, and linearization
around an unstable equilibrium. It also supports advanced controls such as
LQR, with observer/Kalman-filter examples left for future work. Matplotlib
animations make the cart motion, pendulum angle, and unstable departure easier
to understand visually.

### Inverted Pendulum LQR Control

Shows modern state-space feedback control by using a Linear Quadratic
Regulator to stabilize the nonlinear inverted pendulum near the upright
equilibrium. Animation examples make the stabilizing effect easier to show in
a portfolio or walkthrough.

### Linear Model Predictive Control

Shows constrained optimal control by solving a finite-horizon optimization
problem at each time step. The double-integrator example tracks a target
position while respecting acceleration limits, demonstrating the core
receding-horizon idea without adding nonlinear or UAV-specific MPC complexity.

### Kalman Filter State Estimation

Shows practical observer design by estimating hidden states from noisy sensor
measurements. The DC motor example estimates current and speed from noisy
speed measurements, while the RLC example reconstructs current from noisy
capacitor-voltage measurements.

### Extended Kalman Filter / Nonlinear Observer

Shows nonlinear state estimation by using an Extended Kalman Filter to estimate
pendulum angle and hidden angular velocity from noisy angle measurements. This
demonstrates the observer idea beyond linear state-space models.

### Unscented Kalman Filter / Nonlinear Observer

Shows nonlinear state estimation by using an Unscented Kalman Filter to
estimate pendulum angle and hidden angular velocity from noisy angle
measurements. This demonstrates nonlinear observer design without manually
deriving Jacobians.

### Particle Filter / Nonlinear Observer

Shows probabilistic nonlinear state estimation by using weighted particles and
systematic resampling to estimate pendulum angle and hidden angular velocity
from noisy angle measurements. This demonstrates an observer that does not
assume the state uncertainty must stay Gaussian.

### Quadcopter Altitude

Starts the UAV/drone simulation part of the project with a focused
one-dimensional vertical dynamics model. The examples show hover thrust,
descent below hover, climb above hover, and a thrust-step altitude response
without introducing full attitude or 6-DOF dynamics yet.

### Quadcopter Altitude PID Control

Connects the UAV altitude plant to practical drone control by using a sampled
PID loop to adjust thrust around hover thrust. The examples show target
altitude tracking, thrust saturation, tracking error, and recovery from a
downward force disturbance without adding attitude coupling or full 6-DOF
dynamics.

### Quadcopter Attitude

Extends the UAV/drone simulation track beyond vertical motion with simplified
roll, pitch, and yaw rotational dynamics. The examples show how open-loop body
torques create angular acceleration, increasing body rates, and changing
attitude angles before adding attitude PID control or full 6-DOF dynamics.

### Quadcopter Attitude PID Control

Connects the simplified quadcopter rotational dynamics to practical drone
stabilization ideas by using independent sampled PID controllers for roll,
pitch, and yaw. The examples show commanded attitude tracking, torque
saturation, tracking error, and recovery from an external roll disturbance
without adding full 6-DOF dynamics or trajectory control.

### Full 6-DOF Quadcopter Dynamics

Combines translational and rotational UAV motion in one rigid-body model. The
examples show hover, tilted thrust producing horizontal acceleration, and body
torques changing attitude, which creates a foundation for future UAV
trajectory, waypoint, and control examples.

### Quadcopter Trajectory Tracking

Demonstrates controlled UAV motion using the full 6-DOF model and a simplified
cascaded PD controller. The examples compare actual and desired hover/circular
paths, position error, thrust, attitude commands, and body torques while
leaving advanced autopilot topics such as MPC and waypoint planning for later.

### Quadcopter Waypoint Following

Demonstrates discrete UAV navigation goals converted into smooth controlled
motion. The example turns a list of 3D waypoints into a linear or smoothstep
reference trajectory, then tracks it with the existing full 6-DOF cascaded
controller without adding obstacle avoidance, MPC, or rotor-level motor
mixing.

### Quadcopter Static Obstacle Avoidance

Demonstrates navigation behavior beyond waypoint tracking by adding static
spherical obstacles and a local repulsive acceleration term. The example shows
how a UAV can reactively deviate from a reference path near an obstacle while
still moving toward the final waypoint, without adding SLAM, global planning,
or moving-obstacle logic.

### Quadcopter Animation

Improves visual presentation of the 6-DOF UAV examples by showing the drone
body attitude, path trail, reference trajectory, and waypoint markers in a 3D
Matplotlib animation. This makes the waypoint and circular tracking examples
easier to present on GitHub, LinkedIn, and short demo videos.

### DC Motor

Shows coupled electrical-mechanical dynamics.
The load disturbance rejection comparison shows the practical value of
feedback control when external load torque changes: open-loop speed sags,
while PI and discrete PID controllers increase voltage and current to recover
toward the target.

### PI Motor Speed Control

Shows closed-loop feedback control, reference tracking, voltage saturation, and
motor speed regulation. The demo also includes load disturbance rejection,
where the controller increases voltage and current to recover speed after a
torque step.

### Discrete PID Motor Control

Shows embedded-style sampled control: fixed update time, held voltage commands,
output saturation, anti-windup, derivative-on-measurement, and DC motor speed
tracking. The disturbance response example shows practical closed-loop
behavior by applying a step load torque and observing speed drop, voltage and
current increase, and recovery toward the target.

### PID Tuning Examples

Shows practical control-engineering intuition by comparing P, PI, and PID
responses and demonstrating how `Kp`, `Ki`, and `Kd` affect speed tracking,
overshoot, settling time, and control effort.

### Streamlit GUI

Shows selected simulations in a browser UI with interactive inputs and
Matplotlib plots while reusing the tested model modules.

### Frequency Response

Shows Bode magnitude and phase plots for first-order, second-order, and RLC
low-pass transfer functions.

### Transfer Functions

Strengthens the control-systems toolkit with reusable transfer function
models, step response analysis, impulse response analysis, and common low-pass
helpers.

### State-Space Models

Shows modern control-system modeling using `x_dot = A*x + B*u` and
`y = C*x + D*u` for mechanical, electrical, and electromechanical systems.

### CSV Export

Shows practical engineering workflow support by exporting simulation results
for external analysis in Excel, MATLAB, LibreOffice Calc, or Python.

### Plot Screenshots

Selected simulation plots are saved as screenshots for GitHub, README, and
portfolio presentation.

### Animation Examples

Selected Matplotlib animations can show nonlinear motion and feedback
stabilization, especially for the inverted pendulum, LQR, and quadcopter
trajectory examples.

## Freelance Relevance

The project supports offers such as:

- engineering simulation scripts
- ODE modeling
- plotting and visualization
- control-system response analysis
- converting equations into Python tools
- educational simulation apps

## Interview Talking Points

- I built a modular simulation library with one model per file.
- I used SciPy `solve_ivp` for numerical integration.
- I added tests that check engineering behavior.
- I implemented reusable step response metrics.
- I added reusable frequency response utilities and Bode plot examples.
- I added reusable state-space simulation utilities for linear systems.
- I added CSV export so simulation data can be reused in external analysis tools.
- I modeled both open-loop and closed-loop systems.
- I added an inverted pendulum/cart-pole model that demonstrates nonlinear
  instability and upright state-space linearization.
- I added LQR control for the inverted pendulum to demonstrate modern
  state-space feedback stabilization.
- I added linear Model Predictive Control to demonstrate constrained optimal
  control with receding-horizon simulation.
- I added Matplotlib animation examples for open-loop and LQR inverted
  pendulum trajectories.
- I added Kalman filter examples that estimate hidden states from noisy
  measurements.
- I added an Extended Kalman Filter example for nonlinear pendulum state
  estimation from noisy angle measurements.
- I added an Unscented Kalman Filter example that estimates nonlinear pendulum
  state without manually deriving Jacobians.
- I added a Particle Filter example that estimates nonlinear pendulum state
  with weighted particles and resampling.
- I added a 1D quadcopter altitude model as the foundation for future UAV
  control examples.
- I added PID altitude control to show thrust-based target tracking and
  disturbance rejection for the 1D quadcopter model.
- I added a simplified quadcopter attitude model for open-loop roll, pitch,
  and yaw rotational dynamics.
- I added attitude PID control to show roll, pitch, and yaw tracking with
  body torque commands and disturbance rejection.
- I added a full 6-DOF quadcopter model that combines translational and
  rotational rigid-body dynamics for future UAV control examples.
- I added quadcopter trajectory tracking with hover-point and circular-path
  examples using the full 6-DOF model.
- I added quadcopter waypoint following to show how discrete 3D navigation
  goals can be converted into smooth reference motion for a controlled 6-DOF
  UAV simulation.
- I added static quadcopter obstacle avoidance to demonstrate local reactive
  navigation around spherical obstacles.
- I added 3D quadcopter animation examples to visualize 6-DOF position,
  attitude, reference paths, and waypoint tracking.
- I added an embedded-style discrete PID controller for motor speed control.
- I added a discrete PID disturbance response example to show load rejection.
- I added PID tuning examples that show how controller gains affect response
  quality.
- I added a Streamlit GUI MVP so selected simulations can be explored from a
  browser.
- I documented the project for future extension.
- I added a 1D heat equation solver to demonstrate PDE simulation with finite
  differences, stability checks, boundary conditions, and heatmap visualization.
- I added a 1D wave equation solver to demonstrate propagation, reflection,
  CFL stability, and second-order PDE time integration.
- I added reusable finite difference derivative utilities to demonstrate
  numerical differentiation, error metrics, and convergence-order analysis.

## Future Portfolio Upgrades

- screenshots
- hosted demo
- blog post series
- LinkedIn posts
- short demo video
