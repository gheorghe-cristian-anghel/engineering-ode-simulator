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
- CSV export for Excel, MATLAB, LibreOffice Calc, or Python analysis
- saved plot screenshots for GitHub and README presentation
- pytest testing
- Git/GitHub workflow
- differential equation modeling
- control systems
- optimal state-space feedback with LQR
- Kalman filtering and noisy-measurement state estimation
- Extended Kalman filtering for nonlinear observer design
- nonlinear instability and linearization
- UAV altitude dynamics
- UAV altitude PID control
- UAV attitude dynamics
- UAV attitude PID control
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

### Kalman Filter State Estimation

Shows practical observer design by estimating hidden states from noisy sensor
measurements. The DC motor example estimates current and speed from noisy
speed measurements, while the RLC example reconstructs current from noisy
capacitor-voltage measurements.

### Extended Kalman Filter / Nonlinear Observer

Shows nonlinear state estimation by using an Extended Kalman Filter to estimate
pendulum angle and hidden angular velocity from noisy angle measurements. This
demonstrates the observer idea beyond linear state-space models.

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
stabilization, especially for the inverted pendulum and LQR examples.

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
- I added Matplotlib animation examples for open-loop and LQR inverted
  pendulum trajectories.
- I added Kalman filter examples that estimate hidden states from noisy
  measurements.
- I added an Extended Kalman Filter example for nonlinear pendulum state
  estimation from noisy angle measurements.
- I added a 1D quadcopter altitude model as the foundation for future UAV
  control examples.
- I added PID altitude control to show thrust-based target tracking and
  disturbance rejection for the 1D quadcopter model.
- I added a simplified quadcopter attitude model for open-loop roll, pitch,
  and yaw rotational dynamics.
- I added attitude PID control to show roll, pitch, and yaw tracking with
  body torque commands and disturbance rejection.
- I added an embedded-style discrete PID controller for motor speed control.
- I added a discrete PID disturbance response example to show load rejection.
- I added PID tuning examples that show how controller gains affect response
  quality.
- I added a Streamlit GUI MVP so selected simulations can be explored from a
  browser.
- I documented the project for future extension.

## Future Portfolio Upgrades

- screenshots
- hosted demo
- blog post series
- LinkedIn posts
- short demo video
