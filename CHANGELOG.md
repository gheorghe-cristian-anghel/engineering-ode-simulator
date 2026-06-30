# Changelog

All notable changes to this project will be documented here.

## Unreleased

### Added

#### Basic First-Order Models

- RC circuit charging model
- RL circuit step response model
- Newton cooling model
- First-order control system step response

#### Second-Order Physical Systems

- Mass-spring-damper free vibration model
- RLC circuit step response model
- Simple pendulum nonlinear model with small-angle comparison

#### Control Systems

- Second-order control system step response
- Nonlinear inverted pendulum / cart-pole model
- Linearized inverted pendulum upright state-space model
- Open-loop inverted pendulum instability examples
- LQR stabilization examples for the nonlinear inverted pendulum
- PI motor speed control with PID-compatible API
- Discrete PID controller with anti-windup and derivative-on-measurement
- Discrete PID disturbance response example for DC motor speed control
- PID tuning examples for P, PI, and PID comparison

#### DC Motor and PI Control

- DC motor open-loop speed response
- DC motor open-loop load disturbance example
- Closed-loop PI motor speed control with voltage saturation
- Load disturbance response example for PI motor speed control
- Disturbance rejection comparison for open-loop and feedback control
- Discrete PID motor speed-control example
- Discrete PID load disturbance metrics and plots
- Kp, Ki, and Kd tuning demonstrations for DC motor speed control

#### Analysis Utilities

- Reusable step response metrics
- Generic CSV export utility for simulation results
- CSV export from the discrete PID motor example
- Output directory support for generated files
- Tests for CSV export validation and file output
- DC motor load disturbance metrics helper
- PI gain sweep utility for motor speed-control tuning
- PI gain sweep example with speed and voltage comparison plots
- RLC parameter sweep examples for resistance, capacitance, and inductance
- RLC transient-response metrics for comparing component effects
- Reusable PID tuning helper for discrete motor speed-control examples
- Frequency response utility for continuous-time transfer functions
- Bode plot examples for first-order, second-order, and RLC low-pass systems
- Reusable transfer function utilities
- Step, impulse, and comparison examples for transfer function models
- Reusable continuous-time state-space simulation utilities
- Reusable continuous-time LQR utility
- State-space examples for mass-spring-damper, RLC circuit, and DC motor
- pytest test coverage for implemented models

#### Interactive App

- Streamlit GUI MVP for RC circuit charging, RLC circuit step response, and
  discrete PID motor speed control

#### Documentation and Portfolio Assets

- Added `docs/screenshots` folder
- Selected examples now save plot screenshots for README and portfolio use
