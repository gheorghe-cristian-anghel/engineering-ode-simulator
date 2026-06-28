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
- PI motor speed control with PID-compatible API
- Discrete PID controller with anti-windup and derivative-on-measurement

#### DC Motor and PI Control

- DC motor open-loop speed response
- Closed-loop PI motor speed control with voltage saturation
- Load disturbance response example for PI motor speed control
- Discrete PID motor speed-control example

#### Analysis Utilities

- Reusable step response metrics
- PI gain sweep utility for motor speed-control tuning
- PI gain sweep example with speed and voltage comparison plots
- pytest test coverage for implemented models
