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
- CSV export utilities

`analysis/transfer_function.py` provides reusable continuous-time transfer
function infrastructure, including validated coefficient storage, SciPy signal
conversion, common low-pass helpers, and step/impulse response simulation.

`analysis/pid_tuning.py` provides reusable PID tuning infrastructure for
educational examples built on the discrete PID motor simulation.

`analysis/lqr.py` provides reusable continuous-time Linear Quadratic
Regulator infrastructure based on the continuous algebraic Riccati equation.

`analysis/kalman_filter.py` provides reusable discrete-time linear Kalman
filter infrastructure for estimating hidden states from noisy measurements.

### visualization/

Reusable visualization helpers for plotting and animation code that should not
live in the physics model modules. `visualization/inverted_pendulum_animation.py`
animates cart-pole trajectories using Matplotlib while preserving the state and
angle convention from `models/inverted_pendulum.py`.

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
