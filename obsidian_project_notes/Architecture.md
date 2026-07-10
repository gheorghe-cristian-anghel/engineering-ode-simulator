---
title: Architecture
tags:
  - engineering-simulation-toolkit
  - architecture
---

# Architecture

The repository uses a simple layered structure. Core equations live outside the UI, examples demonstrate usage, and tests verify behavior.

## Models Layer

`models/` contains physical systems, ODE/PDE right-hand sides, simulation functions, validation helpers, and formulas.

Representative modules:

- circuits: RC, RL, RLC
- thermal/mechanical: cooling, mass-spring-damper, pendulum
- control plants: DC motor, first/second-order systems, inverted pendulum
- UAV: altitude, attitude, simplified 6-DOF
- PDEs: 1D/2D heat and wave equations

## Analysis Layer

`analysis/` contains reusable engineering tools:

- step-response metrics
- finite differences
- FEM 1D axial bar helpers
- parameter sweeps
- frequency response, transfer functions, state space
- LQR, MPC, PID tuning
- Kalman, EKF, UKF, Particle Filter
- quadcopter controllers, waypoint tracking, obstacle avoidance
- CSV export helpers

## Visualization Layer

`visualization/` contains reusable Matplotlib helpers:

- `plot_style.py` for shared styling, figure sizing, colorbars, legends, and Streamlit display
- inverted pendulum animation helpers
- quadcopter 3D animation helpers

See [[Visualization and Animations]].

## Examples Layer

`examples/` contains runnable scripts that import `models/`, `analysis/`, and `visualization/`, print parameters or metrics, and generate plots, animations, or CSV outputs.

## Tests Layer

`tests/` contains pytest tests for model behavior, validation, numerical formulas, controller behavior, state estimation, exports, and visualization helpers.

## Streamlit App Layer

`streamlit_app.py` is the interactive UI. It should orchestrate controls, call existing simulation helpers, compute/display metrics, and render plots. It should not become the source of solver logic.

See [[Streamlit App]].

## Documentation Layer

`docs/`, `README.md`, and handoff files explain theory, architecture, limitations, screenshots, and future plans.

## Data Flow

```text
parameters
  -> models/ simulations
  -> analysis/ metrics, estimators, controllers, sweeps
  -> visualization/ plots and animations
  -> examples/ or streamlit_app.py
  -> tests/ regression and scientific checks
```

## Architecture Rule

Keep solver/scientific behavior in `models/` or `analysis/`; keep Streamlit as presentation and orchestration.
