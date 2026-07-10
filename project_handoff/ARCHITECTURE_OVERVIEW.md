# Architecture Overview

## Purpose

The repository is a compact educational engineering simulation toolkit. The architecture favors readable equations, small modules, runnable examples, tests, and a Streamlit UI over heavy framework structure.

## Directory Roles

### `models/`

Holds physical systems and numerical simulation functions. Most modules define parameters, validation, ODE/PDE right-hand sides, simulation helpers, and formulas. Examples include circuits, DC motor, pendulum, inverted pendulum, heat/wave equations, and quadcopter models.

### `analysis/`

Holds reusable engineering analysis tools that are not tied to a single presentation. This includes step metrics, finite differences, FEM basics, parameter sweeps, frequency response, transfer functions, state space, LQR, MPC, PID tuning, disturbance metrics, Kalman/EKF/UKF/particle filters, quadcopter controllers, obstacle avoidance, and CSV export helpers.

### `visualization/`

Holds shared Matplotlib presentation helpers and animation utilities. `plot_style.py` centralizes engineering plot defaults, Streamlit figure sizing, axes formatting, colorbars, legends, color limits, and figure display. Animation modules cover inverted pendulum and quadcopter trajectory visualization.

### `examples/`

Runnable scripts that import `models/`, `analysis/`, and `visualization/`, print parameters/metrics, and generate plots or animations. Selected scripts save PNGs for README/docs/portfolio use and CSV outputs under `outputs/`.

### `tests/`

Pytest tests for implemented behavior. Tests check formulas, initial conditions, expected trends, stability checks, convergence behavior, estimator behavior, controller behavior, exports, and selected visualization helper runtime behavior.

### `docs/`

Longer written context: architecture, equations, theory overview, control systems, numerical methods, PDE methods, FEM basics, UAV models, state estimation, screenshots, and future ideas.

### `streamlit_app.py`

Single-file Streamlit browser UI. It imports existing model/analysis helpers, exposes controls, computes results, renders metrics, and displays Matplotlib plots with `st.pyplot`. It should remain a presentation layer, not a place for core solver logic.

## Data Flow

```text
User or example selects parameters
  -> model module simulates states/time histories
  -> analysis module computes metrics, controller signals, estimator outputs, sweeps, or exports
  -> visualization helpers format Matplotlib figures
  -> examples save/display plots, or Streamlit displays them interactively
  -> tests verify expected behavior and regression boundaries
```

For Streamlit:

```text
sidebar/tabs/widgets
  -> existing model and analysis functions
  -> NumPy arrays and metric dictionaries
  -> Matplotlib figures using visualization.plot_style helpers
  -> st.pyplot plus captions/tables/metrics
```

## Recommended Future Architecture

- Keep solver and scientific logic in `models/` and `analysis/`.
- Keep `streamlit_app.py` as UI orchestration; gradually extract repeated UI helpers only when duplication becomes painful.
- Continue using focused tests before changing model or analysis behavior.
- Add shared validation utilities only where repeated checks are noisy.
- Keep examples runnable and close to the code they demonstrate.
- Consider a structured gallery asset generation flow if screenshots/animations grow.
- Consider packaging/release polish after tests, docs, and Streamlit are stable.
