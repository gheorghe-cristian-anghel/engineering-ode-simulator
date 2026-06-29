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
derivative-on-measurement.

### analysis/

Reusable analysis and output tools independent of any one model.

Current example:

- step response metrics
- parameter sweep utilities
- frequency response and Bode plot utilities
- state-space simulation utilities
- CSV export utilities

### examples/

Runnable scripts that demonstrate models, print important parameters, and
generate plots. Selected examples can also export simulation arrays to CSV
files under `outputs/`.

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
