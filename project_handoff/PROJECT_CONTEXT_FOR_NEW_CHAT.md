# Paste-Ready Project Context For New Chat

You are helping with the **Engineering Simulation Toolkit**.

This is an educational/prototype/portfolio Python engineering simulation toolkit moving toward professional quality. It is not production-grade industrial simulation software, certified control software, flight-ready UAV software, or a replacement for Simulink, ANSYS, COMSOL, autopilot stacks, or specialized numerical tools.

## Project Goal

Build and polish a readable engineering simulation toolkit that demonstrates:

- ODE/PDE modeling and numerical methods
- circuits, mechanical systems, control systems, state estimation, MPC, UAV/quadcopter examples, FEM basics
- scientific Python structure with tests, examples, plots, docs, and a Streamlit UI
- portfolio-quality explanations and visual outputs without overclaiming scientific scope

## Current Technical Focus

Current focus is handoff, stabilization, documentation, validation, Streamlit reliability, and graph polish. Do not start new solver work unless explicitly asked.

## Repo Path

```bash
cd /d/engineering-ode-simulator
```

Windows path:

```text
D:\engineering-ode-simulator
```

## Standard Bash Workflow

Use Git Bash when possible:

```bash
cd /d/engineering-ode-simulator
source .venv/Scripts/activate
python -m pip install -e ".[dev]"
python -m pytest -v --basetemp=".pytest_tmp_$(date +%s)" -p no:cacheprovider
streamlit run streamlit_app.py
```

## Testing Command

Use this command for full local verification:

```bash
python -m pytest -v --basetemp=".pytest_tmp_$(date +%s)" -p no:cacheprovider
```

The `--basetemp` and `-p no:cacheprovider` flags avoid common Windows pytest cache/temp cleanup problems.

## Streamlit Command

```bash
streamlit run streamlit_app.py
```

Recent logs showed Streamlit starting at `http://localhost:8501`; verify the current process before assuming it is still running.

## Current Architecture

- `models/`: physical systems, ODE/PDE models, simulation functions, and helper formulas.
- `analysis/`: reusable metrics, controllers, estimators, MPC, FEM, parameter sweeps, export helpers, and control-analysis utilities.
- `visualization/`: Matplotlib styling, formatting, colorbar, figure, and animation helpers.
- `examples/`: runnable scripts that demonstrate models and generate plots or outputs.
- `tests/`: pytest coverage for implemented numerical and engineering behavior.
- `docs/`: architecture, equations, theory notes, screenshots, and future ideas.
- `streamlit_app.py`: single-file interactive browser UI over selected existing simulations.

Data flow:

```text
models/ simulation arrays
  -> analysis/ metrics, controllers, estimators, sweeps, exports
  -> visualization/ and examples/ plots
  -> streamlit_app.py interactive controls and st.pyplot display
  -> tests/ verify behavior and regressions
```

## Implemented Features

Implemented areas include RC/RL/RLC circuits, Newton cooling, mass-spring-damper, nonlinear pendulum, inverted pendulum with LQR, first/second-order control systems, DC motor PI/discrete PID control, load disturbance examples, step metrics, parameter sweeps, frequency response, transfer functions, state space, Kalman/EKF/UKF/particle filters, linear MPC, quadcopter altitude/attitude/6-DOF/tracking/waypoints/obstacle avoidance, heat and wave equations in 1D/2D, finite differences, 1D axial bar FEM, Matplotlib animations, CSV export, tests, docs, screenshots, and a Streamlit UI.

## Current Bug Status

- No known failing tests at the time this handoff was prepared.
- Handoff verification on 2026-07-10: `651 passed in 16.29s`.
- Recent work focused on Streamlit plot style helpers and plot sizing/readability.
- Streamlit helper issues around `st.pyplot` sizing and figure display were recently fixed; treat this area as worth regression testing after UI edits.

## Recent Fixes

Visible recent commits:

- `6058797 Fix Streamlit plot style helpers`
- `2e8f0f1 Improve Streamlit UI and UX`
- `a590c7d Add scientific validation tests`
- `92a67b7 Improve scientific validation and robustness`
- `ac8d983 Clean up Streamlit helpers`

## Next Planned Phases

1. Fix any current bugs and keep tests green.
2. Continue scientific robustness and validation tests.
3. Improve Streamlit UI/UX without redesigning the app.
4. Polish graph readability, color choices, labels, captions, and layout.
5. Add animations and export/report polish where requested.
6. Do final QA for docs, examples, tests, and portfolio presentation.
7. Consider advanced features later: 2D truss FEM, beam bending FEM, packaging/release polish.

## What Not To Do

- Do not modify numerical solvers unless explicitly requested.
- Do not change code behavior during documentation-only tasks.
- Do not redesign the app unless explicitly requested.
- Do not add features by default.
- Do not make production, certification, or flight-readiness claims.
- Do not overstate scientific fidelity.
- Do not commit automatically.
- Do not delete user changes or unrelated files.

## Preferred Assistant / Codex Behavior

- Be surgical and verify before claiming success.
- Use installed Codex skills where useful: `python-testing`, `developing-with-streamlit`, `scientific-visualization`, `python-patterns`, `codebase-onboarding`, `code-tour`, `strategic-compact`, and `delivery-gate`.
- Prefer tests before behavior changes.
- Keep changes small, readable, and consistent with the existing simple architecture.
- Preserve educational clarity over abstraction.
- For Streamlit work, run or verify `streamlit run streamlit_app.py` when relevant.
- For plot work, preserve scientific readability: clear labels, units where known, colorblind-aware palettes, no misleading styling.
- For handoff/context work, write compact Markdown that a new chat can use immediately.
