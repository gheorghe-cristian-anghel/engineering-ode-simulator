# New Project Chat Starter

You are helping with the **Engineering Simulation Toolkit** in `D:\engineering-ode-simulator`.

## 1. What This Project Is

This is an educational/prototype/portfolio Python engineering simulation toolkit moving toward professional quality. It demonstrates ODE/PDE modeling, numerical methods, circuits, mechanical systems, control systems, state estimation, MPC, UAV/quadcopter examples, FEM basics, scientific visualization, examples, tests, docs, and a Streamlit UI.

It is not production-grade industrial simulation software, certified control software, flight-ready UAV software, or a replacement for Simulink, ANSYS, COMSOL, autopilot stacks, or specialized numerical tools.

## 2. Current Repo Path

Windows path:

```text
D:\engineering-ode-simulator
```

Git Bash path:

```bash
cd /d/engineering-ode-simulator
```

## 3. Current Technical State

- Package structure is organized around `models/`, `analysis/`, `visualization/`, `examples/`, `tests/`, `docs/`, and `streamlit_app.py`.
- Implemented areas include RC/RL/RLC circuits, cooling, mass-spring-damper, pendulums, control systems, DC motor control, state estimation, MPC, quadcopter examples, 1D/2D PDE examples, finite differences, introductory 1D FEM, Matplotlib helpers, CSV export, docs, screenshots, and Streamlit.
- Current focus is stabilization, documentation, validation, Streamlit reliability, and graph polish.
- Recent fragile area: Streamlit plot rendering and sizing, especially `st.pyplot`, constrained layout, legends, colorbars, and multi-panel plots.

## 4. Current Bug Status

- No known failing tests at handoff time.
- Last handoff verification on 2026-07-10: `651 passed in 16.29s`.
- Recent work fixed Streamlit plot style helpers and figure display behavior.
- Regression-test Streamlit after any UI, plotting, layout, or figure-helper edits.

## 5. Standard Bash Commands

Use Git Bash when possible:

```bash
cd /d/engineering-ode-simulator
source .venv/Scripts/activate
python -m pip install -e ".[dev]"
python -m pytest -v --basetemp=".pytest_tmp_$(date +%s)" -p no:cacheprovider
streamlit run streamlit_app.py
```

The pytest flags avoid common Windows pytest cache/temp cleanup issues. Streamlit recently started at `http://localhost:8501`, but verify the current process before assuming it is still running.

## 6. Installed Codex Skills

Use installed skills where useful:

- `python-testing`
- `developing-with-streamlit`
- `scientific-visualization`
- `python-patterns`
- `codebase-onboarding`
- `code-tour`
- `strategic-compact`
- `delivery-gate`

Use `code-tour` only when creating `.tour` artifacts.

## 7. How I Want Help

- Be surgical and verify before claiming success.
- Prefer tests before behavior changes.
- Keep changes small, readable, and consistent with the existing simple architecture.
- Preserve educational clarity over abstraction.
- For Streamlit work, run or verify `streamlit run streamlit_app.py` when relevant.
- For plot work, preserve scientific readability: clear labels, units where known, colorblind-aware palettes, and no misleading styling.
- For handoff/context work, write compact Markdown that a new chat can use immediately.

## 8. Immediate Next Priorities

1. Fix confirmed bugs while keeping tests green.
2. Continue scientific robustness and validation tests.
3. Improve Streamlit UI/UX without redesigning the app.
4. Polish graph readability, labels, captions, legends, color choices, and layout.
5. Add animations, exports, or report polish only when requested.
6. Final QA: run pytest, smoke-test Streamlit, check representative examples, docs links, screenshots, and claims.

## 9. What Not To Do

- Do not modify numerical solvers unless explicitly requested.
- Do not change code behavior during documentation-only tasks.
- Do not redesign the Streamlit app unless explicitly requested.
- Do not add features by default.
- Do not make production, certification, flight-readiness, or high-fidelity industrial claims.
- Do not overstate scientific fidelity.
- Do not commit automatically.
- Do not delete user changes or unrelated files.

## 10. Response Style Preferences

Be concise, practical, and context-aware. State assumptions when needed, surface tradeoffs briefly, and give exact commands or file paths when useful. Lead with findings or outcomes, avoid unnecessary refactors, and keep explanations compact enough to act on.
