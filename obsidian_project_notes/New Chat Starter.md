---
title: New Chat Starter
tags:
  - engineering-simulation-toolkit
  - handoff
  - chat-starter
---

# New Chat Starter

Paste this into a new ChatGPT Project chat:

```text
You are helping with the Engineering Simulation Toolkit in D:\engineering-ode-simulator.

This is an educational/prototype/portfolio Python engineering simulation toolkit moving toward professional quality. It demonstrates ODE/PDE modeling, control systems, numerical methods, state estimation, UAV/quadcopter examples, FEM basics, scientific visualization, tests, examples, docs, and a Streamlit UI.

Important boundaries:
- Do not modify numerical solvers unless explicitly requested.
- Do not redesign Streamlit unless explicitly requested.
- Do not add features during documentation or cleanup tasks.
- Do not make production, certification, flight-ready, or high-fidelity industrial claims.
- Do not commit automatically.

Architecture:
- models/: physical systems, ODE/PDE models, simulation functions, validation helpers.
- analysis/: metrics, controllers, estimators, MPC, FEM, sweeps, export helpers.
- visualization/: Matplotlib styling, plot helpers, animations.
- examples/: runnable demos and generated plots.
- tests/: pytest coverage for numerical and engineering behavior.
- docs/: theory, architecture, screenshots, future ideas.
- streamlit_app.py: interactive UI over selected simulations.

Standard Git Bash workflow:
cd /d/engineering-ode-simulator
source .venv/Scripts/activate
python -m pytest -v --basetemp=".pytest_tmp_$(date +%s)" -p no:cacheprovider
streamlit run streamlit_app.py

Recent verified status:
651 tests passed in 16.29s during handoff verification.

Current focus:
stabilization, documentation, scientific validation, Streamlit reliability, graph polish, animations/exports only when requested.

Use installed Codex skills where useful: codebase-onboarding, strategic-compact, python-patterns, developing-with-streamlit, scientific-visualization, python-testing, and code-tour only for .tour artifacts.
```

Related notes:

- [[00 Start Here]]
- [[Engineering Simulation Toolkit]]
- [[Architecture]]
- [[Codex Workflow]]
