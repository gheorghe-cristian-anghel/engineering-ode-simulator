---
title: Known Bugs and Fixes
tags:
  - engineering-simulation-toolkit
  - bugs
  - fixes
---

# Known Bugs and Fixes

## Recent Streamlit Helper Bug

Recent commits fixed Streamlit plot style helpers and plot sizing:

- `6058797 Fix Streamlit plot style helpers`
- `2e8f0f1 Improve Streamlit UI and UX`
- `1e43458 Improve Streamlit plot sizing`

The sensitive area is interaction between `streamlit_app.py` and `visualization/plot_style.py`.

## How It Was Diagnosed

The issue was treated as a UI/rendering helper problem rather than a numerical solver problem. The relevant checks are:

- Does the app render figures without helper errors?
- Does `st.pyplot` receive finalized figures?
- Are figures closed after display?
- Do legends/colorbars/constrained layout behave in Streamlit?
- Do plot helper tests pass?

## What To Check If It Returns

Inspect:

- `visualization/plot_style.py`
- `streamlit_app.py`
- tests around plot style imports/runtime/helpers
- recent changes to figure creation or display

Run:

```bash
python -m pytest -v --basetemp=".pytest_tmp_$(date +%s)" -p no:cacheprovider
streamlit run streamlit_app.py
```

Manually smoke-test:

- RC/RLC plots
- DC motor PID plots
- UAV tabs
- state-estimation tabs
- 2D heat/wave plots
- FEM plot

## Current Known Fragile Areas

- Streamlit plot rendering and sizing.
- Long/high-resolution PDE demos.
- Legends outside axes.
- Colorbars in multi-panel layouts.
- Quadcopter limitation language.
- Explicit PDE stability settings.

Related: [[Streamlit App]], [[Visualization and Animations]], [[Scientific Correctness]].
