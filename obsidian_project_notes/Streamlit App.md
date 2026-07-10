---
title: Streamlit App
tags:
  - engineering-simulation-toolkit
  - streamlit
  - ui
---

# Streamlit App

The Streamlit UI lives in `streamlit_app.py`. It is a single-file browser interface over selected simulations.

## Current App Structure

The app uses shared render helpers for:

- page headers and intros
- feature cards
- metric rows/grids
- control panels
- assumptions expanders
- parameter summaries
- safe dataframes
- figure display

Main routing uses sidebar controls:

- `st.sidebar.radio("Section", ...)`
- `st.sidebar.selectbox("Demo", ...)`

## Pages / Demo Areas

Current areas include:

- Home and About
- RC/RLC circuits
- DC motor PID control
- state estimation: KF, UKF, optional PF
- UAV/quadcopter: altitude, trajectory, waypoint, obstacle avoidance
- PDEs: 1D/2D heat, 1D/2D wave
- finite-difference convergence
- 1D axial bar FEM

Related notes:

- [[UAV Quadcopter Module]]
- [[State Estimation Module]]
- [[PDE Solvers]]
- [[FEM Basics]]

## UI / UX Goals

- Keep the current structure recognizable.
- Improve readability, default parameter choices, captions, and grouping.
- Avoid expensive default simulations.
- Keep limitations visible but concise.
- Do not redesign the app unless explicitly requested.

## Plotting Helper Architecture

Streamlit plots should use `visualization/plot_style.py` where possible:

- `create_streamlit_figure`
- `create_streamlit_subplots`
- `format_engineering_axes`
- `add_clean_colorbar`
- `place_legend_outside`
- `finalize_streamlit_figure`
- `display_streamlit_figure`

See [[Visualization and Animations]].

## Current Known Issues

- Plot display paths are regression-sensitive after recent helper fixes.
- Watch figure sizing, legends outside axes, colorbars, constrained layout, and `st.pyplot(width="stretch")`.
- Long/high-resolution PDE demos can become slow if defaults are too ambitious.

See [[Known Bugs and Fixes]].
