---
title: Visualization and Animations
tags:
  - engineering-simulation-toolkit
  - visualization
  - matplotlib
  - animations
---

# Visualization and Animations

Visualization is a key portfolio surface. The goal is readable engineering figures, not decorative plots.

## Plot Style Helpers

`visualization/plot_style.py` centralizes:

- engineering color cycle
- Streamlit figure sizes
- axes formatting
- heatmap/colorbar helpers
- equal-axis limits
- legend placement
- high-DPI save helper
- Streamlit figure finalization/display

Recent helper fixes make this file important for regression checks.

## Graph Improvements

Priorities:

- clear titles and axis labels
- units where known
- readable legends
- colorblind-aware colors
- stable figure sizing in Streamlit
- avoid misleading scales
- avoid unnecessary chart decoration

## Animation Roadmap

Animations should reuse existing simulation outputs. Do not change solvers just to animate.

Good first animations:

- inverted pendulum open-loop vs LQR
- quadcopter waypoint following
- quadcopter circular trajectory
- 1D wave propagation
- 2D heat diffusion snapshots
- 2D wave membrane snapshots

## Existing Animation Areas

- `visualization/inverted_pendulum_animation.py`
- `visualization/quadcopter_animation.py`
- example scripts for selected pendulum/quadcopter animations

## Scientific Visualization Rules

- Make the plotted quantity inspectable.
- Prefer perceptually reasonable colormaps.
- Keep heatmap colorbars labeled.
- Test multi-panel layouts in Streamlit.
- Keep screenshots reproducible where possible.

Related: [[Streamlit App]], [[Scientific Correctness]], [[UAV Quadcopter Module]], [[PDE Solvers]].
