---
title: Engineering Simulation Toolkit
tags:
  - engineering-simulation-toolkit
  - project
  - overview
aliases:
  - EST
---

# Engineering Simulation Toolkit

The Engineering Simulation Toolkit is a Python educational/prototype/portfolio project for representative engineering simulations. It began as an ODE simulator and now covers circuits, mechanics, control, state estimation, UAV/quadcopter dynamics, PDEs, numerical methods, FEM basics, plotting, examples, tests, docs, and a Streamlit UI.

## Purpose

- Demonstrate scientific Python engineering workflows.
- Keep equations readable and examples reproducible.
- Show portfolio-quality structure without overclaiming fidelity.
- Move gradually toward professional polish through tests, docs, validation, and UI refinement.

## Scope

Included:

- ODE models and simulation helpers.
- Explicit finite-difference PDE solvers.
- Introductory 1D FEM.
- Control and state-estimation examples.
- Quadcopter educational models and controllers.
- Matplotlib figures and selected animations.
- Streamlit demos for selected simulations.

Out of scope unless explicitly added later:

- Certified control or safety-critical use.
- Flight-ready UAV/autopilot software.
- High-fidelity CFD/aerodynamics.
- Production PDE/FEM solvers.
- Replacement for Simulink, ANSYS, COMSOL, or flight stacks.

## Current Status

- Tests recently passed: `651 passed in 16.29s`.
- Handoff docs were added in recent commit `a38637b Add project handoff documentation`.
- Recent work focused on Streamlit UI/UX, plot helper fixes, and scientific validation.
- Current focus: documentation, stabilization, validation, Streamlit reliability, and graph polish.

## Important Links

- [[Architecture]]
- [[Streamlit App]]
- [[Scientific Correctness]]
- [[Roadmap]]
- [[Known Bugs and Fixes]]
- [[Codex Workflow]]
- [[New Chat Starter]]
