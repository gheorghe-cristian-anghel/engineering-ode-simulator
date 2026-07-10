---
title: PDE Solvers
tags:
  - engineering-simulation-toolkit
  - pde
  - numerical-methods
---

# PDE Solvers

The PDE modules demonstrate explicit finite-difference methods for educational heat and wave simulations.

## 1D Heat

`models/heat_equation_1d.py` includes:

- explicit finite-difference solver
- stability-number checks
- fixed-temperature and insulated-end boundary options
- educational initial conditions

## 2D Heat

`models/heat_equation_2d.py` includes:

- explicit rectangular-grid heat diffusion
- 2D stability checks
- stored temperature snapshots for heatmap display
- fixed or optional insulated boundaries

## 1D Wave

`models/wave_equation_1d.py` includes:

- explicit central finite-difference scheme
- CFL checks
- initial displacement and velocity helpers
- fixed/free boundary options

## 2D Wave

`models/wave_equation_2d.py` includes:

- membrane-style propagation
- 2D CFL checks
- Gaussian, ring, and sine initial shapes
- stored displacement snapshots

## Stability / CFL

Do not bypass stability checks. Explicit methods are educational and become unstable outside safe time-step/grid settings.

## Possible Animations

Good candidates:

- 1D heat profile over time
- 2D heat heatmap snapshots
- 1D wave propagation
- 2D membrane displacement heatmap

Keep animations lightweight and optional. See [[Visualization and Animations]].

## Limitations

- Not a production PDE solver.
- No adaptive meshing.
- No implicit methods currently.
- No high-performance sparse/distributed backend.

Related: [[Scientific Correctness]], [[Roadmap]].
