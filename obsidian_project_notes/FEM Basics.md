---
title: FEM Basics
tags:
  - engineering-simulation-toolkit
  - fem
  - computational-mechanics
---

# FEM Basics

The FEM area currently focuses on an educational 1D axial bar solver.

## 1D Axial Bar

Relevant area:

- `analysis/finite_element_1d.py`
- `examples/run_fem_1d_bar.py`
- Streamlit axial bar FEM demo

## Stiffness Assembly

The module demonstrates:

- uniform mesh generation
- linear element stiffness
- global stiffness matrix assembly
- displacement boundary conditions

## Displacement

The solver computes nodal displacement for a fixed-free bar under load.

## Stress

Post-processing includes:

- element strain
- element stress
- support reaction recovery

## Analytical Validation

The example compares against the analytical fixed-free axial bar solution. Tests verify displacement, stress, strain, and reaction behavior.

## Limitations

- Introductory 1D only.
- No 2D truss yet.
- No beam bending yet.
- No complex materials, contact, nonlinearities, or meshing workflows.

Related: [[Scientific Correctness]], [[Roadmap]].
