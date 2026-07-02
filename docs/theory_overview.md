# Theory Overview

This project is an educational engineering simulation toolkit. It connects
governing equations, numerical methods, and compact examples for circuits,
mechanics, control systems, estimation, PDEs, FEM, and UAV dynamics.

## Algorithm Idea

Most examples follow the same pattern:

```text
continuous model -> discretization or ODE solver -> simulation arrays -> metrics/plots
```

ODE models are usually integrated with `scipy.integrate.solve_ivp`. PDE and
finite-difference examples use explicit grid-based updates. Control and
estimation examples use small reusable analysis utilities around state-space
models, filters, and optimization.

## Assumptions

- Models are simplified for clarity and reproducibility.
- Parameters are idealized and usually constant.
- Noise, disturbances, and constraints are included only where an example
  explicitly models them.
- The repository demonstrates engineering intuition, not certified design or
  production-grade accuracy.

## Numerical Methods

- Continuous ODEs are solved with SciPy time integration.
- Heat and wave PDEs use explicit finite differences on uniform grids.
- FEM basics use linear 1D axial bar elements and dense matrix assembly.
- Estimators use discrete-time prediction/update recursions.
- Control examples use PID updates, Riccati-based LQR, or finite-horizon MPC.

## What the Repository Demonstrates

- How physical equations become executable simulations.
- How stability, convergence, and error metrics affect numerical behavior.
- How feedback controllers and estimators wrap around dynamic models.
- How simplified UAV and structural examples can be used for portfolio-level
  engineering discussion.

## Relevant Files and Examples

- `models/`: physical ODE/PDE model implementations.
- `analysis/`: reusable numerical, control, estimation, MPC, and FEM utilities.
- `examples/`: runnable scripts that generate plots and printed summaries.
- `tests/`: focused checks for formulas, stability, behavior, and dimensions.
- `docs/equations.md`: equation reference for several core models.

## Limitations

- The project is educational and intentionally compact.
- It does not replace specialized tools such as Simulink, ANSYS, COMSOL, or
  real flight-control stacks.
- Many examples use ideal boundary conditions, known parameters, and modest
  problem sizes.
- Numerical accuracy depends on time step, grid spacing, model assumptions,
  and solver settings.
