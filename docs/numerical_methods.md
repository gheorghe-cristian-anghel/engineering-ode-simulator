# Numerical Methods

The numerical-methods examples focus on finite differences, convergence,
stability, and practical error metrics.

## Governing Ideas

Finite differences approximate derivatives from nearby grid samples. For a
uniform grid spacing `dx`:

```text
forward:  f'(x_i) ~= (f_{i+1} - f_i) / dx
backward: f'(x_i) ~= (f_i - f_{i-1}) / dx
central:  f'(x_i) ~= (f_{i+1} - f_{i-1}) / (2 dx)
second:   f''(x_i) ~= (f_{i+1} - 2 f_i + f_{i-1}) / dx^2
```

Convergence order describes how error changes as grid spacing decreases:

```text
error ~= C dx^p
```

The estimated order `p` is computed from the slope of `log(error)` versus
`log(dx)`.

## Assumptions

- Grids are one-dimensional and uniform for derivative utilities.
- Test functions are smooth enough for the expected convergence behavior.
- Boundary derivative values use simple one-sided or copied approximations.
- PDE stability checks apply to the explicit schemes implemented here.

## Numerical Method

- Finite-difference derivative arrays are computed directly from neighboring
  values.
- Dense derivative matrices are available for small educational examples.
- Maximum absolute error and RMS error compare numerical results to exact
  reference values.
- Explicit heat and wave solvers expose stability numbers such as `r`,
  `rx + ry`, CFL, and `lambda_x^2 + lambda_y^2`.

## What the Repository Demonstrates

- Forward, backward, central, and second-derivative finite differences.
- Empirical convergence order from grid refinement.
- Error metrics for numerical-versus-analytical comparison.
- Stability constraints for explicit heat and wave equation solvers.
- How grid spacing and time step affect numerical reliability.

## Relevant Files and Examples

- `analysis/finite_difference.py`
- `models/heat_equation_1d.py`
- `models/heat_equation_2d.py`
- `models/wave_equation_1d.py`
- `models/wave_equation_2d.py`
- `examples/run_finite_difference_derivatives.py`
- `examples/run_finite_difference_convergence.py`
- `examples/run_heat_equation_1d.py`
- `examples/run_wave_equation_1d.py`

## Limitations

- Finite-difference utilities are intentionally small and 1D.
- Dense matrices are not suitable for large-scale PDE work.
- Stability checks do not guarantee physical model fidelity; they only check
  constraints for the implemented explicit schemes.
- No adaptive time stepping, adaptive mesh refinement, implicit PDE solvers, or
  spectral methods are included.
