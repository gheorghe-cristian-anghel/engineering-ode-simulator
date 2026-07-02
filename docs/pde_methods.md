# PDE Methods

The PDE examples demonstrate explicit finite-difference methods for heat
diffusion and wave propagation on uniform grids.

## Governing Equations

The 1D heat equation is:

```text
u_t = alpha u_xx
```

The 2D heat equation is:

```text
u_t = alpha (u_xx + u_yy)
```

The 1D wave equation is:

```text
u_tt = c^2 u_xx
```

The repository also includes a 2D wave equation:

```text
u_tt = c^2 (u_xx + u_yy)
```

Heat diffusion smooths temperature gradients and dissipates peaks. Wave
propagation transports displacement energy through the domain and can reflect
from boundaries.

## Assumptions

- Grids are uniform in space and time.
- Material properties `alpha` or wave speed `c` are constant.
- Boundary conditions are idealized.
- Source terms and nonlinear material effects are not included.

## Numerical Method

Heat examples use explicit forward Euler in time and central differences in
space. In 1D:

```text
u_i^{n+1} = u_i^n + r (u_{i+1}^n - 2u_i^n + u_{i-1}^n)
r = alpha dt / dx^2
```

The 1D explicit heat stability condition is `r <= 0.5`. In 2D the repository
checks `rx + ry <= 0.5`.

Wave examples use central differences in time and space. In 1D:

```text
u_i^{n+1} = 2u_i^n - u_i^{n-1}
            + lambda^2 (u_{i+1}^n - 2u_i^n + u_{i-1}^n)
lambda = c dt / dx
```

The 1D wave CFL check is `lambda <= 1`. In 2D the repository checks
`lambda_x^2 + lambda_y^2 <= 1`.

## What the Repository Demonstrates

- 1D heat diffusion from smooth or step-like initial temperature profiles.
- 2D heat diffusion on a rectangular plate.
- 1D wave propagation from Gaussian, sine, or triangular displacement profiles.
- 2D membrane-style wave propagation with fixed boundaries.
- Stability numbers are returned with simulation results and printed by
  example scripts.

## Relevant Files and Examples

- `models/heat_equation_1d.py`
- `models/heat_equation_2d.py`
- `models/wave_equation_1d.py`
- `models/wave_equation_2d.py`
- `examples/run_heat_equation_1d.py`
- `examples/run_heat_equation_2d.py`
- `examples/run_wave_equation_1d.py`
- `examples/run_wave_equation_2d.py`

## Limitations

- Explicit methods require small stable time steps.
- Boundary conditions are simple Dirichlet or Neumann variants, with 2D wave
  currently using fixed Dirichlet boundaries.
- No adaptive grids, implicit solvers, source terms, heterogeneous materials,
  or high-performance sparse PDE solvers are included.
