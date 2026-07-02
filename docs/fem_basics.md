# FEM Basics

The FEM example is a compact 1D axial bar solver. It demonstrates the main
finite element workflow without extending into beams, plates, shells, contact,
or nonlinear materials.

## Governing Equation

For a uniform axial bar with displacement `u(x)`, Young's modulus `E`, area
`A`, and axial force `F`, the static equilibrium relation is:

```text
d/dx(E A du/dx) + b = 0
```

The repository example focuses on a fixed-free bar with a nodal end load and
no distributed body force.

## Element Stiffness Matrix

A two-node linear axial element of length `Le` has stiffness:

```text
ke = (E A / Le) [[ 1, -1],
                 [-1,  1]]
```

The element strain is constant:

```text
epsilon = (u_right - u_left) / Le
sigma = E epsilon
```

## Assumptions

- Small displacement, linear elastic axial deformation.
- Constant `E` and `A` over the bar.
- Linear two-node elements on a uniform 1D mesh.
- Static loading only.

## Numerical Method

The solver creates a uniform mesh, computes each element stiffness matrix, and
adds local stiffness contributions into the global matrix:

```text
K u = F
```

Dirichlet displacement boundary conditions are applied by modifying copies of
`K` and `F`. Reactions are recovered afterward from the original system:

```text
reaction = K_original u - F_original
```

## What the Repository Demonstrates

- Uniform 1D mesh generation.
- Element stiffness construction.
- Global stiffness assembly.
- Fixed displacement boundary condition at the left end.
- Nodal displacement solution under a right-end force.
- Reaction force, strain, and stress recovery.
- Analytical comparison with:

```text
u_tip = F L / (E A)
```

## Relevant Files and Examples

- `analysis/finite_element_1d.py`
- `examples/run_fem_1d_bar.py`
- `tests/test_finite_element_1d.py`
- `examples/fem_1d_bar.png`

## Limitations

- Only a 1D axial bar is implemented.
- No beam bending, trusses, 2D/3D elements, nonlinear materials, buckling, or
  dynamic FEM is included.
- Dense matrices are acceptable for the small educational example but are not
  intended for large structural models.
