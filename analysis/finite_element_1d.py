"""Reusable one-dimensional finite element utilities for axial bars."""

import numpy as np


def create_uniform_bar_mesh(length=1.0, num_elements=10):
    """Return node coordinates and two-node element connectivity.

    Parameters
    ----------
    length : float, optional
        Total bar length.
    num_elements : int, optional
        Number of linear finite elements.

    Returns
    -------
    tuple
        ``(nodes, elements)`` where ``nodes`` has shape ``(num_nodes,)`` and
        ``elements`` has shape ``(num_elements, 2)``.
    """
    length = float(length)
    num_elements = int(num_elements)

    if length <= 0:
        raise ValueError("length must be positive")

    if num_elements < 1:
        raise ValueError("num_elements must be at least 1")

    nodes = np.linspace(0.0, length, num_elements + 1)
    elements = np.column_stack(
        (
            np.arange(num_elements, dtype=int),
            np.arange(1, num_elements + 1, dtype=int),
        )
    )

    return nodes, elements


def bar_element_stiffness(E, A, length):
    """Return the 2x2 stiffness matrix for a linear axial bar element."""
    E, A, length = _validate_bar_parameters(E, A, length)

    return (E * A / length) * np.array(
        [
            [1.0, -1.0],
            [-1.0, 1.0],
        ]
    )


def assemble_global_stiffness(nodes, elements, E, A):
    """Assemble and return the global stiffness matrix for a 1D bar mesh."""
    nodes, elements = _validate_mesh(nodes, elements)
    E, A = _validate_material_area(E, A)

    num_nodes = len(nodes)
    stiffness = np.zeros((num_nodes, num_nodes), dtype=float)

    for element in elements:
        left_node, right_node = element
        element_length = nodes[right_node] - nodes[left_node]

        if element_length <= 0:
            raise ValueError("element lengths must be positive")

        element_stiffness = bar_element_stiffness(E, A, element_length)
        dofs = np.array([left_node, right_node])

        for local_row, global_row in enumerate(dofs):
            for local_col, global_col in enumerate(dofs):
                stiffness[global_row, global_col] += element_stiffness[
                    local_row,
                    local_col,
                ]

    return stiffness


def apply_dirichlet_boundary_conditions(K, F, fixed_dofs, fixed_values=None):
    """Return copies of ``K`` and ``F`` with prescribed displacements applied.

    The input arrays are not mutated. Reactions should be computed later from
    the original, unmodified global stiffness matrix and force vector.
    """
    K, F = _validate_linear_system(K, F)
    fixed_dofs = _validate_fixed_dofs(fixed_dofs, len(F))
    fixed_values = _validate_fixed_values(fixed_values, len(fixed_dofs))

    K_modified = K.copy()
    F_modified = F.copy()
    K_original = K.copy()

    for dof, value in zip(fixed_dofs, fixed_values):
        F_modified -= K_original[:, dof] * value

    for dof, value in zip(fixed_dofs, fixed_values):
        K_modified[dof, :] = 0.0
        K_modified[:, dof] = 0.0
        K_modified[dof, dof] = 1.0
        F_modified[dof] = value

    return K_modified, F_modified


def solve_displacements(K, F, fixed_dofs=None, fixed_values=None):
    """Solve the global finite element system for nodal displacements."""
    K, F = _validate_linear_system(K, F)

    if fixed_dofs is not None:
        K, F = apply_dirichlet_boundary_conditions(
            K,
            F,
            fixed_dofs,
            fixed_values,
        )

    return np.linalg.solve(K, F)


def compute_element_strains(nodes, elements, displacements):
    """Return the constant strain in each linear axial bar element."""
    nodes, elements = _validate_mesh(nodes, elements)
    displacements = _validate_displacements(displacements, len(nodes))

    strains = np.empty(len(elements), dtype=float)

    for element_index, element in enumerate(elements):
        left_node, right_node = element
        element_length = nodes[right_node] - nodes[left_node]

        if element_length <= 0:
            raise ValueError("element lengths must be positive")

        strains[element_index] = (
            displacements[right_node] - displacements[left_node]
        ) / element_length

    return strains


def compute_element_stresses(strains, E):
    """Return axial stress values from strain and Young's modulus."""
    E = float(E)
    strains = np.asarray(strains, dtype=float)

    if E <= 0:
        raise ValueError("E must be positive")

    if strains.ndim != 1:
        raise ValueError("strains must be one-dimensional")

    if not np.all(np.isfinite(strains)):
        raise ValueError("strains must contain only finite values")

    return E * strains


def compute_reaction_forces(K_original, displacements, F_original):
    """Return reaction forces from the original system ``K*u - F``."""
    K_original, F_original = _validate_linear_system(K_original, F_original)
    displacements = _validate_displacements(displacements, len(F_original))

    return K_original @ displacements - F_original


def analytical_tip_displacement_axial_bar(force, length, E, A):
    """Return the fixed-free axial bar tip displacement ``F*L/(E*A)``."""
    force = float(force)
    E, A, length = _validate_bar_parameters(E, A, length)

    if not np.isfinite(force):
        raise ValueError("force must be finite")

    return force * length / (E * A)


def simulate_axial_bar_fem(
    length=1.0,
    E=200e9,
    A=1e-4,
    force=1000.0,
    num_elements=10,
    fixed_left=True,
):
    """Solve a fixed-free 1D axial bar with a nodal force at the right end."""
    length = float(length)
    E, A = _validate_material_area(E, A)
    force = float(force)

    if length <= 0:
        raise ValueError("length must be positive")

    if not np.isfinite(force):
        raise ValueError("force must be finite")

    nodes, elements = create_uniform_bar_mesh(length, num_elements)
    stiffness = assemble_global_stiffness(nodes, elements, E, A)

    force_vector = np.zeros(len(nodes), dtype=float)
    force_vector[-1] = force

    fixed_dofs = [0] if fixed_left else None
    fixed_values = [0.0] if fixed_left else None
    displacements = solve_displacements(
        stiffness,
        force_vector,
        fixed_dofs=fixed_dofs,
        fixed_values=fixed_values,
    )

    strains = compute_element_strains(nodes, elements, displacements)
    stresses = compute_element_stresses(strains, E)
    reactions = compute_reaction_forces(stiffness, displacements, force_vector)
    analytical_tip = analytical_tip_displacement_axial_bar(force, length, E, A)
    tip_displacement = float(displacements[-1])

    if analytical_tip == 0.0:
        relative_tip_error = abs(tip_displacement - analytical_tip)
    else:
        relative_tip_error = abs(
            (tip_displacement - analytical_tip) / analytical_tip
        )

    return {
        "nodes": nodes,
        "elements": elements,
        "K": stiffness,
        "F": force_vector,
        "displacements": displacements,
        "strains": strains,
        "stresses": stresses,
        "reactions": reactions,
        "analytical_tip_displacement": analytical_tip,
        "tip_displacement": tip_displacement,
        "relative_tip_error": float(relative_tip_error),
        "parameters": {
            "length": length,
            "E": E,
            "A": A,
            "force": force,
            "num_elements": int(num_elements),
            "fixed_left": bool(fixed_left),
        },
    }


def _validate_material_area(E, A):
    """Return validated Young's modulus and cross-sectional area."""
    E = float(E)
    A = float(A)

    if E <= 0:
        raise ValueError("E must be positive")

    if A <= 0:
        raise ValueError("A must be positive")

    if not np.isfinite([E, A]).all():
        raise ValueError("E and A must be finite")

    return E, A


def _validate_bar_parameters(E, A, length):
    """Return validated scalar bar parameters."""
    E, A = _validate_material_area(E, A)
    length = float(length)

    if length <= 0:
        raise ValueError("length must be positive")

    if not np.isfinite(length):
        raise ValueError("length must be finite")

    return E, A, length


def _validate_mesh(nodes, elements):
    """Return validated node coordinates and element connectivity."""
    nodes = np.asarray(nodes, dtype=float)
    raw_elements = np.asarray(elements)

    if nodes.ndim != 1:
        raise ValueError("nodes must be one-dimensional")

    if len(nodes) < 2:
        raise ValueError("nodes must contain at least two points")

    if not np.all(np.isfinite(nodes)):
        raise ValueError("nodes must contain only finite values")

    if raw_elements.ndim != 2 or raw_elements.shape[1] != 2:
        raise ValueError("elements must have shape (num_elements, 2)")

    if len(raw_elements) < 1:
        raise ValueError("elements must contain at least one element")

    if not np.all(np.isfinite(raw_elements)):
        raise ValueError("elements must contain only finite values")

    if not np.all(raw_elements == np.floor(raw_elements)):
        raise ValueError("element indices must be integers")

    elements = raw_elements.astype(int)

    if np.any(elements < 0) or np.any(elements >= len(nodes)):
        raise ValueError("element indices must refer to valid nodes")

    if np.any(elements[:, 0] == elements[:, 1]):
        raise ValueError("elements must connect two distinct nodes")

    return nodes, elements


def _validate_linear_system(K, F):
    """Return validated stiffness matrix and force vector copies."""
    K = np.asarray(K, dtype=float)
    F = np.asarray(F, dtype=float)

    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError("K must be a square matrix")

    if F.ndim != 1:
        raise ValueError("F must be one-dimensional")

    if K.shape[0] != len(F):
        raise ValueError("K row count must match F length")

    if not np.all(np.isfinite(K)):
        raise ValueError("K must contain only finite values")

    if not np.all(np.isfinite(F)):
        raise ValueError("F must contain only finite values")

    return K, F


def _validate_fixed_dofs(fixed_dofs, num_dofs):
    """Return validated constrained degree-of-freedom indices."""
    raw_dofs = np.asarray(fixed_dofs)

    if raw_dofs.ndim == 0:
        raw_dofs = raw_dofs.reshape(1)

    if raw_dofs.ndim != 1:
        raise ValueError("fixed_dofs must be one-dimensional")

    if not np.all(np.isfinite(raw_dofs)):
        raise ValueError("fixed_dofs must contain only finite values")

    if not np.all(raw_dofs == np.floor(raw_dofs)):
        raise ValueError("fixed_dofs must contain integer indices")

    dofs = raw_dofs.astype(int)

    if np.any(dofs < 0) or np.any(dofs >= num_dofs):
        raise ValueError("fixed_dofs must refer to valid indices")

    if len(np.unique(dofs)) != len(dofs):
        raise ValueError("fixed_dofs must not contain duplicates")

    return dofs


def _validate_fixed_values(fixed_values, num_fixed):
    """Return validated prescribed displacement values."""
    if fixed_values is None:
        return np.zeros(num_fixed, dtype=float)

    values = np.asarray(fixed_values, dtype=float)

    if values.ndim == 0:
        values = values.reshape(1)

    if values.ndim != 1:
        raise ValueError("fixed_values must be one-dimensional")

    if len(values) != num_fixed:
        raise ValueError("fixed_values length must match fixed_dofs length")

    if not np.all(np.isfinite(values)):
        raise ValueError("fixed_values must contain only finite values")

    return values


def _validate_displacements(displacements, num_nodes):
    """Return a validated displacement vector."""
    displacements = np.asarray(displacements, dtype=float)

    if displacements.ndim != 1:
        raise ValueError("displacements must be one-dimensional")

    if len(displacements) != num_nodes:
        raise ValueError("displacements length must match the number of nodes")

    if not np.all(np.isfinite(displacements)):
        raise ValueError("displacements must contain only finite values")

    return displacements
