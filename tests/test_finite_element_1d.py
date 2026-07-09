import numpy as np
import pytest

from analysis.finite_element_1d import (
    analytical_tip_displacement_axial_bar,
    apply_dirichlet_boundary_conditions,
    assemble_global_stiffness,
    bar_element_stiffness,
    compute_element_strains,
    compute_element_stresses,
    compute_reaction_forces,
    create_uniform_bar_mesh,
    simulate_axial_bar_fem,
    solve_displacements,
)


def test_create_uniform_bar_mesh_returns_expected_shapes():
    """Uniform mesh helper should return num_elements + 1 nodes."""
    nodes, elements = create_uniform_bar_mesh(length=1.0, num_elements=4)

    assert nodes.shape == (5,)
    assert elements.shape == (4, 2)
    assert nodes == pytest.approx([0.0, 0.25, 0.5, 0.75, 1.0])
    assert np.array_equal(
        elements,
        np.array(
            [
                [0, 1],
                [1, 2],
                [2, 3],
                [3, 4],
            ]
        ),
    )


def test_invalid_mesh_length_raises_value_error():
    """Bar length must be positive."""
    with pytest.raises(ValueError):
        create_uniform_bar_mesh(length=0.0, num_elements=4)


def test_invalid_num_elements_raises_value_error():
    """The mesh needs at least one element."""
    with pytest.raises(ValueError):
        create_uniform_bar_mesh(length=1.0, num_elements=0)


def test_bar_element_stiffness_returns_two_by_two_matrix():
    """A linear axial bar element should have a 2x2 stiffness matrix."""
    stiffness = bar_element_stiffness(E=200.0, A=0.5, length=2.0)

    assert stiffness.shape == (2, 2)


def test_bar_element_stiffness_is_symmetric():
    """Element stiffness should be symmetric."""
    stiffness = bar_element_stiffness(E=200.0, A=0.5, length=2.0)

    assert stiffness == pytest.approx(stiffness.T)


def test_bar_element_stiffness_row_sums_are_zero():
    """Unconstrained element stiffness should have zero row sums."""
    stiffness = bar_element_stiffness(E=200.0, A=0.5, length=2.0)

    assert np.sum(stiffness, axis=1) == pytest.approx([0.0, 0.0])


@pytest.mark.parametrize(
    "E,A,length",
    [
        (0.0, 1.0, 1.0),
        (1.0, 0.0, 1.0),
        (1.0, 1.0, 0.0),
    ],
)
def test_invalid_element_parameters_raise_value_error(E, A, length):
    """Element material, area, and length parameters should be positive."""
    with pytest.raises(ValueError):
        bar_element_stiffness(E=E, A=A, length=length)


def test_assemble_global_stiffness_returns_correct_shape():
    """Global stiffness should be square with one DOF per node."""
    nodes, elements = create_uniform_bar_mesh(length=1.0, num_elements=3)

    stiffness = assemble_global_stiffness(nodes, elements, E=200.0, A=0.5)

    assert stiffness.shape == (4, 4)


def test_global_stiffness_matrix_is_symmetric():
    """Assembled stiffness should preserve symmetry."""
    nodes, elements = create_uniform_bar_mesh(length=1.0, num_elements=3)

    stiffness = assemble_global_stiffness(nodes, elements, E=200.0, A=0.5)

    assert stiffness == pytest.approx(stiffness.T)


def test_unconstrained_global_stiffness_row_sums_are_zero():
    """Assembled stiffness should preserve rigid-body equilibrium before supports."""
    nodes, elements = create_uniform_bar_mesh(length=1.0, num_elements=4)

    stiffness = assemble_global_stiffness(nodes, elements, E=200.0, A=0.5)

    assert np.sum(stiffness, axis=1) == pytest.approx(np.zeros(len(nodes)))


def test_fixed_free_one_element_bar_matches_analytical_tip_displacement():
    """A one-element fixed-free bar should match the exact tip displacement."""
    length = 2.0
    E = 200.0
    A = 0.5
    force = 10.0
    nodes, elements = create_uniform_bar_mesh(length, num_elements=1)
    stiffness = assemble_global_stiffness(nodes, elements, E, A)
    force_vector = np.array([0.0, force])

    displacements = solve_displacements(
        stiffness,
        force_vector,
        fixed_dofs=[0],
    )

    expected_tip = analytical_tip_displacement_axial_bar(force, length, E, A)
    assert displacements[-1] == pytest.approx(expected_tip)


def test_multi_element_bar_matches_analytical_tip_displacement():
    """A uniform multi-element bar should match the exact tip displacement."""
    result = simulate_axial_bar_fem(
        length=1.5,
        E=210e9,
        A=2.0e-4,
        force=1200.0,
        num_elements=6,
    )

    assert result["tip_displacement"] == pytest.approx(
        result["analytical_tip_displacement"]
    )


def test_computed_strain_is_constant_for_uniform_bar():
    """A uniform fixed-free bar with end load should have constant strain."""
    result = simulate_axial_bar_fem(
        length=1.0,
        E=200e9,
        A=1e-4,
        force=1000.0,
        num_elements=8,
    )

    expected_strain = result["parameters"]["force"] / (
        result["parameters"]["E"] * result["parameters"]["A"]
    )

    assert result["strains"] == pytest.approx(
        np.full(result["parameters"]["num_elements"], expected_strain)
    )


def test_computed_stress_matches_force_over_area():
    """Axial stress should be force divided by cross-sectional area."""
    result = simulate_axial_bar_fem(
        length=1.0,
        E=200e9,
        A=1e-4,
        force=1000.0,
        num_elements=8,
    )

    expected_stress = result["parameters"]["force"] / result["parameters"]["A"]

    assert result["stresses"] == pytest.approx(
        np.full(result["parameters"]["num_elements"], expected_stress)
    )


def test_computed_strain_matches_stress_over_youngs_modulus():
    """Recovered strain should be consistent with Hooke's law."""
    result = simulate_axial_bar_fem(
        length=1.0,
        E=200e9,
        A=1e-4,
        force=1000.0,
        num_elements=8,
    )

    assert result["strains"] == pytest.approx(
        result["stresses"] / result["parameters"]["E"]
    )


def test_reaction_force_at_fixed_node_balances_applied_force():
    """The fixed support reaction should balance the applied end force."""
    result = simulate_axial_bar_fem(
        length=1.0,
        E=200e9,
        A=1e-4,
        force=1000.0,
        num_elements=8,
    )

    assert result["reactions"][0] == pytest.approx(-1000.0)
    assert np.sum(result["reactions"] + result["F"]) == pytest.approx(0.0, abs=1e-9)


def test_apply_boundary_conditions_does_not_mutate_inputs():
    """Boundary-condition helper should return modified copies."""
    stiffness = np.array([[2.0, -2.0], [-2.0, 2.0]])
    force_vector = np.array([0.0, 5.0])
    stiffness_original = stiffness.copy()
    force_original = force_vector.copy()

    apply_dirichlet_boundary_conditions(
        stiffness,
        force_vector,
        fixed_dofs=[0],
    )

    assert stiffness == pytest.approx(stiffness_original)
    assert force_vector == pytest.approx(force_original)


def test_invalid_element_indices_raise_value_error():
    """Assembly should reject element indices outside the node array."""
    nodes = np.array([0.0, 1.0])
    elements = np.array([[0, 2]])

    with pytest.raises(ValueError):
        assemble_global_stiffness(nodes, elements, E=200.0, A=0.5)


def test_analytical_tip_displacement_returns_expected_value():
    """Analytical tip helper should return F*L/(E*A)."""
    assert analytical_tip_displacement_axial_bar(
        force=10.0,
        length=2.0,
        E=200.0,
        A=0.5,
    ) == pytest.approx(0.2)


def test_simulate_axial_bar_fem_returns_expected_keys():
    """High-level FEM helper should return documented result fields."""
    result = simulate_axial_bar_fem()

    expected_keys = {
        "nodes",
        "elements",
        "K",
        "F",
        "displacements",
        "strains",
        "stresses",
        "reactions",
        "analytical_tip_displacement",
        "tip_displacement",
        "relative_tip_error",
        "parameters",
    }

    assert expected_keys.issubset(result.keys())


def test_compute_element_strains_uses_nodal_displacements():
    """Strain recovery should use displacement difference over element length."""
    nodes = np.array([0.0, 0.5, 1.0])
    elements = np.array([[0, 1], [1, 2]])
    displacements = np.array([0.0, 0.1, 0.2])

    strains = compute_element_strains(nodes, elements, displacements)

    assert strains == pytest.approx([0.2, 0.2])


def test_compute_element_stresses_multiplies_strain_by_E():
    """Stress recovery should use Hooke's law for an axial bar."""
    stresses = compute_element_stresses(np.array([0.01, 0.02]), E=200.0)

    assert stresses == pytest.approx([2.0, 4.0])


def test_compute_reaction_forces_matches_original_system_residual():
    """Reaction helper should compute K*u - F."""
    stiffness = np.array([[2.0, -2.0], [-2.0, 2.0]])
    displacement = np.array([0.0, 2.5])
    force_vector = np.array([0.0, 5.0])

    reactions = compute_reaction_forces(stiffness, displacement, force_vector)

    assert reactions == pytest.approx([-5.0, 0.0])
