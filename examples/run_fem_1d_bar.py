"""Run and plot a 1D axial bar finite element example."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.finite_element_1d import simulate_axial_bar_fem
from visualization.plot_style import (
    apply_plot_style,
    format_axes,
    place_legends_outside,
    save_figure,
)


def _analytical_displacement(nodes, force, E, A):
    """Return the exact displacement field for a fixed-free axial bar."""
    return force * nodes / (E * A)


def _draw_plots(result):
    """Draw displacement, stress, and exaggerated deformed-shape plots."""
    apply_plot_style()

    nodes = result["nodes"]
    elements = result["elements"]
    displacements = result["displacements"]
    stresses = result["stresses"]
    parameters = result["parameters"]

    analytical = _analytical_displacement(
        nodes,
        parameters["force"],
        parameters["E"],
        parameters["A"],
    )
    element_centers = 0.5 * (
        nodes[elements[:, 0]] + nodes[elements[:, 1]]
    )

    figure, axes = plt.subplots(3, 1, figsize=(8, 9))

    axes[0].plot(nodes, displacements, "o-", label="FEM displacement")
    axes[0].plot(nodes, analytical, "--", label="Analytical displacement")
    format_axes(
        axes[0],
        title="1D Axial Bar FEM: Displacement",
        xlabel="Position x (m)",
        ylabel="Displacement u (m)",
    )

    axes[1].step(
        element_centers,
        stresses,
        where="mid",
        label="Element stress",
    )
    axes[1].axhline(
        parameters["force"] / parameters["A"],
        color="tab:orange",
        linestyle="--",
        label="Analytical stress",
    )
    format_axes(
        axes[1],
        title="Element Stress Distribution",
        xlabel="Element center x (m)",
        ylabel="Stress (Pa)",
    )

    max_displacement = np.max(np.abs(displacements))
    scale = 0.1 * parameters["length"] / max_displacement
    deformed_nodes = nodes + scale * displacements

    axes[2].plot(nodes, np.zeros_like(nodes), "o-", label="Original bar")
    axes[2].plot(
        deformed_nodes,
        np.zeros_like(deformed_nodes) + 0.02,
        "o-",
        label=f"Deformed shape ({scale:.2e}x)",
    )
    axes[2].set_ylim(-0.04, 0.08)
    axes[2].set_yticks([])
    format_axes(
        axes[2],
        title="Exaggerated Deformed Shape",
        xlabel="Position x (m)",
    )
    place_legends_outside(axes, location="right")

    figure.tight_layout()
    return figure


def _plot_response(result):
    """Save and display the FEM example plots."""
    output_path = PROJECT_ROOT / "examples" / "fem_1d_bar.png"

    figure = _draw_plots(result)
    save_figure(figure, output_path)
    print(f"Plot saved to: {output_path}")
    plt.show()


def main():
    """Solve a fixed-free axial bar using linear finite elements."""
    length = 1.0
    E = 200e9
    A = 1e-4
    force = 1000.0
    num_elements = 8

    result = simulate_axial_bar_fem(
        length=length,
        E=E,
        A=A,
        force=force,
        num_elements=num_elements,
    )

    fixed_reaction = result["reactions"][0]
    average_stress = np.mean(result["stresses"])

    print("1D Axial Bar Finite Element Method Example:")
    print(f"Bar length L: {length:.3f} m")
    print(f"Young's modulus E: {E:.6e} Pa")
    print(f"Cross-sectional area A: {A:.6e} m^2")
    print(f"Applied force: {force:.6f} N")
    print(f"Number of elements: {num_elements}")
    print(f"FEM tip displacement: {result['tip_displacement']:.6e} m")
    print(
        "Analytical tip displacement: "
        f"{result['analytical_tip_displacement']:.6e} m"
    )
    print(f"Relative tip error: {result['relative_tip_error']:.6e}")
    print(f"Reaction force at fixed support: {fixed_reaction:.6f} N")
    print(f"Average stress: {average_stress:.6e} Pa")
    print()
    print(
        "The FEM assembles element stiffness matrices into a global system, "
        "applies boundary conditions, and solves for nodal displacements."
    )
    print(
        "For a uniform axial bar with an end load, the FEM solution matches "
        "the analytical linear displacement field."
    )

    _plot_response(result)


if __name__ == "__main__":
    main()
