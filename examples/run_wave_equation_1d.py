"""Run and plot a 1D wave equation finite-difference example."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.wave_equation_1d import (
    gaussian_displacement,
    simulate_wave_equation_1d,
    zero_initial_velocity,
)
from visualization.plot_style import (
    add_colorbar,
    apply_plot_style,
    format_axes,
    save_figure,
)


def _profile_indices(t):
    """Return indices for 0%, 25%, 50%, 75%, and 100% of the simulation."""
    fractions = [0.0, 0.25, 0.5, 0.75, 1.0]
    return [int(round(fraction * (len(t) - 1))) for fraction in fractions]


def _draw_plots(result):
    """Draw displacement profiles and a heatmap."""
    apply_plot_style()

    x = result["x"]
    t = result["t"]
    displacement = result["displacement"]

    figure, axes = plt.subplots(2, 1, figsize=(8, 7))

    for index in _profile_indices(t):
        axes[0].plot(
            x,
            displacement[index],
            label=f"t = {t[index]:.2f} s",
        )

    format_axes(
        axes[0],
        title="1D Wave Equation: Displacement Profiles",
        xlabel="Position x (m)",
        ylabel="Displacement",
    )

    peak = np.max(np.abs(displacement))
    color_limit = peak if peak > 0 else 1.0
    heatmap = axes[1].imshow(
        displacement,
        aspect="auto",
        origin="lower",
        extent=[x[0], x[-1], t[0], t[-1]],
        cmap="coolwarm",
        vmin=-color_limit,
        vmax=color_limit,
    )
    format_axes(
        axes[1],
        title="Displacement Heatmap",
        xlabel="Position x (m)",
        ylabel="Time (s)",
        grid=False,
    )
    add_colorbar(figure, heatmap, axes[1], label="Displacement")

    figure.tight_layout()
    return figure


def _plot_response(result):
    """Save and display the wave equation plots."""
    output_path = PROJECT_ROOT / "examples" / "wave_equation_1d.png"

    figure = _draw_plots(result)
    save_figure(figure, output_path)
    print(f"Plot saved to: {output_path}")
    plt.show()


def main():
    """Simulate wave propagation from a Gaussian displacement pulse."""
    length = 1.0
    c = 1.0
    t_final = 2.0
    num_points = 201
    boundary_type = "dirichlet"
    boundary_values = (0.0, 0.0)

    initial_displacement = lambda x: gaussian_displacement(
        x,
        center=0.5 * length,
        width=0.08,
        amplitude=1.0,
    )

    result = simulate_wave_equation_1d(
        length=length,
        c=c,
        t_final=t_final,
        num_points=num_points,
        initial_displacement=initial_displacement,
        initial_velocity=zero_initial_velocity,
        boundary_type=boundary_type,
        boundary_values=boundary_values,
    )

    displacement = result["displacement"]

    print("1D Wave Equation Finite-Difference Example:")
    print(f"String/rod length: {length:.3f} m")
    print(f"Wave speed c: {result['c']:.5f} m/s")
    print(f"Spatial step dx: {result['dx']:.5f} m")
    print(f"Time step dt: {result['dt']:.5f} s")
    print(f"CFL number lambda: {result['cfl_number']:.5f}")
    print(f"Boundary type: {result['boundary_type']}")
    print(f"Boundary values: {result['boundary_values']}")
    print(f"Initial max displacement: {np.max(displacement[0]):.5f}")
    print(f"Final max displacement: {np.max(displacement[-1]):.5f}")
    print(f"Final min displacement: {np.min(displacement[-1]):.5f}")
    print()
    print("The wave equation propagates disturbances through space.")
    print(
        "A Gaussian displacement splits into traveling waves that reflect "
        "from fixed boundaries."
    )

    _plot_response(result)


if __name__ == "__main__":
    main()
