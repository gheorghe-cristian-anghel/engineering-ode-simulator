"""Run and plot a 2D wave equation finite-difference example."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.wave_equation_2d import (
    gaussian_displacement_2d,
    simulate_wave_equation_2d,
    zero_initial_velocity_2d,
)
from visualization.plot_style import (
    add_colorbar,
    apply_plot_style,
    format_axes,
    save_figure,
    set_equal_2d_axes,
)


def _snapshot_indices(t):
    """Return indices for initial, intermediate, and final snapshots."""
    fractions = [0.0, 0.25, 0.5, 0.75, 1.0]
    return [int(round(fraction * (len(t) - 1))) for fraction in fractions]


def _draw_plots(result):
    """Draw displacement heatmaps and centerline profiles."""
    apply_plot_style()

    x = result["x"]
    y = result["y"]
    t = result["t"]
    displacement = result["displacement"]

    figure, axes = plt.subplots(2, 3, figsize=(14, 8))
    flat_axes = axes.ravel()
    heatmap_axes = flat_axes[:4]
    centerline_axis = flat_axes[4]
    figure.delaxes(flat_axes[5])

    snapshot_indices = [
        0,
        int(round(0.25 * (len(t) - 1))),
        int(round(0.5 * (len(t) - 1))),
        len(t) - 1,
    ]

    peak = float(np.max(np.abs(displacement)))
    color_limit = peak if peak > 0 else 1.0

    for axis, index in zip(heatmap_axes, snapshot_indices):
        heatmap = axis.imshow(
            displacement[index],
            origin="lower",
            extent=[x[0], x[-1], y[0], y[-1]],
            aspect="auto",
            cmap="coolwarm",
            vmin=-color_limit,
            vmax=color_limit,
        )
        format_axes(
            axis,
            title=f"Displacement at t = {t[index]:.3f} s",
            xlabel="x (m)",
            ylabel="y (m)",
            grid=False,
        )
        set_equal_2d_axes(axis)
        add_colorbar(figure, heatmap, axis, label="Displacement")

    center_y_index = len(y) // 2
    for index in _snapshot_indices(t):
        centerline_axis.plot(
            x,
            displacement[index, center_y_index, :],
            label=f"t = {t[index]:.3f} s",
        )

    format_axes(
        centerline_axis,
        title="Centerline Displacement Profile",
        xlabel="x (m)",
        ylabel="Displacement",
    )

    figure.tight_layout()
    return figure


def _plot_response(result):
    """Save and display the 2D wave equation plots."""
    output_path = PROJECT_ROOT / "examples" / "wave_equation_2d.png"

    figure = _draw_plots(result)
    save_figure(figure, output_path)
    print(f"Plot saved to: {output_path}")
    plt.show()


def main():
    """Simulate wave propagation from a Gaussian membrane displacement."""
    width = 1.0
    height = 1.0
    c = 1.0
    nx = 81
    ny = 81
    t_final = 1.2
    boundary_type = "dirichlet"
    boundary_values = 0.0

    initial_displacement = lambda X, Y: gaussian_displacement_2d(
        X,
        Y,
        center=(0.45 * width, 0.5 * height),
        width=0.06,
        amplitude=1.0,
    )

    result = simulate_wave_equation_2d(
        width=width,
        height=height,
        c=c,
        t_final=t_final,
        nx=nx,
        ny=ny,
        initial_displacement=initial_displacement,
        initial_velocity=zero_initial_velocity_2d,
        boundary_type=boundary_type,
        boundary_values=boundary_values,
        store_every=10,
    )

    displacement = result["displacement"]

    print("2D Wave Equation Finite-Difference Example:")
    print(f"Membrane width: {width:.3f} m")
    print(f"Membrane height: {height:.3f} m")
    print(f"Wave speed c: {result['c']:.5f} m/s")
    print(f"Grid size: nx = {nx}, ny = {ny}")
    print(f"Spatial step dx: {result['dx']:.5f} m")
    print(f"Spatial step dy: {result['dy']:.5f} m")
    print(f"Time step dt: {result['dt']:.6f} s")
    print(f"CFL lambda_x: {result['lambda_x']:.5f}")
    print(f"CFL lambda_y: {result['lambda_y']:.5f}")
    print(f"Stability number rx: {result['rx']:.5f}")
    print(f"Stability number ry: {result['ry']:.5f}")
    print(f"Stability sum rx + ry: {result['stability_sum']:.5f}")
    print(f"Boundary type: {result['boundary_type']}")
    print(f"Boundary values: {result['boundary_values']}")
    print(f"Initial max displacement: {np.max(displacement[0]):.5f}")
    print(f"Initial min displacement: {np.min(displacement[0]):.5f}")
    print(f"Final max displacement: {np.max(displacement[-1]):.5f}")
    print(f"Final min displacement: {np.min(displacement[-1]):.5f}")
    print()
    print(
        "The 2D wave equation propagates a localized membrane displacement "
        "outward as waves."
    )
    print(
        "Unlike the heat equation, the solution oscillates and reflects from "
        "fixed boundaries instead of simply diffusing away."
    )

    _plot_response(result)


if __name__ == "__main__":
    main()
