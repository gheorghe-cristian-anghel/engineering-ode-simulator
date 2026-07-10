"""Run and plot a 2D heat equation finite-difference example."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.heat_equation_2d import gaussian_hotspot_2d, simulate_heat_equation_2d
from visualization.plot_style import (
    THERMAL_COLORMAP,
    add_colorbar,
    apply_plot_style,
    format_axes,
    place_legend_outside,
    save_figure,
    set_equal_2d_axes,
)


def _snapshot_indices(t):
    """Return indices for initial, intermediate, and final snapshots."""
    fractions = [0.0, 0.33, 0.67, 1.0]
    return [int(round(fraction * (len(t) - 1))) for fraction in fractions]


def _draw_plots(result):
    """Draw heatmaps and centerline temperature profiles."""
    apply_plot_style()

    x = result["x"]
    y = result["y"]
    t = result["t"]
    temperature = result["temperature"]

    figure, axes = plt.subplots(2, 2, figsize=(11, 9))
    snapshot_axes = [axes[0, 0], axes[0, 1], axes[1, 0]]
    snapshot_indices = [
        0,
        int(round(0.5 * (len(t) - 1))),
        len(t) - 1,
    ]

    color_min = float(np.min(temperature))
    color_max = float(np.max(temperature))

    for axis, index in zip(snapshot_axes, snapshot_indices):
        heatmap = axis.imshow(
            temperature[index],
            origin="lower",
            extent=[x[0], x[-1], y[0], y[-1]],
            aspect="auto",
            cmap=THERMAL_COLORMAP,
            vmin=color_min,
            vmax=color_max,
        )
        format_axes(
            axis,
            title=f"Temperature at t = {t[index]:.2f} s",
            xlabel="x (m)",
            ylabel="y (m)",
            grid=False,
        )
        set_equal_2d_axes(axis)
        add_colorbar(figure, heatmap, axis, label="Temperature")

    centerline_axis = axes[1, 1]
    center_y_index = len(y) // 2

    for index in _snapshot_indices(t):
        centerline_axis.plot(
            x,
            temperature[index, center_y_index, :],
            label=f"t = {t[index]:.2f} s",
        )

    format_axes(
        centerline_axis,
        title="Centerline Temperature Profile",
        xlabel="x (m)",
        ylabel="Temperature",
    )
    place_legend_outside(centerline_axis, location="right")

    figure.tight_layout()
    return figure


def _plot_response(result):
    """Save and display the 2D heat equation plots."""
    output_path = PROJECT_ROOT / "examples" / "heat_equation_2d.png"

    figure = _draw_plots(result)
    save_figure(figure, output_path)
    print(f"Plot saved to: {output_path}")
    plt.show()


def main():
    """Simulate diffusion of a Gaussian hot spot on a 2D plate."""
    width = 1.0
    height = 1.0
    alpha = 0.01
    nx = 61
    ny = 61
    t_final = 1.0
    boundary_type = "dirichlet"
    boundary_values = 0.0

    initial_condition = lambda X, Y: gaussian_hotspot_2d(
        X,
        Y,
        center=(0.5 * width, 0.5 * height),
        width=0.08,
        amplitude=1.0,
    )

    result = simulate_heat_equation_2d(
        width=width,
        height=height,
        alpha=alpha,
        t_final=t_final,
        nx=nx,
        ny=ny,
        initial_condition=initial_condition,
        boundary_type=boundary_type,
        boundary_values=boundary_values,
        store_every=20,
    )

    temperature = result["temperature"]

    print("2D Heat Equation Finite-Difference Example:")
    print(f"Plate width: {width:.3f} m")
    print(f"Plate height: {height:.3f} m")
    print(f"Thermal diffusivity alpha: {result['alpha']:.5f}")
    print(f"Grid size: nx = {nx}, ny = {ny}")
    print(f"Spatial step dx: {result['dx']:.5f} m")
    print(f"Spatial step dy: {result['dy']:.5f} m")
    print(f"Time step dt: {result['dt']:.6f} s")
    print(f"Stability number rx: {result['rx']:.5f}")
    print(f"Stability number ry: {result['ry']:.5f}")
    print(f"Stability sum rx + ry: {result['stability_sum']:.5f}")
    print(f"Boundary type: {result['boundary_type']}")
    print(f"Boundary values: {result['boundary_values']}")
    print(f"Initial max temperature: {np.max(temperature[0]):.5f}")
    print(f"Initial min temperature: {np.min(temperature[0]):.5f}")
    print(f"Final max temperature: {np.max(temperature[-1]):.5f}")
    print(f"Final min temperature: {np.min(temperature[-1]):.5f}")
    print()
    print(
        "The 2D heat equation diffuses a localized hot spot across the plate "
        "while fixed boundaries remain at prescribed temperature."
    )

    _plot_response(result)


if __name__ == "__main__":
    main()
