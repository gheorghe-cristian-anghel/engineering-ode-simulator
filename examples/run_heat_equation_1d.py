"""Run and plot a 1D heat equation finite-difference example."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.heat_equation_1d import (
    gaussian_initial_condition,
    simulate_heat_equation_1d,
)
from visualization.plot_style import (
    THERMAL_COLORMAP,
    add_colorbar,
    apply_plot_style,
    format_axes,
    place_legend_outside,
    save_figure,
)


def _profile_indices(t):
    """Return indices for 0%, 25%, 50%, and 100% of the simulation."""
    fractions = [0.0, 0.25, 0.5, 1.0]
    return [int(round(fraction * (len(t) - 1))) for fraction in fractions]


def _draw_plots(result):
    """Draw temperature profiles and a heatmap."""
    apply_plot_style()

    x = result["x"]
    t = result["t"]
    temperature = result["temperature"]

    figure, axes = plt.subplots(2, 1, figsize=(8, 7))

    for index in _profile_indices(t):
        axes[0].plot(
            x,
            temperature[index],
            label=f"t = {t[index]:.2f} s",
        )

    format_axes(
        axes[0],
        title="1D Heat Equation: Temperature Profiles",
        xlabel="Position x (m)",
        ylabel="Temperature",
    )
    place_legend_outside(axes[0], location="right")

    heatmap = axes[1].imshow(
        temperature,
        aspect="auto",
        origin="lower",
        extent=[x[0], x[-1], t[0], t[-1]],
        cmap=THERMAL_COLORMAP,
    )
    format_axes(
        axes[1],
        title="Temperature Heatmap",
        xlabel="Position x (m)",
        ylabel="Time (s)",
        grid=False,
    )
    add_colorbar(figure, heatmap, axes[1], label="Temperature")

    figure.tight_layout()
    return figure


def _plot_response(result):
    """Save and display the heat equation plots."""
    output_path = PROJECT_ROOT / "examples" / "heat_equation_1d.png"

    figure = _draw_plots(result)
    save_figure(figure, output_path)
    print(f"Plot saved to: {output_path}")
    plt.show()


def main():
    """Simulate heat diffusion from a Gaussian pulse in a 1D rod."""
    length = 1.0
    alpha = 0.01
    t_final = 2.0
    num_points = 101
    boundary_type = "dirichlet"
    boundary_values = (0.0, 0.0)

    initial_condition = lambda x: gaussian_initial_condition(
        x,
        center=0.5 * length,
        width=0.08,
        amplitude=1.0,
    )

    result = simulate_heat_equation_1d(
        length=length,
        alpha=alpha,
        t_final=t_final,
        num_points=num_points,
        initial_condition=initial_condition,
        boundary_type=boundary_type,
        boundary_values=boundary_values,
    )

    temperature = result["temperature"]

    print("1D Heat Equation Finite-Difference Example:")
    print(f"Rod length: {length:.3f} m")
    print(f"Thermal diffusivity alpha: {result['alpha']:.5f}")
    print(f"Spatial step dx: {result['dx']:.5f} m")
    print(f"Time step dt: {result['dt']:.5f} s")
    print(f"Stability number r: {result['stability_number']:.5f}")
    print(f"Boundary type: {result['boundary_type']}")
    print(f"Boundary values: {result['boundary_values']}")
    print(f"Initial max temperature: {np.max(temperature[0]):.5f}")
    print(f"Final max temperature: {np.max(temperature[-1]):.5f}")
    print(f"Final min temperature: {np.min(temperature[-1]):.5f}")
    print()
    print("The heat equation smooths temperature gradients over time.")
    print("A hot Gaussian pulse diffuses and its peak temperature decreases.")

    _plot_response(result)


if __name__ == "__main__":
    main()
