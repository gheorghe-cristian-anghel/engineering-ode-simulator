"""Run a finite difference derivative convergence study."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.finite_difference import (
    backward_difference,
    central_difference,
    estimate_convergence_order,
    forward_difference,
    rms_error,
    uniform_grid_1d,
)
from visualization.plot_style import (
    apply_plot_style,
    format_axes,
    place_legend_outside,
    save_figure,
)


def _function(x):
    """Return f(x) = sin(2*pi*x)."""
    return np.sin(2.0 * np.pi * x)


def _exact_first_derivative(x):
    """Return the exact first derivative of sin(2*pi*x)."""
    return 2.0 * np.pi * np.cos(2.0 * np.pi * x)


def _draw_plot(dx_values, forward_errors, backward_errors, central_errors):
    """Draw log-log convergence curves for first derivative methods."""
    apply_plot_style()

    figure, ax = plt.subplots(figsize=(7, 5))

    ax.loglog(dx_values, forward_errors, "o-", label="Forward")
    ax.loglog(dx_values, backward_errors, "s-", label="Backward")
    ax.loglog(dx_values, central_errors, "^-", label="Central")

    reference_dx = np.asarray(dx_values)
    reference_first_order = forward_errors[-1] * (
        reference_dx / reference_dx[-1]
    )
    reference_second_order = central_errors[-1] * (
        reference_dx / reference_dx[-1]
    ) ** 2

    ax.loglog(reference_dx, reference_first_order, ":", label="O(dx)")
    ax.loglog(reference_dx, reference_second_order, "--", label="O(dx^2)")

    format_axes(
        ax,
        title="Finite Difference Convergence",
        xlabel="Grid spacing dx",
        ylabel="RMS error",
    )
    ax.grid(True, which="both", linestyle="--", linewidth=0.7, alpha=0.8)
    ax.invert_xaxis()
    place_legend_outside(ax, location="right")

    figure.tight_layout()
    return figure


def _plot_response(dx_values, forward_errors, backward_errors, central_errors):
    """Save and display the convergence plot."""
    output_path = PROJECT_ROOT / "examples" / "finite_difference_convergence.png"

    figure = _draw_plot(dx_values, forward_errors, backward_errors, central_errors)
    save_figure(figure, output_path)
    print(f"Plot saved to: {output_path}")
    plt.show()


def main():
    """Estimate convergence order for first derivative formulas."""
    num_points_values = [21, 41, 81, 161, 321]
    dx_values = []
    forward_errors = []
    backward_errors = []
    central_errors = []

    print("Finite Difference Convergence Study:")
    print("Function: f(x) = sin(2*pi*x)")
    print()
    print("Grid results:")

    for num_points in num_points_values:
        x, dx = uniform_grid_1d(0.0, 1.0, num_points)
        y = _function(x)
        exact = _exact_first_derivative(x)

        forward = forward_difference(y, dx)
        backward = backward_difference(y, dx)
        central = central_difference(y, dx)

        interior = slice(1, -1)
        forward_error = rms_error(forward[interior], exact[interior])
        backward_error = rms_error(backward[interior], exact[interior])
        central_error = rms_error(central[interior], exact[interior])

        dx_values.append(dx)
        forward_errors.append(forward_error)
        backward_errors.append(backward_error)
        central_errors.append(central_error)

        print(
            f"  n = {num_points:3d}, dx = {dx:.6f}, "
            f"RMS errors: forward = {forward_error:.6e}, "
            f"backward = {backward_error:.6e}, central = {central_error:.6e}"
        )

    forward_order = estimate_convergence_order(dx_values, forward_errors)
    backward_order = estimate_convergence_order(dx_values, backward_errors)
    central_order = estimate_convergence_order(dx_values, central_errors)

    print()
    print("Estimated Convergence Orders:")
    print(f"Forward difference: {forward_order:.3f}")
    print(f"Backward difference: {backward_order:.3f}")
    print(f"Central difference: {central_order:.3f}")

    _plot_response(
        dx_values,
        forward_errors,
        backward_errors,
        central_errors,
    )


if __name__ == "__main__":
    main()
