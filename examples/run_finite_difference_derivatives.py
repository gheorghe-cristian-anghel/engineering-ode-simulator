"""Compare finite difference derivatives with analytical derivatives."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.finite_difference import (
    backward_difference,
    central_difference,
    forward_difference,
    max_abs_error,
    rms_error,
    second_derivative_central,
    uniform_grid_1d,
)


def _function(x):
    """Return f(x) = sin(2*pi*x)."""
    return np.sin(2.0 * np.pi * x)


def _exact_first_derivative(x):
    """Return the exact first derivative of sin(2*pi*x)."""
    return 2.0 * np.pi * np.cos(2.0 * np.pi * x)


def _exact_second_derivative(x):
    """Return the exact second derivative of sin(2*pi*x)."""
    return -(2.0 * np.pi) ** 2 * np.sin(2.0 * np.pi * x)


def _draw_plots(
    x,
    exact_first,
    forward,
    backward,
    central,
    exact_second,
    second,
):
    """Draw derivative comparison and absolute error plots."""
    figure, axes = plt.subplots(2, 2, figsize=(11, 8))

    axes[0, 0].plot(x, exact_first, label="Exact f'(x)", linewidth=2)
    axes[0, 0].plot(x, forward, "--", label="Forward")
    axes[0, 0].plot(x, backward, "--", label="Backward")
    axes[0, 0].plot(x, central, "--", label="Central")
    axes[0, 0].set_title("First Derivative")
    axes[0, 0].set_xlabel("x")
    axes[0, 0].set_ylabel("Derivative")
    axes[0, 0].grid(True)
    axes[0, 0].legend()

    axes[0, 1].plot(x, np.abs(forward - exact_first), label="Forward")
    axes[0, 1].plot(x, np.abs(backward - exact_first), label="Backward")
    axes[0, 1].plot(x, np.abs(central - exact_first), label="Central")
    axes[0, 1].set_title("First Derivative Absolute Error")
    axes[0, 1].set_xlabel("x")
    axes[0, 1].set_ylabel("Absolute error")
    axes[0, 1].grid(True)
    axes[0, 1].legend()

    axes[1, 0].plot(x, exact_second, label="Exact f''(x)", linewidth=2)
    axes[1, 0].plot(x, second, "--", label="Central")
    axes[1, 0].set_title("Second Derivative")
    axes[1, 0].set_xlabel("x")
    axes[1, 0].set_ylabel("Second derivative")
    axes[1, 0].grid(True)
    axes[1, 0].legend()

    axes[1, 1].plot(x, np.abs(second - exact_second), label="Second derivative")
    axes[1, 1].set_title("Second Derivative Absolute Error")
    axes[1, 1].set_xlabel("x")
    axes[1, 1].set_ylabel("Absolute error")
    axes[1, 1].grid(True)
    axes[1, 1].legend()

    figure.tight_layout()


def _plot_response(
    x,
    exact_first,
    forward,
    backward,
    central,
    exact_second,
    second,
):
    """Save and display the finite difference derivative comparison plots."""
    output_path = PROJECT_ROOT / "examples" / "finite_difference_derivatives.png"

    _draw_plots(
        x,
        exact_first,
        forward,
        backward,
        central,
        exact_second,
        second,
    )
    plt.savefig(output_path, dpi=150)
    print(f"Plot saved to: {output_path}")
    plt.show()


def _print_error_line(label, numerical, exact):
    """Print max and RMS errors for one derivative method."""
    print(
        f"{label}: max error = {max_abs_error(numerical, exact):.6e}, "
        f"RMS error = {rms_error(numerical, exact):.6e}"
    )


def main():
    """Compare finite difference derivatives against exact derivatives."""
    num_points = 101
    x, dx = uniform_grid_1d(0.0, 1.0, num_points)
    y = _function(x)
    exact_first = _exact_first_derivative(x)
    exact_second = _exact_second_derivative(x)

    forward = forward_difference(y, dx)
    backward = backward_difference(y, dx)
    central = central_difference(y, dx)
    second = second_derivative_central(y, dx)

    print("Finite Difference Derivative Comparison:")
    print("Function: f(x) = sin(2*pi*x)")
    print(f"Grid size: {num_points}")
    print(f"dx: {dx:.6f}")
    print()
    print("First Derivative Errors:")
    _print_error_line("Forward difference", forward, exact_first)
    _print_error_line("Backward difference", backward, exact_first)
    _print_error_line("Central difference", central, exact_first)
    print()
    print("Second Derivative Errors:")
    _print_error_line("Central second derivative", second, exact_second)

    _plot_response(
        x,
        exact_first,
        forward,
        backward,
        central,
        exact_second,
        second,
    )


if __name__ == "__main__":
    main()
