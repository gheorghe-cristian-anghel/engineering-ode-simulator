"""Run and plot the linearized open-loop inverted pendulum example."""

import sys
from pathlib import Path
from _tkinter import TclError

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.state_space import simulate_state_space
from models.inverted_pendulum import linearized_inverted_pendulum_state_space


def _zero_input(t):
    """Return zero cart force for open-loop simulation."""
    return 0.0


def _draw_plots(t, cart_position, theta):
    """Draw linearized cart-pole open-loop response plots."""
    figure, axes = plt.subplots(2, 1, sharex=True)

    axes[0].plot(t, cart_position, label="Cart position")
    axes[0].set_ylabel("Position (m)")
    axes[0].set_title("Linearized Inverted Pendulum: Cart Position")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(t, np.degrees(theta), label="Pendulum angle", color="tab:orange")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Angle (deg)")
    axes[1].set_title("Linearized Pendulum Angle from Upright: Unstable Growth")
    axes[1].grid(True)
    axes[1].legend()

    figure.tight_layout()


def _plot_response(t, cart_position, theta):
    """Plot response, falling back gracefully if Tk is unavailable."""
    output_path = PROJECT_ROOT / "examples" / "inverted_pendulum_linearized.png"

    try:
        _draw_plots(t, cart_position, theta)
        plt.savefig(output_path, dpi=150)
        plt.show()
    except TclError:
        plt.switch_backend("Agg")
        plt.close("all")
        _draw_plots(t, cart_position, theta)
        plt.savefig(output_path, dpi=150)
        print("Interactive Matplotlib window is unavailable in this environment.")
        print(f"Plot saved to: {output_path}")


def main():
    """Simulate the upright linearized model after a small perturbation."""
    M = 1.0
    m = 0.1
    l = 0.5
    g = 9.81
    b = 0.0
    initial_angle_degrees = 0.5
    x0 = [0.0, 0.0, np.radians(initial_angle_degrees), 0.0]
    t_span = (0.0, 1.0)
    num_points = 1000

    A, B, C, D = linearized_inverted_pendulum_state_space(M, m, l, g, b)
    eigenvalues = np.linalg.eigvals(A)

    t, states, outputs = simulate_state_space(
        A,
        B,
        C,
        D,
        _zero_input,
        x0,
        t_span,
        num_points,
    )

    cart_position = outputs[:, 0]
    theta = outputs[:, 1]

    print("Linearized Inverted Pendulum State-Space Model:")
    print("State: [x, x_dot, theta, theta_dot]")
    print("Input: cart force F")
    print("Outputs: cart position and pendulum angle")
    print()
    print("A matrix:")
    print(np.array2string(A, precision=4, suppress_small=True))
    print()
    print("B matrix:")
    print(np.array2string(B, precision=4, suppress_small=True))
    print()
    print("C matrix:")
    print(np.array2string(C, precision=4, suppress_small=True))
    print()
    print("D matrix:")
    print(np.array2string(D, precision=4, suppress_small=True))
    print()
    print("Eigenvalues:")
    print(np.array2string(eigenvalues, precision=4, suppress_small=True))
    print(f"Simulation time: {t_span[1]:.2f} s")
    print(f"Initial angle: {initial_angle_degrees:.3f} degrees")
    print(f"Final angle: {np.degrees(theta[-1]):.3f} degrees")
    print()
    print("A positive real eigenvalue shows open-loop upright instability.")
    print("The short simulation window keeps the linearized angle scale readable.")
    print("No stabilizing controller is used in this example.")

    _plot_response(t, cart_position, theta)


if __name__ == "__main__":
    main()
