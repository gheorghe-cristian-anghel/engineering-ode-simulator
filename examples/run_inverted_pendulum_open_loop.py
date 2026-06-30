"""Run and plot the nonlinear open-loop inverted pendulum example."""

import sys
from pathlib import Path
from _tkinter import TclError

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.inverted_pendulum import simulate_inverted_pendulum


def _draw_plots(t, cart_position, cart_velocity, theta, theta_dot):
    """Draw cart-pole open-loop response plots."""
    figure, axes = plt.subplots(4, 1, sharex=True)

    axes[0].plot(t, cart_position, label="Cart position")
    axes[0].set_ylabel("Position (m)")
    axes[0].set_title("Open-Loop Inverted Pendulum: Cart Position")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(t, np.degrees(theta), label="Pendulum angle", color="tab:orange")
    axes[1].set_ylabel("Angle (deg)")
    axes[1].set_title("Pendulum Angle from Upright: Early Unstable Departure")
    axes[1].grid(True)
    axes[1].legend()

    axes[2].plot(t, cart_velocity, label="Cart velocity", color="tab:green")
    axes[2].set_ylabel("Velocity (m/s)")
    axes[2].set_title("Cart Velocity")
    axes[2].grid(True)
    axes[2].legend()

    axes[3].plot(t, theta_dot, label="Angular velocity", color="tab:red")
    axes[3].set_xlabel("Time (s)")
    axes[3].set_ylabel("Angular velocity (rad/s)")
    axes[3].set_title("Pendulum Angular Velocity")
    axes[3].grid(True)
    axes[3].legend()

    figure.tight_layout()


def _plot_response(t, cart_position, cart_velocity, theta, theta_dot):
    """Plot response, falling back gracefully if Tk is unavailable."""
    output_path = PROJECT_ROOT / "examples" / "inverted_pendulum_open_loop.png"

    try:
        _draw_plots(t, cart_position, cart_velocity, theta, theta_dot)
        plt.savefig(output_path, dpi=150)
        plt.show()
    except TclError:
        plt.switch_backend("Agg")
        plt.close("all")
        _draw_plots(t, cart_position, cart_velocity, theta, theta_dot)
        plt.savefig(output_path, dpi=150)
        print("Interactive Matplotlib window is unavailable in this environment.")
        print(f"Plot saved to: {output_path}")


def main():
    """Simulate a small open-loop perturbation from upright."""
    M = 1.0
    m = 0.1
    l = 0.5
    g = 9.81
    b = 0.0
    initial_angle_degrees = 5.0
    x0 = [0.0, 0.0, np.radians(initial_angle_degrees), 0.0]
    t_span = (0.0, 0.75)
    num_points = 800

    t, states = simulate_inverted_pendulum(
        x0,
        t_span=t_span,
        num_points=num_points,
        M=M,
        m=m,
        l=l,
        g=g,
        b=b,
    )

    cart_position = states[:, 0]
    cart_velocity = states[:, 1]
    theta = states[:, 2]
    theta_dot = states[:, 3]

    print("Open-Loop Inverted Pendulum:")
    print(f"Cart mass M: {M} kg")
    print(f"Pendulum mass m: {m} kg")
    print(f"Pendulum length l: {l} m")
    print(f"Gravity g: {g} m/s^2")
    print(f"Cart damping b: {b} N*s/m")
    print(f"Simulation time: {t_span[1]:.2f} s")
    print(f"Initial angle: {initial_angle_degrees:.3f} degrees")
    print(f"Final angle: {np.degrees(theta[-1]):.3f} degrees")
    print(f"Maximum absolute angle: {np.max(np.abs(np.degrees(theta))):.3f} degrees")
    print()
    print("The upright equilibrium is unstable without feedback control.")
    print("This short time window keeps the nonlinear departure readable.")

    _plot_response(t, cart_position, cart_velocity, theta, theta_dot)


if __name__ == "__main__":
    main()
