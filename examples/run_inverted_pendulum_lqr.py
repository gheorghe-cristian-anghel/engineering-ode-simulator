"""Run and plot nonlinear inverted pendulum stabilization with LQR."""

import sys
from pathlib import Path
from _tkinter import TclError

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.inverted_pendulum_lqr import (
    design_inverted_pendulum_lqr,
    simulate_inverted_pendulum_lqr,
)


def _draw_plots(t, states, control_force):
    """Draw cart-pole LQR stabilization plots."""
    cart_position = states[:, 0]
    cart_velocity = states[:, 1]
    theta = states[:, 2]
    theta_dot = states[:, 3]

    figure, axes = plt.subplots(5, 1, sharex=True)

    axes[0].plot(t, cart_position, label="Cart position")
    axes[0].set_ylabel("Position (m)")
    axes[0].set_title("Inverted Pendulum LQR: Cart Position")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(t, np.degrees(theta), label="Pendulum angle", color="tab:orange")
    axes[1].axhline(0.0, color="gray", linestyle=":", label="Upright")
    axes[1].set_ylabel("Angle (deg)")
    axes[1].set_title("Pendulum Angle Stabilized Near Upright")
    axes[1].grid(True)
    axes[1].legend()

    axes[2].plot(t, control_force, label="Control force", color="tab:red")
    axes[2].set_ylabel("Force (N)")
    axes[2].set_title("LQR Control Force")
    axes[2].grid(True)
    axes[2].legend()

    axes[3].plot(t, cart_velocity, label="Cart velocity", color="tab:green")
    axes[3].set_ylabel("Velocity (m/s)")
    axes[3].set_title("Cart Velocity")
    axes[3].grid(True)
    axes[3].legend()

    axes[4].plot(t, theta_dot, label="Angular velocity", color="tab:purple")
    axes[4].set_xlabel("Time (s)")
    axes[4].set_ylabel("Angular velocity (rad/s)")
    axes[4].set_title("Pendulum Angular Velocity")
    axes[4].grid(True)
    axes[4].legend()

    figure.tight_layout()


def _plot_response(t, states, control_force):
    """Plot response, falling back gracefully if Tk is unavailable."""
    output_path = PROJECT_ROOT / "examples" / "inverted_pendulum_lqr.png"

    try:
        _draw_plots(t, states, control_force)
        plt.savefig(output_path, dpi=150)
        plt.show()
    except TclError:
        plt.switch_backend("Agg")
        plt.close("all")
        _draw_plots(t, states, control_force)
        plt.savefig(output_path, dpi=150)
        print("Interactive Matplotlib window is unavailable in this environment.")
        print(f"Plot saved to: {output_path}")


def main():
    """Design an LQR gain and simulate nonlinear cart-pole stabilization."""
    M = 1.0
    m = 0.1
    l = 0.5
    g = 9.81
    b = 0.0
    Q = np.diag([1.0, 1.0, 100.0, 10.0])
    R = np.array([[0.1]])
    force_limit = 20.0
    initial_angle_degrees = 5.0
    x0 = [0.0, 0.0, np.radians(initial_angle_degrees), 0.0]
    t_span = (0.0, 5.0)
    num_points = 2000

    K, closed_loop_eigenvalues, _, _, _, _ = design_inverted_pendulum_lqr(
        M=M,
        m=m,
        l=l,
        g=g,
        b=b,
        Q=Q,
        R=R,
    )
    t, states, control_force = simulate_inverted_pendulum_lqr(
        x0,
        K,
        t_span=t_span,
        num_points=num_points,
        M=M,
        m=m,
        l=l,
        g=g,
        b=b,
        force_limit=force_limit,
    )

    theta = states[:, 2]
    cart_position = states[:, 0]

    print("Inverted Pendulum LQR Stabilization:")
    print(f"Initial angle: {initial_angle_degrees:.3f} degrees")
    print(f"Final angle: {np.degrees(theta[-1]):.3f} degrees")
    print(f"Maximum cart displacement: {np.max(np.abs(cart_position)):.3f} m")
    print(f"Maximum control force: {np.max(np.abs(control_force)):.3f} N")
    print()
    print("LQR gain K:")
    print(np.array2string(K, precision=4, suppress_small=True))
    print()
    print("Closed-loop eigenvalues:")
    print(np.array2string(closed_loop_eigenvalues, precision=4, suppress_small=True))
    print()
    print("LQR stabilizes the upright equilibrium locally.")
    print("This controller is designed from the linearized upright model.")

    _plot_response(t, states, control_force)


if __name__ == "__main__":
    main()
