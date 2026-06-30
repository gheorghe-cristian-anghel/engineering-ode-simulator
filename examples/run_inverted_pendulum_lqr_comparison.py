"""Compare open-loop and LQR-controlled inverted pendulum responses."""

import sys
from pathlib import Path
from _tkinter import TclError

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.inverted_pendulum import simulate_inverted_pendulum
from models.inverted_pendulum_lqr import (
    design_inverted_pendulum_lqr,
    simulate_inverted_pendulum_lqr,
)


def _draw_plots(t_open, open_loop_states, t_lqr, lqr_states, control_force):
    """Draw open-loop versus LQR response plots."""
    figure, axes = plt.subplots(3, 1, sharex=True)

    axes[0].plot(
        t_open,
        np.degrees(open_loop_states[:, 2]),
        label="Open-loop angle",
    )
    axes[0].plot(
        t_lqr,
        np.degrees(lqr_states[:, 2]),
        label="LQR angle",
        linestyle="--",
    )
    axes[0].axhline(0.0, color="gray", linestyle=":", label="Upright")
    axes[0].set_ylabel("Angle (deg)")
    axes[0].set_title("Inverted Pendulum Angle: Open Loop vs LQR")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(t_open, open_loop_states[:, 0], label="Open-loop cart position")
    axes[1].plot(
        t_lqr,
        lqr_states[:, 0],
        label="LQR cart position",
        linestyle="--",
    )
    axes[1].set_ylabel("Position (m)")
    axes[1].set_title("Cart Position")
    axes[1].grid(True)
    axes[1].legend()

    axes[2].plot(t_lqr, control_force, label="LQR control force", color="tab:red")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Force (N)")
    axes[2].set_title("Control Force Applied by LQR")
    axes[2].grid(True)
    axes[2].legend()

    figure.tight_layout()


def _plot_response(t_open, open_loop_states, t_lqr, lqr_states, control_force):
    """Plot response, falling back gracefully if Tk is unavailable."""
    output_path = PROJECT_ROOT / "examples" / "inverted_pendulum_lqr_comparison.png"

    try:
        _draw_plots(t_open, open_loop_states, t_lqr, lqr_states, control_force)
        plt.savefig(output_path, dpi=150)
        plt.show()
    except TclError:
        plt.switch_backend("Agg")
        plt.close("all")
        _draw_plots(t_open, open_loop_states, t_lqr, lqr_states, control_force)
        plt.savefig(output_path, dpi=150)
        print("Interactive Matplotlib window is unavailable in this environment.")
        print(f"Plot saved to: {output_path}")


def main():
    """Compare unstable open-loop motion with LQR stabilization."""
    M = 1.0
    m = 0.1
    l = 0.5
    g = 9.81
    b = 0.0
    force_limit = 20.0
    initial_angle_degrees = 5.0
    x0 = [0.0, 0.0, np.radians(initial_angle_degrees), 0.0]
    t_span = (0.0, 0.9)
    num_points = 1200

    K, closed_loop_eigenvalues, _, _, _, _ = design_inverted_pendulum_lqr(
        M=M,
        m=m,
        l=l,
        g=g,
        b=b,
    )
    t_open, open_loop_states = simulate_inverted_pendulum(
        x0,
        t_span=t_span,
        num_points=num_points,
        M=M,
        m=m,
        l=l,
        g=g,
        b=b,
    )
    t_lqr, lqr_states, control_force = simulate_inverted_pendulum_lqr(
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

    print("Open Loop vs LQR Inverted Pendulum:")
    print(f"Initial angle: {initial_angle_degrees:.3f} degrees")
    print(f"Open-loop final angle: {np.degrees(open_loop_states[-1, 2]):.3f} degrees")
    print(f"LQR final angle: {np.degrees(lqr_states[-1, 2]):.3f} degrees")
    print(f"Maximum LQR control force: {np.max(np.abs(control_force)):.3f} N")
    print()
    print("LQR gain K:")
    print(np.array2string(K, precision=4, suppress_small=True))
    print()
    print("Closed-loop eigenvalues:")
    print(np.array2string(closed_loop_eigenvalues, precision=4, suppress_small=True))
    print()
    print("The open-loop pendulum departs from upright.")
    print("The LQR-controlled nonlinear model stabilizes near upright locally.")

    _plot_response(t_open, open_loop_states, t_lqr, lqr_states, control_force)


if __name__ == "__main__":
    main()
