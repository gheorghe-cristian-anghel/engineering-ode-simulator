"""Run and plot quadcopter roll response to a constant body torque."""

import sys
from pathlib import Path
from _tkinter import TclError

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.quadcopter_attitude import (
    constant_torque,
    simulate_quadcopter_attitude,
)


def _draw_plots(t, phi, p, tau_phi):
    """Draw roll angle, roll rate, and torque command plots."""
    figure, axes = plt.subplots(3, 1, sharex=True)

    axes[0].plot(t, np.degrees(phi), label="Roll angle")
    axes[0].set_ylabel("Roll angle (deg)")
    axes[0].set_title("Quadcopter Roll Response to Constant Torque")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(t, np.degrees(p), label="Roll rate", color="tab:orange")
    axes[1].set_ylabel("Roll rate (deg/s)")
    axes[1].set_title("Roll Rate")
    axes[1].grid(True)
    axes[1].legend()

    axes[2].plot(t, tau_phi, label="Roll torque", color="tab:green")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Torque (N*m)")
    axes[2].set_title("Applied Body Torque")
    axes[2].grid(True)
    axes[2].legend()

    figure.tight_layout()


def _plot_response(t, phi, p, tau_phi):
    """Plot response, falling back gracefully if Tk is unavailable."""
    output_path = PROJECT_ROOT / "examples" / "quadcopter_attitude_roll_torque.png"

    try:
        _draw_plots(t, phi, p, tau_phi)
        plt.savefig(output_path, dpi=150)
        plt.show()
    except TclError:
        plt.switch_backend("Agg")
        plt.close("all")
        _draw_plots(t, phi, p, tau_phi)
        plt.savefig(output_path, dpi=150)
        print("Interactive Matplotlib window is unavailable in this environment.")
        print(f"Plot saved to: {output_path}")


def main():
    """Simulate roll motion from a constant roll torque."""
    Ixx = 0.02
    Iyy = 0.02
    Izz = 0.04
    tau_phi = 0.001
    t_span = (0.0, 3.0)
    num_points = 1000
    torque_func = constant_torque(tau_phi=tau_phi)

    t, states, torques = simulate_quadcopter_attitude(
        t_span=t_span,
        num_points=num_points,
        Ixx=Ixx,
        Iyy=Iyy,
        Izz=Izz,
        torque_func=torque_func,
    )

    phi = states[:, 0]
    p = states[:, 3]

    print("Quadcopter Attitude Roll Torque Response:")
    print("State: [phi, theta, psi, p, q, r]")
    print(f"Ixx: {Ixx:.4f} kg*m^2")
    print(f"Iyy: {Iyy:.4f} kg*m^2")
    print(f"Izz: {Izz:.4f} kg*m^2")
    print(f"Applied roll torque: {tau_phi:.4f} N*m")
    print(f"Final roll angle: {np.degrees(phi[-1]):.3f} deg")
    print(f"Final roll rate: {np.degrees(p[-1]):.3f} deg/s")
    print()
    print("A constant torque causes angular acceleration, so roll rate")
    print("increases and roll angle grows in this open-loop model.")

    _plot_response(t, phi, p, torques[:, 0])


if __name__ == "__main__":
    main()
