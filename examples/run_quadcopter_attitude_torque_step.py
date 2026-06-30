"""Run and plot quadcopter attitude response to a body torque step."""

import sys
from pathlib import Path
from _tkinter import TclError

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.quadcopter_attitude import (
    simulate_quadcopter_attitude,
    torque_step,
)


def _draw_plots(t, angles, rates, torques, step_time):
    """Draw attitude angle, body rate, and torque command plots."""
    figure, axes = plt.subplots(3, 1, sharex=True)

    axes[0].plot(t, np.degrees(angles[:, 0]), label="Roll phi")
    axes[0].plot(t, np.degrees(angles[:, 1]), label="Pitch theta")
    axes[0].plot(t, np.degrees(angles[:, 2]), label="Yaw psi")
    axes[0].axvline(step_time, color="black", linestyle="--", label="Torque step")
    axes[0].set_ylabel("Angle (deg)")
    axes[0].set_title("Quadcopter Attitude Response to Torque Step")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(t, np.degrees(rates[:, 0]), label="p roll rate")
    axes[1].plot(t, np.degrees(rates[:, 1]), label="q pitch rate")
    axes[1].plot(t, np.degrees(rates[:, 2]), label="r yaw rate")
    axes[1].axvline(step_time, color="black", linestyle="--", label="Torque step")
    axes[1].set_ylabel("Rate (deg/s)")
    axes[1].set_title("Body Rates")
    axes[1].grid(True)
    axes[1].legend()

    axes[2].plot(t, torques[:, 0], label="tau_phi")
    axes[2].plot(t, torques[:, 1], label="tau_theta")
    axes[2].plot(t, torques[:, 2], label="tau_psi")
    axes[2].axvline(step_time, color="black", linestyle="--", label="Torque step")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Torque (N*m)")
    axes[2].set_title("Applied Body Torques")
    axes[2].grid(True)
    axes[2].legend()

    figure.tight_layout()


def _plot_response(t, angles, rates, torques, step_time):
    """Plot response, falling back gracefully if Tk is unavailable."""
    output_path = PROJECT_ROOT / "examples" / "quadcopter_attitude_torque_step.png"

    try:
        _draw_plots(t, angles, rates, torques, step_time)
        plt.savefig(output_path, dpi=150)
        plt.show()
    except TclError:
        plt.switch_backend("Agg")
        plt.close("all")
        _draw_plots(t, angles, rates, torques, step_time)
        plt.savefig(output_path, dpi=150)
        print("Interactive Matplotlib window is unavailable in this environment.")
        print(f"Plot saved to: {output_path}")


def main():
    """Simulate attitude motion from a multi-axis body torque step."""
    Ixx = 0.02
    Iyy = 0.02
    Izz = 0.04
    step_time = 1.0
    torque_after = (0.001, -0.0008, 0.0005)
    t_span = (0.0, 4.0)
    num_points = 1200
    torque_func = torque_step(
        step_time,
        before=(0.0, 0.0, 0.0),
        after=torque_after,
    )

    t, states, torques = simulate_quadcopter_attitude(
        t_span=t_span,
        num_points=num_points,
        Ixx=Ixx,
        Iyy=Iyy,
        Izz=Izz,
        torque_func=torque_func,
    )

    angles = states[:, 0:3]
    rates = states[:, 3:6]

    print("Quadcopter Attitude Torque Step Response:")
    print("State: [phi, theta, psi, p, q, r]")
    print(f"Ixx: {Ixx:.4f} kg*m^2")
    print(f"Iyy: {Iyy:.4f} kg*m^2")
    print(f"Izz: {Izz:.4f} kg*m^2")
    print(f"Torque step time: {step_time:.3f} s")
    print(
        "Torque after step: "
        f"tau_phi={torque_after[0]:.4f} N*m, "
        f"tau_theta={torque_after[1]:.4f} N*m, "
        f"tau_psi={torque_after[2]:.4f} N*m"
    )
    print(
        "Final angles: "
        f"roll={np.degrees(angles[-1, 0]):.3f} deg, "
        f"pitch={np.degrees(angles[-1, 1]):.3f} deg, "
        f"yaw={np.degrees(angles[-1, 2]):.3f} deg"
    )
    print(
        "Final angular rates: "
        f"p={np.degrees(rates[-1, 0]):.3f} deg/s, "
        f"q={np.degrees(rates[-1, 1]):.3f} deg/s, "
        f"r={np.degrees(rates[-1, 2]):.3f} deg/s"
    )
    print()
    print("The torque step starts angular acceleration after the step time.")

    _plot_response(t, angles, rates, torques, step_time)


if __name__ == "__main__":
    main()
