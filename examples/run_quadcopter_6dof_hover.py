"""Run and plot a full 6-DOF quadcopter hover response."""

import sys
from pathlib import Path
from _tkinter import TclError

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.quadcopter_6dof import (
    constant_control,
    hover_control,
    simulate_quadcopter_6dof,
)


def _draw_plots(t, states):
    """Draw position, velocity, and Euler-angle histories."""
    positions = states[:, 0:3]
    velocities = states[:, 3:6]
    angles_deg = np.degrees(states[:, 6:9])

    figure, axes = plt.subplots(3, 1, sharex=True)

    for axis, label in enumerate(("x", "y", "z")):
        axes[0].plot(t, positions[:, axis], label=label)
    axes[0].set_ylabel("Position (m)")
    axes[0].set_title("Full 6-DOF Quadcopter Hover: Position")
    axes[0].grid(True)
    axes[0].legend()

    for axis, label in enumerate(("vx", "vy", "vz")):
        axes[1].plot(t, velocities[:, axis], label=label)
    axes[1].set_ylabel("Velocity (m/s)")
    axes[1].set_title("Velocity")
    axes[1].grid(True)
    axes[1].legend()

    for axis, label in enumerate(("roll phi", "pitch theta", "yaw psi")):
        axes[2].plot(t, angles_deg[:, axis], label=label)
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Angle (deg)")
    axes[2].set_title("Euler Angles")
    axes[2].grid(True)
    axes[2].legend()

    figure.tight_layout()


def _plot_response(t, states):
    """Plot response, falling back gracefully if Tk is unavailable."""
    output_path = PROJECT_ROOT / "examples" / "quadcopter_6dof_hover.png"

    try:
        _draw_plots(t, states)
        plt.savefig(output_path, dpi=150)
        plt.show()
    except TclError:
        plt.switch_backend("Agg")
        plt.close("all")
        _draw_plots(t, states)
        plt.savefig(output_path, dpi=150)
        print("Interactive Matplotlib window is unavailable in this environment.")
        print(f"Plot saved to: {output_path}")


def main():
    """Simulate level hover with total thrust equal to vehicle weight."""
    m = 1.0
    g = 9.81
    hover = hover_control(m=m, g=g)
    control_func = constant_control(*hover)

    t, states, _ = simulate_quadcopter_6dof(
        initial_state=np.zeros(12),
        t_span=(0.0, 5.0),
        num_points=1000,
        m=m,
        g=g,
        control_func=control_func,
    )

    final_position = states[-1, 0:3]
    final_velocity = states[-1, 3:6]
    final_angles_deg = np.degrees(states[-1, 6:9])

    print("Full 6-DOF Quadcopter Hover:")
    print("State: [x, y, z, vx, vy, vz, phi, theta, psi, p, q, r]")
    print("Inertial z is positive upward.")
    print(f"Mass: {m:.3f} kg")
    print(f"Gravity: {g:.3f} m/s^2")
    print(f"Hover thrust: {hover[0]:.3f} N")
    print(
        "Final position: "
        f"x={final_position[0]:.6f} m, "
        f"y={final_position[1]:.6f} m, "
        f"z={final_position[2]:.6f} m"
    )
    print(
        "Final velocity: "
        f"vx={final_velocity[0]:.6f} m/s, "
        f"vy={final_velocity[1]:.6f} m/s, "
        f"vz={final_velocity[2]:.6f} m/s"
    )
    print(
        "Final Euler angles: "
        f"roll={final_angles_deg[0]:.6f} deg, "
        f"pitch={final_angles_deg[1]:.6f} deg, "
        f"yaw={final_angles_deg[2]:.6f} deg"
    )
    print()
    print("At zero attitude, hover thrust balances gravity in the 6-DOF model.")

    _plot_response(t, states)


if __name__ == "__main__":
    main()
