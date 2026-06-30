"""Run and plot a full 6-DOF quadcopter response to body torque steps."""

import sys
from pathlib import Path
from _tkinter import TclError

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.quadcopter_6dof import (
    control_step,
    hover_control,
    simulate_quadcopter_6dof,
)


def _draw_plots(t, states, controls, step_time):
    """Draw attitude, rate, position, and control histories."""
    angles_deg = np.degrees(states[:, 6:9])
    rates_deg = np.degrees(states[:, 9:12])
    positions = states[:, 0:3]

    figure, axes = plt.subplots(4, 1, sharex=True)

    for axis, label in enumerate(("roll phi", "pitch theta", "yaw psi")):
        axes[0].plot(t, angles_deg[:, axis], label=label)
    axes[0].axvline(step_time, color="black", linestyle="--", label="Torque step")
    axes[0].set_ylabel("Angle (deg)")
    axes[0].set_title("Full 6-DOF Quadcopter Torque Response: Euler Angles")
    axes[0].grid(True)
    axes[0].legend()

    for axis, label in enumerate(("p", "q", "r")):
        axes[1].plot(t, rates_deg[:, axis], label=label)
    axes[1].axvline(step_time, color="black", linestyle="--", label="Torque step")
    axes[1].set_ylabel("Rate (deg/s)")
    axes[1].set_title("Body Rates")
    axes[1].grid(True)
    axes[1].legend()

    for axis, label in enumerate(("x", "y", "z")):
        axes[2].plot(t, positions[:, axis], label=label)
    axes[2].axvline(step_time, color="black", linestyle="--", label="Torque step")
    axes[2].set_ylabel("Position (m)")
    axes[2].set_title("Position Response")
    axes[2].grid(True)
    axes[2].legend()

    axes[3].plot(t, controls[:, 0], label="T")
    axes[3].plot(t, controls[:, 1], label="tau_phi")
    axes[3].plot(t, controls[:, 2], label="tau_theta")
    axes[3].plot(t, controls[:, 3], label="tau_psi")
    axes[3].axvline(step_time, color="black", linestyle="--", label="Torque step")
    axes[3].set_xlabel("Time (s)")
    axes[3].set_ylabel("Control")
    axes[3].set_title("Thrust and Body Torques")
    axes[3].grid(True)
    axes[3].legend()

    figure.tight_layout()


def _plot_response(t, states, controls, step_time):
    """Plot response, falling back gracefully if Tk is unavailable."""
    output_path = PROJECT_ROOT / "examples" / "quadcopter_6dof_torque_response.png"

    try:
        _draw_plots(t, states, controls, step_time)
        plt.savefig(output_path, dpi=150)
        plt.show()
    except TclError:
        plt.switch_backend("Agg")
        plt.close("all")
        _draw_plots(t, states, controls, step_time)
        plt.savefig(output_path, dpi=150)
        print("Interactive Matplotlib window is unavailable in this environment.")
        print(f"Plot saved to: {output_path}")


def main():
    """Simulate hover thrust with a small multi-axis body torque step."""
    m = 1.0
    g = 9.81
    step_time = 1.0
    hover = hover_control(m=m, g=g)
    control_after = hover.copy()
    control_after[1:] = [0.0008, -0.0006, 0.0004]
    control_func = control_step(step_time, before=hover, after=control_after)

    t, states, controls = simulate_quadcopter_6dof(
        initial_state=np.zeros(12),
        t_span=(0.0, 4.0),
        num_points=1000,
        m=m,
        g=g,
        control_func=control_func,
    )

    final_angles_deg = np.degrees(states[-1, 6:9])
    final_rates_deg = np.degrees(states[-1, 9:12])
    final_position = states[-1, 0:3]

    print("Full 6-DOF Quadcopter Torque Response:")
    print("Hover thrust is held while body torques are stepped after 1 second.")
    print(
        "Torque after step: "
        f"tau_phi={control_after[1]:.4f} N*m, "
        f"tau_theta={control_after[2]:.4f} N*m, "
        f"tau_psi={control_after[3]:.4f} N*m"
    )
    print(
        "Final Euler angles: "
        f"roll={final_angles_deg[0]:.3f} deg, "
        f"pitch={final_angles_deg[1]:.3f} deg, "
        f"yaw={final_angles_deg[2]:.3f} deg"
    )
    print(
        "Final body rates: "
        f"p={final_rates_deg[0]:.3f} deg/s, "
        f"q={final_rates_deg[1]:.3f} deg/s, "
        f"r={final_rates_deg[2]:.3f} deg/s"
    )
    print(
        "Final position: "
        f"x={final_position[0]:.3f} m, "
        f"y={final_position[1]:.3f} m, "
        f"z={final_position[2]:.3f} m"
    )

    _plot_response(t, states, controls, step_time)


if __name__ == "__main__":
    main()
