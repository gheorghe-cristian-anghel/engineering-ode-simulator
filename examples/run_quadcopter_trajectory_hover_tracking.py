"""Run and plot 6-DOF quadcopter hover-point trajectory tracking."""

import sys
from pathlib import Path
from _tkinter import TclError

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.quadcopter_trajectory_tracking import (
    hover_trajectory,
    simulate_quadcopter_trajectory_tracking,
)


def _draw_plots(result):
    """Draw position tracking, error, trajectory, and control plots."""
    time = result["time"]
    states = result["states"]
    controls = result["controls"]
    reference_positions = result["reference_positions"]
    error_norm = result["tracking_error_norm"]
    positions = states[:, 0:3]

    figure = plt.figure(figsize=(10, 10))
    axes_position = figure.add_subplot(4, 1, 1)
    axes_error = figure.add_subplot(4, 1, 2)
    axes_trajectory = figure.add_subplot(4, 1, 3, projection="3d")
    axes_control = figure.add_subplot(4, 1, 4)

    for axis, label in enumerate(("x", "y", "z")):
        axes_position.plot(time, positions[:, axis], label=f"{label} actual")
        axes_position.plot(
            time,
            reference_positions[:, axis],
            linestyle="--",
            label=f"{label} reference",
        )
    axes_position.set_ylabel("Position (m)")
    axes_position.set_title("6-DOF Quadcopter Hover Tracking")
    axes_position.grid(True)
    axes_position.legend(ncol=3)

    axes_error.plot(time, error_norm, color="tab:red", label="Position error norm")
    axes_error.set_ylabel("Error (m)")
    axes_error.set_title("Tracking Error")
    axes_error.grid(True)
    axes_error.legend()

    axes_trajectory.plot(
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
        label="Actual trajectory",
    )
    axes_trajectory.scatter(
        reference_positions[-1, 0],
        reference_positions[-1, 1],
        reference_positions[-1, 2],
        color="tab:red",
        label="Hover target",
    )
    axes_trajectory.set_xlabel("x (m)")
    axes_trajectory.set_ylabel("y (m)")
    axes_trajectory.set_zlabel("z (m)")
    axes_trajectory.set_title("3D Trajectory")
    axes_trajectory.legend()

    axes_control.plot(time, controls[:, 0], label="Thrust T")
    axes_control.plot(time, controls[:, 1], label="tau_phi")
    axes_control.plot(time, controls[:, 2], label="tau_theta")
    axes_control.plot(time, controls[:, 3], label="tau_psi")
    axes_control.set_xlabel("Time (s)")
    axes_control.set_ylabel("Control")
    axes_control.set_title("Thrust and Torque Commands")
    axes_control.grid(True)
    axes_control.legend(ncol=4)

    figure.tight_layout()


def _plot_response(result):
    """Plot response, falling back gracefully if Tk is unavailable."""
    output_path = PROJECT_ROOT / "examples" / "quadcopter_trajectory_hover_tracking.png"

    try:
        _draw_plots(result)
        plt.savefig(output_path, dpi=150)
        plt.show()
    except TclError:
        plt.switch_backend("Agg")
        plt.close("all")
        _draw_plots(result)
        plt.savefig(output_path, dpi=150)
        print("Interactive Matplotlib window is unavailable in this environment.")
        print(f"Plot saved to: {output_path}")


def main():
    """Track a fixed hover point from an offset initial position."""
    target_position = (0.0, 0.0, 2.0)
    initial_state = np.zeros(12)
    initial_state[0:3] = [0.5, -0.5, 0.0]

    result = simulate_quadcopter_trajectory_tracking(
        hover_trajectory(target_position),
        initial_state=initial_state,
        t_span=(0.0, 8.0),
        dt=0.02,
    )
    metrics = result["tracking_metrics"]

    print("6-DOF Quadcopter Hover Trajectory Tracking:")
    print("The simplified cascaded controller tracks a fixed hover point.")
    print()
    print(
        "Final position: "
        f"x={metrics['final_position'][0]:.3f} m, "
        f"y={metrics['final_position'][1]:.3f} m, "
        f"z={metrics['final_position'][2]:.3f} m"
    )
    print(
        "Final reference position: "
        f"x={metrics['final_reference_position'][0]:.3f} m, "
        f"y={metrics['final_reference_position'][1]:.3f} m, "
        f"z={metrics['final_reference_position'][2]:.3f} m"
    )
    print(f"Final error norm: {metrics['final_position_error_norm']:.4f} m")
    print(f"RMS position error: {metrics['rms_position_error']:.4f} m")
    print(f"Max thrust: {metrics['max_thrust']:.3f} N")
    print(f"Max absolute torque: {metrics['max_abs_torque']:.4f} N*m")

    _plot_response(result)


if __name__ == "__main__":
    main()
