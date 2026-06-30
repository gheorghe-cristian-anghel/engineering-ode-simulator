"""Run and plot 6-DOF quadcopter waypoint following."""

import sys
from pathlib import Path
from _tkinter import TclError

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.quadcopter_waypoint_following import (  # noqa: E402
    simulate_quadcopter_waypoint_following,
    total_waypoint_time,
)


def _draw_plots(result):
    """Draw waypoint path, position tracking, error, and command plots."""
    time = result["time"]
    states = result["states"]
    controls = result["controls"]
    reference_positions = result["reference_positions"]
    error_norm = result["tracking_error_norm"]
    attitude_commands_deg = np.degrees(result["attitude_commands"])
    waypoints = result["waypoints"]
    positions = states[:, 0:3]

    figure = plt.figure(figsize=(10, 12))
    axes_trajectory = figure.add_subplot(4, 1, 1, projection="3d")
    axes_position = figure.add_subplot(4, 1, 2)
    axes_error = figure.add_subplot(4, 1, 3)
    axes_commands = figure.add_subplot(4, 1, 4)

    axes_trajectory.plot(
        reference_positions[:, 0],
        reference_positions[:, 1],
        reference_positions[:, 2],
        linestyle="--",
        label="Reference path",
    )
    axes_trajectory.plot(
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
        label="Actual trajectory",
    )
    axes_trajectory.plot(
        waypoints[:, 0],
        waypoints[:, 1],
        waypoints[:, 2],
        marker="o",
        linestyle=":",
        color="tab:red",
        label="Waypoints",
    )
    axes_trajectory.set_xlabel("x (m)")
    axes_trajectory.set_ylabel("y (m)")
    axes_trajectory.set_zlabel("z (m)")
    axes_trajectory.set_title("6-DOF Quadcopter Waypoint Following")
    axes_trajectory.legend()

    for axis, label in enumerate(("x", "y", "z")):
        axes_position.plot(time, positions[:, axis], label=f"{label} actual")
        axes_position.plot(
            time,
            reference_positions[:, axis],
            linestyle="--",
            label=f"{label} reference",
        )
    axes_position.set_ylabel("Position (m)")
    axes_position.set_title("Position Tracking")
    axes_position.grid(True)
    axes_position.legend(ncol=3)

    axes_error.plot(time, error_norm, color="tab:red", label="Position error norm")
    axes_error.set_ylabel("Error (m)")
    axes_error.set_title("Tracking Error")
    axes_error.grid(True)
    axes_error.legend()

    axes_commands.plot(time, controls[:, 0], label="Thrust T")
    axes_commands.plot(time, attitude_commands_deg[:, 0], label="phi cmd (deg)")
    axes_commands.plot(time, attitude_commands_deg[:, 1], label="theta cmd (deg)")
    axes_commands.plot(time, attitude_commands_deg[:, 2], label="psi cmd (deg)")
    axes_commands.set_xlabel("Time (s)")
    axes_commands.set_ylabel("Command")
    axes_commands.set_title("Thrust and Attitude Commands")
    axes_commands.grid(True)
    axes_commands.legend(ncol=4)

    figure.tight_layout()


def _plot_response(result):
    """Plot response, falling back gracefully if Tk is unavailable."""
    output_path = PROJECT_ROOT / "examples" / "quadcopter_waypoint_following.png"

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
    """Track a smooth reference path through several 3D waypoints."""
    waypoints = [
        (0.0, 0.0, 1.0),
        (1.0, 0.0, 1.5),
        (1.0, 1.0, 2.0),
        (0.0, 1.0, 1.5),
        (0.0, 0.0, 2.0),
    ]
    segment_time = 4.0
    smoothing = "smoothstep"

    result = simulate_quadcopter_waypoint_following(
        waypoints,
        segment_time=segment_time,
        smoothing=smoothing,
        dt=0.02,
        hold_time=2.0,
    )
    metrics = result["waypoint_metrics"]
    final_position = metrics["final_position"]
    final_waypoint = metrics["final_waypoint"]

    print("6-DOF Quadcopter Waypoint Following:")
    print(
        "The waypoint follower converts discrete waypoints into a smooth "
        "reference trajectory and tracks it using the 6-DOF quadcopter "
        "controller."
    )
    print()
    print(f"Number of waypoints: {metrics['number_of_waypoints']}")
    print(f"Segment time: {segment_time:.3f} s")
    print(f"Smoothing: {smoothing}")
    print(f"Total trajectory time: {total_waypoint_time(waypoints, segment_time):.3f} s")
    print(
        "Final waypoint: "
        f"x={final_waypoint[0]:.3f} m, "
        f"y={final_waypoint[1]:.3f} m, "
        f"z={final_waypoint[2]:.3f} m"
    )
    print(
        "Final position: "
        f"x={final_position[0]:.3f} m, "
        f"y={final_position[1]:.3f} m, "
        f"z={final_position[2]:.3f} m"
    )
    print(f"Final waypoint error: {metrics['final_waypoint_error']:.4f} m")
    print(f"RMS position error: {metrics['rms_position_error']:.4f} m")
    print(f"Max position error: {metrics['max_position_error']:.4f} m")
    print(f"Max thrust: {metrics['max_thrust']:.3f} N")
    print(f"Max absolute torque: {metrics['max_abs_torque']:.4f} N*m")

    _plot_response(result)


if __name__ == "__main__":
    main()
