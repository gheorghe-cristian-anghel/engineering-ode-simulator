"""Run and plot 6-DOF quadcopter circular trajectory tracking."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.quadcopter_trajectory_tracking import (  # noqa: E402
    circular_trajectory,
    simulate_quadcopter_trajectory_tracking,
)
from visualization.plot_style import apply_plot_style, format_axes, save_figure  # noqa: E402


def _draw_plots(result):
    """Draw 3D trajectory, position tracking, error, and command plots."""
    apply_plot_style()

    time = result["time"]
    states = result["states"]
    controls = result["controls"]
    reference_positions = result["reference_positions"]
    error_norm = result["tracking_error_norm"]
    attitude_commands_deg = np.degrees(result["attitude_commands"])
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
        label="Reference trajectory",
    )
    axes_trajectory.plot(
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
        label="Actual trajectory",
    )
    axes_trajectory.set_xlabel("x (m)")
    axes_trajectory.set_ylabel("y (m)")
    axes_trajectory.set_zlabel("z (m)")
    axes_trajectory.set_title("6-DOF Quadcopter Circular Trajectory Tracking")
    axes_trajectory.legend()

    for axis, label in enumerate(("x", "y", "z")):
        axes_position.plot(time, positions[:, axis], label=f"{label} actual")
        axes_position.plot(
            time,
            reference_positions[:, axis],
            linestyle="--",
            label=f"{label} reference",
        )
    format_axes(axes_position, title="Position Tracking", ylabel="Position (m)")
    axes_position.legend(ncol=3)

    axes_error.plot(time, error_norm, color="tab:red", label="Position error norm")
    format_axes(axes_error, title="Tracking Error", ylabel="Error (m)")

    axes_commands.plot(time, controls[:, 0], label="Thrust T")
    axes_commands.plot(time, attitude_commands_deg[:, 0], label="phi cmd (deg)")
    axes_commands.plot(time, attitude_commands_deg[:, 1], label="theta cmd (deg)")
    axes_commands.plot(time, attitude_commands_deg[:, 2], label="psi cmd (deg)")
    format_axes(
        axes_commands,
        title="Thrust and Attitude Commands",
        xlabel="Time (s)",
        ylabel="Command",
    )
    axes_commands.legend(ncol=4)

    figure.tight_layout()
    return figure


def _plot_response(result):
    """Save and display the circular trajectory tracking plots."""
    output_path = PROJECT_ROOT / "examples" / "quadcopter_trajectory_circle_tracking.png"

    figure = _draw_plots(result)
    save_figure(figure, output_path)
    print(f"Plot saved to: {output_path}")
    plt.show()


def main():
    """Track a slow circular path at constant altitude."""
    radius = 1.0
    altitude = 2.0
    angular_speed = 0.3
    initial_state = np.zeros(12)
    initial_state[0:3] = [radius, 0.0, altitude]

    result = simulate_quadcopter_trajectory_tracking(
        circular_trajectory(
            radius=radius,
            altitude=altitude,
            angular_speed=angular_speed,
        ),
        initial_state=initial_state,
        t_span=(0.0, 20.0),
        dt=0.02,
    )
    metrics = result["tracking_metrics"]

    print("6-DOF Quadcopter Circular Trajectory Tracking:")
    print("The simplified cascaded controller tracks a slow horizontal circle.")
    print()
    print(f"Radius: {radius:.3f} m")
    print(f"Altitude: {altitude:.3f} m")
    print(f"Angular speed: {angular_speed:.3f} rad/s")
    print(f"Final error norm: {metrics['final_position_error_norm']:.4f} m")
    print(f"RMS position error: {metrics['rms_position_error']:.4f} m")
    print(f"Max position error: {metrics['max_position_error']:.4f} m")
    print(f"Max thrust: {metrics['max_thrust']:.3f} N")
    print(f"Max absolute torque: {metrics['max_abs_torque']:.4f} N*m")

    _plot_response(result)


if __name__ == "__main__":
    main()
