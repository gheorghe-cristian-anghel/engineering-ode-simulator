"""Run and plot 6-DOF quadcopter static obstacle avoidance."""

import sys
from pathlib import Path
from _tkinter import TclError

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.quadcopter_obstacle_avoidance import (  # noqa: E402
    SphericalObstacle,
    simulate_quadcopter_obstacle_avoidance,
)
from analysis.quadcopter_waypoint_following import waypoint_trajectory  # noqa: E402
from visualization.plot_style import (  # noqa: E402
    apply_plot_style,
    format_axes,
    save_figure,
)


def _plot_obstacle_sphere(axis, obstacle):
    """Draw a transparent spherical obstacle."""
    u = np.linspace(0.0, 2.0 * np.pi, 32)
    v = np.linspace(0.0, np.pi, 16)
    x = obstacle.center[0] + obstacle.radius * np.outer(np.cos(u), np.sin(v))
    y = obstacle.center[1] + obstacle.radius * np.outer(np.sin(u), np.sin(v))
    z = obstacle.center[2] + obstacle.radius * np.outer(np.ones_like(u), np.cos(v))

    axis.plot_wireframe(
        x,
        y,
        z,
        color="tab:red",
        alpha=0.45,
        linewidth=0.8,
    )


def _draw_plots(result, waypoints):
    """Draw trajectory, tracking, clearance, avoidance, and command plots."""
    apply_plot_style()

    time = result["time"]
    states = result["states"]
    controls = result["controls"]
    reference_positions = result["reference_positions"]
    nearest_clearances = result["nearest_clearances"]
    avoidance_norm = result["avoidance_acceleration_norm"]
    positions = states[:, 0:3]
    obstacles = result["obstacles"]

    figure = plt.figure(figsize=(11, 15))
    axes_trajectory = figure.add_subplot(5, 1, 1, projection="3d")
    axes_position = figure.add_subplot(5, 1, 2)
    axes_clearance = figure.add_subplot(5, 1, 3)
    axes_avoidance = figure.add_subplot(5, 1, 4)
    axes_control = figure.add_subplot(5, 1, 5)

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
        color="tab:green",
        label="Waypoints",
    )
    for obstacle in obstacles:
        _plot_obstacle_sphere(axes_trajectory, obstacle)
    axes_trajectory.set_zlabel("z (m)")
    format_axes(
        axes_trajectory,
        title="Quadcopter Static Obstacle Avoidance",
        xlabel="x (m)",
        ylabel="y (m)",
    )

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

    axes_clearance.plot(
        time,
        nearest_clearances,
        color="tab:red",
        label="Nearest obstacle clearance",
    )
    axes_clearance.axhline(0.0, color="black", linestyle=":", label="Obstacle surface")
    format_axes(
        axes_clearance,
        title="Obstacle Clearance",
        ylabel="Clearance (m)",
    )

    axes_avoidance.plot(
        time,
        avoidance_norm,
        color="tab:purple",
        label="Avoidance acceleration magnitude",
    )
    format_axes(
        axes_avoidance,
        title="Repulsive Avoidance Command",
        ylabel="Acceleration (m/s^2)",
    )

    axes_control.plot(time, controls[:, 0], label="Thrust T")
    axes_control.plot(time, controls[:, 1], label="tau_phi")
    axes_control.plot(time, controls[:, 2], label="tau_theta")
    axes_control.plot(time, controls[:, 3], label="tau_psi")
    format_axes(
        axes_control,
        title="Thrust and Torque Commands",
        xlabel="Time (s)",
        ylabel="Control",
    )
    axes_control.legend(ncol=4)

    figure.tight_layout()
    return figure


def _plot_response(result, waypoints):
    """Plot response, falling back gracefully if Tk is unavailable."""
    output_path = PROJECT_ROOT / "examples" / "quadcopter_obstacle_avoidance.png"

    try:
        figure = _draw_plots(result, waypoints)
        save_figure(figure, output_path)
        print(f"Plot saved to: {output_path}")
        plt.show()
    except TclError:
        plt.switch_backend("Agg")
        plt.close("all")
        figure = _draw_plots(result, waypoints)
        save_figure(figure, output_path)
        print("Interactive Matplotlib window is unavailable in this environment.")
        print(f"Plot saved to: {output_path}")


def main():
    """Track a waypoint path while avoiding one static spherical obstacle."""
    waypoints = np.array(
        [
            [0.0, 0.0, 1.5],
            [2.0, 0.0, 1.5],
        ]
    )
    segment_time = 8.0
    obstacle = SphericalObstacle(
        center=np.array([1.0, 0.28, 1.5]),
        radius=0.25,
        influence_radius=1.0,
    )
    initial_state = np.zeros(12)
    initial_state[0:3] = waypoints[0]

    trajectory_func = waypoint_trajectory(
        waypoints,
        segment_time=segment_time,
        smoothing="smoothstep",
    )
    result = simulate_quadcopter_obstacle_avoidance(
        trajectory_func,
        [obstacle],
        initial_state=initial_state,
        t_span=(0.0, 10.0),
        dt=0.02,
        Kd_pos=(2.0, 2.0, 3.2),
        Kp_att=(0.25, 0.25, 0.16),
        Kd_att=(0.12, 0.12, 0.08),
        avoidance_gain=0.03,
        max_avoidance_acceleration=1.5,
    )
    metrics = result["obstacle_metrics"]
    final_position = result["states"][-1, 0:3]
    final_target = waypoints[-1]

    print("6-DOF Quadcopter Static Obstacle Avoidance:")
    print(
        "The obstacle avoidance term adds a repulsive acceleration near "
        "static obstacles while the trajectory tracker still drives the "
        "drone toward the goal."
    )
    print()
    print(
        "Obstacle center: "
        f"x={obstacle.center[0]:.3f} m, "
        f"y={obstacle.center[1]:.3f} m, "
        f"z={obstacle.center[2]:.3f} m"
    )
    print(f"Obstacle radius: {obstacle.radius:.3f} m")
    print(f"Obstacle influence radius: {obstacle.influence_radius:.3f} m")
    print(
        "Final target: "
        f"x={final_target[0]:.3f} m, "
        f"y={final_target[1]:.3f} m, "
        f"z={final_target[2]:.3f} m"
    )
    print(
        "Final position: "
        f"x={final_position[0]:.3f} m, "
        f"y={final_position[1]:.3f} m, "
        f"z={final_position[2]:.3f} m"
    )
    print(f"Final position error: {metrics['final_position_error_norm']:.4f} m")
    print(f"Minimum obstacle clearance: {metrics['min_clearance']:.4f} m")
    print(f"RMS position error: {metrics['rms_position_error']:.4f} m")
    print(
        "Max avoidance acceleration: "
        f"{metrics['max_avoidance_acceleration']:.4f} m/s^2"
    )
    print(f"Max thrust: {metrics['max_thrust']:.3f} N")
    print(f"Max absolute torque: {metrics['max_abs_torque']:.4f} N*m")

    _plot_response(result, waypoints)


if __name__ == "__main__":
    main()
