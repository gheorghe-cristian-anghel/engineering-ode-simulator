"""Run and plot quadcopter attitude PID disturbance rejection."""

import sys
from pathlib import Path
from _tkinter import TclError

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.quadcopter_attitude_control import (
    AXIS_NAMES,
    disturbance_torque_step,
    simulate_attitude_pid_control,
    summarize_attitude_response,
)


def _format_optional(value, unit=""):
    """Format optional metric values for printing."""
    if value is None:
        return "None"

    return f"{value:.3f}{unit}"


def _draw_plots(result, disturbance_time):
    """Draw attitude, control torque, and disturbance torque plots."""
    time = result["time"]
    angles_deg = result["angles_deg"]
    control_torques = result["control_torques"]
    disturbance_torques = result["disturbance_torques"]
    target_angles_deg = result["target_angles_deg"]

    figure, axes = plt.subplots(3, 1, sharex=True)

    for axis, axis_name in enumerate(AXIS_NAMES):
        axes[0].plot(time, angles_deg[:, axis], label=axis_name)
        axes[0].axhline(
            target_angles_deg[axis],
            color=f"C{axis}",
            linestyle=":",
            label=f"{axis_name} target",
        )
    axes[0].axvline(
        disturbance_time,
        color="black",
        linestyle="--",
        label="Disturbance",
    )
    axes[0].set_ylabel("Angle (deg)")
    axes[0].set_title("Quadcopter Attitude PID Disturbance Rejection")
    axes[0].grid(True)
    axes[0].legend()

    for axis, torque_name in enumerate(("tau_phi", "tau_theta", "tau_psi")):
        axes[1].plot(time, control_torques[:, axis], label=torque_name)
    axes[1].axvline(disturbance_time, color="black", linestyle="--")
    axes[1].axhline(0.0, color="gray", linestyle=":")
    axes[1].set_ylabel("Torque (N*m)")
    axes[1].set_title("PID Control Torques")
    axes[1].grid(True)
    axes[1].legend()

    for axis, torque_name in enumerate(("roll disturbance", "pitch disturbance", "yaw disturbance")):
        axes[2].plot(time, disturbance_torques[:, axis], label=torque_name)
    axes[2].axvline(disturbance_time, color="black", linestyle="--")
    axes[2].axhline(0.0, color="gray", linestyle=":")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Torque (N*m)")
    axes[2].set_title("External Disturbance Torque")
    axes[2].grid(True)
    axes[2].legend()

    figure.tight_layout()


def _plot_response(result, disturbance_time):
    """Plot response, falling back gracefully if Tk is unavailable."""
    output_path = (
        PROJECT_ROOT / "examples" / "quadcopter_attitude_pid_disturbance.png"
    )

    try:
        _draw_plots(result, disturbance_time)
        plt.savefig(output_path, dpi=150)
        plt.show()
    except TclError:
        plt.switch_backend("Agg")
        plt.close("all")
        _draw_plots(result, disturbance_time)
        plt.savefig(output_path, dpi=150)
        print("Interactive Matplotlib window is unavailable in this environment.")
        print(f"Plot saved to: {output_path}")


def main():
    """Simulate attitude control with a step roll torque disturbance."""
    target_angles = (0.0, 0.0, 0.0)
    disturbance_time = 2.5
    disturbance_after = (0.02, 0.0, 0.0)
    torque_limits = (-0.2, 0.2)
    disturbance_torque_func = disturbance_torque_step(
        disturbance_time,
        before=(0.0, 0.0, 0.0),
        after=disturbance_after,
    )

    result = simulate_attitude_pid_control(
        target_angles=target_angles,
        initial_state=np.zeros(6),
        t_span=(0.0, 12.0),
        dt=0.01,
        Kp=(0.12, 0.10, 0.14),
        Ki=(0.05, 0.001, 0.0015),
        Kd=(0.12, 0.09, 0.14),
        torque_limits=torque_limits,
        disturbance_torque_func=disturbance_torque_func,
    )
    metrics = summarize_attitude_response(result)

    print("Quadcopter Attitude PID Disturbance Response:")
    print("An external roll disturbance torque is applied after 2.5 seconds.")
    print("The attitude controller adjusts torque to recover toward level attitude.")
    print()
    print(f"Target angles: {target_angles} deg")
    print(f"Disturbance time: {disturbance_time:.3f} s")
    print(
        "Disturbance torque after step: "
        f"roll={disturbance_after[0]:.4f} N*m, "
        f"pitch={disturbance_after[1]:.4f} N*m, "
        f"yaw={disturbance_after[2]:.4f} N*m"
    )
    print(f"Torque limits: {torque_limits[0]:.3f} to {torque_limits[1]:.3f} N*m")
    print()

    for axis_name in AXIS_NAMES:
        print(f"{axis_name.title()} final angle: "
              f"{metrics.final_angle_deg[axis_name]:.3f} deg")
        print(f"{axis_name.title()} final error: "
              f"{metrics.final_error_deg[axis_name]:.3f} deg")
        print(f"{axis_name.title()} settling time: "
              f"{_format_optional(metrics.settling_time[axis_name], ' s')}")
        print(f"{axis_name.title()} max abs control torque: "
              f"{metrics.max_abs_torque[axis_name]:.4f} N*m")

    _plot_response(result, disturbance_time)


if __name__ == "__main__":
    main()
