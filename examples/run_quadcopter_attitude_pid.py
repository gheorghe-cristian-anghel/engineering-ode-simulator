"""Run and plot PID-controlled quadcopter attitude tracking."""

import sys
from pathlib import Path
from _tkinter import TclError

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.quadcopter_attitude_control import (
    AXIS_NAMES,
    simulate_attitude_pid_control,
    summarize_attitude_response,
)


def _format_optional(value, unit=""):
    """Format optional metric values for printing."""
    if value is None:
        return "None"

    return f"{value:.3f}{unit}"


def _draw_plots(result):
    """Draw attitude angles, rates, torques, and tracking errors."""
    time = result["time"]
    angles_deg = result["angles_deg"]
    body_rates_deg = np.degrees(result["body_rates"])
    torques = result["torques"]
    errors_deg = result["errors_deg"]
    target_angles_deg = result["target_angles_deg"]

    figure, axes = plt.subplots(4, 1, sharex=True)

    for axis, axis_name in enumerate(AXIS_NAMES):
        axes[0].plot(time, angles_deg[:, axis], label=axis_name)
        axes[0].axhline(
            target_angles_deg[axis],
            color=f"C{axis}",
            linestyle=":",
            label=f"{axis_name} target",
        )
    axes[0].set_ylabel("Angle (deg)")
    axes[0].set_title("Quadcopter Attitude PID Tracking")
    axes[0].grid(True)
    axes[0].legend()

    for axis, rate_name in enumerate(("p", "q", "r")):
        axes[1].plot(time, body_rates_deg[:, axis], label=rate_name)
    axes[1].axhline(0.0, color="gray", linestyle=":")
    axes[1].set_ylabel("Rate (deg/s)")
    axes[1].set_title("Body Rates")
    axes[1].grid(True)
    axes[1].legend()

    for axis, torque_name in enumerate(("tau_phi", "tau_theta", "tau_psi")):
        axes[2].plot(time, torques[:, axis], label=torque_name)
    axes[2].axhline(0.0, color="gray", linestyle=":")
    axes[2].set_ylabel("Torque (N*m)")
    axes[2].set_title("PID Body Torque Commands")
    axes[2].grid(True)
    axes[2].legend()

    for axis, axis_name in enumerate(AXIS_NAMES):
        axes[3].plot(time, errors_deg[:, axis], label=f"{axis_name} error")
    axes[3].axhline(0.0, color="gray", linestyle=":")
    axes[3].set_xlabel("Time (s)")
    axes[3].set_ylabel("Error (deg)")
    axes[3].set_title("Attitude Tracking Error")
    axes[3].grid(True)
    axes[3].legend()

    figure.tight_layout()


def _plot_response(result):
    """Plot response, falling back gracefully if Tk is unavailable."""
    output_path = PROJECT_ROOT / "examples" / "quadcopter_attitude_pid.png"

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
    """Simulate PID tracking of commanded roll, pitch, and yaw angles."""
    target_angles = (10.0, -5.0, 15.0)
    Kp = (0.10, 0.10, 0.14)
    Ki = (0.001, 0.001, 0.0015)
    Kd = (0.09, 0.09, 0.14)
    torque_limits = (-0.2, 0.2)

    result = simulate_attitude_pid_control(
        target_angles=target_angles,
        initial_state=np.zeros(6),
        t_span=(0.0, 5.0),
        dt=0.01,
        Ixx=0.02,
        Iyy=0.02,
        Izz=0.04,
        Kp=Kp,
        Ki=Ki,
        Kd=Kd,
        torque_limits=torque_limits,
    )
    metrics = summarize_attitude_response(result)

    print("Quadcopter Attitude PID Control:")
    print("The PID controllers generate body torques to track commanded")
    print("roll, pitch, and yaw angles.")
    print()
    print(
        "Target angles: "
        f"roll={target_angles[0]:.3f} deg, "
        f"pitch={target_angles[1]:.3f} deg, "
        f"yaw={target_angles[2]:.3f} deg"
    )
    print(f"PID gains Kp: {Kp}")
    print(f"PID gains Ki: {Ki}")
    print(f"PID gains Kd: {Kd}")
    print(f"Torque limits: {torque_limits[0]:.3f} to {torque_limits[1]:.3f} N*m")
    print()

    for axis_name in AXIS_NAMES:
        print(f"{axis_name.title()} final angle: "
              f"{metrics.final_angle_deg[axis_name]:.3f} deg")
        print(f"{axis_name.title()} final error: "
              f"{metrics.final_error_deg[axis_name]:.3f} deg")
        print(f"{axis_name.title()} settling time: "
              f"{_format_optional(metrics.settling_time[axis_name], ' s')}")
        print(f"{axis_name.title()} max abs torque: "
              f"{metrics.max_abs_torque[axis_name]:.4f} N*m")

    _plot_response(result)


if __name__ == "__main__":
    main()
