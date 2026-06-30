"""Run and plot PID altitude control with a downward force disturbance."""

import sys
from pathlib import Path
from _tkinter import TclError

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.quadcopter_altitude_control import (
    downward_force_step,
    simulate_altitude_pid_control,
    summarize_altitude_response,
)


def _format_optional(value, unit=""):
    """Format optional metric values for printing."""
    if value is None:
        return "None"

    return f"{value:.3f}{unit}"


def _draw_plots(result, disturbance_time):
    """Draw altitude, thrust, and disturbance force plots."""
    time = result["time"]
    altitude = result["altitude"]
    thrust = result["thrust"]
    disturbance_force = result["disturbance_force"]
    target_altitude = result["target_altitude"]
    hover = result["hover_thrust"]

    figure, axes = plt.subplots(3, 1, sharex=True)

    axes[0].plot(time, altitude, label="Altitude")
    axes[0].axhline(target_altitude, color="gray", linestyle=":", label="Target")
    axes[0].axvline(disturbance_time, color="black", linestyle="--", label="Disturbance")
    axes[0].set_ylabel("Altitude (m)")
    axes[0].set_title("Quadcopter Altitude PID Disturbance Rejection")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(time, thrust, label="Thrust command", color="tab:green")
    axes[1].axhline(hover, color="gray", linestyle=":", label="Hover thrust")
    axes[1].axvline(disturbance_time, color="black", linestyle="--", label="Disturbance")
    axes[1].set_ylabel("Thrust (N)")
    axes[1].set_title("PID Thrust Command")
    axes[1].grid(True)
    axes[1].legend()

    axes[2].plot(time, disturbance_force, label="Downward force", color="tab:red")
    axes[2].axvline(disturbance_time, color="black", linestyle="--", label="Disturbance")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Force (N)")
    axes[2].set_title("Downward Force Disturbance")
    axes[2].grid(True)
    axes[2].legend()

    figure.tight_layout()


def _plot_response(result, disturbance_time):
    """Plot response, falling back gracefully if Tk is unavailable."""
    output_path = PROJECT_ROOT / "examples" / "quadcopter_altitude_pid_disturbance.png"

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
    """Simulate altitude control with a step downward force disturbance."""
    target_altitude = 5.0
    disturbance_time = 10.0
    disturbance_force_after = 1.0
    Kp = 2.5
    Ki = 0.8
    Kd = 3.0
    thrust_limits = (0.0, 25.0)
    disturbance_force_func = downward_force_step(
        disturbance_time,
        force_before=0.0,
        force_after=disturbance_force_after,
    )

    result = simulate_altitude_pid_control(
        target_altitude=target_altitude,
        t_span=(0.0, 16.0),
        dt=0.01,
        m=1.0,
        g=9.81,
        c_drag=0.2,
        Kp=Kp,
        Ki=Ki,
        Kd=Kd,
        thrust_limits=thrust_limits,
        disturbance_force_func=disturbance_force_func,
    )
    metrics = summarize_altitude_response(result)

    print("Quadcopter Altitude PID Disturbance Response:")
    print("A downward force step pulls the vehicle below target, then the")
    print("controller increases thrust to recover altitude.")
    print()
    print(f"Target altitude: {target_altitude:.3f} m")
    print(f"PID gains: Kp={Kp}, Ki={Ki}, Kd={Kd}")
    print(f"Disturbance time: {disturbance_time:.3f} s")
    print(f"Downward force after disturbance: {disturbance_force_after:.3f} N")
    print(f"Hover thrust: {result['hover_thrust']:.3f} N")
    print(f"Final altitude: {metrics.final_altitude:.3f} m")
    print(f"Final altitude error: {metrics.final_error:.3f} m")
    print(f"Settling time: {_format_optional(metrics.settling_time, ' s')}")
    print(f"Max thrust: {metrics.max_thrust:.3f} N")
    print(f"Min thrust: {metrics.min_thrust:.3f} N")

    _plot_response(result, disturbance_time)


if __name__ == "__main__":
    main()
