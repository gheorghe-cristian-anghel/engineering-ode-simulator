"""Run and plot PID-controlled quadcopter altitude tracking."""

import sys
from pathlib import Path
from _tkinter import TclError

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.quadcopter_altitude_control import (
    simulate_altitude_pid_control,
    summarize_altitude_response,
)


def _format_optional(value, unit=""):
    """Format optional metric values for printing."""
    if value is None:
        return "None"

    return f"{value:.3f}{unit}"


def _draw_plots(result):
    """Draw altitude, velocity, thrust, and tracking error plots."""
    time = result["time"]
    altitude = result["altitude"]
    velocity = result["velocity"]
    thrust = result["thrust"]
    error = result["error"]
    target_altitude = result["target_altitude"]
    hover = result["hover_thrust"]

    figure, axes = plt.subplots(4, 1, sharex=True)

    axes[0].plot(time, altitude, label="Altitude")
    axes[0].axhline(target_altitude, color="gray", linestyle=":", label="Target")
    axes[0].set_ylabel("Altitude (m)")
    axes[0].set_title("Quadcopter Altitude PID Tracking")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(time, velocity, label="Vertical velocity", color="tab:orange")
    axes[1].axhline(0.0, color="gray", linestyle=":")
    axes[1].set_ylabel("Velocity (m/s)")
    axes[1].set_title("Vertical Velocity")
    axes[1].grid(True)
    axes[1].legend()

    axes[2].plot(time, thrust, label="Thrust command", color="tab:green")
    axes[2].axhline(hover, color="gray", linestyle=":", label="Hover thrust")
    axes[2].set_ylabel("Thrust (N)")
    axes[2].set_title("PID Thrust Command")
    axes[2].grid(True)
    axes[2].legend()

    axes[3].plot(time, error, label="Altitude error", color="tab:red")
    axes[3].axhline(0.0, color="gray", linestyle=":")
    axes[3].set_xlabel("Time (s)")
    axes[3].set_ylabel("Error (m)")
    axes[3].set_title("Altitude Tracking Error")
    axes[3].grid(True)
    axes[3].legend()

    figure.tight_layout()


def _plot_response(result):
    """Plot response, falling back gracefully if Tk is unavailable."""
    output_path = PROJECT_ROOT / "examples" / "quadcopter_altitude_pid.png"

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
    """Simulate PID altitude tracking to a 5 m target."""
    target_altitude = 5.0
    m = 1.0
    g = 9.81
    c_drag = 0.2
    Kp = 2.5
    Ki = 0.8
    Kd = 3.0
    dt = 0.01
    thrust_limits = (0.0, 25.0)

    result = simulate_altitude_pid_control(
        target_altitude=target_altitude,
        z0=0.0,
        v0=0.0,
        t_span=(0.0, 10.0),
        dt=dt,
        m=m,
        g=g,
        c_drag=c_drag,
        Kp=Kp,
        Ki=Ki,
        Kd=Kd,
        thrust_limits=thrust_limits,
    )
    metrics = summarize_altitude_response(result)

    print("Quadcopter Altitude PID Control:")
    print("The PID controller adjusts thrust around hover thrust to climb")
    print("and settle near the target altitude.")
    print()
    print(f"Target altitude: {target_altitude:.3f} m")
    print(f"Mass: {m:.3f} kg")
    print(f"Gravity: {g:.3f} m/s^2")
    print(f"Linear drag coefficient: {c_drag:.3f} N*s/m")
    print(f"PID gains: Kp={Kp}, Ki={Ki}, Kd={Kd}")
    print(f"Sample time dt: {dt:.3f} s")
    print(f"Thrust limits: {thrust_limits[0]:.3f} N to {thrust_limits[1]:.3f} N")
    print(f"Hover thrust: {result['hover_thrust']:.3f} N")
    print(f"Final altitude: {metrics.final_altitude:.3f} m")
    print(f"Final altitude error: {metrics.final_error:.3f} m")
    print(f"Overshoot: {metrics.overshoot_percent:.3f}%")
    print(f"Settling time: {_format_optional(metrics.settling_time, ' s')}")
    print(f"Max thrust: {metrics.max_thrust:.3f} N")
    print(f"Min thrust: {metrics.min_thrust:.3f} N")

    _plot_response(result)


if __name__ == "__main__":
    main()
