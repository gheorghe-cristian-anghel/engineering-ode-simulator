"""Run and plot discrete PID disturbance rejection for a DC motor."""

import sys
from pathlib import Path
from _tkinter import TclError

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
SCREENSHOT_DIR = PROJECT_ROOT / "docs" / "screenshots"
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

from models.discrete_pid import (
    simulate_discrete_pid_motor_with_disturbance,
    summarize_disturbance_response,
)


def _format_optional(value, unit=""):
    """Format optional metric values for printing."""
    if value is None:
        return "None"

    return f"{value:.3f}{unit}"


def _print_metrics(metrics):
    """Print a compact disturbance response metrics table."""
    print("Discrete PID DC Motor Load Disturbance Response:")
    print(
        "The load disturbance causes speed to drop, then the discrete PID "
        "controller increases voltage/current to recover toward the target speed."
    )
    print()
    print("Metric                              Value")
    print(f"Speed before disturbance            {metrics['speed_before_disturbance']:.3f} rad/s")
    print(
        "Minimum speed after disturbance     "
        f"{metrics['minimum_speed_after_disturbance']:.3f} rad/s"
    )
    print(f"Maximum speed drop                   {metrics['speed_drop']:.3f} rad/s")
    print(f"Final speed                          {metrics['final_speed']:.3f} rad/s")
    print(f"Final error                          {metrics['final_error']:.3f} rad/s")
    print(
        "Voltage before disturbance          "
        f"{metrics['voltage_before_disturbance']:.3f} V"
    )
    print(
        "Voltage after recovery/final        "
        f"{metrics['voltage_after_recovery_or_final']:.3f} V"
    )
    print(
        "Current before disturbance          "
        f"{metrics['current_before_disturbance']:.3f} A"
    )
    print(
        "Current after recovery/final        "
        f"{metrics['current_after_recovery_or_final']:.3f} A"
    )
    print(f"Recovery time                        {_format_optional(metrics['recovery_time'], ' s')}")


def _plot_response(result, disturbance_time, voltage_limits):
    """Plot disturbance response, falling back to saving if needed."""
    output_path = PROJECT_ROOT / "examples" / "discrete_pid_disturbance_response.png"
    screenshot_path = SCREENSHOT_DIR / "discrete_pid_disturbance_response.png"

    try:
        _draw_plots(result, disturbance_time, voltage_limits)
        plt.savefig(screenshot_path, dpi=150, bbox_inches="tight")
        print(f"Saved screenshot to {screenshot_path.relative_to(PROJECT_ROOT)}")
        plt.savefig(output_path, dpi=150)
        plt.show()
    except TclError:
        plt.switch_backend("Agg")
        plt.close("all")
        _draw_plots(result, disturbance_time, voltage_limits)
        plt.savefig(screenshot_path, dpi=150, bbox_inches="tight")
        print(f"Saved screenshot to {screenshot_path.relative_to(PROJECT_ROOT)}")
        plt.savefig(output_path, dpi=150)
        print("Interactive Matplotlib window is unavailable in this environment.")
        print(f"Plot saved to: {output_path}")


def _draw_plots(result, disturbance_time, voltage_limits):
    """Draw speed, voltage, load torque, and current plots."""
    time = result["time"]
    speed = result["speed"]
    voltage = result["voltage"]
    load_torque = result["load_torque"]
    current = result["current"]
    target_speed = result["target_speed"]
    voltage_min, voltage_max = voltage_limits

    figure, axes = plt.subplots(4, 1, sharex=True)

    axes[0].plot(time, speed, label="Motor speed")
    axes[0].axhline(target_speed, color="gray", linestyle=":", label="Target speed")
    axes[0].axvline(
        disturbance_time,
        color="black",
        linestyle="--",
        label="Disturbance",
    )
    axes[0].set_ylabel("Speed (rad/s)")
    axes[0].set_title("Discrete PID Speed with Load Disturbance")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(time, voltage, label="Control voltage", color="tab:orange")
    axes[1].axhline(voltage_max, color="gray", linestyle=":", label="Voltage limits")
    axes[1].axhline(voltage_min, color="gray", linestyle=":")
    axes[1].axvline(
        disturbance_time,
        color="black",
        linestyle="--",
        label="Disturbance",
    )
    axes[1].set_ylabel("Voltage (V)")
    axes[1].set_title("Held Voltage Command")
    axes[1].grid(True)
    axes[1].legend()

    axes[2].plot(time, load_torque, label="Load torque", color="tab:red")
    axes[2].axvline(
        disturbance_time,
        color="black",
        linestyle="--",
        label="Disturbance",
    )
    axes[2].set_ylabel("Torque (N*m)")
    axes[2].set_title("Step Load Torque Disturbance")
    axes[2].grid(True)
    axes[2].legend()

    axes[3].plot(time, current, label="Armature current", color="tab:green")
    axes[3].axvline(
        disturbance_time,
        color="black",
        linestyle="--",
        label="Disturbance",
    )
    axes[3].set_xlabel("Time (s)")
    axes[3].set_ylabel("Current (A)")
    axes[3].set_title("Armature Current")
    axes[3].grid(True)
    axes[3].legend()

    figure.tight_layout()


def main():
    """Simulate discrete PID motor speed control with load torque disturbance."""
    target_speed = 80.0
    Kp = 0.5
    Ki = 0.2
    Kd = 0.02
    dt = 0.01
    t_span = (0.0, 25.0)
    disturbance_time = 12.0
    load_torque_initial = 0.0
    load_torque_final = 0.03
    voltage_limits = (0.0, 24.0)

    result = simulate_discrete_pid_motor_with_disturbance(
        target_speed=target_speed,
        Kp=Kp,
        Ki=Ki,
        Kd=Kd,
        dt=dt,
        t_span=t_span,
        disturbance_time=disturbance_time,
        load_torque_initial=load_torque_initial,
        load_torque_final=load_torque_final,
        voltage_limits=voltage_limits,
    )
    metrics = summarize_disturbance_response(
        result,
        disturbance_time=disturbance_time,
        target_speed=target_speed,
    )

    print(f"Target speed: {target_speed:.3f} rad/s")
    print(f"PID gains: Kp={Kp}, Ki={Ki}, Kd={Kd}")
    print(f"Sample time dt: {dt:.3f} s")
    print(f"Disturbance time: {disturbance_time:.3f} s")
    print(f"Load torque step: {load_torque_initial:.3f} to {load_torque_final:.3f} N*m")
    print(f"Voltage limits: {voltage_limits[0]:.3f} V to {voltage_limits[1]:.3f} V")
    print()
    _print_metrics(metrics)
    _plot_response(result, disturbance_time, voltage_limits)


if __name__ == "__main__":
    main()
