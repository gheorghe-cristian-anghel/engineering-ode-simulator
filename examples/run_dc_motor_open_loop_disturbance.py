"""Run and plot an open-loop DC motor load disturbance response."""

import sys
from pathlib import Path
from _tkinter import TclError

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.motor_disturbance import (
    load_torque_step,
    summarize_motor_disturbance_response,
)
from models.dc_motor import simulate_dc_motor, steady_state_current, steady_state_speed


def _format_optional(value, unit=""):
    """Format optional metric values for printing."""
    if value is None:
        return "None"

    return f"{value:.3f}{unit}"


def _print_metrics(metrics, fixed_voltage, load_before, load_after):
    """Print open-loop disturbance response metrics."""
    print("Open-Loop DC Motor Load Disturbance:")
    print(
        "Without feedback control, the motor settles to a lower speed after "
        "the load torque increases."
    )
    print()
    print(f"Fixed voltage: {fixed_voltage:.3f} V")
    print(f"Load torque step: {load_before:.3f} to {load_after:.3f} N*m")
    print(f"Speed before disturbance: {metrics['speed_before_disturbance']:.3f} rad/s")
    print(
        "Minimum speed after disturbance: "
        f"{metrics['minimum_speed_after_disturbance']:.3f} rad/s"
    )
    print(f"Speed drop: {metrics['speed_drop']:.3f} rad/s")
    print(f"Final speed: {metrics['final_speed']:.3f} rad/s")
    print(f"Final error from reference: {metrics['final_error']:.3f} rad/s")
    print(f"Recovery time: {_format_optional(metrics['recovery_time'], ' s')}")
    print(f"Current before disturbance: {metrics['current_before_disturbance']:.3f} A")
    print(f"Current final: {metrics['current_final']:.3f} A")


def _draw_plots(t, speed, current, load_torque, target_speed, disturbance_time):
    """Draw speed, current, and load torque for the open-loop response."""
    figure, axes = plt.subplots(3, 1, sharex=True)

    axes[0].plot(t, speed, label="Open-loop speed")
    axes[0].axhline(target_speed, color="gray", linestyle=":", label="Reference speed")
    axes[0].axvline(disturbance_time, color="black", linestyle="--", label="Disturbance")
    axes[0].set_ylabel("Speed (rad/s)")
    axes[0].set_title("Open-Loop DC Motor Speed with Load Disturbance")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(t, current, label="Armature current", color="tab:orange")
    axes[1].axvline(disturbance_time, color="black", linestyle="--", label="Disturbance")
    axes[1].set_ylabel("Current (A)")
    axes[1].set_title("Armature Current")
    axes[1].grid(True)
    axes[1].legend()

    axes[2].plot(t, load_torque, label="Load torque", color="tab:red")
    axes[2].axvline(disturbance_time, color="black", linestyle="--", label="Disturbance")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Torque (N*m)")
    axes[2].set_title("Step Load Torque")
    axes[2].grid(True)
    axes[2].legend()

    figure.tight_layout()


def _plot_response(t, speed, current, load_torque, target_speed, disturbance_time):
    """Plot the response, falling back gracefully if Tk is unavailable."""
    try:
        _draw_plots(t, speed, current, load_torque, target_speed, disturbance_time)
        plt.show()
    except TclError:
        plt.switch_backend("Agg")
        plt.close("all")
        _draw_plots(t, speed, current, load_torque, target_speed, disturbance_time)
        print("Interactive Matplotlib window is unavailable in this environment.")


def main():
    """Simulate an open-loop DC motor under a load torque step."""
    R = 1.0
    L = 0.5
    J = 0.01
    b = 0.001
    Kt = 0.01
    Ke = 0.01
    target_speed = 80.0
    load_before = 0.0
    load_after = 0.03
    disturbance_time = 12.0
    t_span = (0.0, 25.0)
    num_points = 3000

    fixed_voltage = target_speed * (Ke + R * b / Kt)
    voltage_func = lambda t: fixed_voltage
    load_torque_func = lambda t: load_torque_step(
        t,
        step_time=disturbance_time,
        initial=load_before,
        final=load_after,
    )

    no_load_speed = steady_state_speed(R, b, Kt, Ke, fixed_voltage, load_before)
    no_load_current = steady_state_current(b, Kt, no_load_speed, load_before)
    loaded_speed = steady_state_speed(R, b, Kt, Ke, fixed_voltage, load_after)
    loaded_current = steady_state_current(b, Kt, loaded_speed, load_after)

    t, current, speed = simulate_dc_motor(
        R,
        L,
        J,
        b,
        Kt,
        Ke,
        no_load_current,
        no_load_speed,
        t_span,
        num_points,
        voltage_func=voltage_func,
        load_torque_func=load_torque_func,
    )
    load_torque = np.array([load_torque_func(sample_time) for sample_time in t])
    voltage = np.full_like(t, fixed_voltage)
    metrics = summarize_motor_disturbance_response(
        t,
        speed,
        target_speed,
        disturbance_time,
        voltage=voltage,
        current=current,
    )

    print(f"No-load steady speed estimate: {no_load_speed:.3f} rad/s")
    print(f"Loaded steady speed estimate: {loaded_speed:.3f} rad/s")
    print(f"Loaded steady current estimate: {loaded_current:.3f} A")
    print()
    _print_metrics(metrics, fixed_voltage, load_before, load_after)
    _plot_response(t, speed, current, load_torque, target_speed, disturbance_time)


if __name__ == "__main__":
    main()
