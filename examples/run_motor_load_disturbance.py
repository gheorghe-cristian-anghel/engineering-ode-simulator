"""Run and plot PI-controlled DC motor load disturbance response."""

import sys
from pathlib import Path
from _tkinter import TclError

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.pid_motor_control import (
    load_torque_step,
    simulate_pi_motor_control,
    speed_reference_step,
)


def _disturbance_indices(t, disturbance_time):
    """Return indices for samples near and before the disturbance."""
    disturbance_index = int(np.argmin(np.abs(t - disturbance_time)))
    before_index = max(0, disturbance_index - 1)

    return disturbance_index, before_index


def _print_disturbance_metrics(
    t,
    current,
    omega,
    voltage,
    target_speed,
    disturbance_time,
    load_before,
    load_after,
):
    """Print disturbance rejection metrics."""
    disturbance_index, before_index = _disturbance_indices(t, disturbance_time)
    speed_before = omega[before_index]
    minimum_speed_after = np.min(omega[disturbance_index:])
    maximum_speed_drop = speed_before - minimum_speed_after
    final_speed = omega[-1]
    final_error = target_speed - final_speed

    print("PI Motor Load Disturbance Response:")
    print(f"Target speed: {target_speed:.3f} rad/s")
    print(f"Disturbance time: {disturbance_time:.3f} s")
    print(f"Load torque before disturbance: {load_before:.3f} N*m")
    print(f"Load torque after disturbance: {load_after:.3f} N*m")
    print(f"Speed before disturbance: {speed_before:.3f} rad/s")
    print(f"Minimum speed after disturbance: {minimum_speed_after:.3f} rad/s")
    print(f"Maximum speed drop: {maximum_speed_drop:.3f} rad/s")
    print(f"Final speed: {final_speed:.3f} rad/s")
    print(f"Final tracking error: {final_error:.3f} rad/s")
    print(f"Voltage before disturbance: {voltage[before_index]:.3f} V")
    print(f"Voltage near final recovery: {voltage[-1]:.3f} V")
    print(f"Current before disturbance: {current[before_index]:.3f} A")
    print(f"Current near final recovery: {current[-1]:.3f} A")


def _plot_response(
    t,
    current,
    omega,
    voltage,
    load_torque,
    target_speed,
    disturbance_time,
    voltage_min,
    voltage_max,
):
    """Plot disturbance response, falling back to saving if needed."""
    output_path = PROJECT_ROOT / "examples" / "motor_load_disturbance_response.png"

    try:
        _draw_plots(
            t,
            current,
            omega,
            voltage,
            load_torque,
            target_speed,
            disturbance_time,
            voltage_min,
            voltage_max,
        )
        plt.savefig(output_path, dpi=150)
        plt.show()
    except TclError:
        plt.switch_backend("Agg")
        plt.close("all")
        _draw_plots(
            t,
            current,
            omega,
            voltage,
            load_torque,
            target_speed,
            disturbance_time,
            voltage_min,
            voltage_max,
        )
        plt.savefig(output_path, dpi=150)
        print("Interactive Matplotlib window is unavailable in this environment.")
        print(f"Plot saved to: {output_path}")


def _draw_plots(
    t,
    current,
    omega,
    voltage,
    load_torque,
    target_speed,
    disturbance_time,
    voltage_min,
    voltage_max,
):
    """Draw speed, voltage, current, and load torque plots."""
    figure, axes = plt.subplots(4, 1, sharex=True)

    axes[0].plot(t, omega, label="Motor speed")
    axes[0].axhline(target_speed, color="gray", linestyle=":", label="Target speed")
    axes[0].set_ylabel("Speed (rad/s)")
    axes[0].set_title("PI Motor Speed with Load Disturbance")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(t, voltage, label="Control voltage", color="tab:orange")
    axes[1].axhline(voltage_max, color="gray", linestyle=":", label="Voltage limits")
    axes[1].axhline(voltage_min, color="gray", linestyle=":")
    axes[1].set_ylabel("Voltage (V)")
    axes[1].set_title("PI Control Voltage")
    axes[1].grid(True)
    axes[1].legend()

    axes[2].plot(t, current, label="Armature current", color="tab:green")
    axes[2].set_ylabel("Current (A)")
    axes[2].set_title("Armature Current")
    axes[2].grid(True)
    axes[2].legend()

    axes[3].plot(t, load_torque, label="Load torque", color="tab:red")
    axes[3].set_xlabel("Time (s)")
    axes[3].set_ylabel("Torque (N*m)")
    axes[3].set_title("Load Torque Disturbance")
    axes[3].grid(True)
    axes[3].legend()

    for axis in axes[:3]:
        axis.axvline(
            disturbance_time,
            color="black",
            linestyle="--",
            label="Disturbance",
        )

    figure.tight_layout()


def main():
    """Simulate PI motor speed control with a load torque disturbance."""
    R = 1.0
    L = 0.5
    J = 0.01
    b = 0.001
    Kt = 0.01
    Ke = 0.01
    target_speed = 80.0
    Kp = 0.3
    Ki = 0.04
    Kd = 0.0
    voltage_min = 0.0
    voltage_max = 24.0
    i0 = 0.0
    omega0 = 0.0
    integral_error0 = 0.0
    t_span = (0.0, 25.0)
    num_points = 3000
    disturbance_time = 12.0
    load_before = 0.0
    load_after = 0.03

    reference_func = lambda t: speed_reference_step(t, target_speed)
    load_torque_func = lambda t: load_torque_step(
        t,
        disturbance_time,
        load_before,
        load_after,
    )

    t, current, omega, _, voltage = simulate_pi_motor_control(
        R,
        L,
        J,
        b,
        Kt,
        Ke,
        Kp,
        Ki,
        i0,
        omega0,
        integral_error0,
        t_span,
        num_points,
        reference_func=reference_func,
        load_torque_func=load_torque_func,
        voltage_min=voltage_min,
        voltage_max=voltage_max,
        Kd=Kd,
    )

    load_torque = np.array([load_torque_func(sample_time) for sample_time in t])

    _print_disturbance_metrics(
        t,
        current,
        omega,
        voltage,
        target_speed,
        disturbance_time,
        load_before,
        load_after,
    )
    _plot_response(
        t,
        current,
        omega,
        voltage,
        load_torque,
        target_speed,
        disturbance_time,
        voltage_min,
        voltage_max,
    )


if __name__ == "__main__":
    main()
