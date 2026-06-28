"""Run and plot closed-loop PI speed control for a DC motor."""

import sys
from pathlib import Path
from _tkinter import TclError

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.step_response import calculate_step_info
from models.dc_motor import rad_per_sec_to_rpm
from models.pid_motor_control import (
    simulate_pi_motor_control,
    speed_reference_step,
    zero_load_torque,
)


def _format_optional(value, unit=""):
    """Format optional metric values for printing."""
    if value is None:
        return "None"

    return f"{value:.3f}{unit}"


def _plot_response(t, current, omega, voltage, target_speed, voltage_min, voltage_max):
    """Plot speed, voltage, and current, falling back to saving if needed."""
    output_path = PROJECT_ROOT / "examples" / "pid_motor_control_response.png"

    try:
        _draw_plots(t, current, omega, voltage, target_speed, voltage_min, voltage_max)
        plt.savefig(output_path, dpi=150)
        plt.show()
    except TclError:
        plt.switch_backend("Agg")
        plt.close("all")
        _draw_plots(t, current, omega, voltage, target_speed, voltage_min, voltage_max)
        plt.savefig(output_path, dpi=150)
        print("Interactive Matplotlib window is unavailable in this environment.")
        print(f"Plot saved to: {output_path}")


def _draw_plots(t, current, omega, voltage, target_speed, voltage_min, voltage_max):
    """Draw closed-loop motor speed, voltage, and current plots."""
    figure, axes = plt.subplots(3, 1, sharex=True)

    axes[0].plot(t, omega, label="Motor speed")
    axes[0].axhline(target_speed, color="gray", linestyle=":", label="Target speed")
    axes[0].set_ylabel("Speed (rad/s)")
    axes[0].set_title("Closed-Loop DC Motor Speed")
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
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Current (A)")
    axes[2].set_title("Armature Current")
    axes[2].grid(True)
    axes[2].legend()

    figure.tight_layout()


def main():
    """Simulate closed-loop DC motor speed control."""
    R = 1.0
    L = 0.5
    J = 0.01
    b = 0.001
    Kt = 0.01
    Ke = 0.01
    target_speed = 80.0
    Kp = 0.5
    Ki = 0.05
    Kd = 0.0
    voltage_min = 0.0
    voltage_max = 24.0
    i0 = 0.0
    omega0 = 0.0
    integral_error0 = 0.0
    t_span = (0, 8)
    num_points = 2000

    reference_func = lambda t: speed_reference_step(t, target_speed)
    load_torque_func = zero_load_torque

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

    step_info = calculate_step_info(t, omega)
    final_error = target_speed - omega[-1]

    print("Closed-Loop DC Motor PI Control:")
    print(f"Target speed: {target_speed:.3f} rad/s")
    print(f"Target speed: {rad_per_sec_to_rpm(target_speed):.3f} rpm")
    print(f"Kp: {Kp}")
    print(f"Ki: {Ki}")
    print(f"Kd: {Kd}")
    print(f"Voltage limits: {voltage_min} V to {voltage_max} V")
    print(f"Final speed: {omega[-1]:.3f} rad/s")
    print(f"Final error: {final_error:.3f} rad/s")
    print(f"Peak speed: {step_info.peak_value:.3f} rad/s")
    print(f"Overshoot: {step_info.overshoot_percent:.3f}%")
    print(f"Settling time: {_format_optional(step_info.settling_time, ' s')}")

    _plot_response(t, current, omega, voltage, target_speed, voltage_min, voltage_max)


if __name__ == "__main__":
    main()
