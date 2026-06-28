"""Run and plot discrete PID speed control for a DC motor."""

import sys
from pathlib import Path
from _tkinter import TclError

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.export_utils import export_simulation_to_csv
from models.discrete_pid import DiscretePID, simulate_discrete_pid_motor_control


def _plot_response(t, current, speed, voltage, error, target_speed, output_min, output_max):
    """Plot discrete PID motor response, falling back to saving if needed."""
    output_path = PROJECT_ROOT / "examples" / "discrete_pid_motor_response.png"

    try:
        _draw_plots(t, current, speed, voltage, error, target_speed, output_min, output_max)
        plt.savefig(output_path, dpi=150)
        plt.show()
    except TclError:
        plt.switch_backend("Agg")
        plt.close("all")
        _draw_plots(t, current, speed, voltage, error, target_speed, output_min, output_max)
        plt.savefig(output_path, dpi=150)
        print("Interactive Matplotlib window is unavailable in this environment.")
        print(f"Plot saved to: {output_path}")


def _draw_plots(t, current, speed, voltage, error, target_speed, output_min, output_max):
    """Draw speed, voltage, current, and tracking error plots."""
    figure, axes = plt.subplots(4, 1, sharex=True)

    axes[0].plot(t, speed, label="Motor speed")
    axes[0].axhline(target_speed, color="gray", linestyle=":", label="Target speed")
    axes[0].set_ylabel("Speed (rad/s)")
    axes[0].set_title("Discrete PID DC Motor Speed Control")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(t, voltage, label="Control voltage", color="tab:orange")
    axes[1].axhline(output_max, color="gray", linestyle=":", label="Voltage limits")
    axes[1].axhline(output_min, color="gray", linestyle=":")
    axes[1].set_ylabel("Voltage (V)")
    axes[1].set_title("Held Voltage Command")
    axes[1].grid(True)
    axes[1].legend()

    axes[2].plot(t, current, label="Armature current", color="tab:green")
    axes[2].set_ylabel("Current (A)")
    axes[2].set_title("Armature Current")
    axes[2].grid(True)
    axes[2].legend()

    axes[3].plot(t, error, label="Tracking error", color="tab:red")
    axes[3].axhline(0.0, color="gray", linestyle=":")
    axes[3].set_xlabel("Time (s)")
    axes[3].set_ylabel("Error (rad/s)")
    axes[3].set_title("Speed Tracking Error")
    axes[3].grid(True)
    axes[3].legend()

    figure.tight_layout()


def main():
    """Simulate DC motor speed control with a discrete PID controller."""
    R = 1.0
    L = 0.5
    J = 0.01
    b = 0.001
    Kt = 0.01
    Ke = 0.01
    target_speed = 80.0
    Kp = 0.16
    Ki = 0.018
    Kd = 0.12
    output_min = 0.0
    output_max = 24.0
    anti_windup = True
    i0 = 0.0
    omega0 = 0.0
    t_final = 25.0
    dt = 0.01

    pid = DiscretePID(
        Kp,
        Ki,
        Kd,
        output_min=output_min,
        output_max=output_max,
        anti_windup=anti_windup,
    )

    t, current, speed, voltage, error = simulate_discrete_pid_motor_control(
        R,
        L,
        J,
        b,
        Kt,
        Ke,
        pid,
        target_speed,
        i0,
        omega0,
        t_final,
        dt,
    )

    final_error = target_speed - speed[-1]
    peak_speed = np.max(speed)
    overshoot_percent = max(0.0, (peak_speed - target_speed) / target_speed * 100)

    print("Discrete PID DC Motor Speed Control:")
    print(f"Target speed: {target_speed:.3f} rad/s")
    print(f"Kp: {Kp}")
    print(f"Ki: {Ki}")
    print(f"Kd: {Kd}")
    print(f"Sample time dt: {dt:.3f} s")
    print(f"Voltage limits: {output_min} V to {output_max} V")
    print(f"Final speed: {speed[-1]:.3f} rad/s")
    print(f"Final tracking error: {final_error:.3f} rad/s")
    print(f"Peak speed: {peak_speed:.3f} rad/s")
    print(f"Maximum overshoot: {overshoot_percent:.3f}%")
    print(f"Max voltage: {np.max(voltage):.3f} V")
    print(f"Max current: {np.max(current):.3f} A")

    csv_path = export_simulation_to_csv(
        PROJECT_ROOT / "outputs" / "discrete_pid_motor.csv",
        {
            "time_s": t,
            "speed_rad_s": speed,
            "voltage_v": voltage,
            "current_a": current,
            "error_rad_s": error,
        },
    )
    print(f"Exported simulation data to {csv_path.relative_to(PROJECT_ROOT)}")

    _plot_response(t, current, speed, voltage, error, target_speed, output_min, output_max)


if __name__ == "__main__":
    main()
