"""Compare P, PI, and PID control for DC motor speed tuning."""

import sys
from pathlib import Path
from _tkinter import TclError

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.pid_tuning import compare_pid_cases


def _format_optional(value, unit=""):
    """Format optional metric values for printing."""
    if value is None:
        return "None"

    return f"{value:.3f}{unit}"


def _print_results(results):
    """Print a compact PID comparison table."""
    print("P vs PI vs PID DC Motor Speed Control:")
    print("P control may have steady-state error.")
    print("PI control reduces steady-state error.")
    print("PID control can improve transient behavior when tuned well.")
    print()
    print("Controller   Final Speed   Final Error   Overshoot   Settling Time")
    print("             (rad/s)       (rad/s)       (%)         (s)")

    for result in results:
        print(
            f"{result.label:<13}"
            f"{result.final_speed:<14.3f}"
            f"{result.final_error:<14.3f}"
            f"{result.overshoot_percent:<12.3f}"
            f"{_format_optional(result.settling_time):<14}"
        )


def _draw_plots(simulations, target_speed):
    """Draw speed and voltage plots for each controller."""
    figure, axes = plt.subplots(2, 1, sharex=True)

    for label, simulation in simulations.items():
        axes[0].plot(simulation["t"], simulation["speed"], label=label)
        axes[1].plot(simulation["t"], simulation["voltage"], label=label)

    axes[0].axhline(target_speed, color="gray", linestyle=":", label="Target speed")
    axes[0].set_title("P vs PI vs PID Speed Response")
    axes[0].set_ylabel("Speed (rad/s)")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].set_title("Control Voltage")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Voltage (V)")
    axes[1].grid(True)
    axes[1].legend()

    figure.tight_layout()


def _plot_response(simulations, target_speed):
    """Plot comparison results, falling back when Tk is unavailable."""
    try:
        _draw_plots(simulations, target_speed)
        plt.show()
    except TclError:
        plt.switch_backend("Agg")
        plt.close("all")
        _draw_plots(simulations, target_speed)
        print("Interactive Matplotlib window is unavailable in this environment.")


def main():
    """Run the P, PI, and PID tuning comparison."""
    target_speed = 80.0
    cases = [
        {"label": "P", "Kp": 0.16, "Ki": 0.0, "Kd": 0.0},
        {"label": "PI", "Kp": 0.16, "Ki": 0.018, "Kd": 0.0},
        {"label": "PID", "Kp": 0.16, "Ki": 0.018, "Kd": 0.12},
    ]

    results, simulations = compare_pid_cases(
        cases,
        target_speed=target_speed,
        t_span=(0.0, 25.0),
        dt=0.01,
    )

    _print_results(results)
    _plot_response(simulations, target_speed)


if __name__ == "__main__":
    main()
