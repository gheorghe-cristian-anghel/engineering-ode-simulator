"""Show how integral gain Ki affects DC motor speed control."""

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
    """Print Ki tuning metrics."""
    print("Ki Tuning for DC Motor Speed Control:")
    print("Higher Ki reduces steady-state error.")
    print("Too much Ki can cause overshoot or slower settling due to integral accumulation.")
    print()
    print("Ki      Final Speed   Final Error   Overshoot   Settling Time   Max Voltage")
    print("        (rad/s)       (rad/s)       (%)         (s)             (V)")

    for result in results:
        print(
            f"{result.Ki:<8.3f}"
            f"{result.final_speed:<14.3f}"
            f"{result.final_error:<14.3f}"
            f"{result.overshoot_percent:<12.3f}"
            f"{_format_optional(result.settling_time):<16}"
            f"{result.max_voltage:.3f}"
        )


def _draw_plots(simulations, target_speed):
    """Draw speed responses for the Ki sweep."""
    figure, axes = plt.subplots(2, 1, sharex=True)

    for label, simulation in simulations.items():
        axes[0].plot(simulation["t"], simulation["speed"], label=label)
        axes[1].plot(simulation["t"], simulation["voltage"], label=label)

    axes[0].axhline(target_speed, color="gray", linestyle=":", label="Target speed")
    axes[0].set_title("Effect of Ki on Motor Speed Response")
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
    """Run an integral gain tuning comparison."""
    target_speed = 80.0
    Kp = 0.16
    Ki_values = [0.0, 0.006, 0.018, 0.04]
    cases = [
        {"label": f"Ki = {Ki}", "Kp": Kp, "Ki": Ki, "Kd": 0.0}
        for Ki in Ki_values
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
