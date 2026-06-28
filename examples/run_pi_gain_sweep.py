"""Run and plot a PI motor speed-control gain sweep."""

import sys
from pathlib import Path
from _tkinter import TclError

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.parameter_sweep import run_pi_motor_gain_sweep


def _format_optional(value, unit=""):
    """Format optional metric values for printing."""
    if value is None:
        return "None"

    return f"{value:.3f}{unit}"


def _print_results(results):
    """Print a comparison table for PI gain sweep results."""
    print("PI Motor Gain Sweep:")
    print(
        "Kp     Final Speed   Final Error   Peak Speed   Overshoot   "
        "Settling Time   Max Voltage   Max Current"
    )
    print(
        "       (rad/s)       (rad/s)       (rad/s)      (%)         "
        "(s)             (V)           (A)"
    )

    for result in results:
        print(
            f"{result.parameter_value:<6.2f}"
            f"{result.final_value:<14.3f}"
            f"{result.final_error:<14.3f}"
            f"{result.peak_value:<13.3f}"
            f"{result.overshoot_percent:<12.3f}"
            f"{_format_optional(result.settling_time):<16}"
            f"{result.max_control_effort:<14.3f}"
            f"{result.max_current:.3f}"
        )


def _plot_sweep(simulations, target_speed, voltage_min, voltage_max):
    """Plot speed and voltage responses for all sweep runs."""
    output_path = PROJECT_ROOT / "examples" / "pi_gain_sweep_response.png"

    try:
        _draw_plots(simulations, target_speed, voltage_min, voltage_max)
        plt.savefig(output_path, dpi=150)
        plt.show()
    except TclError:
        plt.switch_backend("Agg")
        plt.close("all")
        _draw_plots(simulations, target_speed, voltage_min, voltage_max)
        plt.savefig(output_path, dpi=150)
        print("Interactive Matplotlib window is unavailable in this environment.")
        print(f"Plot saved to: {output_path}")


def _draw_plots(simulations, target_speed, voltage_min, voltage_max):
    """Draw PI gain sweep speed and voltage plots."""
    figure, axes = plt.subplots(2, 1, sharex=True)

    for parameter_value, simulation in simulations.items():
        label = f"Kp = {parameter_value}"
        axes[0].plot(simulation["t"], simulation["speed"], label=label)
        axes[1].plot(simulation["t"], simulation["voltage"], label=label)

    axes[0].axhline(target_speed, color="gray", linestyle=":", label="Target speed")
    axes[0].set_ylabel("Speed (rad/s)")
    axes[0].set_title("PI Motor Speed Response for Kp Sweep")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].axhline(voltage_max, color="gray", linestyle=":", label="Voltage limits")
    axes[1].axhline(voltage_min, color="gray", linestyle=":")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Voltage (V)")
    axes[1].set_title("PI Control Voltage for Kp Sweep")
    axes[1].grid(True)
    axes[1].legend()

    figure.tight_layout()


def main():
    """Sweep Kp values for closed-loop PI motor speed control."""
    motor_params = {
        "R": 1.0,
        "L": 0.5,
        "J": 0.01,
        "b": 0.001,
        "Kt": 0.01,
        "Ke": 0.01,
    }
    controller_params = {
        "Kp": 0.2,
        "Ki": 0.04,
        "Kd": 0.0,
        "target_speed": 80.0,
        "voltage_min": 0.0,
        "voltage_max": 24.0,
    }
    simulation_params = {
        "i0": 0.0,
        "omega0": 0.0,
        "integral_error0": 0.0,
        "t_span": (0.0, 25.0),
        "num_points": 3000,
    }
    parameter_name = "Kp"
    parameter_values = [0.10, 0.15, 0.20, 0.25, 0.30]

    results, simulations = run_pi_motor_gain_sweep(
        parameter_name,
        parameter_values,
        motor_params,
        controller_params,
        simulation_params,
    )

    _print_results(results)
    _plot_sweep(
        simulations,
        controller_params["target_speed"],
        controller_params["voltage_min"],
        controller_params["voltage_max"],
    )


if __name__ == "__main__":
    main()
