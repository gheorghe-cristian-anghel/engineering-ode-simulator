"""Run and plot a resistance sweep for a series RLC step response."""

import sys
from pathlib import Path
from _tkinter import TclError

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.rlc_sweep import run_rlc_sweep


def _format_optional_time(value):
    """Format an optional time value for table output."""
    if value is None:
        return "None"

    return f"{value:.3f}"


def _print_results(results):
    """Print resistance sweep metrics."""
    print("RLC Resistance Sweep:")
    print("Lower R gives lower damping and more oscillation or overshoot.")
    print("Higher R gives stronger damping and less overshoot.")
    print()
    print(
        "R (ohm)   Zeta      Omega_n    Final V    Peak V     "
        "Overshoot   Settling"
    )
    print(
        "          (-)       (rad/s)    (V)        (V)        "
        "(%)         (s)"
    )

    for result in results:
        print(
            f"{result['parameter_value']:<10.3f}"
            f"{result['damping_ratio']:<10.3f}"
            f"{result['natural_frequency']:<11.3f}"
            f"{result['final_voltage']:<11.3f}"
            f"{result['peak_voltage']:<11.3f}"
            f"{result['overshoot_percent']:<12.3f}"
            f"{_format_optional_time(result['settling_time'])}"
        )


def _draw_plot(simulations, vin):
    """Draw capacitor-voltage responses for all resistance values."""
    plt.figure()

    for simulation in simulations:
        label = f"R = {simulation['parameter_value']:.2f} ohm"
        plt.plot(simulation["t"], simulation["capacitor_voltage"], label=label)

    plt.axhline(vin, color="gray", linestyle=":", label="Input voltage")
    plt.title("RLC Capacitor Voltage Response: Resistance Sweep")
    plt.xlabel("Time (s)")
    plt.ylabel("Capacitor voltage (V)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()


def _plot_sweep(simulations, vin):
    """Plot the sweep and keep a graceful fallback for headless environments."""
    try:
        _draw_plot(simulations, vin)
        plt.show()
    except TclError:
        plt.switch_backend("Agg")
        plt.close("all")
        _draw_plot(simulations, vin)
        print("Interactive Matplotlib window is unavailable in this environment.")


def main():
    """Sweep resistance while keeping inductance and capacitance fixed."""
    R_values = [0.5, 1.0, 2.0, 5.0]
    L = 1.0
    C = 0.25
    Vin = 5.0

    results, simulations = run_rlc_sweep(
        "R",
        R_values,
        R=2.0,
        L=L,
        C=C,
        Vin=Vin,
        t_span=(0.0, 12.0),
        num_points=1200,
    )

    _print_results(results)
    _plot_sweep(simulations, Vin)


if __name__ == "__main__":
    main()
