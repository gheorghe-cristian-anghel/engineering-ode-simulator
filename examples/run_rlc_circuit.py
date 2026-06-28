"""Run and plot the series RLC circuit step response example."""

import sys
from pathlib import Path
from _tkinter import TclError

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.step_response import calculate_step_info
from models.rlc_circuit import (
    damping_ratio,
    natural_frequency,
    simulate_rlc,
    steady_state_voltage,
)


def response_type(zeta):
    """Return a readable damping classification from damping ratio."""
    if zeta < 1:
        return "underdamped"

    if zeta == 1:
        return "critically damped"

    return "overdamped"


def _format_optional_time(value):
    """Format an optional time value for printing."""
    if value is None:
        return "None"

    return f"{value:.3f} s"


def _plot_response(t, capacitor_voltage, current, Vin):
    """Plot capacitor voltage and current, falling back to saving if needed."""
    output_path = PROJECT_ROOT / "examples" / "rlc_circuit_response.png"

    try:
        _draw_plots(t, capacitor_voltage, current, Vin)
        plt.savefig(output_path, dpi=150)
        plt.show()
    except TclError:
        plt.switch_backend("Agg")
        plt.close("all")
        _draw_plots(t, capacitor_voltage, current, Vin)
        plt.savefig(output_path, dpi=150)
        print("Interactive Matplotlib window is unavailable in this environment.")
        print(f"Plot saved to: {output_path}")


def _draw_plots(t, capacitor_voltage, current, Vin):
    """Draw the RLC voltage and current plots."""
    figure, axes = plt.subplots(2, 1, sharex=True)

    axes[0].plot(t, capacitor_voltage, label="Capacitor voltage")
    axes[0].axhline(
        steady_state_voltage(Vin),
        color="gray",
        linestyle=":",
        label="Steady state",
    )
    axes[0].set_ylabel("Voltage (V)")
    axes[0].set_title("RLC Circuit Capacitor Voltage Step Response")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(t, current, label="Current", color="tab:orange")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Current (A)")
    axes[1].set_title("RLC Circuit Current")
    axes[1].grid(True)
    axes[1].legend()

    figure.tight_layout()


def main():
    """Simulate a series RLC circuit step response."""
    R = 2
    L = 1
    C = 0.25
    Vin = 5
    Vc0 = 0
    i0 = 0
    t_span = (0, 10)
    num_points = 2000

    omega_n = natural_frequency(L, C)
    zeta = damping_ratio(R, L, C)

    t, capacitor_voltage, current = simulate_rlc(
        R,
        L,
        C,
        Vin,
        Vc0,
        i0,
        t_span,
        num_points,
    )
    step_info = calculate_step_info(t, capacitor_voltage)

    print("RLC Parameters:")
    print(f"Resistance: {R} ohms")
    print(f"Inductance: {L} H")
    print(f"Capacitance: {C} F")
    print(f"Input voltage: {Vin} V")
    print(f"Natural frequency: {omega_n:.3f} rad/s")
    print(f"Damping ratio: {zeta:.3f}")
    print(f"Response type: {response_type(zeta)}")
    print()
    print("Step Response Metrics:")
    print(f"Final value: {step_info.final_value:.3f} V")
    print(f"Steady-state estimate: {step_info.steady_state_value:.3f} V")
    print(f"Rise time: {_format_optional_time(step_info.rise_time)}")
    print(f"Settling time: {_format_optional_time(step_info.settling_time)}")
    print(f"Peak voltage: {step_info.peak_value:.3f} V")
    print(f"Peak time: {step_info.peak_time:.3f} s")
    print(f"Overshoot: {step_info.overshoot_percent:.3f}%")

    _plot_response(t, capacitor_voltage, current, Vin)


if __name__ == "__main__":
    main()
