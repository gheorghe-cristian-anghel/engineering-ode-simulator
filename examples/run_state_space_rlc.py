"""Run a series RLC step response using state-space form."""

import sys
from pathlib import Path
from _tkinter import TclError

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.state_space import rlc_state_space, simulate_state_space, step_input
from models.rlc_circuit import damping_ratio, natural_frequency


def _draw_plots(t, capacitor_voltage, current, input_voltage):
    """Draw capacitor voltage and current plots."""
    figure, axes = plt.subplots(2, 1, sharex=True)

    axes[0].plot(t, capacitor_voltage, label="Capacitor voltage")
    axes[0].axhline(input_voltage, color="gray", linestyle=":", label="Input voltage")
    axes[0].set_ylabel("Voltage (V)")
    axes[0].set_title("RLC State-Space Capacitor Voltage Response")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(t, current, label="Current", color="tab:orange")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Current (A)")
    axes[1].set_title("RLC Current")
    axes[1].grid(True)
    axes[1].legend()

    figure.tight_layout()


def _plot_response(t, capacitor_voltage, current, input_voltage):
    """Plot response, falling back gracefully if Tk is unavailable."""
    try:
        _draw_plots(t, capacitor_voltage, current, input_voltage)
        plt.show()
    except TclError:
        plt.switch_backend("Agg")
        plt.close("all")
        _draw_plots(t, capacitor_voltage, current, input_voltage)
        print("Interactive Matplotlib window is unavailable in this environment.")


def main():
    """Simulate an RLC voltage-step response in state-space form."""
    R = 2.0
    L = 1.0
    C_value = 0.25
    input_voltage = 5.0
    x0 = [0.0, 0.0]
    t_span = (0.0, 10.0)
    num_points = 1000

    A, B, C, D = rlc_state_space(R, L, C_value)
    t, states, output = simulate_state_space(
        A,
        B,
        C,
        D,
        step_input(input_voltage),
        x0,
        t_span,
        num_points,
    )

    capacitor_voltage = output[:, 0]
    current = states[:, 1]

    print("RLC State-Space Response:")
    print(f"Resistance: {R} ohms")
    print(f"Inductance: {L} H")
    print(f"Capacitance: {C_value} F")
    print(f"Input voltage: {input_voltage} V")
    print(f"Natural frequency: {natural_frequency(L, C_value):.3f} rad/s")
    print(f"Damping ratio: {damping_ratio(R, L, C_value):.3f}")
    print(f"Final capacitor voltage: {capacitor_voltage[-1]:.3f} V")
    print(f"Final current: {current[-1]:.3f} A")

    _plot_response(t, capacitor_voltage, current, input_voltage)


if __name__ == "__main__":
    main()
