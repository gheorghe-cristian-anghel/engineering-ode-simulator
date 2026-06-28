"""Run and plot the RL circuit step response example."""

import sys
from pathlib import Path
from _tkinter import TclError

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.rl_circuit import (
    analytical_rl,
    simulate_rl,
    steady_state_current,
    time_constant,
)


def _plot_response(t, numerical_current, analytical_current, current_ss):
    """Plot current response, falling back to saving if Tk is unavailable."""
    output_path = PROJECT_ROOT / "examples" / "rl_circuit_response.png"

    try:
        plt.plot(t, numerical_current, label="Numerical solution")
        plt.plot(t, analytical_current, "--", label="Analytical solution")
        plt.axhline(current_ss, color="gray", linestyle=":", label="Steady state")
        plt.xlabel("Time (s)")
        plt.ylabel("Current (A)")
        plt.title("RL Circuit Step Response")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.show()
    except TclError:
        plt.switch_backend("Agg")
        plt.close("all")
        plt.plot(t, numerical_current, label="Numerical solution")
        plt.plot(t, analytical_current, "--", label="Analytical solution")
        plt.axhline(current_ss, color="gray", linestyle=":", label="Steady state")
        plt.xlabel("Time (s)")
        plt.ylabel("Current (A)")
        plt.title("RL Circuit Step Response")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        print("Interactive Matplotlib window is unavailable in this environment.")
        print(f"Plot saved to: {output_path}")


def main():
    """Simulate an RL circuit and compare numerical and analytical results."""
    R = 10
    L = 2
    Vin = 5
    i0 = 0
    t_span = (0, 1.5)
    num_points = 1000

    tau = time_constant(R, L)
    current_ss = steady_state_current(R, Vin)

    print(f"Resistance R: {R} ohms")
    print(f"Inductance L: {L} H")
    print(f"Input voltage Vin: {Vin} V")
    print(f"Time constant tau: {tau:.3f} s")
    print(f"Steady-state current: {current_ss:.3f} A")

    t, numerical_current = simulate_rl(R, L, Vin, i0, t_span, num_points)
    analytical_current = analytical_rl(t, R, L, Vin, i0)

    _plot_response(t, numerical_current, analytical_current, current_ss)


if __name__ == "__main__":
    main()
