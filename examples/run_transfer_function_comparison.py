"""Compare transfer function responses for different parameter values."""

import sys
from pathlib import Path
from _tkinter import TclError

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.transfer_function import (
    second_order_lowpass_tf,
    simulate_step_response,
)


def _draw_damping_comparison(omega_n, damping_ratios):
    """Draw second-order step responses for several damping ratios."""
    figure, axis = plt.subplots()

    for zeta in damping_ratios:
        model = second_order_lowpass_tf(K=1.0, omega_n=omega_n, zeta=zeta)
        t, y = simulate_step_response(model, t_span=(0.0, 8.0), num_points=1000)
        axis.plot(t, y, label=f"zeta = {zeta}")
        print(f"zeta = {zeta}: final value = {y[-1]:.3f}, peak = {np.max(y):.3f}")

    axis.set_title("Effect of Damping Ratio on Step Response")
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Output")
    axis.grid(True)
    axis.legend()
    figure.tight_layout()


def main():
    """Compare how damping ratio changes second-order step response."""
    omega_n = 4.0
    damping_ratios = [0.2, 0.5, 1.0]

    print("Second-Order Damping Ratio Comparison:")
    try:
        _draw_damping_comparison(omega_n, damping_ratios)
        plt.show()
    except TclError:
        plt.switch_backend("Agg")
        plt.close("all")
        _draw_damping_comparison(omega_n, damping_ratios)
        print("Interactive Matplotlib window is unavailable in this environment.")


if __name__ == "__main__":
    main()
