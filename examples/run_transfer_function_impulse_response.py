"""Run and plot impulse responses for transfer function models."""

import sys
from pathlib import Path
from _tkinter import TclError

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.transfer_function import (
    first_order_lowpass_tf,
    second_order_lowpass_tf,
    simulate_impulse_response,
)


def _draw_impulse_responses(models):
    """Draw impulse responses for the supplied transfer function models."""
    figure, axis = plt.subplots()

    for model in models:
        t, y = simulate_impulse_response(model, t_span=(0.0, 8.0), num_points=1000)
        axis.plot(t, y, label=model.name)
        print(f"{model.name}: peak response = {np.max(y):.3f}")

    axis.set_title("Transfer Function Impulse Responses")
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Output")
    axis.grid(True)
    axis.legend()
    figure.tight_layout()


def main():
    """Simulate impulse responses for first- and second-order systems."""
    models = [
        first_order_lowpass_tf(K=1.0, tau=1.0),
        second_order_lowpass_tf(K=1.0, omega_n=4.0, zeta=0.3),
    ]

    print("Transfer Function Impulse Responses:")
    try:
        _draw_impulse_responses(models)
        plt.show()
    except TclError:
        plt.switch_backend("Agg")
        plt.close("all")
        _draw_impulse_responses(models)
        print("Interactive Matplotlib window is unavailable in this environment.")


if __name__ == "__main__":
    main()
